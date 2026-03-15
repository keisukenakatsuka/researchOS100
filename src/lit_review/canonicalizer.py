# src/lit_review/canonicalizer.py
"""Claim Canonicalization (086).

Collects run-local Claims from multiple Block 3 runs, groups
semantically identical claims, and generates canonical Claims
for the KML Claims DB.

Two-stage dedup:
  Stage 1: Jaccard similarity pre-filter (no LLM)
  Stage 2: LLM semantic grouping

Usage::

    from src.lit_review.canonicalizer import canonicalize

    result = canonicalize(run_dirs, llm_client=client)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class CollectedClaim:
    """A claim collected from a run's lit_review.json."""
    claim_id: str
    statement: str
    category: str  # established | emerging | contested
    run_id: str
    rq_title: str
    supporting_papers: List[str] = field(default_factory=list)
    evidence_summary: str = ""


@dataclass
class CanonicalClaim:
    """A canonical (deduplicated) claim."""
    canonical_id: str
    canonical_statement: str
    confidence: str
    confidence_reason: str
    member_claim_ids: List[str] = field(default_factory=list)
    supporting_runs: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    majority_category: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_claims_db_record(self, now_iso: str) -> Dict[str, Any]:
        """Convert to format expected by claims_repo.upsert_claim()."""
        return {
            "claim_id": self.canonical_id,
            "statement": self.canonical_statement,
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "tags": self.tags,
            "created_at": now_iso,
        }


@dataclass
class CanonicalizationResult:
    """Result of canonicalization."""
    canonicalization_id: str
    input_run_ids: List[str] = field(default_factory=list)
    total_input_claims: int = 0
    groups_formed: int = 0
    singletons: int = 0
    canonical_claims_total: int = 0
    canonical_claims: List[CanonicalClaim] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["canonical_claims"] = [c.to_dict() for c in self.canonical_claims]
        return d

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# Step 1: Collect Claims
# ------------------------------------------------------------------

def collect_claims(run_dirs: List[Path]) -> List[CollectedClaim]:
    """Collect claims from multiple runs' lit_review.json files."""
    all_claims: List[CollectedClaim] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        lr_path = run_dir / "lit_review.json"
        if not lr_path.exists():
            logger.warning("lit_review.json not found in %s, skipping", run_id)
            continue

        lr = json.loads(lr_path.read_text())
        rq_title = lr.get("rq_context", {}).get("title", "")
        findings = lr.get("empirical_findings", {})

        for i, f in enumerate(findings.get("established", [])):
            all_claims.append(CollectedClaim(
                claim_id=f"{run_id}__cl_established_{i:03d}",
                statement=f.get("statement", ""),
                category="established",
                run_id=run_id,
                rq_title=rq_title,
                supporting_papers=f.get("supporting_papers", []),
                evidence_summary=f.get("evidence_summary", ""),
            ))

        for i, f in enumerate(findings.get("emerging", [])):
            all_claims.append(CollectedClaim(
                claim_id=f"{run_id}__cl_emerging_{i:03d}",
                statement=f.get("statement", ""),
                category="emerging",
                run_id=run_id,
                rq_title=rq_title,
                supporting_papers=f.get("supporting_papers", []),
                evidence_summary=f.get("evidence_summary", ""),
            ))

        for i, c in enumerate(findings.get("contested", [])):
            for j, pos in enumerate(c.get("positions", [])):
                all_claims.append(CollectedClaim(
                    claim_id=f"{run_id}__cl_contested_{i * 10 + j:03d}",
                    statement=pos.get("statement", ""),
                    category="contested",
                    run_id=run_id,
                    rq_title=rq_title,
                ))

    logger.info("Collected %d claims from %d runs", len(all_claims), len(run_dirs))
    return all_claims


# ------------------------------------------------------------------
# Step 2: Semantic Grouping
# ------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text


def _tokenize(text: str) -> Set[str]:
    """Simple word tokenization for Jaccard."""
    return set(_normalize_text(text).split())


def _jaccard(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def prefilter_candidates(
    claims: List[CollectedClaim],
    threshold: float = 0.2,
) -> List[Tuple[int, int, float]]:
    """Stage 1: Find candidate pairs by Jaccard similarity."""
    pairs = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            sim = _jaccard(claims[i].statement, claims[j].statement)
            if sim >= threshold:
                pairs.append((i, j, sim))
    logger.info("Stage 1: %d candidate pairs (threshold=%.2f) from %d claims",
                len(pairs), threshold, len(claims))
    return pairs


def _parse_json_response(text: str) -> Optional[Any]:
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


_GROUPING_SYSTEM = """\
あなたは学術的な主張の同一性を判定する専門家です。
複数の Research Run から生成された Claims のリストが与えられます。
意味的に同じ知見を述べている Claims をグループ化してください。

重要な指示:
- 表現が異なっていても、本質的に同じ主張であればグループ化してください
- 異なる知見は絶対に統合しないでください
- 各グループに対して、正規化された canonical statement を日本語で生成してください
- 統合されなかった Claims は singletons として出力してください"""


def group_claims(
    claims: List[CollectedClaim],
    *,
    llm_client: Any,
) -> Optional[Dict]:
    """Stage 2: LLM-based semantic grouping."""
    claims_input = []
    for i, c in enumerate(claims):
        claims_input.append({
            "index": i,
            "claim_id": c.claim_id,
            "statement": c.statement,
            "category": c.category,
            "run_id": c.run_id,
            "rq_title": c.rq_title[:50],
        })

    user_msg = (
        f"## Claims ({len(claims)} 件)\n\n"
        f"{json.dumps(claims_input, ensure_ascii=False, indent=2)}\n\n"
        f"## 指示\n"
        f"上記の Claims を意味的にグルーピングしてください。\n"
        f"同じ知見を述べている Claims は 1 つのグループにまとめてください。\n"
        f"グループ化されない Claims は singletons として出力してください。\n\n"
        f"出力形式 (JSON):\n"
        f'{{\n'
        f'  "groups": [\n'
        f'    {{\n'
        f'      "canonical_statement": "正規化された主張（日本語）",\n'
        f'      "member_indices": [0, 5],\n'
        f'      "rationale": "なぜこれらが同一知見か"\n'
        f'    }}\n'
        f'  ],\n'
        f'  "singletons": [\n'
        f'    {{\n'
        f'      "canonical_statement": "正規化された主張（日本語）",\n'
        f'      "member_indices": [2]\n'
        f'    }}\n'
        f'  ]\n'
        f'}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _GROUPING_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Grouping LLM call failed: %s", e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Grouping LLM: in=%d, out=%d tokens", usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    parsed = _parse_json_response(resp_text)
    if not parsed:
        logger.error("Grouping JSON parse failed")
    return parsed


# ------------------------------------------------------------------
# Step 3: Canonical Claim Generation
# ------------------------------------------------------------------

def canonical_claim_id(statement: str) -> str:
    """Generate canonical Claim ID from normalized statement.

    Same canonical statement always produces the same ID,
    enabling idempotent upsert.
    """
    normalized = statement.strip().lower()
    content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:12]
    return f"canonical__{content_hash}"


def determine_confidence(
    supporting_runs: int,
    categories: List[str],
) -> str:
    """Determine canonical claim confidence.

    Conservative: high only when 2+ runs AND established is included.
    """
    has_established = "established" in categories
    is_contested_only = all(c == "contested" for c in categories)

    if is_contested_only:
        return "low"
    if supporting_runs >= 2 and has_established:
        return "high"
    return "medium"


def _majority_category(categories: List[str]) -> str:
    """Determine the majority category."""
    from collections import Counter
    if not categories:
        return "unknown"
    counts = Counter(categories)
    return counts.most_common(1)[0][0]


def generate_canonical_claims(
    grouping_result: Dict,
    claims: List[CollectedClaim],
    now_iso: str,
) -> List[CanonicalClaim]:
    """Generate canonical claims from grouping result."""
    canonical_claims = []

    for group_type in ["groups", "singletons"]:
        for group in grouping_result.get(group_type, []):
            canonical_stmt = group.get("canonical_statement", "")
            if not canonical_stmt:
                continue

            member_indices = group.get("member_indices", [])
            members = [claims[i] for i in member_indices if i < len(claims)]
            if not members:
                continue

            member_ids = [m.claim_id for m in members]
            run_ids = list(set(m.run_id for m in members))
            categories = [m.category for m in members]
            maj_cat = _majority_category(categories)

            confidence = determine_confidence(len(run_ids), categories)

            # Build confidence reason
            reason_parts = [
                f"Supported by {len(run_ids)} run(s): {', '.join(run_ids)}",
                f"Categories: {', '.join(categories)}",
                f"Member claims: {', '.join(member_ids)}",
            ]
            confidence_reason = "\n".join(reason_parts)

            tags = ["canonical", "block3", "cross_run", maj_cat]

            canonical_claims.append(CanonicalClaim(
                canonical_id=canonical_claim_id(canonical_stmt),
                canonical_statement=canonical_stmt,
                confidence=confidence,
                confidence_reason=confidence_reason[:2000],
                member_claim_ids=member_ids,
                supporting_runs=run_ids,
                categories=categories,
                majority_category=maj_cat,
                tags=tags,
            ))

    logger.info("Generated %d canonical claims (%d groups, %d singletons)",
                len(canonical_claims),
                len(grouping_result.get("groups", [])),
                len(grouping_result.get("singletons", [])))
    return canonical_claims


# ------------------------------------------------------------------
# Step 4: Claims DB Writeback
# ------------------------------------------------------------------

def write_canonical_claims(
    canonical_claims: List[CanonicalClaim],
    *,
    claims_repo: Any,
    now_iso: str,
) -> Dict[str, Any]:
    """Write canonical claims to Claims DB."""
    page_ids = []
    errors = []

    for cc in canonical_claims:
        record = cc.to_claims_db_record(now_iso)
        try:
            page = claims_repo.upsert_claim(record)
            page_ids.append(page["id"])
        except Exception as e:
            logger.warning("Canonical claim write failed %s: %s", cc.canonical_id, e)
            errors.append(f"{cc.canonical_id}: {e}")

    logger.info("Canonical claims written: %d succeeded, %d failed", len(page_ids), len(errors))
    return {"page_ids": page_ids, "errors": errors}


# ------------------------------------------------------------------
# Main orchestration
# ------------------------------------------------------------------

def canonicalize(
    run_dirs: List[Path],
    *,
    llm_client: Any,
    dry_run: bool = False,
) -> CanonicalizationResult:
    """Run full canonicalization pipeline."""
    canon_id = f"canon_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Step 1: Collect
    claims = collect_claims(run_dirs)
    if len(claims) < 2:
        logger.warning("Need at least 2 claims to canonicalize, got %d", len(claims))
        return CanonicalizationResult(
            canonicalization_id=canon_id,
            input_run_ids=[d.name for d in run_dirs],
            total_input_claims=len(claims),
        )

    # Step 2: Group
    grouping = group_claims(claims, llm_client=llm_client)
    if not grouping:
        logger.error("Grouping failed")
        return CanonicalizationResult(
            canonicalization_id=canon_id,
            input_run_ids=[d.name for d in run_dirs],
            total_input_claims=len(claims),
            metadata={"error": "grouping_failed"},
        )

    # Step 3: Generate canonical claims
    canonical_claims = generate_canonical_claims(grouping, claims, now_iso)

    groups_count = len(grouping.get("groups", []))
    singletons_count = len(grouping.get("singletons", []))

    return CanonicalizationResult(
        canonicalization_id=canon_id,
        input_run_ids=[d.name for d in run_dirs],
        total_input_claims=len(claims),
        groups_formed=groups_count,
        singletons=singletons_count,
        canonical_claims_total=len(canonical_claims),
        canonical_claims=canonical_claims,
        metadata={
            "created_at": now_iso,
            "model": _MODEL,
        },
    )


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: CanonicalizationResult) -> str:
    lines = [
        f"# Claim Canonicalization Results",
        f"",
        f"## Summary",
        f"",
        f"- Input runs: {len(result.input_run_ids)}",
        f"- Input claims: {result.total_input_claims}",
        f"- Groups formed: {result.groups_formed}",
        f"- Singletons: {result.singletons}",
        f"- **Canonical claims: {result.canonical_claims_total}**",
        f"",
    ]

    # By confidence
    from collections import Counter
    conf_counts = Counter(cc.confidence for cc in result.canonical_claims)
    lines.append(f"## Confidence Distribution")
    lines.append(f"")
    for conf in ["high", "medium", "low"]:
        count = conf_counts.get(conf, 0)
        lines.append(f"- {conf}: {count}")
    lines.append(f"")

    # Grouped claims (2+ members)
    grouped = [cc for cc in result.canonical_claims if len(cc.member_claim_ids) >= 2]
    if grouped:
        lines.extend([f"## Grouped Claims (統合された知見)", f""])
        for cc in grouped:
            lines.append(f"### [{cc.confidence}] {cc.canonical_statement}")
            lines.append(f"")
            lines.append(f"- Canonical ID: `{cc.canonical_id}`")
            lines.append(f"- Supporting runs: {len(cc.supporting_runs)} ({', '.join(cc.supporting_runs)})")
            lines.append(f"- Categories: {', '.join(cc.categories)}")
            lines.append(f"- Members: {', '.join(cc.member_claim_ids)}")
            lines.append(f"")

    # Singletons
    singletons = [cc for cc in result.canonical_claims if len(cc.member_claim_ids) == 1]
    if singletons:
        lines.extend([f"## Singletons (統合されなかった知見)", f""])
        for cc in singletons:
            lines.append(f"- [{cc.confidence}] {cc.canonical_statement} ({cc.majority_category})")
        lines.append(f"")

    return "\n".join(lines)
