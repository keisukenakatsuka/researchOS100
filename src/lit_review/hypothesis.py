# src/lit_review/hypothesis.py
"""Block 4: Hypothesis Generation (087).

Generates testable research hypotheses from Canonical Claims,
Open Questions, Blindspots, and Cross-RQ Opportunities.

Four generation strategies:
  - gap_driven: gap × theory → hypothesis
  - claim_combination: claim1 + claim2 → new hypothesis
  - contested_resolution: contested claim → resolution hypothesis
  - cross_rq: cross-RQ opportunity → hypothesis

Usage::

    from src.lit_review.hypothesis import generate_hypotheses_pipeline

    result = generate_hypotheses_pipeline(run_dir, llm_client=client)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Hypothesis:
    hypothesis_id: str
    hypothesis_statement: str
    rationale: str
    strategy: str  # gap_driven | claim_combination | contested_resolution | cross_rq
    testability: str  # high | medium | low
    suggested_test: str
    source_claim_ids: List[str] = field(default_factory=list)
    source_gaps: List[str] = field(default_factory=list)
    novelty_rationale: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_claims_db_record(self, now_iso: str) -> Dict[str, Any]:
        """Convert to format expected by claims_repo.upsert_claim()."""
        reason_parts = [
            f"[hypothesis] strategy={self.strategy}, testability={self.testability}",
            f"Rationale: {self.rationale[:500]}",
            f"Suggested test: {self.suggested_test[:500]}",
        ]
        if self.source_claim_ids:
            reason_parts.append(f"Supporting claims: {', '.join(self.source_claim_ids)}")
        if self.source_gaps:
            reason_parts.append(f"Source gaps: {'; '.join(g[:80] for g in self.source_gaps)}")

        return {
            "claim_id": self.hypothesis_id,
            "statement": self.hypothesis_statement,
            "confidence": "medium",  # hypotheses are unverified
            "confidence_reason": "\n".join(reason_parts)[:2000],
            "tags": self.tags,
            "created_at": now_iso,
        }


@dataclass
class HypothesisResult:
    run_id: str
    rq_title: str = ""
    input_summary: Dict[str, int] = field(default_factory=dict)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        by_type = Counter(h.strategy for h in self.hypotheses)
        by_test = Counter(h.testability for h in self.hypotheses)
        return {
            "run_id": self.run_id,
            "rq_title": self.rq_title,
            "input_summary": self.input_summary,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "metadata": {
                **self.metadata,
                "total_hypotheses": len(self.hypotheses),
                "by_strategy": dict(by_type),
                "by_testability": dict(by_test),
            },
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# ID generation
# ------------------------------------------------------------------

def hypothesis_id(statement: str) -> str:
    """Generate hypothesis ID from content hash. Same statement → same ID."""
    normalized = statement.strip().lower()
    content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:12]
    return f"hyp__{content_hash}"


# ------------------------------------------------------------------
# Input collection
# ------------------------------------------------------------------

def collect_inputs(
    run_dir: Path,
    *,
    canon_dir: Optional[Path] = None,
    cross_rq_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Collect all inputs for hypothesis generation."""
    inputs: Dict[str, Any] = {
        "canonical_claims": [],
        "open_questions": [],
        "blindspots": [],
        "cross_rq_opportunities": [],
        "theoretical_streams": [],
        "rq_title": "",
    }

    # RQ context
    rq_path = run_dir / "rq_context.json"
    if rq_path.exists():
        rq = json.loads(rq_path.read_text())
        inputs["rq_title"] = rq.get("title", "")

    # Canonical claims
    if canon_dir:
        canon_path = canon_dir / "canonicalization_result.json"
        if canon_path.exists():
            canon = json.loads(canon_path.read_text())
            inputs["canonical_claims"] = canon.get("canonical_claims", [])

    # Always load open_questions and theoretical_streams from lit_review.json
    lr_path = run_dir / "lit_review.json"
    if lr_path.exists():
        lr = json.loads(lr_path.read_text())
        inputs["open_questions"] = lr.get("open_questions", [])
        inputs["theoretical_streams"] = lr.get("theoretical_streams", [])

        # If no canon_dir, also extract claims from lit_review.json
        if not inputs["canonical_claims"]:
            findings = lr.get("empirical_findings", {})
            for cat in ["established", "emerging", "contested"]:
                for f in findings.get(cat, []):
                    if cat == "contested":
                        for pos in f.get("positions", []):
                            inputs["canonical_claims"].append({
                                "canonical_statement": pos.get("statement", ""),
                                "confidence": "low",
                                "majority_category": "contested",
                            })
                    else:
                        inputs["canonical_claims"].append({
                            "canonical_statement": f.get("statement", ""),
                            "confidence": "high" if cat == "established" else "medium",
                            "majority_category": cat,
                        })

    # Landscape blindspots
    ls_path = run_dir / "landscape.json"
    if ls_path.exists():
        ls = json.loads(ls_path.read_text())
        inputs["blindspots"] = ls.get("blindspots", [])

    # Cross-RQ opportunities
    if cross_rq_dir:
        cmp_path = cross_rq_dir / "cross_rq_comparison.json"
        if cmp_path.exists():
            cmp = json.loads(cmp_path.read_text())
            inputs["cross_rq_opportunities"] = cmp.get("cross_rq_opportunities", [])

    logger.info(
        "Collected inputs: %d claims, %d gaps, %d blindspots, %d cross-RQ opportunities",
        len(inputs["canonical_claims"]),
        len(inputs["open_questions"]),
        len(inputs["blindspots"]),
        len(inputs["cross_rq_opportunities"]),
    )
    return inputs


# ------------------------------------------------------------------
# LLM hypothesis generation
# ------------------------------------------------------------------

def _parse_json_response(text: str) -> Optional[Any]:
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


_GENERATION_SYSTEM = """\
あなたは研究仮説生成の専門家です。
既存の知見（Canonical Claims）と研究ギャップ（Open Questions / Blindspots）をもとに、
検証可能な研究仮説を生成してください。

重要な指示:
- 各仮説は具体的で検証可能でなければなりません
- 「〜が影響する」のような漠然とした仮説は避け、方向性（正/負/条件付き）を含めてください
- 仮説ごとに、具体的な検証アプローチ（suggested_test）を示してください
- 新規性（novelty）の根拠を明示してください
- 5〜10件の仮説を生成してください

4つの生成戦略を使ってください:
1. gap_driven: 研究ギャップから仮説を生成
2. claim_combination: 複数の既存知見を組み合わせて新しい仮説を導出
3. contested_resolution: 対立する知見の条件依存性を仮説化
4. cross_rq: 横断的な研究機会を仮説に変換（該当する場合）"""


def generate_hypotheses(
    inputs: Dict[str, Any],
    *,
    llm_client: Any,
) -> List[Dict[str, Any]]:
    """Generate hypotheses using LLM."""
    # Build input summary
    claims_text = []
    for i, c in enumerate(inputs["canonical_claims"]):
        stmt = c.get("canonical_statement", c.get("statement", ""))
        conf = c.get("confidence", "?")
        cat = c.get("majority_category", "?")
        claims_text.append(f"[{i}] ({cat}, conf={conf}) {stmt}")

    gaps_text = []
    for q in inputs["open_questions"]:
        desc = q.get("description", "")
        why = q.get("why_unresolved", "")
        gaps_text.append(f"- {desc}" + (f" (理由: {why})" if why else ""))

    blindspots_text = []
    for b in inputs["blindspots"]:
        area = b.get("area", "")
        what = b.get("what_is_missing", "")
        blindspots_text.append(f"- {area}: {what}")

    streams_text = []
    for s in inputs["theoretical_streams"]:
        streams_text.append(f"- {s.get('name', '')}: {s.get('description', '')[:100]}")

    cross_rq_text = []
    for opp in inputs["cross_rq_opportunities"]:
        cross_rq_text.append(f"- {opp.get('theme', '')}: {opp.get('rationale', '')[:100]}")

    user_msg = (
        f"## RQ: {inputs.get('rq_title', '')}\n\n"
        f"## Canonical Claims ({len(claims_text)})\n" + "\n".join(claims_text) + "\n\n"
        f"## 研究ギャップ ({len(gaps_text)})\n" + "\n".join(gaps_text or ["(なし)"]) + "\n\n"
        f"## ブラインドスポット ({len(blindspots_text)})\n" + "\n".join(blindspots_text or ["(なし)"]) + "\n\n"
        f"## 理論的系譜\n" + "\n".join(streams_text or ["(なし)"]) + "\n\n"
    )
    if cross_rq_text:
        user_msg += f"## Cross-RQ Research Opportunities\n" + "\n".join(cross_rq_text) + "\n\n"

    user_msg += (
        f"## 指示\n"
        f"上記の知見・ギャップ・ブラインドスポットをもとに、検証可能な研究仮説を5〜10件生成してください。\n"
        f"4つの戦略（gap_driven, claim_combination, contested_resolution, cross_rq）を使ってください。\n\n"
        f"出力形式 (JSON):\n"
        f'{{"hypotheses": [\n'
        f'  {{\n'
        f'    "hypothesis_statement": "具体的な仮説（日本語、検証可能に）",\n'
        f'    "rationale": "この仮説の論理的根拠",\n'
        f'    "strategy": "gap_driven | claim_combination | contested_resolution | cross_rq",\n'
        f'    "testability": "high | medium | low",\n'
        f'    "suggested_test": "検証に必要な手法・データ・条件",\n'
        f'    "source_claim_indices": [0, 2],\n'
        f'    "source_gaps": ["gap の記述"],\n'
        f'    "novelty_rationale": "なぜ新規性があるか"\n'
        f'  }}\n'
        f']}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 8192,
        "system": _GENERATION_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Hypothesis generation LLM call failed: %s", e)
        return []

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Hypothesis LLM: in=%d, out=%d tokens", usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    parsed = _parse_json_response(resp_text)
    if not parsed or "hypotheses" not in parsed:
        logger.error("Hypothesis JSON parse failed")
        return []

    return parsed["hypotheses"]


# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------

def _build_tags(strategy: str, testability: str) -> List[str]:
    return [
        "hypothesis",
        "block4",
        strategy,
        f"testability_{testability}",
    ]


def build_hypotheses(
    raw_hypotheses: List[Dict[str, Any]],
    canonical_claims: List[Dict[str, Any]],
) -> List[Hypothesis]:
    """Convert raw LLM output to Hypothesis objects."""
    results = []
    for raw in raw_hypotheses:
        stmt = raw.get("hypothesis_statement", "")
        if not stmt:
            continue

        strategy = raw.get("strategy", "gap_driven")
        testability = raw.get("testability", "medium")

        # Resolve claim indices to IDs
        source_claim_ids = []
        for idx in raw.get("source_claim_indices", []):
            if isinstance(idx, int) and idx < len(canonical_claims):
                cid = canonical_claims[idx].get("canonical_id", "")
                if cid:
                    source_claim_ids.append(cid)

        results.append(Hypothesis(
            hypothesis_id=hypothesis_id(stmt),
            hypothesis_statement=stmt,
            rationale=raw.get("rationale", ""),
            strategy=strategy,
            testability=testability,
            suggested_test=raw.get("suggested_test", ""),
            source_claim_ids=source_claim_ids,
            source_gaps=raw.get("source_gaps", []),
            novelty_rationale=raw.get("novelty_rationale", ""),
            tags=_build_tags(strategy, testability),
        ))

    logger.info("Built %d hypotheses", len(results))
    return results


# ------------------------------------------------------------------
# Claims DB writeback
# ------------------------------------------------------------------

def write_hypotheses(
    hypotheses: List[Hypothesis],
    *,
    claims_repo: Any,
    now_iso: str,
) -> Dict[str, Any]:
    """Write hypotheses to Claims DB as type=hypothesis."""
    page_ids = []
    errors = []
    for h in hypotheses:
        record = h.to_claims_db_record(now_iso)
        try:
            page = claims_repo.upsert_claim(record)
            page_ids.append(page["id"])
        except Exception as e:
            logger.warning("Hypothesis write failed %s: %s", h.hypothesis_id, e)
            errors.append(f"{h.hypothesis_id}: {e}")

    logger.info("Hypotheses written: %d succeeded, %d failed", len(page_ids), len(errors))
    return {"page_ids": page_ids, "errors": errors}


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def generate_hypotheses_pipeline(
    run_dir: Path,
    *,
    llm_client: Any,
    canon_dir: Optional[Path] = None,
    cross_rq_dir: Optional[Path] = None,
) -> HypothesisResult:
    """Full hypothesis generation pipeline."""
    run_id = run_dir.name
    now_iso = datetime.now(timezone.utc).isoformat()

    # Collect inputs
    inputs = collect_inputs(run_dir, canon_dir=canon_dir, cross_rq_dir=cross_rq_dir)

    if not inputs["canonical_claims"] and not inputs["open_questions"]:
        logger.error("No claims or gaps found")
        return HypothesisResult(run_id=run_id, rq_title=inputs.get("rq_title", ""))

    # Generate
    raw = generate_hypotheses(inputs, llm_client=llm_client)

    # Build
    hypotheses = build_hypotheses(raw, inputs["canonical_claims"])

    return HypothesisResult(
        run_id=run_id,
        rq_title=inputs.get("rq_title", ""),
        input_summary={
            "canonical_claims": len(inputs["canonical_claims"]),
            "open_questions": len(inputs["open_questions"]),
            "blindspots": len(inputs["blindspots"]),
            "cross_rq_opportunities": len(inputs["cross_rq_opportunities"]),
        },
        hypotheses=hypotheses,
        metadata={"created_at": now_iso, "model": _MODEL},
    )


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: HypothesisResult) -> str:
    lines = [
        f"# Hypothesis Generation Results",
        f"",
        f"## RQ: {result.rq_title}",
        f"",
        f"## Input Summary",
        f"",
    ]
    for k, v in result.input_summary.items():
        lines.append(f"- {k}: {v}")
    lines.extend([f"", f"**Total hypotheses: {len(result.hypotheses)}**", f""])

    # By strategy
    by_strat = Counter(h.strategy for h in result.hypotheses)
    lines.extend([f"## By Strategy", f""])
    for s, c in by_strat.most_common():
        lines.append(f"- {s}: {c}")
    lines.append(f"")

    # By testability
    by_test = Counter(h.testability for h in result.hypotheses)
    lines.extend([f"## By Testability", f""])
    for t in ["high", "medium", "low"]:
        lines.append(f"- {t}: {by_test.get(t, 0)}")
    lines.append(f"")

    # Each hypothesis
    lines.extend([f"## Hypotheses", f""])
    for i, h in enumerate(result.hypotheses, 1):
        lines.append(f"### H{i}: {h.hypothesis_statement}")
        lines.append(f"")
        lines.append(f"- **Strategy**: {h.strategy}")
        lines.append(f"- **Testability**: {h.testability}")
        lines.append(f"- **Rationale**: {h.rationale}")
        lines.append(f"- **Suggested test**: {h.suggested_test}")
        if h.source_claim_ids:
            lines.append(f"- **Source claims**: {', '.join(h.source_claim_ids)}")
        if h.source_gaps:
            lines.append(f"- **Source gaps**: {'; '.join(h.source_gaps[:3])}")
        if h.novelty_rationale:
            lines.append(f"- **Novelty**: {h.novelty_rationale}")
        lines.append(f"")

    return "\n".join(lines)
