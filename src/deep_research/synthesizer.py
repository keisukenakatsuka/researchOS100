"""071 Synthesis Reasoning — service logic.

Reads credibility.json (annotated evidence) and plan.json,
clusters evidence by topic, generates Claims, and produces claims.json.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.deep_research import load_step_output, make_claim_id
from src.deep_research.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    CONFIDENCE_ORDER,
    TAG_ALIASES,
)

logger = logging.getLogger("071_synthesizer")

# -- constants -----------------------------------------------------------

_MODEL = "claude-sonnet-4-20250514"

MAX_CLAIMS = 10

# Confidence scoring thresholds.
_HIGH_MIN_EVIDENCE = 3
_MEDIUM_MIN_EVIDENCE = 2

# -- LLM prompt ----------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a research synthesis assistant.
Given a set of evidence items (each with a statement, confidence level,
and source domain), synthesize ONE concise, evidence-based claim.

A claim is a higher-level insight or conclusion that follows logically
from the evidence. It should:
- Be a single, clear assertion (1-2 sentences)
- Be grounded in the evidence (not speculative)
- Go beyond restating a single fact — combine or generalize

Output a JSON object with:
- statement: the synthesized claim (1-2 sentences)
- tags: list of 1-3 topic tags

Return ONLY a JSON object. No markdown fences, no explanation."""

# -- clustering ----------------------------------------------------------

# Tag synonyms for merging small clusters (from shared constants).
_TAG_ALIASES = TAG_ALIASES


def _primary_tag(ev: Dict[str, Any]) -> str:
    """Determine the primary clustering tag for an evidence item."""
    tags = ev.get("tags", [])
    if not tags:
        return "misc"
    raw = tags[0]
    return _TAG_ALIASES.get(raw, raw)


def cluster_evidence(
    evidence: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group evidence by primary tag.

    - Uses the first tag of each evidence item.
    - Applies tag aliases to merge small related clusters.
    - Falls back to ``"misc"`` when tags are empty.
    """
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in evidence:
        clusters[_primary_tag(ev)].append(ev)
    return dict(clusters)


def prioritize_clusters(
    clusters: Dict[str, List[Dict[str, Any]]],
) -> List[tuple[str, List[Dict[str, Any]]]]:
    """Sort clusters by importance: size * avg confidence score, descending."""
    def _score(items: List[Dict[str, Any]]) -> float:
        if not items:
            return 0.0
        conf_sum = sum(
            CONFIDENCE_ORDER.get(ev.get("confidence", "low"), 1) for ev in items
        )
        return len(items) * (conf_sum / len(items))

    ranked = sorted(clusters.items(), key=lambda kv: _score(kv[1]), reverse=True)
    return ranked


# -- source_ids aggregation ----------------------------------------------


def _collect_source_ids(
    evidence_ids: List[str],
    ev_index: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Derive unique source_ids from evidence, preserving order."""
    seen: set[str] = set()
    out: List[str] = []
    for eid in evidence_ids:
        sid = ev_index.get(eid, {}).get("source_id", "")
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


# -- key_question matching -----------------------------------------------


def _match_key_questions(
    statement: str,
    tags: List[str],
    key_questions: List[str],
) -> List[int]:
    """Return indices (0-based) of key_questions that relate to the claim.

    Simple keyword overlap heuristic.
    """
    claim_words = set(re.findall(r'[a-zA-Z\u3000-\u9fff]{2,}', statement.lower()))
    claim_words.update(t.lower() for t in tags)

    refs: List[int] = []
    for i, q in enumerate(key_questions):
        q_words = set(re.findall(r'[a-zA-Z\u3000-\u9fff]{2,}', q.lower()))
        overlap = claim_words & q_words
        if len(overlap) >= 2:
            refs.append(i)
    return refs


# -- claim confidence ----------------------------------------------------


def _assess_claim_confidence(
    evidence_ids: List[str],
    ev_index: Dict[str, Dict[str, Any]],
) -> tuple[str, List[Dict[str, str]], Dict[str, int]]:
    """Compute claim confidence from supporting evidence.

    Rules:
        high   — supporting evidence >= 3 AND at least one high-confidence
        medium — supporting evidence >= 2
        low    — supporting evidence == 1

    Returns (confidence, confidence_reason, confidence_meta).
    confidence_reason is a list of ``{"signal": str, "value": str}`` dicts,
    matching the format used by 070 credibility.
    """
    n = len(evidence_ids)
    confidences = [
        ev_index.get(eid, {}).get("confidence", CONFIDENCE_LOW) for eid in evidence_ids
    ]
    high_count = confidences.count(CONFIDENCE_HIGH)
    med_count = confidences.count(CONFIDENCE_MEDIUM)
    low_count = confidences.count(CONFIDENCE_LOW)

    meta = {
        "evidence_count": n,
        "high_count": high_count,
        "medium_count": med_count,
        "low_count": low_count,
    }

    signals: List[Dict[str, str]] = [
        {"signal": "evidence_count", "value": str(n)},
        {"signal": "high_confidence", "value": str(high_count)},
        {"signal": "medium_confidence", "value": str(med_count)},
    ]

    if n >= _HIGH_MIN_EVIDENCE and high_count > 0:
        signals.append({"signal": "rule", "value": f">={_HIGH_MIN_EVIDENCE} evidence with high"})
        return CONFIDENCE_HIGH, signals, meta
    if n >= _MEDIUM_MIN_EVIDENCE:
        signals.append({"signal": "rule", "value": f">={_MEDIUM_MIN_EVIDENCE} evidence"})
        return CONFIDENCE_MEDIUM, signals, meta
    signals.append({"signal": "rule", "value": "insufficient evidence"})
    return CONFIDENCE_LOW, signals, meta


# -- claim generation (LLM) ---------------------------------------------


def _build_llm_input(cluster_items: List[Dict[str, Any]]) -> str:
    """Format evidence cluster as text for LLM."""
    lines: List[str] = []
    for i, ev in enumerate(cluster_items, 1):
        lines.append(
            f"{i}. [{ev.get('confidence', '?')}] "
            f"{ev.get('statement', '')} "
            f"(source: {ev.get('source_title', 'unknown')})"
        )
    return "\n".join(lines)


def generate_claim_llm(
    tag: str,
    cluster_items: List[Dict[str, Any]],
    llm_client: Any,
) -> Optional[Dict[str, Any]]:
    """Use LLM to synthesize a claim from an evidence cluster."""
    user_msg = (
        f"Topic: {tag}\n"
        f"Evidence items:\n"
        f"{_build_llm_input(cluster_items)}"
    )

    body = {
        "model": _MODEL,
        "max_tokens": 1024,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.warning("LLM synthesis failed for tag=%s: %s", tag, e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    if not resp_text:
        return None

    try:
        result = json.loads(resp_text)
    except json.JSONDecodeError:
        cleaned = resp_text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        try:
            result = json.loads(cleaned.strip())
        except json.JSONDecodeError as e2:
            logger.warning("LLM JSON parse failed for tag=%s: %s", tag, e2)
            return None

    if isinstance(result, list):
        result = result[0] if result else None
    return result


# -- claim generation (fallback) -----------------------------------------

# Tag → summary template.  {target} = research subject, {details} = key facts.
_CLAIM_TEMPLATES: Dict[str, str] = {
    "product": "{target} has been expanding its product portfolio, {details}.",
    "funding": "{target} has secured substantial financial backing, {details}.",
    "strategy": "{target}'s strategic direction has evolved significantly, {details}.",
    "partnership": "{target} has established key partnerships, {details}.",
    "governance": "{target} faces ongoing governance and leadership challenges, {details}.",
    "research": "{target} continues to push research boundaries, {details}.",
    "market": "{target} has achieved significant market traction, {details}.",
    "misc": "Multiple developments relate to {target}, {details}.",
}

# Default target name when plan.json is unavailable.
_DEFAULT_TARGET = "The research subject"


def _extract_key_facts(items: List[Dict[str, Any]], max_facts: int = 3) -> str:
    """Extract short key facts from evidence for template filling."""
    sorted_items = sorted(
        items,
        key=lambda ev: CONFIDENCE_ORDER.get(ev.get("confidence", "low"), 1),
        reverse=True,
    )

    facts: List[str] = []
    for ev in sorted_items[:max_facts]:
        stmt = ev.get("statement", "").strip()
        # Strip trailing period for embedding in template sentence
        if stmt.endswith("."):
            stmt = stmt[:-1]
        # Shorten to key clause
        if len(stmt) > 80:
            stmt = stmt[:77].rstrip() + "…"
        facts.append(stmt)

    if not facts:
        return "with notable recent developments"

    if len(facts) == 1:
        return f"notably that {facts[0]}"
    return "including: " + "; ".join(facts)


def generate_claim_fallback(
    tag: str,
    cluster_items: List[Dict[str, Any]],
    target: str = "",
) -> Dict[str, Any]:
    """Rule-based fallback: generate a synthesized claim from template."""
    if len(cluster_items) == 1:
        # Single evidence → re-use statement directly
        statement = cluster_items[0].get("statement", "")
    else:
        template = _CLAIM_TEMPLATES.get(tag, _CLAIM_TEMPLATES["misc"])
        details = _extract_key_facts(cluster_items)
        subject = target or _DEFAULT_TARGET
        statement = template.format(target=subject, details=details)

    # Collect all tags from cluster
    all_tags: List[str] = []
    for ev in cluster_items:
        for t in ev.get("tags", []):
            if t not in all_tags:
                all_tags.append(t)

    return {
        "statement": statement,
        "tags": all_tags[:3],
    }


# -- main pipeline -------------------------------------------------------


def synthesize_claims(
    credibility_data: Dict[str, Any],
    run_id: str,
    llm_client: Any = None,
    key_questions: Optional[List[str]] = None,
    target: str = "",
) -> List[Dict[str, Any]]:
    """Cluster evidence, generate claims, and link evidence.

    Returns a list of claim dicts.
    """
    evidence_list = credibility_data.get("annotated_evidence", [])
    if not evidence_list:
        logger.warning("No annotated evidence found")
        return []

    # Build evidence index for lookups
    ev_index: Dict[str, Dict[str, Any]] = {
        ev["evidence_id"]: ev for ev in evidence_list
    }

    # 1. Cluster
    clusters = cluster_evidence(evidence_list)
    ranked = prioritize_clusters(clusters)

    logger.info(
        "Clusters: %s",
        {tag: len(items) for tag, items in ranked},
    )

    # 2. Generate claims (up to MAX_CLAIMS)
    claims: List[Dict[str, Any]] = []
    seq = 0
    now = datetime.now().isoformat()

    for tag, cluster_items in ranked:
        if seq >= MAX_CLAIMS:
            break

        # Try LLM first
        claim_raw = None
        if llm_client is not None:
            claim_raw = generate_claim_llm(tag, cluster_items, llm_client)

        if claim_raw is None:
            claim_raw = generate_claim_fallback(tag, cluster_items, target=target)

        statement = claim_raw.get("statement", "").strip()
        if not statement:
            continue

        seq += 1

        # 3. Link evidence + source_ids
        evidence_ids = [ev["evidence_id"] for ev in cluster_items]
        source_ids = _collect_source_ids(evidence_ids, ev_index)

        # 4. Assess confidence
        confidence, confidence_reason, confidence_meta = _assess_claim_confidence(
            evidence_ids, ev_index,
        )

        # Tags from LLM or cluster
        tags = claim_raw.get("tags", [tag])
        if not tags:
            tags = [tag]

        # 5. key_question_refs
        kq_refs = (
            _match_key_questions(statement, tags, key_questions)
            if key_questions
            else []
        )

        claims.append({
            "claim_id": make_claim_id(run_id, seq),
            "statement": statement,
            "evidence_ids": evidence_ids,
            "source_ids": source_ids,
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "confidence_meta": confidence_meta,
            "tags": tags[:3],
            "key_question_refs": kq_refs,
            "created_at": now,
        })

        logger.info(
            "Claim %s: tag=%s, evidence=%d, sources=%d, confidence=%s, kq=%s",
            claims[-1]["claim_id"],
            tag,
            len(evidence_ids),
            len(source_ids),
            confidence,
            kq_refs,
        )

    return claims


def build_output(
    run_id: str,
    claims: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the claims.json envelope."""
    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "total_claims": len(claims),
        "claims": claims,
    }


# -- entry point ---------------------------------------------------------

def run(
    run_id: str,
    llm_client: Any = None,
) -> Dict[str, Any]:
    """Execute the full 071 synthesis pipeline.

    1. Load credibility.json and plan.json.
    2. Cluster evidence by tag.
    3. Generate claims (LLM or fallback).
    4. Link evidence/sources and assess confidence.
    5. Match key_questions.
    6. Build and return claims.json dict.

    Args:
        run_id: Research run identifier.
        llm_client: Optional ClaudeClient.  If None, uses rule-based fallback.

    Returns:
        Dict ready to be saved as ``claims.json``.
    """
    credibility_data = load_step_output(run_id, "070")

    # Load plan for key_questions and target (optional)
    key_questions: Optional[List[str]] = None
    target: str = ""
    try:
        plan_data = load_step_output(run_id, "067")
        key_questions = plan_data.get("key_questions", [])
        targets = plan_data.get("targets", [])
        target = targets[0] if targets else ""
        logger.info("Loaded plan: %d key_questions, target=%s", len(key_questions), target or "(none)")
    except FileNotFoundError:
        logger.info("plan.json not found — skipping key_question matching")

    logger.info(
        "=== 071 Synthesizer: run_id=%s, evidence=%d ===",
        run_id,
        len(credibility_data.get("annotated_evidence", [])),
    )

    claims = synthesize_claims(
        credibility_data, run_id, llm_client, key_questions, target=target,
    )
    result = build_output(run_id, claims)

    logger.info(
        "Synthesis done: %d claims generated",
        result["total_claims"],
    )
    return result
