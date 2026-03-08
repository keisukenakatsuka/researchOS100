"""070 Credibility Analysis — service logic.

Reads evidence.json and sources.json, annotates each Evidence item
with confidence (high/medium/low) and confidence_reason based on
source metadata heuristics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from src.deep_research import load_step_output
from src.deep_research.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    CONFIDENCE_ORDER,
    SOURCE_TYPE_CONFIDENCE,
)

logger = logging.getLogger("070_credibility")

# -- domain / source_type heuristics ------------------------------------

# Domains known to be high-credibility sources.
_HIGH_DOMAINS = {
    "openai.com", "anthropic.com", "deepmind.google", "ai.meta.com",
    "microsoft.com", "google.com", "apple.com", "amazon.com",
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "wsj.com", "ft.com",
    "nikkei.com", "bloomberg.com",
    "arxiv.org", "nature.com", "science.org", "ieee.org",
}

# TLD patterns for high credibility.
_HIGH_TLD_PATTERNS = (".gov", ".go.jp", ".edu", ".ac.jp", ".ac.uk")

# Domains typically medium credibility.
_MEDIUM_DOMAINS = {
    "techcrunch.com", "theverge.com", "wired.com", "arstechnica.com",
    "venturebeat.com", "zdnet.com", "thenextweb.com",
    "medium.com", "substack.com", "towardsdatascience.com",
    "cnbc.com", "forbes.com", "businessinsider.com",
    "en.wikipedia.org", "ja.wikipedia.org",
}

# source_type → base confidence mapping (from shared constants).
_TYPE_CONFIDENCE = SOURCE_TYPE_CONFIDENCE

# Minimum char count for medium/high confidence.
_MIN_CHARS_MEDIUM = 200


# -- core logic ----------------------------------------------------------

def _build_source_index(
    sources_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Build source_id → source dict lookup."""
    idx: Dict[str, Dict[str, Any]] = {}
    for src in sources_data.get("sources", []):
        sid = src.get("source_id", "")
        if sid:
            idx[sid] = src
    return idx


def assess_confidence(
    evidence: Dict[str, Any],
    source: Dict[str, Any],
) -> tuple[str, List[Dict[str, str]]]:
    """Determine confidence and structured signals for a single evidence item.

    Returns (confidence, signals) where signals is a list of
    ``{"signal": str, "value": str}`` dicts.
    """
    domain = source.get("domain", "").lower()
    source_type = source.get("source_type", "unknown")
    fetch_status = source.get("fetch_status", "")
    char_count = source.get("fetched_char_count", 0)

    signals: List[Dict[str, str]] = []

    # 1. Domain-based assessment
    if domain in _HIGH_DOMAINS:
        domain_score = CONFIDENCE_HIGH
        signals.append({"signal": "domain", "value": f"authoritative ({domain})"})
    elif any(domain.endswith(tld) for tld in _HIGH_TLD_PATTERNS):
        domain_score = CONFIDENCE_HIGH
        signals.append({"signal": "domain", "value": f"institutional TLD ({domain})"})
    elif domain in _MEDIUM_DOMAINS:
        domain_score = CONFIDENCE_MEDIUM
        signals.append({"signal": "domain", "value": f"established ({domain})"})
    else:
        domain_score = CONFIDENCE_LOW
        signals.append({"signal": "domain", "value": f"unrecognized ({domain})"})

    # 2. Source-type assessment
    type_score = _TYPE_CONFIDENCE.get(source_type, CONFIDENCE_LOW)
    signals.append({"signal": "source_type", "value": source_type})

    # 3. Content quality signals
    if fetch_status != "success":
        signals.append({"signal": "fetch_status", "value": fetch_status})
        return CONFIDENCE_LOW, signals

    signals.append({"signal": "content_length", "value": f"{char_count} chars"})

    if char_count < _MIN_CHARS_MEDIUM:
        signals.append({"signal": "content_quality", "value": "sparse"})
        return CONFIDENCE_LOW, signals

    # 4. Combine scores — take the higher of domain vs type
    _SCORE_TO_LEVEL = {v: k for k, v in CONFIDENCE_ORDER.items()}
    final_score_num = max(CONFIDENCE_ORDER[domain_score], CONFIDENCE_ORDER[type_score])
    final = _SCORE_TO_LEVEL[final_score_num]

    return final, signals


def annotate_evidence(
    evidence_data: Dict[str, Any],
    sources_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Annotate all evidence items with confidence.

    Returns a new list of evidence dicts with confidence/confidence_reason
    populated.
    """
    source_idx = _build_source_index(sources_data)
    annotated: List[Dict[str, Any]] = []

    for ev in evidence_data.get("evidence", []):
        ev_copy = dict(ev)
        source_id = ev_copy.get("source_id", "")
        source = source_idx.get(source_id, {})

        if not source:
            ev_copy["confidence"] = CONFIDENCE_LOW
            ev_copy["confidence_reason"] = [
                {"signal": "source_lookup", "value": f"{source_id} not found"},
            ]
        else:
            conf, signals = assess_confidence(ev_copy, source)
            ev_copy["confidence"] = conf
            ev_copy["confidence_reason"] = signals

        annotated.append(ev_copy)

    return annotated


def build_output(
    run_id: str,
    annotated: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the credibility.json envelope."""
    counts = {CONFIDENCE_HIGH: 0, CONFIDENCE_MEDIUM: 0, CONFIDENCE_LOW: 0}
    for ev in annotated:
        c = ev.get("confidence", CONFIDENCE_LOW)
        counts[c] = counts.get(c, 0) + 1

    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "total_annotated": len(annotated),
        "confidence_distribution": counts,
        "annotated_evidence": annotated,
    }


# -- main entry point ----------------------------------------------------

def run(run_id: str) -> Dict[str, Any]:
    """Execute the full 070 credibility pipeline.

    1. Load evidence.json and sources.json.
    2. Annotate each evidence with confidence/confidence_reason.
    3. Build and return credibility.json dict.

    Args:
        run_id: Research run identifier.

    Returns:
        Dict ready to be saved as ``credibility.json``.
    """
    evidence_data = load_step_output(run_id, "069")
    sources_data = load_step_output(run_id, "068")

    logger.info(
        "=== 070 Credibility: run_id=%s, evidence=%d, sources=%d ===",
        run_id,
        len(evidence_data.get("evidence", [])),
        len(sources_data.get("sources", [])),
    )

    annotated = annotate_evidence(evidence_data, sources_data)
    result = build_output(run_id, annotated)

    logger.info(
        "Credibility done: %d annotated — high=%d, medium=%d, low=%d",
        result["total_annotated"],
        result["confidence_distribution"].get(CONFIDENCE_HIGH, 0),
        result["confidence_distribution"].get(CONFIDENCE_MEDIUM, 0),
        result["confidence_distribution"].get(CONFIDENCE_LOW, 0),
    )
    return result
