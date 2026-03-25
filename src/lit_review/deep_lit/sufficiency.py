# src/lit_review/deep_lit/sufficiency.py
"""119b Evidence Sufficiency Gate — rule-based check.

Determines whether the literature base provides sufficient evidence
for a hypothesis's core mechanism. No LLM calls — pure rule-based
logic on synthesis outputs.

Usage::

    from src.lit_review.deep_lit.sufficiency import check_sufficiency

    result = check_sufficiency(synthesis, hypothesis)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Keywords that signal a gap directly undermines the hypothesis core
_DIRECT_GAP_MARKERS = [
    "直接的",
    "direct",
    "not evidenced",
    "未検証",
    "不足",
    "存在しない",
    "absence",
    "no empirical",
    "実証分析が不足",
    "定量的証拠",
]


@dataclass
class SufficiencyResult:
    """Result of evidence sufficiency check."""
    hypothesis_id: str = ""
    result: str = ""  # "sufficient" / "weak" / "insufficient"
    consensus_support: Dict[str, Any] = field(default_factory=dict)
    gap_concerns: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""  # "proceed" / "proceed_with_caution" / "halt"
    suggested_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_sufficiency(
    synthesis: Dict[str, Any],
    hypothesis: Dict[str, Any],
) -> SufficiencyResult:
    """Check if literature provides sufficient evidence for the hypothesis.

    Rules:
    1. Consensus check: Are there consensus findings that support the
       hypothesis direction? Strong = sufficient, moderate = weak, none = insufficient.
    2. Gap check: Do any unknown_gaps directly undermine the core mechanism?
       If so, downgrade by one level.
    """
    hyp_id = hypothesis.get("hypothesis_id", hypothesis.get("id", ""))
    stmt = hypothesis.get("hypothesis_statement", "")
    result = SufficiencyResult(hypothesis_id=hyp_id)

    # --- Rule 1: Consensus support ---
    established = synthesis.get("known_established", [])
    consensus_level, relevant_findings = _assess_consensus(established, stmt)
    result.consensus_support = {
        "count": len(relevant_findings),
        "max_strength": consensus_level,
        "relevant_findings": relevant_findings,
    }

    # --- Rule 2: Gap concerns ---
    gaps = synthesis.get("unknown_gaps", [])
    critical_gaps = _find_critical_gaps(gaps, stmt)
    result.gap_concerns = {
        "count": len(critical_gaps),
        "critical_gaps": critical_gaps,
    }

    # --- Combine ---
    base_level = consensus_level  # "strong" / "moderate" / "none"

    # Downgrade if critical gaps exist
    if critical_gaps:
        if base_level == "strong":
            base_level = "moderate"
        elif base_level == "moderate":
            base_level = "none"

    # Map to result
    level_map = {
        "strong": ("sufficient", "proceed"),
        "moderate": ("weak", "proceed_with_caution"),
        "none": ("insufficient", "halt"),
    }
    result.result, result.recommendation = level_map[base_level]

    # Generate suggested queries on insufficient
    if result.result == "insufficient":
        result.suggested_queries = _suggest_queries(critical_gaps, stmt)

    logger.info("Evidence sufficiency for %s: %s (consensus=%s, critical_gaps=%d)",
                hyp_id[:12], result.result, consensus_level, len(critical_gaps))
    return result


def _assess_consensus(
    established: List[Any],
    hypothesis_stmt: str,
) -> tuple[str, List[str]]:
    """Assess consensus findings for hypothesis support.

    Returns (strength_level, relevant_finding_texts).
    strength_level: "strong" / "moderate" / "none"
    """
    relevant: List[str] = []
    max_strength = "none"

    # Extract key terms from hypothesis for matching
    hyp_terms = _extract_key_terms(hypothesis_stmt)

    for item in established:
        if isinstance(item, dict):
            finding = item.get("finding", "")
            strength = item.get("strength", item.get("direction", "")).lower()
        elif isinstance(item, str):
            finding = item
            strength = ""
        else:
            continue

        if not finding:
            continue

        # Check if finding is relevant to hypothesis
        finding_lower = finding.lower()
        if any(term in finding_lower for term in hyp_terms):
            relevant.append(finding[:200])
            if "strong" in strength:
                max_strength = "strong"
            elif "moderate" in strength and max_strength != "strong":
                max_strength = "moderate"
            elif max_strength == "none":
                # Any relevant finding without explicit strength = moderate
                max_strength = "moderate"

    # If no term-matched findings but established list is non-empty,
    # treat as weak support (topic is studied but not directly matching)
    if not relevant and established:
        max_strength = "moderate"
        for item in established[:2]:
            text = item.get("finding", str(item)) if isinstance(item, dict) else str(item)
            relevant.append(f"(indirect) {text[:200]}")

    return max_strength, relevant


def _extract_key_terms(stmt: str) -> List[str]:
    """Extract key terms from hypothesis statement for matching."""
    terms = []
    # Japanese key terms
    jp_patterns = [
        r"政府系", r"固定給", r"報酬", r"投資", r"VC", r"ディープテック",
        r"技術成果", r"CVC", r"制度的支援", r"リスク軽減", r"長期",
    ]
    for pat in jp_patterns:
        if re.search(pat, stmt):
            terms.append(pat.lower())

    # English key terms from statement
    en_words = re.findall(r"[a-zA-Z]{4,}", stmt.lower())
    terms.extend(en_words)

    # Common research terms that should match
    terms.extend(["government", "venture", "investment", "long-term", "institutional"])

    return list(set(terms))


def _find_critical_gaps(
    gaps: List[Any],
    hypothesis_stmt: str,
) -> List[str]:
    """Find gaps that directly undermine the hypothesis core mechanism."""
    critical: List[str] = []

    for gap in gaps:
        if isinstance(gap, dict):
            gap_text = gap.get("gap", "") + " " + gap.get("description", gap.get("importance", ""))
        elif isinstance(gap, str):
            gap_text = gap
        else:
            continue

        gap_lower = gap_text.lower()

        # Check if gap contains direct-evidence-absence markers
        has_marker = any(marker in gap_lower for marker in _DIRECT_GAP_MARKERS)
        if has_marker:
            critical.append(gap_text[:200].strip())

    return critical


def _suggest_queries(
    critical_gaps: List[str],
    hypothesis_stmt: str,
) -> List[str]:
    """Generate targeted search queries from critical gaps (template-based, no LLM)."""
    queries: List[str] = []
    for gap in critical_gaps[:3]:
        # Extract the core topic from the gap
        # Remove common filler and create a search query
        clean = re.sub(r"[（(].*?[）)]", "", gap)  # Remove parenthetical
        clean = clean.strip()[:100]
        queries.append(f"empirical evidence: {clean}")

    if not queries:
        queries.append(f"direct empirical test: {hypothesis_stmt[:80]}")

    return queries[:5]
