# src/discovery/scoring.py
"""Scoring, categorization, and rationale generation for discovery candidates.

Pure functions — no I/O, no LLM.  All heuristics are documented and tunable
via module-level constants.

Usage::

    from src.discovery.scoring import (
        score_candidates,
        filter_already_tracked,
    )

    scored = score_candidates(merged_entities, prev_week=None)
    scored = filter_already_tracked(scored, existing_target_names)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

# ----------------------------------------------------------------
# Scoring constants (documented, easy to tune)
# ----------------------------------------------------------------

# Final score = weighted sum of three components
FREQUENCY_WEIGHT: float = 0.50   # how often the entity appears this week
DIVERSITY_WEIGHT: float = 0.30   # spread across event_types / source IDs
GROWTH_WEIGHT: float = 0.20     # week-over-week change (0 if no prior data)

# Normalisation caps
FREQUENCY_CAP: int = 15          # mentions above this → frequency_score = 1.0
DIVERSITY_CAP: int = 5           # unique sources above this → diversity_score = 1.0

# Filtering
MIN_MENTION_COUNT: int = 2       # candidates below this are dropped
MIN_FINAL_SCORE: float = 0.10   # candidates below this are dropped
DEFAULT_TOP_K: int = 50          # default number of candidates in output

# ----------------------------------------------------------------
# Categorization rules (rule-based, explainable)
# ----------------------------------------------------------------

# Keyword cues for each type (lowercased, matched against normalized name
# or event_types evidence)
_VC_CUES = frozenset({
    "capital", "ventures", "venture", "partners", "fund", "funds",
    "investment", "investments", "advisory", "holdings", "asset",
    "management", "equity",
})
_VC_SUFFIXES = ("capital", "ventures", "partners", "fund", "investments",
                "holdings", "advisors", "advisory")

_STARTUP_CUES = frozenset({
    "labs", "tech", "technologies", "platform", "software",
    "therapeutics", "biotech", "robotics", "dynamics",
    "health", "genomics", "sciences",
})

# Known product/brand names that should NOT be classified as People
# even if they appear in PEOPLE-typed events
_KNOWN_PRODUCTS = frozenset({
    "chatgpt", "openai", "anthropic", "deepseek", "gemini",
    "copilot", "midjourney", "perplexity", "claude", "grok",
    "nvidia", "tesla", "google", "microsoft", "apple", "meta",
    "amazon", "spacex", "xai",
})

_POLICY_CUES = frozenset({
    "commission", "agency", "ministry", "government", "council",
    "department", "bureau", "authority", "regulation", "regulatory",
    "foundation", "institute", "institution", "university",
    "organization", "organisation",
    # Japanese
    "省", "庁", "機構", "委員会", "研究所", "大学", "銀行", "基金",
    "内閣", "政策",
})

# Person-name heuristic: 2-3 tokens, all Title Case, no org cues
_PERSON_RE = re.compile(
    r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}$"
)


def categorize_candidate(
    name: str,
    event_types: List[str],
    *,
    _name_lower: Optional[str] = None,
) -> str:
    """Assign a coarse type label to a candidate entity.

    Returns one of: ``"VC"``, ``"Startup"``, ``"Policy"``, ``"People"``,
    ``"Other"``, ``"Unknown"``.

    Strategy (evaluated in order):
    1. Event-type majority vote (if >50% of mentions come from one type)
    2. Name-based cue matching
    3. Person-name pattern
    4. Fallback to "Unknown"
    """
    low = _name_lower or name.lower()
    tokens = low.split()

    # --- 0. Known products / companies (override event-type) ---
    # Single-token entities that are known companies or products should
    # NOT be classified as People even if they appear in PEOPLE-typed events.
    if low in _KNOWN_PRODUCTS:
        return "Startup"

    # --- 1. Name-based cues (higher priority than event-type vote) ---
    # VC
    if any(tok in _VC_CUES for tok in tokens):
        return "VC"
    if low.endswith(_VC_SUFFIXES):
        return "VC"

    # Policy
    if any(tok in _POLICY_CUES for tok in tokens):
        return "Policy"
    # Japanese policy orgs
    if any(c in low for c in ("省", "庁", "機構", "委員会", "研究所", "大学", "基金", "内閣")):
        return "Policy"

    # Startup (name cues)
    if any(tok in _STARTUP_CUES for tok in tokens):
        return "Startup"

    # --- 2. Person pattern (2-3 Title Case words, no org cues) ---
    if _PERSON_RE.match(name):
        return "People"

    # --- 3. Event-type majority vote ---
    if event_types:
        from collections import Counter
        type_counts = Counter(event_types)
        top_type, top_count = type_counts.most_common(1)[0]
        if top_count / len(event_types) > 0.50:
            top_upper = top_type.upper()
            if top_upper in ("VC",):
                return "VC"
            if top_upper in ("STARTUP",):
                return "Startup"
            if top_upper in ("POLICY",):
                return "Policy"
            if top_upper in ("PEOPLE",):
                # Only classify as People if it also looks like a name
                # (to avoid products being classified as People)
                return "People"

    # --- 4. Fallback ---
    if event_types:
        return "Other"
    return "Unknown"


# ----------------------------------------------------------------
# Scoring functions
# ----------------------------------------------------------------

def _frequency_score(mention_count: int) -> float:
    """Normalised frequency: count / FREQUENCY_CAP, capped at 1.0."""
    return min(mention_count / FREQUENCY_CAP, 1.0)


def _diversity_score(source_count: int) -> float:
    """Normalised diversity: unique sources / DIVERSITY_CAP, capped at 1.0."""
    return min(source_count / DIVERSITY_CAP, 1.0)


def _growth_score(
    current_count: int,
    prev_count: Optional[int],
) -> Optional[float]:
    """Week-over-week growth score.

    Returns ``None`` if no prior data (growth_score not applicable).
    Growth formula: (current - prev) / max(prev, 1), capped at [-1, 1].
    Positive = growing, negative = declining, 0 = stable.
    """
    if prev_count is None:
        return None
    denom = max(prev_count, 1)
    raw = (current_count - prev_count) / denom
    return max(min(raw, 1.0), -1.0)


def score_candidates(
    merged_entities: List[Dict[str, Any]],
    *,
    prev_week_entities: Optional[Dict[str, int]] = None,
    min_count: int = MIN_MENTION_COUNT,
    min_score: float = MIN_FINAL_SCORE,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Score and rank merged entities.

    Parameters
    ----------
    merged_entities:
        Output of :func:`merge_raw_entities`.
    prev_week_entities:
        Optional mapping of ``{normalized_name: mention_count}`` from
        the previous week's candidates.json.  Used for growth_score.
    min_count:
        Minimum mentions to keep a candidate.
    min_score:
        Minimum final_score to keep.
    top_k:
        Max candidates to return.

    Returns a list of scored candidate dicts, sorted by final_score desc.
    """
    prev = prev_week_entities or {}

    scored: List[Dict[str, Any]] = []
    for ent in merged_entities:
        count = ent["mention_count"]
        if count < min_count:
            continue

        freq = _frequency_score(count)
        div = _diversity_score(ent["source_count"])
        growth = _growth_score(count, prev.get(ent["candidate_name_normalized"]))

        # Final score: use growth if available, otherwise redistribute weight
        if growth is not None:
            # Normalise growth to [0, 1] range for weighting: (growth + 1) / 2
            growth_norm = (growth + 1.0) / 2.0
            final = (
                FREQUENCY_WEIGHT * freq
                + DIVERSITY_WEIGHT * div
                + GROWTH_WEIGHT * growth_norm
            )
        else:
            # No prior data: redistribute growth weight proportionally
            w_total = FREQUENCY_WEIGHT + DIVERSITY_WEIGHT
            final = (
                (FREQUENCY_WEIGHT / w_total) * freq
                + (DIVERSITY_WEIGHT / w_total) * div
            )

        final = round(final, 4)
        if final < min_score:
            continue

        # Categorize
        cat = categorize_candidate(
            ent["candidate_name"],
            ent["event_types"],
        )

        # Generate rationale
        why = _generate_rationale(ent, freq, div, growth, cat)

        scored.append({
            "candidate_name": ent["candidate_name"],
            "candidate_name_normalized": ent["candidate_name_normalized"],
            "aliases": ent.get("aliases", []),
            "type": cat,
            "frequency_score": round(freq, 4),
            "diversity_score": round(div, 4),
            "growth_score": round(growth, 4) if growth is not None else None,
            "final_score": final,
            "mention_count": count,
            "source_count": ent["source_count"],
            "event_types": ent["event_types"],
            "why_notable": why,
            "evidence": {
                "sample_event_titles": ent.get("sample_event_titles", []),
                "sample_paper_titles": ent.get("sample_paper_titles", []),
                "event_ids": ent.get("event_ids", []),
            },
            "already_tracked": False,  # set later by filter_already_tracked
            "llm_override": None,       # set later by LLM step
        })

    # Sort by final_score descending
    scored.sort(key=lambda c: (-c["final_score"], -c["mention_count"]))
    return scored[:top_k]


def _generate_rationale(
    ent: Dict[str, Any],
    freq: float,
    div: float,
    growth: Optional[float],
    cat: str,
) -> str:
    """Generate a one-line, evidence-grounded rationale."""
    count = ent["mention_count"]
    source_count = ent["source_count"]
    event_types = ent.get("event_types", [])
    action_count = ent.get("action_needed_count", 0)

    parts: List[str] = []

    # Frequency + diversity
    if source_count > 1:
        type_str = "/".join(event_types[:3]) if event_types else "mixed"
        parts.append(f"Appeared in {count} mentions across {source_count} sources ({type_str})")
    else:
        parts.append(f"Appeared {count} times")

    # Growth
    if growth is not None:
        if growth > 0.1:
            parts.append(f"spike vs last week (+{growth:.0%})")
        elif growth < -0.1:
            parts.append(f"declining vs last week ({growth:.0%})")
        else:
            parts.append("stable vs last week")
    else:
        parts.append("new this week")

    # Action-needed signal
    if action_count >= 2:
        parts.append(f"{action_count} action-needed events")

    return "; ".join(parts) + "."


def filter_already_tracked(
    candidates: List[Dict[str, Any]],
    existing_names: Set[str],
) -> List[Dict[str, Any]]:
    """Mark candidates that are already tracked as monitoring targets.

    Parameters
    ----------
    existing_names:
        Set of normalised target names (lowercase, stripped).
    """
    for c in candidates:
        norm = c["candidate_name_normalized"]
        # Check: exact match or substring match (e.g., "sam altman" in targets)
        if norm in existing_names:
            c["already_tracked"] = True
        elif any(norm in en or en in norm for en in existing_names):
            c["already_tracked"] = True
    return candidates
