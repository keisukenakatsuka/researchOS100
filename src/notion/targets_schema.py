# src/notion/targets_schema.py
"""Schema definition for the Monitoring Targets Notion database.

Extracted from notebook 043_weekly_targets_review Cells 05/07/09/10.
Reusable by 050+ scripts and any targets pipeline.

Property groups
---------------
- **TARGETS_CORE_PROPERTIES**: identity, status, priority, cadence.
- **TARGETS_SEARCH_PROPERTIES**: search-configuration fields.
- **TARGETS_OPERATIONAL_PROPERTIES**: scheduling and error tracking.

Scoring, threshold, and keyword-proposal constants are also defined here
so that all heuristic parameters are centralised and easy to tune.

Usage::

    from src.notion.targets_schema import (
        TARGETS_ALL_PROPERTIES,
        MIN_SIGNAL_THRESHOLD,
        KEYWORD_STOPWORDS,
    )
"""

from __future__ import annotations

from typing import Dict


# ----------------------------------------------------------------
# Property groups  (property name -> Notion property type)
# ----------------------------------------------------------------

TARGETS_CORE_PROPERTIES: Dict[str, str] = {
    "Name": "title",
    "Type": "select",         # VC / Startup / Policy / People
    "Status": "select",       # Active / Paused / Archived
    "Priority": "select",     # Low / Medium / High / Critical
    "Cadence": "select",      # Daily / Weekly / Bi-weekly / Monthly
    "Enabled": "checkbox",
}

TARGETS_SEARCH_PROPERTIES: Dict[str, str] = {
    "Search Keywords": "rich_text",
    "Source URLs": "rich_text",
    "Source Type": "select",
}

TARGETS_OPERATIONAL_PROPERTIES: Dict[str, str] = {
    "Last Checked": "date",
    "Next Check": "date",
    "Last Error": "rich_text",
    "Error Count": "number",
    "Last Hit At": "date",
    "Consecutive Misses": "number",
    "Cadence Reason": "rich_text",
    "Created By": "select",       # manual / 076_session / 050_review
    "Source Session": "rich_text",
}

# Union of all groups.
TARGETS_ALL_PROPERTIES: Dict[str, str] = {
    **TARGETS_CORE_PROPERTIES,
    **TARGETS_SEARCH_PROPERTIES,
    **TARGETS_OPERATIONAL_PROPERTIES,
}


# ----------------------------------------------------------------
# Filtering defaults
# ----------------------------------------------------------------

DEFAULT_TARGET_FILTER_ENABLED: bool = True
DEFAULT_TARGET_EXCLUDE_STATUSES = frozenset({"Archived"})


# ----------------------------------------------------------------
# Signal / noise scoring weights  (from Cell 07)
# ----------------------------------------------------------------

SIGNAL_WEIGHTS = {"volume": 0.45, "action_needed": 0.35, "confidence": 0.20}
NOISE_WEIGHTS = {"dedup_dup_rate": 0.60, "low_confidence": 0.40}

MIN_SIGNAL_THRESHOLD: float = 0.35
HIGH_NOISE_THRESHOLD: float = 0.65
VOLUME_CAP: int = 10


# ----------------------------------------------------------------
# Priority / cadence thresholds  (from Cells 09/10)
# ----------------------------------------------------------------

PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]
CADENCE_OPTIONS = ["Daily", "Weekly", "Bi-weekly", "Monthly"]

# 075 cadence transition thresholds (execution-count based)
WEEKLY_TO_MONTHLY_MISSES: int = 3
MONTHLY_TO_PAUSED_MISSES: int = 3
MONTHLY_CHECK_INTERVAL_DAYS: int = 30

REMOVAL_EFFECTIVENESS_THRESHOLD: float = 0.25
DOWNGRADE_EFFECTIVENESS_THRESHOLD: float = 0.40
UPGRADE_EFFECTIVENESS_THRESHOLD: float = 0.75
NO_EVENT_DAYS_THRESHOLD: int = 14


# ----------------------------------------------------------------
# Keyword proposal controls
# ----------------------------------------------------------------

KEYWORD_MIN_TOKEN_LEN: int = 3
KEYWORD_MAX_SUGGESTIONS: int = 10
KEYWORD_MIN_EVENT_COUNT: int = 2        # token must appear in >= N events to be suggested
KEYWORD_NOISE_CONFIDENCE_THRESHOLD: float = 0.3   # events below this are "low confidence"
KEYWORD_NOISE_RATIO_THRESHOLD: float = 0.6        # >60% low-conf events → noise token

# Stale-keyword safety  (do NOT auto-remove)
# v1 only flags as "stale_candidate"; multi-week tracking is future work.
KEYWORD_STALE_MIN_WEEKS: int = 3  # escalate to "recommend remove" after N consecutive stale weeks

# Stopwords  (EN common + JA particles + URL fragments)
KEYWORD_STOPWORDS = frozenset({
    # English
    "the", "and", "for", "are", "not", "but", "was", "can", "all", "may",
    "its", "does", "how", "this", "that", "with", "from", "into", "over",
    "will", "have", "has", "been", "were", "their", "about", "after",
    "before", "also", "than", "then", "them", "they", "what", "when",
    "where", "which", "while", "said", "says", "new", "more",
    # Japanese particles / common
    "の", "は", "が", "を", "に", "で", "と", "も", "や", "へ", "から",
    "まで", "より", "など", "って", "した", "する", "ある", "いる",
    "この", "その", "あの", "これ", "それ", "あれ",
    # URL fragments
    "http", "https", "www", "com", "org", "net", "html", "php",
})


# ----------------------------------------------------------------
# Schema builder
# ----------------------------------------------------------------

def get_targets_schema(db_id: str) -> dict:
    """Return the full Monitoring Targets schema dict."""
    return {
        "database_id": db_id,
        "all_properties": dict(TARGETS_ALL_PROPERTIES),
        "core_properties": dict(TARGETS_CORE_PROPERTIES),
        "search_properties": dict(TARGETS_SEARCH_PROPERTIES),
        "operational_properties": dict(TARGETS_OPERATIONAL_PROPERTIES),
    }
