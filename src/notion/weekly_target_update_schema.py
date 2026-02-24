# src/notion/weekly_target_update_schema.py
"""Schema definition for the WEEKLY_TARGET_UPDATE_DB Notion database.

Written by:
  - 050_weekly_targets_review.py  (existing target reviews)
  - 051_weekly_discovery_expansion.py  (new discovery candidates)

Each row captures either a weekly target review (050) or a discovery
candidate (051).  The ``Source Script`` field distinguishes them.

Idempotency key formats
-----------------------
- 050: ``"{week_id}::{target_id}"``
- 051: ``"{week_id}::disc::{type}::{candidate_name_normalized}"``

The 051 key includes ``type`` to prevent collisions between e.g.
a "Sequoia" entity categorised as VC vs Startup across runs.

Property list
-------------
The database must have these properties created manually in Notion
before the first ``--write`` run.  The ``EXPECTED_PROPERTIES`` dict
maps each Notion property name to its expected Notion type string.

Note on ``GET /databases/{id}``
-------------------------------
This is the **only** endpoint that returns database property metadata.
It is explicitly allowed in the project's Notion client rules (see
``client.py`` line 10).  All *content* queries use ``data_sources``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Set

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Notion property names  (must match what you create in the UI)
# ----------------------------------------------------------------

PROP_NAME = "Name"                     # title (Notion default)
PROP_KEY = "Key"                       # rich_text — idempotency key
PROP_WEEK = "Week"                     # relation (to weekly digest)
PROP_SOURCE_SCRIPT = "Source Script"   # rich_text — "050" or "051"
PROP_ENTITY_TYPE = "Entity Type"       # select — VC / Startup / Policy / People / Other
PROP_ACTION = "Action"                 # select — keep / drop_candidate / review  (050 only)
PROP_CURRENT_PRIORITY = "Current Priority"   # rich_text  (050 only, may not exist)
PROP_PROPOSED_PRIORITY = "Proposed Priority" # rich_text  (050 only, may not exist)
PROP_CURRENT_CADENCE = "Current Cadence"     # rich_text  (050 only, may not exist)
PROP_PROPOSED_CADENCE = "Proposed Cadence"   # rich_text  (050 only, may not exist)
PROP_SIGNAL_SCORE = "Signal Score"     # number  (050 only)
PROP_NOISE_SCORE = "Noise Score"       # number  (050 only)
PROP_EVENT_COUNT = "Event Count"       # number  (both: number_of_events / mention_count)
PROP_DAYS_SINCE_LAST = "Days Since Last Event"  # number  (050 only)
PROP_RECENCY_SCORE = "Recency Score"   # number  (050 only)
PROP_FINAL_SCORE = "Final Score"       # number  (051 only)
PROP_REASON = "Reason"                 # rich_text  (both: reason / why_notable)
PROP_ALIASES = "Aliases"               # multi_select in actual DB  (051 only)
PROP_ALREADY_TRACKED = "Already Tracked"  # checkbox  (051 only)
PROP_KEYWORDS_TO_ADD = "Keywords To Add"  # multi_select in actual DB  (050 only)
PROP_KEYWORDS_STALE = "Keywords Stale"    # multi_select in actual DB  (050 only)
PROP_EVIDENCE_DETAIL = "Evidence Detail"  # rich_text  (051 only)
PROP_TARGET_RELATION = "Target Relation"  # relation  (050 only — link to Monitoring Targets DB)
PROP_SUGGESTED_KEYWORDS = "Suggested Keywords"  # rich_text  (050 only — external search suggestions)
PROP_SUGGESTED_URLS = "Suggested URLs"          # rich_text  (050 only — external search suggestions)

# Additional properties present in the actual DB
PROP_PRIORITY = "Priority"             # rich_text or select
PROP_STATUS = "Status"                 # select
PROP_CONFIDENCE = "Confidence"         # number or rich_text
PROP_CHANGE_SUMMARY = "Change Summary" # rich_text
PROP_RATIONALE = "Rationale"           # rich_text
PROP_TAGS = "Tags"                     # rich_text or multi_select
PROP_EVIDENCE_COUNT = "Evidence Count"  # number
PROP_EVIDENCE_EVENTS = "Evidence Events"  # relation
PROP_RQ_RELATION = "RQ Relation"       # relation
PROP_RELATED_THEMES = "Related Theme(s)" # relation


# Expected Notion property type per property name.
# Only truly required properties go here (Name + Key).
# Everything else is optional and handled gracefully.
EXPECTED_PROPERTIES: Dict[str, str] = {
    PROP_NAME: "title",
    PROP_KEY: "rich_text",
}

# Properties that are optional (won't fail validation if missing).
# The builders check _skip_props before writing these.
OPTIONAL_PROPERTIES: Set[str] = {
    PROP_WEEK,               # relation in actual DB
    PROP_SOURCE_SCRIPT,      # rich_text
    PROP_ENTITY_TYPE,        # select in actual DB
    PROP_ACTION,             # select in actual DB
    PROP_CURRENT_PRIORITY,   # may not exist
    PROP_PROPOSED_PRIORITY,  # may not exist
    PROP_CURRENT_CADENCE,    # may not exist
    PROP_PROPOSED_CADENCE,   # may not exist
    PROP_SIGNAL_SCORE,       # number
    PROP_NOISE_SCORE,        # number
    PROP_EVENT_COUNT,        # number
    PROP_DAYS_SINCE_LAST,    # number
    PROP_RECENCY_SCORE,      # number
    PROP_FINAL_SCORE,        # number
    PROP_REASON,             # rich_text
    PROP_ALIASES,            # multi_select in actual DB
    PROP_ALREADY_TRACKED,    # checkbox
    PROP_KEYWORDS_TO_ADD,    # multi_select in actual DB
    PROP_KEYWORDS_STALE,     # multi_select in actual DB
    PROP_EVIDENCE_DETAIL,    # rich_text
    PROP_TARGET_RELATION,    # relation
    PROP_SUGGESTED_KEYWORDS, # rich_text
    PROP_SUGGESTED_URLS,     # rich_text
    PROP_PRIORITY,           # present in actual DB
    PROP_STATUS,             # present in actual DB (select)
    PROP_CONFIDENCE,         # present in actual DB
    PROP_CHANGE_SUMMARY,     # present in actual DB
    PROP_RATIONALE,          # present in actual DB
    PROP_TAGS,               # present in actual DB
    PROP_EVIDENCE_COUNT,     # present in actual DB
    PROP_EVIDENCE_EVENTS,    # present in actual DB (relation)
    PROP_RQ_RELATION,        # present in actual DB (relation)
    PROP_RELATED_THEMES,     # present in actual DB (relation)
}


# ----------------------------------------------------------------
# Validation
# ----------------------------------------------------------------

class WeeklyTargetUpdateSchemaError(ValueError):
    """Raised when the Notion DB schema does not match expectations."""


def validate_weekly_target_update_schema(
    db_meta: Mapping[str, Any],
    *,
    allow_missing: Set[str] | None = None,
) -> List[str]:
    """Validate that WEEKLY_TARGET_UPDATE_DB has the required properties.

    Parameters
    ----------
    db_meta:
        Result of ``GET /databases/{database_id}``  (the *only* endpoint
        that exposes property metadata — explicitly allowed by the project's
        Notion client rules; all content queries use ``data_sources``).
    allow_missing:
        Extra property names to treat as optional beyond
        ``OPTIONAL_PROPERTIES``.

    Returns
    -------
    List[str]
        Names of optional properties that are missing or misconfigured
        (callers should skip these in property builders).

    Raises
    ------
    WeeklyTargetUpdateSchemaError
        If any required property is missing or has the wrong type.
    """
    allow = set(OPTIONAL_PROPERTIES)
    if allow_missing:
        allow |= allow_missing
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise WeeklyTargetUpdateSchemaError(
            "WEEKLY_TARGET_UPDATE_DB: db_meta.properties missing or invalid"
        )

    existing = {name: p.get("type") for name, p in props.items()}
    errors: list[str] = []

    for name, expected_type in EXPECTED_PROPERTIES.items():
        if name in allow:
            continue
        if name not in existing:
            errors.append(f"  - Missing property: {name!r} (expected type: {expected_type})")
        elif existing[name] != expected_type:
            errors.append(
                f"  - Wrong type for {name!r}: "
                f"got {existing[name]!r}, expected {expected_type!r}"
            )

    if errors:
        raise WeeklyTargetUpdateSchemaError(
            "WEEKLY_TARGET_UPDATE_DB schema validation failed.\n"
            "Please create the following properties manually in Notion "
            "before running --write:\n"
            + "\n".join(errors)
            + "\n\nExisting properties: "
            + ", ".join(sorted(existing.keys()))
        )

    # Report optional properties: missing ones should be skipped by builders.
    missing_optional: list[str] = []
    for name in OPTIONAL_PROPERTIES:
        if name not in existing:
            logger.info(
                "WEEKLY_TARGET_UPDATE_DB: optional property %r not found — "
                "writes for this property will be skipped.",
                name,
            )
            missing_optional.append(name)
        else:
            logger.debug(
                "WEEKLY_TARGET_UPDATE_DB: optional property %r present (type=%s)",
                name, existing[name],
            )

    return missing_optional
