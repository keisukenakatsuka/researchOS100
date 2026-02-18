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

Actual Notion schema (inferred from sample page — merged superset)
------------------------------------------------------------------
Name                : title
Key                 : rich_text        ← idempotency key
Week                : relation         ← link to WEEKLY_DIGESTS_DB
Status              : select           ← "Proposed" / "Accepted" / …
Source Script       : rich_text        ← "050" or "051"
Entity Type         : select
Action              : select           ← keep / drop_candidate / review
Priority            : select
Confidence          : number
Signal Score        : number           (050)
Noise Score         : number           (050)
Event Count         : number           (both)
Days Since Last Event: number          (050)
Recency Score       : number           (050)
Final Score         : number           (051)
Evidence Count      : number           (049-style, reused)
Reason              : rich_text        (both)
Aliases             : multi_select     (051)
Already Tracked     : checkbox         (051)
Keywords To Add     : multi_select     (050)
Keywords Stale      : multi_select     (050)
Evidence Detail     : rich_text        (051)
Open Gaps           : rich_text        (reused)
Current Approach    : rich_text        (reused)
Current Value       : rich_text        (052 decision)
Proposed Value      : rich_text        (052 decision)
Change Summary      : rich_text        (052 decision)
Rationale           : rich_text        (052 decision)
Proposal Type       : select           (052 decision)
Field               : select           (052 decision)
Decided At          : date             (052 decision)
Tags                : multi_select
Target              : relation         ← link to Monitoring Targets DB
Target Relation     : relation         ← (legacy, kept for compat)
RQ Relation         : relation         ← link to RQ DB
Evidence Events     : relation         ← link to Events DB
Related Theme(s)    : relation         ← link to WEEKLY_THEMES_DB
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Set

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Notion property names  (must match the actual Notion UI)
# ----------------------------------------------------------------

PROP_NAME = "Name"                           # title
PROP_KEY = "Key"                             # rich_text — idempotency key
PROP_WEEK = "Week"                           # relation
PROP_STATUS = "Status"                       # select
PROP_SOURCE_SCRIPT = "Source Script"         # rich_text
PROP_ENTITY_TYPE = "Entity Type"             # select
PROP_ACTION = "Action"                       # select
PROP_PRIORITY = "Priority"                   # select
PROP_CONFIDENCE = "Confidence"               # number
PROP_SIGNAL_SCORE = "Signal Score"           # number
PROP_NOISE_SCORE = "Noise Score"             # number
PROP_EVENT_COUNT = "Event Count"             # number
PROP_DAYS_SINCE_LAST = "Days Since Last Event"  # number
PROP_RECENCY_SCORE = "Recency Score"         # number
PROP_FINAL_SCORE = "Final Score"             # number
PROP_EVIDENCE_COUNT = "Evidence Count"       # number
PROP_REASON = "Reason"                       # rich_text
PROP_ALIASES = "Aliases"                     # multi_select
PROP_ALREADY_TRACKED = "Already Tracked"     # checkbox
PROP_KEYWORDS_TO_ADD = "Keywords To Add"     # multi_select
PROP_KEYWORDS_STALE = "Keywords Stale"       # multi_select
PROP_EVIDENCE_DETAIL = "Evidence Detail"     # rich_text
PROP_OPEN_GAPS = "Open Gaps"                 # rich_text
PROP_CURRENT_APPROACH = "Current Approach"   # rich_text
PROP_CURRENT_VALUE = "Current Value"         # rich_text
PROP_PROPOSED_VALUE = "Proposed Value"       # rich_text
PROP_CHANGE_SUMMARY = "Change Summary"       # rich_text
PROP_RATIONALE = "Rationale"                 # rich_text
PROP_PROPOSAL_TYPE = "Proposal Type"         # select
PROP_FIELD = "Field"                         # select
PROP_DECIDED_AT = "Decided At"               # date
PROP_TAGS = "Tags"                           # multi_select
PROP_TARGET = "Target"                       # relation
PROP_TARGET_RELATION = "Target Relation"     # relation (legacy)
PROP_RQ_RELATION = "RQ Relation"             # relation
PROP_EVIDENCE_EVENTS = "Evidence Events"     # relation
PROP_RELATED_THEMES = "Related Theme(s)"     # relation


# Expected Notion property type per property name.
EXPECTED_PROPERTIES: Dict[str, str] = {
    PROP_NAME: "title",
    PROP_KEY: "rich_text",
    PROP_WEEK: "relation",
    PROP_STATUS: "select",
    PROP_SOURCE_SCRIPT: "rich_text",
    PROP_ENTITY_TYPE: "select",
    PROP_ACTION: "select",
    PROP_PRIORITY: "select",
    PROP_CONFIDENCE: "number",
    PROP_SIGNAL_SCORE: "number",
    PROP_NOISE_SCORE: "number",
    PROP_EVENT_COUNT: "number",
    PROP_DAYS_SINCE_LAST: "number",
    PROP_RECENCY_SCORE: "number",
    PROP_FINAL_SCORE: "number",
    PROP_EVIDENCE_COUNT: "number",
    PROP_REASON: "rich_text",
    PROP_ALIASES: "multi_select",
    PROP_ALREADY_TRACKED: "checkbox",
    PROP_KEYWORDS_TO_ADD: "multi_select",
    PROP_KEYWORDS_STALE: "multi_select",
    PROP_EVIDENCE_DETAIL: "rich_text",
    PROP_OPEN_GAPS: "rich_text",
    PROP_CURRENT_APPROACH: "rich_text",
    PROP_CURRENT_VALUE: "rich_text",
    PROP_PROPOSED_VALUE: "rich_text",
    PROP_CHANGE_SUMMARY: "rich_text",
    PROP_RATIONALE: "rich_text",
    PROP_PROPOSAL_TYPE: "select",
    PROP_FIELD: "select",
    PROP_DECIDED_AT: "date",
    PROP_TAGS: "multi_select",
    PROP_TARGET: "relation",
    PROP_TARGET_RELATION: "relation",
    PROP_RQ_RELATION: "relation",
    PROP_EVIDENCE_EVENTS: "relation",
    PROP_RELATED_THEMES: "relation",
}

# Relation properties — validated as warnings, not errors.
# These require manual setup in the Notion UI.  Missing relation properties
# produce a WARNING log but do NOT fail the schema check.
RELATION_PROPERTIES: Set[str] = {
    PROP_WEEK,
    PROP_TARGET,
    PROP_TARGET_RELATION,
    PROP_RQ_RELATION,
    PROP_EVIDENCE_EVENTS,
    PROP_RELATED_THEMES,
}

# Properties that are optional (won't fail validation if missing).
# Key is NEVER optional — it is the idempotency backbone.
OPTIONAL_PROPERTIES: Set[str] = {
    # 052 decision-engine fields — may not exist yet
    PROP_CURRENT_VALUE,
    PROP_PROPOSED_VALUE,
    PROP_CHANGE_SUMMARY,
    PROP_RATIONALE,
    PROP_PROPOSAL_TYPE,
    PROP_FIELD,
    PROP_DECIDED_AT,
    # Fields used by subset of scripts
    PROP_OPEN_GAPS,
    PROP_CURRENT_APPROACH,
    PROP_EVIDENCE_COUNT,
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

    - Required properties (including Key): raise on missing/wrong type.
    - Relation properties: log WARNING but do NOT raise.
    - Optional properties: silently skipped.

    Parameters
    ----------
    db_meta:
        Result of ``GET /databases/{database_id}`` or inferred schema dict.
    allow_missing:
        Extra property names to treat as optional beyond
        ``OPTIONAL_PROPERTIES``.

    Returns
    -------
    List[str]
        Names of relation properties that are missing or misconfigured.

    Raises
    ------
    WeeklyTargetUpdateSchemaError
        If any required non-relation property is missing or wrong type.
    """
    skip = set(OPTIONAL_PROPERTIES) | RELATION_PROPERTIES
    if allow_missing:
        skip |= allow_missing
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise WeeklyTargetUpdateSchemaError(
            "WEEKLY_TARGET_UPDATE_DB: db_meta.properties missing or invalid"
        )

    existing = {name: p.get("type") for name, p in props.items()}

    # --- Hard errors: required non-relation properties ---
    errors: list[str] = []
    for name, expected_type in EXPECTED_PROPERTIES.items():
        if name in skip:
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

    # --- Warnings: relation properties ---
    missing_relations: list[str] = []
    for name in RELATION_PROPERTIES:
        expected_type = EXPECTED_PROPERTIES.get(name, "relation")
        if name not in existing:
            logger.warning(
                "WEEKLY_TARGET_UPDATE_DB: relation property %r not found — "
                "relation writes will be skipped for this property. "
                "Create it manually in Notion when ready.",
                name,
            )
            missing_relations.append(name)
        elif existing[name] != expected_type:
            logger.warning(
                "WEEKLY_TARGET_UPDATE_DB: property %r has type %r, expected %r — "
                "relation writes will be skipped.",
                name, existing[name], expected_type,
            )
            missing_relations.append(name)

    return missing_relations
