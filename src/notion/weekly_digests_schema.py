# src/notion/weekly_digests_schema.py
"""Schema definition for the WEEKLY_DIGESTS_DB Notion database.

Written by: 047_weekly_papers_review.py (row creation)
Updated by: 052_weekly_decision_and_summary.py (synthesis enrichment)

Each row captures one weekly digest — a single summary page per week.
Idempotency key format: ``"{week_id}"`` (e.g. ``"2026-W08"``).

Actual Notion schema (inferred from sample page)
-------------------------------------------------
Name                : title
Key                 : rich_text        <- idempotency key (= week_id)
Status              : select           <- "generated" / "synthesized"
Generated At        : date
Run ID              : rich_text
Week Start          : date
Week End            : date
Confidence          : number
Executive Summary   : rich_text
Macro Shift         : rich_text
Opportunity Signals : rich_text
Risk Signals        : rich_text
Themes              : relation         <- link to WEEKLY_THEMES_DB
RQ Updates          : relation         <- link to WEEKLY_RQ_UPDATE_DB
Target Updates      : relation         <- link to WEEKLY_TARGET_UPDATE_DB
Paper Updates       : relation         <- link to LIT DB (papers reviewed this week)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Set

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Notion property names  (must match the actual Notion UI)
# ----------------------------------------------------------------

PROP_NAME = "Name"                             # title
PROP_KEY = "Key"                               # rich_text — idempotency key
PROP_STATUS = "Status"                         # select
PROP_GENERATED_AT = "Generated At"             # date
PROP_RUN_ID = "Run ID"                         # rich_text
PROP_WEEK_START = "Week Start"                 # date
PROP_WEEK_END = "Week End"                     # date
PROP_CONFIDENCE = "Confidence"                 # number
PROP_EXECUTIVE_SUMMARY = "Executive Summary"   # rich_text
PROP_MACRO_SHIFT = "Macro Shift"               # rich_text
PROP_OPPORTUNITY_SIGNALS = "Opportunity Signals"  # rich_text
PROP_RISK_SIGNALS = "Risk Signals"             # rich_text
PROP_THEMES = "Themes"                         # relation (to WEEKLY_THEMES_DB)
PROP_RQ_UPDATES = "RQ Updates"                 # relation (to WEEKLY_RQ_UPDATE_DB)
PROP_TARGET_UPDATES = "Target Updates"         # relation (to WEEKLY_TARGET_UPDATE_DB)
PROP_PAPER_UPDATES = "Paper Updates"           # relation (to LIT DB)


# Expected Notion property type per property name.
EXPECTED_PROPERTIES: Dict[str, str] = {
    PROP_NAME: "title",
    PROP_KEY: "rich_text",
    PROP_STATUS: "select",
    PROP_GENERATED_AT: "date",
    PROP_RUN_ID: "rich_text",
    PROP_WEEK_START: "date",
    PROP_WEEK_END: "date",
    PROP_CONFIDENCE: "number",
    PROP_EXECUTIVE_SUMMARY: "rich_text",
    PROP_MACRO_SHIFT: "rich_text",
    PROP_OPPORTUNITY_SIGNALS: "rich_text",
    PROP_RISK_SIGNALS: "rich_text",
    PROP_THEMES: "relation",
    PROP_RQ_UPDATES: "relation",
    PROP_TARGET_UPDATES: "relation",
    PROP_PAPER_UPDATES: "relation",
}

# Relation properties — validated as warnings, not errors.
RELATION_PROPERTIES: Set[str] = {
    PROP_THEMES,            # relation to WEEKLY_THEMES_DB
    PROP_RQ_UPDATES,        # relation to WEEKLY_RQ_UPDATE_DB
    PROP_TARGET_UPDATES,    # relation to WEEKLY_TARGET_UPDATE_DB
    PROP_PAPER_UPDATES,     # relation to LIT DB
}

# Properties that are optional (won't fail validation if missing).
# Key is NEVER optional — it is the idempotency backbone.
OPTIONAL_PROPERTIES: Set[str] = {
    PROP_CONFIDENCE,        # optional enrichment field
    PROP_EXECUTIVE_SUMMARY,
    PROP_MACRO_SHIFT,
    PROP_OPPORTUNITY_SIGNALS,
    PROP_RISK_SIGNALS,
}


# ----------------------------------------------------------------
# Validation
# ----------------------------------------------------------------

class WeeklyDigestsSchemaError(ValueError):
    """Raised when the Notion DB schema does not match expectations."""


def validate_weekly_digests_schema(
    db_meta: Mapping[str, Any],
    *,
    allow_missing: Set[str] | None = None,
) -> List[str]:
    """Validate that WEEKLY_DIGESTS_DB has the required properties.

    - Required properties (including Key): raise on missing/wrong type.
    - Relation properties: log WARNING but do NOT raise.
    - Optional properties: silently skipped.

    Returns
    -------
    List[str]
        Names of relation properties that are missing or misconfigured.

    Raises
    ------
    WeeklyDigestsSchemaError
        If any required non-relation property is missing or wrong type.
    """
    skip = set(OPTIONAL_PROPERTIES) | RELATION_PROPERTIES
    if allow_missing:
        skip |= allow_missing
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise WeeklyDigestsSchemaError(
            "WEEKLY_DIGESTS_DB: db_meta.properties missing or invalid"
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
        raise WeeklyDigestsSchemaError(
            "WEEKLY_DIGESTS_DB schema validation failed.\n"
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
                "WEEKLY_DIGESTS_DB: relation property %r not found — "
                "relation writes will be skipped. "
                "Create it manually in Notion when ready.",
                name,
            )
            missing_relations.append(name)
        elif existing[name] != expected_type:
            logger.warning(
                "WEEKLY_DIGESTS_DB: property %r has type %r, expected %r — "
                "relation writes will be skipped.",
                name, existing[name], expected_type,
            )
            missing_relations.append(name)

    return missing_relations
