# src/notion/weekly_rq_update_schema.py
"""Schema definition for the WEEKLY_RQ_UPDATE_DB Notion database.

Written by: 049_weekly_rq_status.py

Each row captures one Research Question's weekly status snapshot.
Idempotency key format: ``"{week_id}::{rq_id}"``.

Actual Notion schema (inferred from sample page)
-------------------------------------------------
Name            : title
Key             : rich_text          ← idempotency key
Week            : relation           ← link to WEEKLY_DIGESTS_DB
Status          : select             ← "Proposed" / "Accepted" / …
Category        : rich_text
Confidence      : number
Update Summary  : rich_text
Open Questions  : rich_text
Research Question : relation         ← link to RQ DB
Evidence Events : relation           ← link to Events DB
Evidence Papers : relation           ← link to Papers DB
Related Theme(s): relation           ← link to WEEKLY_THEMES_DB
Decided At      : date

Schema validation
-----------------
Call ``validate_weekly_rq_update_schema(db_meta)`` with the result of
``GET /databases/{database_id}`` or inferred-from-sample to fail fast.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Set

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Notion property names  (must match the actual Notion UI)
# ----------------------------------------------------------------

PROP_NAME = "Name"                       # title
PROP_KEY = "Key"                         # rich_text — idempotency key
PROP_WEEK = "Week"                       # relation (to WEEKLY_DIGESTS_DB)
PROP_STATUS = "Status"                   # select
PROP_CATEGORY = "Category"              # rich_text
PROP_CONFIDENCE = "Confidence"           # number
PROP_UPDATE_SUMMARY = "Update Summary"   # rich_text
PROP_OPEN_QUESTIONS = "Open Questions"   # rich_text
PROP_RESEARCH_QUESTION = "Research Question"  # relation (to RQ DB)
PROP_EVIDENCE_EVENTS = "Evidence Events"      # relation (to Events DB)
PROP_EVIDENCE_PAPERS = "Evidence Papers"      # relation (to Papers DB)
PROP_RELATED_THEMES = "Related Theme(s)"      # relation (to WEEKLY_THEMES_DB)
PROP_DECIDED_AT = "Decided At"           # date


# Expected Notion property type per property name.
EXPECTED_PROPERTIES: Dict[str, str] = {
    PROP_NAME: "title",
    PROP_KEY: "rich_text",
    PROP_WEEK: "relation",
    PROP_STATUS: "select",
    PROP_CATEGORY: "rich_text",
    PROP_CONFIDENCE: "number",
    PROP_UPDATE_SUMMARY: "rich_text",
    PROP_OPEN_QUESTIONS: "rich_text",
    PROP_RESEARCH_QUESTION: "relation",
    PROP_EVIDENCE_EVENTS: "relation",
    PROP_EVIDENCE_PAPERS: "relation",
    PROP_RELATED_THEMES: "relation",
    PROP_DECIDED_AT: "date",
}

# Relation properties — validated as warnings, not errors.
# These require manual setup in the Notion UI (creating the relation column
# and linking to the target database).  Missing relation properties produce
# a WARNING log but do NOT fail the schema check, allowing writes of all
# non-relation properties to proceed.
RELATION_PROPERTIES: Set[str] = {
    PROP_RESEARCH_QUESTION,   # relation to RQ DB
    PROP_EVIDENCE_EVENTS,     # relation to Events DB
    PROP_EVIDENCE_PAPERS,     # relation to Papers DB
    PROP_RELATED_THEMES,      # relation to WEEKLY_THEMES_DB
    PROP_WEEK,                # relation to WEEKLY_DIGESTS_DB
}

# Properties that are optional (won't fail validation if missing).
# Key is NEVER optional — it is the idempotency backbone.
OPTIONAL_PROPERTIES: Set[str] = {
    PROP_DECIDED_AT,          # may not exist yet
}


# ----------------------------------------------------------------
# Validation
# ----------------------------------------------------------------

class WeeklyRQUpdateSchemaError(ValueError):
    """Raised when the Notion DB schema does not match expectations."""


def validate_weekly_rq_update_schema(
    db_meta: Mapping[str, Any],
    *,
    allow_missing: Set[str] | None = None,
) -> List[str]:
    """Validate that WEEKLY_RQ_UPDATE_DB has the required properties.

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
        Callers should exclude these from write payloads.

    Raises
    ------
    WeeklyRQUpdateSchemaError
        If any required non-relation property is missing or wrong type.
    """
    skip = set(OPTIONAL_PROPERTIES) | RELATION_PROPERTIES
    if allow_missing:
        skip |= allow_missing
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise WeeklyRQUpdateSchemaError(
            "WEEKLY_RQ_UPDATE_DB: db_meta.properties missing or invalid"
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
        raise WeeklyRQUpdateSchemaError(
            "WEEKLY_RQ_UPDATE_DB schema validation failed.\n"
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
                "WEEKLY_RQ_UPDATE_DB: relation property %r not found — "
                "relation writes will be skipped for this property. "
                "Create it manually in Notion when ready.",
                name,
            )
            missing_relations.append(name)
        elif existing[name] != expected_type:
            logger.warning(
                "WEEKLY_RQ_UPDATE_DB: property %r has type %r, expected %r — "
                "relation writes will be skipped.",
                name, existing[name], expected_type,
            )
            missing_relations.append(name)

    return missing_relations
