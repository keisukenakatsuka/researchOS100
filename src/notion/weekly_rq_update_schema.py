# src/notion/weekly_rq_update_schema.py
"""Schema definition for the WEEKLY_RQ_UPDATE_DB Notion database.

Written by: 049_weekly_rq_status.py

Each row captures one categorized revision proposal for a Research Question.
Multiple rows per RQ per week are expected (one per revision).

Idempotency key format: ``"{week_id}::{rq_id}::{category_abbrev}::{idx}"``.

Property list
-------------
The database must have these properties created manually in Notion
before the first ``--write`` run.  The ``EXPECTED_PROPERTIES`` dict
maps each Notion property name to its expected Notion type string.

Schema validation
-----------------
Call ``validate_weekly_rq_update_schema(db_meta)`` with the result of
``GET /databases/{database_id}`` to fail fast if properties are missing.

Note on ``GET /databases/{id}``
-------------------------------
This is the **only** endpoint that returns database property metadata.
It is explicitly allowed in the project's Notion client rules (see
``client.py`` line 10: "Use GET /v1/databases/{database_id} for
metadata/schema").  All *content* queries use ``data_sources``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Set

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Notion property names  (must match what you create in the UI)
# ----------------------------------------------------------------

PROP_NAME = "Name"                   # title (Notion default)
PROP_KEY = "Key"                     # rich_text — idempotency key
PROP_WEEK = "Week"                   # relation (to weekly digest)
PROP_PRIORITY = "Priority"           # rich_text (may not exist in all DBs)
PROP_STATUS = "Status"               # select
PROP_EVIDENCE_COUNT = "Evidence Count"  # number (may not exist in all DBs)
PROP_OPEN_GAPS = "Open Gaps"         # rich_text (backward compat)
PROP_CURRENT_APPROACH = "Current Approach"  # rich_text (backward compat)
PROP_TAGS = "Tags"                   # rich_text (may not exist in all DBs)
PROP_EVIDENCE_DETAIL = "Evidence Detail"  # rich_text (may not exist in all DBs)
PROP_RQ_RELATION = "RQ Relation"     # relation (to RQ DB — may not exist; see "Research Question")
PROP_RESEARCH_QUESTION = "Research Question"  # relation (actual DB name for RQ link)

# Fields present in actual DB
PROP_CATEGORY = "Category"           # rich_text or select
PROP_CONFIDENCE = "Confidence"       # number or rich_text
PROP_DECIDED_AT = "Decided At"       # date or rich_text
PROP_EVIDENCE_EVENTS = "Evidence Events"  # relation
PROP_EVIDENCE_PAPERS = "Evidence Papers"  # relation
PROP_RELATED_THEMES = "Related Theme(s)"  # relation

# New fields for categorized revision proposals (049 rewrite)
PROP_UPDATE_CATEGORY = "Update Category"   # rich_text — "Rationale / Background" | "Gap Identified" | "Proposed Approach"
PROP_UPDATE_SUMMARY = "Update Summary"     # rich_text — その理由 (reason for the revision, Japanese)
PROP_OPEN_QUESTIONS = "Open Questions"     # rich_text — 修正案 (proposed revision text, Japanese)

# Valid categories for Update Category
UPDATE_CATEGORIES = frozenset({
    "Rationale / Background",
    "Gap Identified",
    "Proposed Approach",
})

# Abbreviations for idempotency keys
CATEGORY_ABBREV: Dict[str, str] = {
    "Rationale / Background": "rat",
    "Gap Identified": "gap",
    "Proposed Approach": "app",
}


# Expected Notion property type per property name.
# Used by validate_weekly_rq_update_schema().
#
# NOTE: Types must match what actually exists in the Notion database.
# The actual DB uses relation for Week, select for Status, etc.
EXPECTED_PROPERTIES: Dict[str, str] = {
    PROP_NAME: "title",
    PROP_KEY: "rich_text",
    PROP_UPDATE_SUMMARY: "rich_text",
    PROP_OPEN_QUESTIONS: "rich_text",
}

# Properties that are optional (won't fail validation if missing or
# having a different type).  Callers should check _skip_props before
# writing these.
OPTIONAL_PROPERTIES: Set[str] = {
    PROP_WEEK,             # relation in actual DB — written as relation when present
    PROP_STATUS,           # select in actual DB
    PROP_PRIORITY,         # may not exist
    PROP_EVIDENCE_COUNT,   # may not exist
    PROP_TAGS,             # may not exist
    PROP_EVIDENCE_DETAIL,  # may not exist
    PROP_RQ_RELATION,      # relation requires manual setup
    PROP_RESEARCH_QUESTION, # relation — actual DB name for RQ link
    PROP_UPDATE_CATEGORY,  # new field
    PROP_UPDATE_SUMMARY,   # new field
    PROP_OPEN_QUESTIONS,   # new field
    PROP_OPEN_GAPS,        # backward compat
    PROP_CURRENT_APPROACH, # backward compat
    PROP_CATEGORY,         # present in actual DB
    PROP_CONFIDENCE,       # present in actual DB
    PROP_DECIDED_AT,       # present in actual DB
    PROP_EVIDENCE_EVENTS,  # relation in actual DB
    PROP_EVIDENCE_PAPERS,  # relation in actual DB
    PROP_RELATED_THEMES,   # relation in actual DB
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

    Parameters
    ----------
    db_meta:
        Result of ``GET /databases/{database_id}``  (the *only* endpoint
        that exposes property metadata — this is explicitly allowed by the
        project's Notion client rules; all content queries use
        ``data_sources``).
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
    WeeklyRQUpdateSchemaError
        If any required property is missing or has the wrong type.
    """
    allow = set(OPTIONAL_PROPERTIES)
    if allow_missing:
        allow |= allow_missing
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise WeeklyRQUpdateSchemaError(
            "WEEKLY_RQ_UPDATE_DB: db_meta.properties missing or invalid"
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
        raise WeeklyRQUpdateSchemaError(
            "WEEKLY_RQ_UPDATE_DB schema validation failed.\n"
            "Please create the following properties manually in Notion "
            "before running --write:\n"
            + "\n".join(errors)
            + "\n\nExisting properties: "
            + ", ".join(sorted(existing.keys()))
        )

    # Report optional properties: missing ones should be skipped by builders,
    # present ones are available (callers can check actual_types for format).
    missing_optional: list[str] = []
    for name in OPTIONAL_PROPERTIES:
        if name not in existing:
            logger.info(
                "WEEKLY_RQ_UPDATE_DB: optional property %r not found — "
                "writes for this property will be skipped.",
                name,
            )
            missing_optional.append(name)
        else:
            logger.debug(
                "WEEKLY_RQ_UPDATE_DB: optional property %r present (type=%s)",
                name, existing[name],
            )

    return missing_optional
