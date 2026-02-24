# src/notion/weekly_target_additional_schema.py
"""Schema definition for the WEEKLY_TARGET_ADDITIONAL_DB Notion database.

Written by: 051_weekly_discovery_expansion.py

Each row captures a NEW entity proposal discovered from weekly events/papers.
These are distinct from existing monitoring targets (050).

Idempotency key format: ``"{week_id}::add::{type}::{candidate_name_normalized}"``

Property list
-------------
The database must have these properties created manually in Notion
before the first ``--write`` run.  The ``EXPECTED_PROPERTIES`` dict
maps each Notion property name to its expected Notion type string.

Schema validation
-----------------
Call ``validate_weekly_target_additional_schema(db_meta)`` with the result of
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

PROP_NAME = "Name"                    # title (Notion default)
PROP_KEY = "Key"                      # rich_text — idempotency key
PROP_WEEK = "Week"                    # rich_text
PROP_ENTITY_TYPE = "Type"             # select: VC / STARTUP / POLICY / PEOPLE / Other
PROP_STATUS = "Status"                # select: ACTIVE
PROP_PRIORITY = "Priority"            # select: High / Medium
PROP_CADENCE = "Cadence"              # select: DAILY / WEEKLY
PROP_SOURCE_TYPE = "Source Type"      # select: HTML / RSS / NEWS
PROP_DECISION = "Decision"            # select: Proposal / Accept / Reject / Defer
PROP_SEARCH_KEYWORDS = "Search Keywords"  # rich_text
PROP_SOURCE_URLS = "Source URLs"      # rich_text
PROP_EVIDENCE = "Evidence"            # rich_text — why this entity matters
PROP_SOURCE_SCRIPT = "Source Script"  # rich_text — "051"
PROP_ALIASES = "Aliases"              # rich_text — alternative names
PROP_MENTION_COUNT = "Mention Count"  # number
PROP_FINAL_SCORE = "Final Score"      # number


# Expected Notion property type per property name.
# Only truly required property is Name (title).
EXPECTED_PROPERTIES: Dict[str, str] = {
    PROP_NAME: "title",
}

# Properties that are optional (won't fail validation if missing).
# Builders check _skip_props before writing these.
OPTIONAL_PROPERTIES: Set[str] = {
    PROP_KEY,            # rich_text — may not exist yet
    PROP_WEEK,           # rich_text — may not exist yet
    PROP_ENTITY_TYPE,    # select
    PROP_STATUS,         # select
    PROP_PRIORITY,       # select
    PROP_CADENCE,        # select
    PROP_SOURCE_TYPE,    # select
    PROP_DECISION,       # select
    PROP_SEARCH_KEYWORDS,# rich_text
    PROP_SOURCE_URLS,    # rich_text
    PROP_EVIDENCE,       # rich_text — may not exist yet
    PROP_SOURCE_SCRIPT,  # rich_text
    PROP_ALIASES,        # rich_text
    PROP_MENTION_COUNT,  # number
    PROP_FINAL_SCORE,    # number
}


# ----------------------------------------------------------------
# Validation
# ----------------------------------------------------------------

class WeeklyTargetAdditionalSchemaError(ValueError):
    """Raised when the Notion DB schema does not match expectations."""


def validate_weekly_target_additional_schema(
    db_meta: Mapping[str, Any],
    *,
    allow_missing: Set[str] | None = None,
) -> List[str]:
    """Validate that WEEKLY_TARGET_ADDITIONAL_DB has the required properties.

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
    WeeklyTargetAdditionalSchemaError
        If any required property is missing or has the wrong type.
    """
    allow = set(OPTIONAL_PROPERTIES)
    if allow_missing:
        allow |= allow_missing
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise WeeklyTargetAdditionalSchemaError(
            "WEEKLY_TARGET_ADDITIONAL_DB: db_meta.properties missing or invalid"
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
        raise WeeklyTargetAdditionalSchemaError(
            "WEEKLY_TARGET_ADDITIONAL_DB schema validation failed.\n"
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
                "WEEKLY_TARGET_ADDITIONAL_DB: optional property %r not found — "
                "writes for this property will be skipped.",
                name,
            )
            missing_optional.append(name)
        else:
            logger.debug(
                "WEEKLY_TARGET_ADDITIONAL_DB: optional property %r present (type=%s)",
                name, existing[name],
            )

    return missing_optional
