# src/notion/weekly_themes_schema.py
"""Schema definition for the WEEKLY_THEMES_DB Notion database.

Written by: 048_weekly_events_digest.py (LLM theme identification)

Each row captures one structural theme identified from the week's events.
Idempotency key format: ``"{week_id}::{theme_name_normalized}"``.

Property list
-------------
Name                : title
Key                 : rich_text        <- idempotency key
Week                : relation         <- link to WEEKLY_DIGESTS_DB (digest row for this week)
Summary             : rich_text
Why It Matters      : rich_text
Source Script       : rich_text        <- "048" (may not exist in all DBs)
Key Events          : relation         <- link to Events DB
Related Targets     : relation         <- link to Monitoring Targets DB
Related RQs         : relation         <- link to RQ DB
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
PROP_SUMMARY = "Summary"                 # rich_text
PROP_WHY_IT_MATTERS = "Why It Matters"   # rich_text
PROP_SOURCE_SCRIPT = "Source Script"     # rich_text (may not exist in all DBs)
PROP_KEY_EVENTS = "Key Events"           # relation (to Events DB)
PROP_RELATED_TARGETS = "Related Targets" # relation (to Monitoring Targets DB)
PROP_RELATED_RQS = "Related RQs"        # relation (to RQ DB)


# Expected Notion property type per property name.
EXPECTED_PROPERTIES: Dict[str, str] = {
    PROP_NAME: "title",
    PROP_KEY: "rich_text",
    PROP_WEEK: "relation",
    PROP_SUMMARY: "rich_text",
    PROP_WHY_IT_MATTERS: "rich_text",
    PROP_SOURCE_SCRIPT: "rich_text",
    PROP_KEY_EVENTS: "relation",
    PROP_RELATED_TARGETS: "relation",
    PROP_RELATED_RQS: "relation",
}

# Relation properties — validated as warnings, not errors.
RELATION_PROPERTIES: Set[str] = {
    PROP_WEEK,
    PROP_KEY_EVENTS,
    PROP_RELATED_TARGETS,
    PROP_RELATED_RQS,
}

# Properties that are optional (won't fail validation if missing).
OPTIONAL_PROPERTIES: Set[str] = {
    PROP_WHY_IT_MATTERS,
    PROP_SOURCE_SCRIPT,
}


# ----------------------------------------------------------------
# Validation
# ----------------------------------------------------------------

class WeeklyThemesSchemaError(ValueError):
    """Raised when the Notion DB schema does not match expectations."""


def validate_weekly_themes_schema(
    db_meta: Mapping[str, Any],
    *,
    allow_missing: Set[str] | None = None,
) -> List[str]:
    """Validate that WEEKLY_THEMES_DB has the required properties.

    - Required properties: raise on missing/wrong type.
    - Relation properties: log WARNING but do NOT raise.
    - Optional properties: silently skipped.

    Returns
    -------
    List[str]
        Names of relation properties that are missing or misconfigured.
    """
    skip = set(OPTIONAL_PROPERTIES) | RELATION_PROPERTIES
    if allow_missing:
        skip |= allow_missing
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise WeeklyThemesSchemaError(
            "WEEKLY_THEMES_DB: db_meta.properties missing or invalid"
        )

    existing = {name: p.get("type") for name, p in props.items()}

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
        raise WeeklyThemesSchemaError(
            "WEEKLY_THEMES_DB schema validation failed.\n"
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
                "WEEKLY_THEMES_DB: relation property %r not found — "
                "relation writes will be skipped.",
                name,
            )
            missing_relations.append(name)
        elif existing[name] != expected_type:
            logger.warning(
                "WEEKLY_THEMES_DB: property %r has type %r, expected %r — "
                "relation writes will be skipped.",
                name, existing[name], expected_type,
            )
            missing_relations.append(name)

    return missing_relations
