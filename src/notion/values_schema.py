# src/notion/values_schema.py
"""Schema definition for the ROS_Values_Codex Notion database.

Written by: 054_values_scale_setup.py (row creation via upsert)

Each row captures one value domain definition. There is one row per
domain per review quarter (12 rows per quarter), with revision support
for within-quarter updates.

Idempotency key format: ``"{review_quarter}:{domain_id}"``
    e.g. ``"2026-Q1:career_work"``

Expected Notion schema (ROS_Values_Codex)
------------------------------------------
Name                        : title           <- domain label
Domain Key                  : rich_text       <- stable domain identifier
Review Quarter              : rich_text       <- "2026-Q1"
Idempotency Key             : rich_text       <- "{quarter}:{domain_id}"
Value Definition            : rich_text
Behavioral Translation      : rich_text
Example Behaviors           : rich_text       <- newline-separated behaviors
Misalignment Description    : rich_text
Reflection Questions        : rich_text       <- newline-separated questions
Micro Habits                : rich_text       <- newline-separated habits
Status                      : select          <- "seed" / "reviewed" / "refined"
Source                      : select          <- "Manual" / "LLM" / "Hybrid"
Version                     : number          <- schema version (int)
Revision                    : number          <- within-quarter revision (int)
Last Updated                : date
Change Notes                : rich_text
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Set

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Notion property names  (must match the actual Notion UI)
# ----------------------------------------------------------------

PROP_NAME = "Name"                                       # title
PROP_DOMAIN_KEY = "Domain Key"                           # rich_text
PROP_REVIEW_QUARTER = "Review Quarter"                   # select
PROP_IDEMPOTENCY_KEY = "Idempotency Key"                 # rich_text
PROP_VALUE_DEFINITION = "Value Definition"               # rich_text
PROP_BEHAVIORAL_TRANSLATION = "Behavioral Translation"   # rich_text
PROP_EXAMPLE_BEHAVIORS = "Example Behaviors"             # rich_text
PROP_MISALIGNMENT = "Misalignment Description"           # rich_text
PROP_REFLECTION_QUESTIONS = "Reflection Questions"       # rich_text
PROP_MICRO_HABITS = "Micro Habits"                       # rich_text
PROP_STATUS = "Status"                                   # select
PROP_SOURCE = "Source"                                   # select
PROP_VERSION = "Version"                                 # number
PROP_REVISION = "Revision"                               # number
PROP_LAST_UPDATED = "Last Updated"                       # date
PROP_CHANGE_NOTES = "Change Notes"                       # rich_text


# Expected Notion property type per property name.
EXPECTED_PROPERTIES: Dict[str, str] = {
    PROP_NAME: "title",
    PROP_DOMAIN_KEY: "rich_text",
    PROP_REVIEW_QUARTER: "select",
    PROP_IDEMPOTENCY_KEY: "rich_text",
    PROP_VALUE_DEFINITION: "rich_text",
    PROP_BEHAVIORAL_TRANSLATION: "rich_text",
    PROP_EXAMPLE_BEHAVIORS: "rich_text",
    PROP_MISALIGNMENT: "rich_text",
    PROP_REFLECTION_QUESTIONS: "rich_text",
    PROP_MICRO_HABITS: "rich_text",
    PROP_STATUS: "select",
    PROP_SOURCE: "select",
    PROP_VERSION: "number",
    PROP_REVISION: "number",
    PROP_LAST_UPDATED: "date",
    PROP_CHANGE_NOTES: "rich_text",
}

# Properties that are optional (won't fail validation if missing).
OPTIONAL_PROPERTIES: Set[str] = {
    PROP_REFLECTION_QUESTIONS,
    PROP_MICRO_HABITS,
    PROP_CHANGE_NOTES,
}

# Valid select options.
STATUS_OPTIONS = ("seed", "reviewed", "refined")
SOURCE_OPTIONS = ("Manual", "LLM", "Hybrid")


# ----------------------------------------------------------------
# Validation
# ----------------------------------------------------------------

class ValuesCodexSchemaError(ValueError):
    """Raised when the Notion DB schema does not match expectations."""


def validate_values_codex_schema(
    db_meta: Mapping[str, Any],
    *,
    allow_missing: Set[str] | None = None,
) -> List[str]:
    """Validate that ROS_Values_Codex has the required properties.

    Returns
    -------
    List[str]
        Names of optional properties that are missing (informational).

    Raises
    ------
    ValuesCodexSchemaError
        If any required property is missing or has the wrong type.
    """
    skip = set(OPTIONAL_PROPERTIES)
    if allow_missing:
        skip |= allow_missing
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise ValuesCodexSchemaError(
            "ROS_Values_Codex: db_meta.properties missing or invalid"
        )

    existing = {name: p.get("type") for name, p in props.items()}

    errors: list[str] = []
    for name, expected_type in EXPECTED_PROPERTIES.items():
        if name in skip:
            continue
        if name not in existing:
            errors.append(
                f"  - Missing property: {name!r} (expected type: {expected_type})"
            )
        elif existing[name] != expected_type:
            errors.append(
                f"  - Wrong type for {name!r}: "
                f"got {existing[name]!r}, expected {expected_type!r}"
            )

    if errors:
        raise ValuesCodexSchemaError(
            "ROS_Values_Codex schema validation failed.\n"
            "Please create the following properties manually in Notion "
            "before running --write:\n"
            + "\n".join(errors)
            + "\n\nExisting properties: "
            + ", ".join(sorted(existing.keys()))
        )

    # Report missing optionals as warnings
    missing_optional: list[str] = []
    for name in OPTIONAL_PROPERTIES:
        if name not in existing:
            logger.warning(
                "ROS_Values_Codex: optional property %r not found — "
                "field will be skipped during writes.",
                name,
            )
            missing_optional.append(name)

    return missing_optional
