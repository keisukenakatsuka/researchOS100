# src/notion/alignment_schema.py
"""Schema definition for the ROS_Alignment_Log Notion database.

Written by: 054_values_scale_setup.py (future: daily/weekly reflection scripts)

Each row captures one reflection entry — a daily, weekly, or quarterly
self-assessment against a specific value domain using the two-dimensional
Value Evaluation Scale:
  - Importance Score (1–5)
  - Alignment Score  (1–5)
  - Gap Score        (computed: Importance − Alignment)
  - Significant Gap  (computed: gap >= 2)

Expected Notion schema (ROS_Alignment_Log)
-------------------------------------------
Name                        : title           <- auto-generated title
Date                        : date            <- reflection date
Review Type                 : select          <- "Daily" / "Weekly" / "Quarterly"
Domain                      : relation        <- relation to ROS_Values_Codex
Importance Score            : number          <- 1–5
Alignment Score             : number          <- 1–5
Gap Score                   : number          <- importance − alignment
Significant Gap             : checkbox        <- gap >= 2
Reflection Text             : rich_text
Misalignment Notes          : rich_text
Next Adjustment             : rich_text
Transcript                  : rich_text       <- voice transcription
Audio URL                   : url
AI Summary                  : rich_text       <- LLM-generated summary
Tags                        : multi_select
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Set

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Notion property names  (must match the actual Notion UI)
# ----------------------------------------------------------------

PROP_NAME = "Name"                                       # title
PROP_DATE = "Date"                                       # date
PROP_REVIEW_TYPE = "Review Type"                         # select
PROP_DOMAIN = "Domain"                                   # relation -> ROS_Values_Codex
PROP_IMPORTANCE_SCORE = "Importance Score"                # number (1-5)
PROP_ALIGNMENT_SCORE = "Alignment Score"                 # number (1-5)
PROP_GAP_SCORE = "Gap Score"                             # number (computed)
PROP_SIGNIFICANT_GAP = "Significant Gap"                 # checkbox
PROP_REFLECTION_TEXT = "Reflection Text"                  # rich_text
PROP_MISALIGNMENT_NOTES = "Misalignment Notes"           # rich_text
PROP_NEXT_ADJUSTMENT = "Next Adjustment"                 # rich_text
PROP_TRANSCRIPT = "Transcript"                           # rich_text
PROP_AUDIO_URL = "Audio URL"                             # url
PROP_AI_SUMMARY = "AI Summary"                           # rich_text
PROP_TAGS = "Tags"                                       # multi_select


# Expected Notion property type per property name.
EXPECTED_PROPERTIES: Dict[str, str] = {
    PROP_NAME: "title",
    PROP_DATE: "date",
    PROP_REVIEW_TYPE: "select",
    PROP_DOMAIN: "relation",
    PROP_IMPORTANCE_SCORE: "number",
    PROP_ALIGNMENT_SCORE: "number",
    PROP_GAP_SCORE: "number",
    PROP_SIGNIFICANT_GAP: "checkbox",
    PROP_REFLECTION_TEXT: "rich_text",
    PROP_MISALIGNMENT_NOTES: "rich_text",
    PROP_NEXT_ADJUSTMENT: "rich_text",
    PROP_TRANSCRIPT: "rich_text",
    PROP_AUDIO_URL: "url",
    PROP_AI_SUMMARY: "rich_text",
    PROP_TAGS: "multi_select",
}

# Relation properties — validated as warnings, not errors.
RELATION_PROPERTIES: Set[str] = {
    PROP_DOMAIN,
}

# Properties that are optional (won't fail validation if missing).
OPTIONAL_PROPERTIES: Set[str] = {
    PROP_TRANSCRIPT,
    PROP_AUDIO_URL,
    PROP_AI_SUMMARY,
    PROP_TAGS,
}

# Valid select options.
REVIEW_TYPE_OPTIONS = ("Daily", "Weekly", "Quarterly")


# ----------------------------------------------------------------
# Validation
# ----------------------------------------------------------------

class AlignmentLogSchemaError(ValueError):
    """Raised when the Notion DB schema does not match expectations."""


def validate_alignment_log_schema(
    db_meta: Mapping[str, Any],
    *,
    allow_missing: Set[str] | None = None,
) -> List[str]:
    """Validate that ROS_Alignment_Log has the required properties.

    Returns list of missing relation + optional property names.
    Raises AlignmentLogSchemaError on missing required properties.
    """
    skip = set(OPTIONAL_PROPERTIES) | RELATION_PROPERTIES
    if allow_missing:
        skip |= allow_missing
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise AlignmentLogSchemaError(
            "ROS_Alignment_Log: db_meta.properties missing or invalid"
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
        raise AlignmentLogSchemaError(
            "ROS_Alignment_Log schema validation failed.\n"
            "Please create the following properties manually in Notion "
            "before running --write:\n"
            + "\n".join(errors)
            + "\n\nExisting properties: "
            + ", ".join(sorted(existing.keys()))
        )

    missing_items: list[str] = []
    for name in RELATION_PROPERTIES:
        expected_type = EXPECTED_PROPERTIES.get(name, "relation")
        if name not in existing:
            logger.warning(
                "ROS_Alignment_Log: relation property %r not found — "
                "relation writes will be skipped.",
                name,
            )
            missing_items.append(name)

    for name in OPTIONAL_PROPERTIES:
        if name not in existing:
            logger.warning(
                "ROS_Alignment_Log: optional property %r not found — "
                "field will be skipped during writes.",
                name,
            )
            missing_items.append(name)

    return missing_items
