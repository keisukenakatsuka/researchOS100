# src/notion/events_schema.py
"""Schema definition for the Events Notion database.

Extracted from notebook 041_weekly_events_digest Cells 02/05.
Reusable by 048+ scripts and any weekly events pipeline.

Property groups
---------------
- **EVENTS_CORE_PROPERTIES**: immutable event metadata (title, dates,
  type, source, summary, URL, confidence, target relations).
- **EVENTS_OPERATIONAL_PROPERTIES**: ingestion/run tracking fields
  (status, dedup key, run ID, ingested-at, action-needed).
- **EVENTS_RELATION_PROPERTIES**: Notion relation fields that link
  events to other databases (related papers, targets).

``EVENTS_REQUIRED_PROPERTIES`` is the union of all groups.

Usage::

    from src.notion.events_schema import (
        get_events_schema,
        EVENTS_CORE_PROPERTIES,
        EVENTS_REQUIRED_PROPERTIES,
    )
"""

from __future__ import annotations

from typing import Dict


# ----------------------------------------------------------------
# Property groups  (property name -> Notion property type)
# ----------------------------------------------------------------

EVENTS_CORE_PROPERTIES: Dict[str, str] = {
    "Name": "title",
    "Date": "date",
    "Detected At": "date",
    "Event Type": "select",
    "Source": "select",
    "Source URL": "url",
    "Summary": "rich_text",
    "Confidence": "number",
}

EVENTS_OPERATIONAL_PROPERTIES: Dict[str, str] = {
    "Status": "select",
    "Dedup Key": "rich_text",
    "Run ID": "rich_text",
    "Ingested At": "date",
    "Action Needed": "checkbox",
}

EVENTS_RELATION_PROPERTIES: Dict[str, str] = {
    "Target": "relation",
    "Related Papers": "relation",
}

# Backward-compatible union of all groups.
EVENTS_REQUIRED_PROPERTIES: Dict[str, str] = {
    **EVENTS_CORE_PROPERTIES,
    **EVENTS_OPERATIONAL_PROPERTIES,
    **EVENTS_RELATION_PROPERTIES,
}

# ----------------------------------------------------------------
# Filtering / noise defaults
# ----------------------------------------------------------------

DEFAULT_MIN_CONFIDENCE: float = 0.3
DEFAULT_EXCLUDED_STATUSES = frozenset({"archived", "deleted", "spam"})
DEFAULT_EXCLUDED_SOURCES = frozenset({"test", "debug"})

# Clustering defaults
DEFAULT_THEME_OVERLAP_THRESHOLD: float = 0.4
DEFAULT_TOP_N_THEMES: int = 5
DEFAULT_MAX_EVENTS_PER_THEME: int = 10


# ----------------------------------------------------------------
# Schema builder
# ----------------------------------------------------------------

def get_events_schema(db_id: str) -> dict:
    """Return the full EVENTS_SCHEMA dict for a given database ID.

    Parameters
    ----------
    db_id:
        The Notion database / data-source ID (``NOTION_EVENTS_DB_ID``).

    Returns
    -------
    dict
        Keys: ``database_id``, ``required_properties``,
        ``core_properties``, ``operational_properties``,
        ``relation_properties``, ``optional_properties``.
    """
    return {
        "database_id": db_id,
        "required_properties": dict(EVENTS_REQUIRED_PROPERTIES),
        "core_properties": dict(EVENTS_CORE_PROPERTIES),
        "operational_properties": dict(EVENTS_OPERATIONAL_PROPERTIES),
        "relation_properties": dict(EVENTS_RELATION_PROPERTIES),
        "optional_properties": {},
    }
