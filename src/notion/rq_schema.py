# src/notion/rq_schema.py
"""Schema definition for the Research Questions (RQ) Notion database.

Extracted from notebook 042_weekly_rq_status Cells 05/08.
Reusable by 049+ scripts and any RQ pipeline.

Property groups
---------------
- **RQ_CORE_PROPERTIES**: identity and status fields
  (name, status, priority, tags).
- **RQ_CONTENT_PROPERTIES**: substantive research-question text fields
  (rationale, approach, gap).

``RQ_ALL_PROPERTIES`` is the union of both groups.

Usage::

    from src.notion.rq_schema import (
        RQ_ALL_PROPERTIES,
        RQ_CONTENT_PROPERTIES,
        DEFAULT_TARGET_PRIORITIES,
    )
"""

from __future__ import annotations

from typing import Dict


# ----------------------------------------------------------------
# Property groups  (property name -> Notion property type)
# ----------------------------------------------------------------

RQ_CORE_PROPERTIES: Dict[str, str] = {
    "Name": "title",
    "Status": "select",      # e.g. "Under Review", "Active"
    "Priority": "select",    # e.g. "High", "Medium", "Low"
    "Tags": "multi_select",
}

RQ_CONTENT_PROPERTIES: Dict[str, str] = {
    "Rationale / Background": "rich_text",
    "Proposed Approach": "rich_text",
    "Gap Identified": "rich_text",
}

# Union of all groups.
RQ_ALL_PROPERTIES: Dict[str, str] = {
    **RQ_CORE_PROPERTIES,
    **RQ_CONTENT_PROPERTIES,
}


# ----------------------------------------------------------------
# Filtering defaults
# ----------------------------------------------------------------

DEFAULT_TARGET_PRIORITIES = frozenset({"High"})


# ----------------------------------------------------------------
# Schema builder
# ----------------------------------------------------------------

def get_rq_schema(db_id: str) -> dict:
    """Return the full RQ schema dict for a given database ID.

    Parameters
    ----------
    db_id:
        The Notion database / data-source ID (``NOTION_RQ_DB_ID``).

    Returns
    -------
    dict
        Keys: ``database_id``, ``all_properties``,
        ``core_properties``, ``content_properties``.
    """
    return {
        "database_id": db_id,
        "all_properties": dict(RQ_ALL_PROPERTIES),
        "core_properties": dict(RQ_CORE_PROPERTIES),
        "content_properties": dict(RQ_CONTENT_PROPERTIES),
    }
