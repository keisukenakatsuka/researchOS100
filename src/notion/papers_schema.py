# src/notion/papers_schema.py
"""Schema definition for the Papers (Literature) Notion database.

Extracted from notebook 040_weekly_papers_review Cell 02.
Reusable by any weekly script (047+) that reads or writes the Papers DB.

Property groups
---------------
- **PAPERS_CORE_PROPERTIES**: immutable metadata that exists from ingestion
  (title, authors, tags, PDF link, type, source, etc.).
- **PAPERS_INGESTION_PROPERTIES**: tracking fields written during ingestion
  (status, dedup key, source UID, ingested-at, run ID, PDF status, slide URL).
- **PAPERS_SCORING_PROPERTIES**: fields written/read by weekly scoring
  notebooks (importance, RQ relevance, weekly priority, decision, reason).
  This is the subset that 047+ scoring scripts care about.

``PAPERS_REQUIRED_PROPERTIES`` is the union of all three groups and is
kept for backward compatibility.

Usage::

    from src.notion.papers_schema import (
        get_papers_schema,
        PAPERS_CORE_PROPERTIES,
        PAPERS_SCORING_PROPERTIES,
    )

    schema = get_papers_schema(db_id="<your-papers-db-id>")
    scoring_cols = list(PAPERS_SCORING_PROPERTIES.keys())
"""

from __future__ import annotations

from typing import Dict


# ----------------------------------------------------------------
# Property groups  (property name → Notion property type)
# ----------------------------------------------------------------

PAPERS_CORE_PROPERTIES: Dict[str, str] = {
    "Name": "title",
    "Created time": "created_time",
    "Authors & Year": "rich_text",
    "Tags": "multi_select",
    "PDF Link": "url",
    "Findings": "rich_text",
    "Core Idea": "rich_text",
    "Notes": "rich_text",
    "Methods": "rich_text",
    "Type": "select",
    "Source": "select",
    "Datasets": "rich_text",
    "Papers": "relation",
}

PAPERS_INGESTION_PROPERTIES: Dict[str, str] = {
    "Status": "select",
    "Dedup Key": "rich_text",
    "Source UID": "rich_text",
    "Ingested At": "date",
    "Run ID": "rich_text",
    "PDF Status": "select",
    "Slide 1 URL": "url",
}

PAPERS_SCORING_PROPERTIES: Dict[str, str] = {
    "Importance": "number",
    "RQ Relevance": "number",
    "Weekly Priority": "number",
    "Decision": "select",
    "Decision Reason": "rich_text",
}

# Backward-compatible union of all groups.
PAPERS_REQUIRED_PROPERTIES: Dict[str, str] = {
    **PAPERS_CORE_PROPERTIES,
    **PAPERS_INGESTION_PROPERTIES,
    **PAPERS_SCORING_PROPERTIES,
}


# ----------------------------------------------------------------
# Schema builder
# ----------------------------------------------------------------

def get_papers_schema(db_id: str) -> dict:
    """Return the full PAPERS_SCHEMA dict for a given database ID.

    Parameters
    ----------
    db_id:
        The Notion database ID (``NOTION_LIT_DB_ID``).

    Returns
    -------
    dict
        Keys: ``database_id``, ``required_properties``, ``scoring_properties``,
        ``core_properties``, ``ingestion_properties``, ``optional_properties``.
    """
    return {
        "database_id": db_id,
        "required_properties": dict(PAPERS_REQUIRED_PROPERTIES),
        "core_properties": dict(PAPERS_CORE_PROPERTIES),
        "ingestion_properties": dict(PAPERS_INGESTION_PROPERTIES),
        "scoring_properties": dict(PAPERS_SCORING_PROPERTIES),
        "optional_properties": {},
    }
