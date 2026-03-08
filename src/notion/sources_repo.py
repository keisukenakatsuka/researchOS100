# src/notion/sources_repo.py
"""Repository for Research Sources Notion DB.

Upserts sources by Source ID (idempotent).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.notion.client import NotionClient, normalize_uuid
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.research_schema import (
    SRC_PROP_TITLE,
    SRC_PROP_URL,
    SRC_PROP_SOURCE_ID,
    SRC_PROP_SOURCE_TYPE,
    SRC_PROP_DOMAIN,
    SRC_PROP_FETCH_STATUS,
    SRC_PROP_FETCHED_CHARS,
    SRC_PROP_RETRIEVED_AT,
)

logger = logging.getLogger(__name__)


# -- property builders ---------------------------------------------------

def _title(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": text}}]}

def _rt(text: str) -> dict:
    return {"rich_text": [{"type": "text", "text": {"content": text}}]}

def _select(name: str) -> dict:
    if not name:
        return {"select": None}
    return {"select": {"name": name}}

def _number(n: Optional[float | int]) -> dict:
    return {"number": None if n is None else float(n)}

def _url(href: Optional[str]) -> dict:
    return {"url": href if href else None}

def _date_iso(iso_str: Optional[str]) -> dict:
    if not iso_str:
        return {"date": None}
    return {"date": {"start": iso_str}}


# -- repo ----------------------------------------------------------------

class SourcesRepo:
    """CRUD for Research Sources DB."""

    def __init__(
        self,
        *,
        client: NotionClient,
        database_id: str,
        data_source_id: str,
    ):
        self.client = client
        self.database_id = normalize_uuid(database_id)
        self.data_source_id = data_source_id

    def upsert_source(
        self,
        source: Dict[str, Any],
        *,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        """Create or update a source page. Returns Notion page object."""
        source_id = source.get("source_id", "")

        # Check existing by Source ID
        existing = self._find_by_source_id(source_id)
        props = self._build_properties(source, tracker=tracker)

        if existing:
            page_id = existing["id"]
            logger.debug("Updating source %s (page=%s)", source_id, page_id)
            return self.client.update_page(page_id=page_id, properties=props)
        else:
            logger.debug("Creating source %s", source_id)
            return self.client.create_page(
                parent_db_id=self.database_id,
                properties=props,
            )

    def _find_by_source_id(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Find an existing page by Source ID property."""
        pages = self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={
                "property": SRC_PROP_SOURCE_ID,
                "rich_text": {"equals": source_id},
            },
            page_size=1,
            fetch_all=False,
        )
        return pages[0] if pages else None

    def _build_properties(
        self,
        source: Dict[str, Any],
        *,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        title = source.get("title", "")
        props: Dict[str, Any] = {
            SRC_PROP_TITLE: _title(notion_truncate(
                title, field_name=SRC_PROP_TITLE, tracker=tracker,
            )),
            SRC_PROP_SOURCE_ID: _rt(source.get("source_id", "")),
            SRC_PROP_SOURCE_TYPE: _select(source.get("source_type", "")),
            SRC_PROP_DOMAIN: _rt(source.get("domain", "")),
            SRC_PROP_FETCH_STATUS: _select(source.get("fetch_status", "")),
            SRC_PROP_FETCHED_CHARS: _number(source.get("fetched_char_count")),
        }

        url = source.get("url")
        if url:
            props[SRC_PROP_URL] = _url(url)

        retrieved_at = source.get("retrieved_at") or source.get("collected_at")
        if retrieved_at:
            props[SRC_PROP_RETRIEVED_AT] = _date_iso(retrieved_at)

        return props
