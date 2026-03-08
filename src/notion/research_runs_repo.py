# src/notion/research_runs_repo.py
"""Repository for Research Runs Notion DB.

Creates research run pages with Sources, Evidence, Claims, Memos relations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.notion.client import NotionClient, normalize_uuid
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.research_schema import (
    RR_PROP_TITLE,
    RR_PROP_RUN_ID,
    RR_PROP_REQUEST,
    RR_PROP_RUN_TYPE,
    RR_PROP_STATUS,
    RR_PROP_STARTED_AT,
    RR_PROP_COMPLETED_AT,
    RR_PROP_SOURCES,
    RR_PROP_EVIDENCE,
    RR_PROP_CLAIMS,
    RR_PROP_MEMOS,
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

def _relation(page_ids: Sequence[str]) -> dict:
    return {"relation": [{"id": normalize_uuid(pid)} for pid in page_ids]}

def _date_iso(iso_str: Optional[str]) -> dict:
    if not iso_str:
        return {"date": None}
    return {"date": {"start": iso_str}}


# -- repo ----------------------------------------------------------------

class ResearchRunsRepo:
    """CRUD for Research Runs DB."""

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

    def upsert_run(
        self,
        run: Dict[str, Any],
        *,
        source_page_ids: Optional[List[str]] = None,
        evidence_page_ids: Optional[List[str]] = None,
        claim_page_ids: Optional[List[str]] = None,
        memo_page_ids: Optional[List[str]] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        """Create or update a research run page. Returns Notion page object."""
        run_id = run.get("run_id", "")
        props = self._build_properties(
            run,
            source_page_ids=source_page_ids,
            evidence_page_ids=evidence_page_ids,
            claim_page_ids=claim_page_ids,
            memo_page_ids=memo_page_ids,
            tracker=tracker,
        )

        existing = self._find_by_run_id(run_id)
        if existing:
            page_id = existing["id"]
            logger.debug("Updating research run %s (page=%s)", run_id, page_id)
            return self.client.update_page(page_id=page_id, properties=props)
        else:
            logger.debug("Creating research run %s", run_id)
            return self.client.create_page(
                parent_db_id=self.database_id,
                properties=props,
            )

    def _find_by_run_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Find an existing page by Run ID property."""
        pages = self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={
                "property": RR_PROP_RUN_ID,
                "rich_text": {"equals": run_id},
            },
            page_size=1,
            fetch_all=False,
        )
        return pages[0] if pages else None

    def _build_properties(
        self,
        run: Dict[str, Any],
        *,
        source_page_ids: Optional[List[str]] = None,
        evidence_page_ids: Optional[List[str]] = None,
        claim_page_ids: Optional[List[str]] = None,
        memo_page_ids: Optional[List[str]] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        run_id = run.get("run_id", "")
        request = run.get("request", "")
        title = f"[{run_id}] {request[:60]}" if request else run_id

        props: Dict[str, Any] = {
            RR_PROP_TITLE: _title(title),
            RR_PROP_RUN_ID: _rt(run_id),
            RR_PROP_REQUEST: _rt(notion_truncate(
                request, field_name=RR_PROP_REQUEST, tracker=tracker,
            )),
            RR_PROP_RUN_TYPE: _select(run.get("run_type", "deep_research")),
            RR_PROP_STATUS: _select(run.get("status", "completed")),
        }

        started_at = run.get("started_at")
        if started_at:
            props[RR_PROP_STARTED_AT] = _date_iso(started_at)

        completed_at = run.get("completed_at")
        if completed_at:
            props[RR_PROP_COMPLETED_AT] = _date_iso(completed_at)

        if source_page_ids:
            props[RR_PROP_SOURCES] = _relation(source_page_ids)
        if evidence_page_ids:
            props[RR_PROP_EVIDENCE] = _relation(evidence_page_ids)
        if claim_page_ids:
            props[RR_PROP_CLAIMS] = _relation(claim_page_ids)
        if memo_page_ids:
            props[RR_PROP_MEMOS] = _relation(memo_page_ids)

        return props
