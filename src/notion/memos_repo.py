# src/notion/memos_repo.py
"""Repository for Research Memos Notion DB.

Creates memo pages with Claims, Evidence, Sources, and Research Run relations.
Optionally writes memo body as page content blocks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.notion.client import NotionClient, normalize_uuid
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.research_schema import (
    MEMO_PROP_TITLE,
    MEMO_PROP_MEMO_ID,
    MEMO_PROP_SUMMARY,
    MEMO_PROP_TYPE,
    MEMO_PROP_CLAIMS,
    MEMO_PROP_EVIDENCE,
    MEMO_PROP_SOURCES,
    MEMO_PROP_RESEARCH_RUN,
    MEMO_PROP_CREATED_AT,
)

logger = logging.getLogger(__name__)

# Notion block API limit: max 100 children per request.
_MAX_BLOCKS_PER_REQUEST = 100


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


# -- markdown → Notion blocks -------------------------------------------

def _markdown_to_blocks(md_text: str) -> List[Dict[str, Any]]:
    """Convert simple markdown to Notion block children.

    Handles: headings (# ## ###), bullet lists (- *), and paragraphs.
    Limited to first 100 blocks (Notion API limit).
    """
    blocks: List[Dict[str, Any]] = []
    lines = md_text.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("### "):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": stripped[4:]}}],
                },
            })
        elif stripped.startswith("## "):
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": stripped[3:]}}],
                },
            })
        elif stripped.startswith("# "):
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": stripped[2:]}}],
                },
            })
        elif stripped.startswith("- ") or stripped.startswith("* "):
            text = stripped[2:]
            # Notion rich_text limit: 2000 chars
            if len(text) > 2000:
                text = text[:1985] + " ...(truncated)"
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": text}}],
                },
            })
        else:
            # Paragraph — truncate at 2000 chars
            text = stripped
            if len(text) > 2000:
                text = text[:1985] + " ...(truncated)"
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": text}}],
                },
            })

        if len(blocks) >= _MAX_BLOCKS_PER_REQUEST:
            break

    return blocks


# -- repo ----------------------------------------------------------------

class MemosRepo:
    """CRUD for Research Memos DB."""

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

    def upsert_memo(
        self,
        memo: Dict[str, Any],
        *,
        memo_body_md: str = "",
        claim_page_ids: Optional[List[str]] = None,
        evidence_page_ids: Optional[List[str]] = None,
        source_page_ids: Optional[List[str]] = None,
        run_page_id: Optional[str] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        """Create or update a memo page. Returns Notion page object."""
        memo_id = memo.get("memo_id", "")
        props = self._build_properties(
            memo,
            claim_page_ids=claim_page_ids,
            evidence_page_ids=evidence_page_ids,
            source_page_ids=source_page_ids,
            run_page_id=run_page_id,
            tracker=tracker,
        )

        existing = self._find_by_memo_id(memo_id)
        if existing:
            page_id = existing["id"]
            logger.debug("Updating memo %s (page=%s)", memo_id, page_id)
            return self.client.update_page(page_id=page_id, properties=props)
        else:
            logger.debug("Creating memo %s", memo_id)

        # Convert markdown body to Notion blocks
        children = None
        if memo_body_md:
            children = _markdown_to_blocks(memo_body_md)
            if children:
                logger.debug("Memo %s: %d content blocks", memo_id, len(children))

        return self.client.create_page(
            parent_db_id=self.database_id,
            properties=props,
            children=children,
        )

    def _find_by_memo_id(self, memo_id: str) -> Optional[Dict[str, Any]]:
        """Find an existing page by Memo ID property."""
        pages = self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={
                "property": MEMO_PROP_MEMO_ID,
                "rich_text": {"equals": memo_id},
            },
            page_size=1,
            fetch_all=False,
        )
        return pages[0] if pages else None

    def _build_properties(
        self,
        memo: Dict[str, Any],
        *,
        claim_page_ids: Optional[List[str]] = None,
        evidence_page_ids: Optional[List[str]] = None,
        source_page_ids: Optional[List[str]] = None,
        run_page_id: Optional[str] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        props: Dict[str, Any] = {
            MEMO_PROP_TITLE: _title(memo.get("title", "")),
            MEMO_PROP_MEMO_ID: _rt(memo.get("memo_id", "")),
            MEMO_PROP_SUMMARY: _rt(notion_truncate(
                memo.get("summary", ""),
                field_name=MEMO_PROP_SUMMARY, tracker=tracker,
            )),
            MEMO_PROP_TYPE: _select(memo.get("type", "research memo")),
        }

        created_at = memo.get("generated_at") or memo.get("created_at")
        if created_at:
            props[MEMO_PROP_CREATED_AT] = _date_iso(created_at)

        if claim_page_ids:
            props[MEMO_PROP_CLAIMS] = _relation(claim_page_ids)
        if evidence_page_ids:
            props[MEMO_PROP_EVIDENCE] = _relation(evidence_page_ids)
        if source_page_ids:
            props[MEMO_PROP_SOURCES] = _relation(source_page_ids)
        if run_page_id:
            props[MEMO_PROP_RESEARCH_RUN] = _relation([run_page_id])

        return props
