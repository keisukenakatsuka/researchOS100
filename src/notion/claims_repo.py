# src/notion/claims_repo.py
"""Repository for Research Claims Notion DB.

Creates claim pages with Evidence and Sources relations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.notion.client import NotionClient, normalize_uuid
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.research_schema import (
    CL_PROP_TITLE,
    CL_PROP_CLAIM_ID,
    CL_PROP_STATEMENT,
    CL_PROP_CONFIDENCE,
    CL_PROP_CONFIDENCE_REASON,
    CL_PROP_TAGS,
    CL_PROP_EVIDENCE,
    CL_PROP_SOURCES,
    CL_PROP_CREATED_AT,
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

def _multi_select(names: Sequence[str]) -> dict:
    return {"multi_select": [{"name": n} for n in names]}

def _relation(page_ids: Sequence[str]) -> dict:
    return {"relation": [{"id": normalize_uuid(pid)} for pid in page_ids]}

def _date_iso(iso_str: Optional[str]) -> dict:
    if not iso_str:
        return {"date": None}
    return {"date": {"start": iso_str}}


# -- repo ----------------------------------------------------------------

class ClaimsRepo:
    """CRUD for Research Claims DB."""

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

    def upsert_claim(
        self,
        claim: Dict[str, Any],
        *,
        evidence_page_ids: Optional[List[str]] = None,
        source_page_ids: Optional[List[str]] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        """Create or update a claim page. Returns Notion page object."""
        claim_id = claim.get("claim_id", "")
        props = self._build_properties(
            claim,
            evidence_page_ids=evidence_page_ids,
            source_page_ids=source_page_ids,
            tracker=tracker,
        )

        existing = self._find_by_claim_id(claim_id)
        if existing:
            page_id = existing["id"]
            logger.debug("Updating claim %s (page=%s)", claim_id, page_id)
            return self.client.update_page(page_id=page_id, properties=props)
        else:
            logger.debug("Creating claim %s", claim_id)
            return self.client.create_page(
                parent_db_id=self.database_id,
                properties=props,
            )

    def _find_by_claim_id(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Find an existing page by Claim ID property."""
        pages = self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={
                "property": CL_PROP_CLAIM_ID,
                "rich_text": {"equals": claim_id},
            },
            page_size=1,
            fetch_all=False,
        )
        return pages[0] if pages else None

    def search_by_keywords(
        self,
        keywords: List[str],
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search claims by keywords against Statement and Tags.

        Returns list of dicts with claim_id, statement, confidence, tags.
        """
        from src.notion.properties import extract_property_value

        seen_ids: set[str] = set()
        results: List[Dict[str, Any]] = []

        for kw in keywords:
            if len(results) >= limit:
                break
            for prop, prop_type in [
                (CL_PROP_STATEMENT, "rich_text"),
                (CL_PROP_TAGS, "multi_select"),
            ]:
                if len(results) >= limit:
                    break
                filt: Dict[str, Any]
                if prop_type == "rich_text":
                    filt = {"property": prop, "rich_text": {"contains": kw}}
                else:
                    filt = {"property": prop, "multi_select": {"contains": kw}}
                try:
                    pages = self.client.query_data_source(
                        data_source_id=self.data_source_id,
                        filter=filt,
                        page_size=min(10, limit - len(results)),
                        fetch_all=False,
                    )
                except Exception as e:
                    logger.debug("Claims search failed for kw=%s prop=%s: %s", kw, prop, e)
                    continue
                for p in pages:
                    cid = extract_property_value(p, CL_PROP_CLAIM_ID) or ""
                    if not cid or cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    tags_raw = extract_property_value(p, CL_PROP_TAGS) or ""
                    results.append({
                        "claim_id": cid,
                        "statement": extract_property_value(p, CL_PROP_STATEMENT) or "",
                        "confidence": extract_property_value(p, CL_PROP_CONFIDENCE) or "",
                        "tags": [t.strip() for t in tags_raw.split(",") if t.strip()],
                    })
                    if len(results) >= limit:
                        break

        return results[:limit]

    def _build_properties(
        self,
        claim: Dict[str, Any],
        *,
        evidence_page_ids: Optional[List[str]] = None,
        source_page_ids: Optional[List[str]] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        statement = claim.get("statement", "")
        title_text = statement[:80] + "…" if len(statement) > 80 else statement

        # Format confidence_reason (list of signals or string)
        conf_reason = claim.get("confidence_reason")
        if isinstance(conf_reason, list):
            reason_text = "; ".join(
                f"{s.get('signal', '')}: {s.get('value', '')}" for s in conf_reason
            )
        elif isinstance(conf_reason, str):
            reason_text = conf_reason
        else:
            reason_text = ""

        props: Dict[str, Any] = {
            CL_PROP_TITLE: _title(title_text),
            CL_PROP_CLAIM_ID: _rt(claim.get("claim_id", "")),
            CL_PROP_STATEMENT: _rt(notion_truncate(
                statement, field_name=CL_PROP_STATEMENT, tracker=tracker,
            )),
            CL_PROP_CONFIDENCE: _select(claim.get("confidence", "")),
            CL_PROP_CONFIDENCE_REASON: _rt(notion_truncate(
                reason_text,
                field_name=CL_PROP_CONFIDENCE_REASON, tracker=tracker,
            )),
            CL_PROP_TAGS: _multi_select(claim.get("tags", [])),
        }

        created_at = claim.get("created_at")
        if created_at:
            props[CL_PROP_CREATED_AT] = _date_iso(created_at)

        if evidence_page_ids:
            props[CL_PROP_EVIDENCE] = _relation(evidence_page_ids)

        if source_page_ids:
            props[CL_PROP_SOURCES] = _relation(source_page_ids)

        return props
