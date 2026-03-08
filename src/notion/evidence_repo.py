# src/notion/evidence_repo.py
"""Repository for Research Evidence Notion DB.

Creates evidence pages with Source relation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from src.notion.client import NotionClient, normalize_uuid
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.research_schema import (
    EV_PROP_TITLE,
    EV_PROP_EVIDENCE_ID,
    EV_PROP_STATEMENT,
    EV_PROP_CONFIDENCE,
    EV_PROP_CONFIDENCE_REASON,
    EV_PROP_TAGS,
    EV_PROP_SOURCE,
    EV_PROP_EXTRACTED_AT,
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

class EvidenceRepo:
    """CRUD for Research Evidence DB."""

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

    def upsert_evidence(
        self,
        evidence: Dict[str, Any],
        *,
        source_page_id: Optional[str] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        """Create or update an evidence page. Returns Notion page object."""
        evidence_id = evidence.get("evidence_id", "")
        props = self._build_properties(
            evidence,
            source_page_id=source_page_id,
            tracker=tracker,
        )

        existing = self._find_by_evidence_id(evidence_id)
        if existing:
            page_id = existing["id"]
            logger.debug("Updating evidence %s (page=%s)", evidence_id, page_id)
            return self.client.update_page(page_id=page_id, properties=props)
        else:
            logger.debug("Creating evidence %s", evidence_id)
            return self.client.create_page(
                parent_db_id=self.database_id,
                properties=props,
            )

    def _find_by_evidence_id(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Find an existing page by Evidence ID property."""
        pages = self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={
                "property": EV_PROP_EVIDENCE_ID,
                "rich_text": {"equals": evidence_id},
            },
            page_size=1,
            fetch_all=False,
        )
        return pages[0] if pages else None

    def search_by_keywords(
        self,
        keywords: List[str],
        *,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search evidence by keywords against Statement and Tags.

        Returns list of dicts with evidence_id, statement, confidence, tags.
        """
        from src.notion.properties import extract_property_value

        seen_ids: set[str] = set()
        results: List[Dict[str, Any]] = []

        for kw in keywords:
            if len(results) >= limit:
                break
            # Search Statement (rich_text contains)
            for prop, prop_type in [
                (EV_PROP_STATEMENT, "rich_text"),
                (EV_PROP_TAGS, "multi_select"),
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
                    logger.debug("Evidence search failed for kw=%s prop=%s: %s", kw, prop, e)
                    continue
                for p in pages:
                    eid = extract_property_value(p, EV_PROP_EVIDENCE_ID) or ""
                    if not eid or eid in seen_ids:
                        continue
                    seen_ids.add(eid)
                    tags_raw = extract_property_value(p, EV_PROP_TAGS) or ""
                    results.append({
                        "evidence_id": eid,
                        "statement": extract_property_value(p, EV_PROP_STATEMENT) or "",
                        "confidence": extract_property_value(p, EV_PROP_CONFIDENCE) or "",
                        "tags": [t.strip() for t in tags_raw.split(",") if t.strip()],
                    })
                    if len(results) >= limit:
                        break

        return results[:limit]

    def _build_properties(
        self,
        evidence: Dict[str, Any],
        *,
        source_page_id: Optional[str] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        statement = evidence.get("statement", "")
        # Title: truncated statement prefix
        title_text = statement[:80] + "…" if len(statement) > 80 else statement

        # Format confidence_reason
        conf_reason = evidence.get("confidence_reason")
        if isinstance(conf_reason, list):
            reason_text = "; ".join(
                f"{s.get('signal', '')}: {s.get('value', '')}" for s in conf_reason
            )
        elif isinstance(conf_reason, str):
            reason_text = conf_reason
        else:
            reason_text = ""

        props: Dict[str, Any] = {
            EV_PROP_TITLE: _title(title_text),
            EV_PROP_EVIDENCE_ID: _rt(evidence.get("evidence_id", "")),
            EV_PROP_STATEMENT: _rt(notion_truncate(
                statement, field_name=EV_PROP_STATEMENT, tracker=tracker,
            )),
            EV_PROP_CONFIDENCE: _select(evidence.get("confidence", "")),
            EV_PROP_CONFIDENCE_REASON: _rt(notion_truncate(
                reason_text, field_name=EV_PROP_CONFIDENCE_REASON, tracker=tracker,
            )),
            EV_PROP_TAGS: _multi_select(evidence.get("tags", [])),
        }

        extracted_at = evidence.get("extracted_at")
        if extracted_at:
            props[EV_PROP_EXTRACTED_AT] = _date_iso(extracted_at)

        if source_page_id:
            props[EV_PROP_SOURCE] = _relation([source_page_id])

        return props
