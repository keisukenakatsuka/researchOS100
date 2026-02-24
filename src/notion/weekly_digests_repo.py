# src/notion/weekly_digests_repo.py
"""Repository helper for WEEKLY_DIGESTS_DB.

Created by: 047_weekly_papers_review.py (row creation via upsert)
Updated by: 052_weekly_decision_and_summary.py (synthesis enrichment)

Each row is one weekly digest page (one per week).
Upsert key format: ``"{week_id}"`` (e.g. ``"2026-W08"``).

Uses sample-page fallback for schema validation (API 2025-09-03).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from src.notion.client import NotionClient, normalize_uuid, infer_schema_types_from_sample_page
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.weekly_digests_schema import (
    validate_weekly_digests_schema,
    PROP_CONFIDENCE,
    PROP_EXECUTIVE_SUMMARY,
    PROP_GENERATED_AT,
    PROP_KEY,
    PROP_MACRO_SHIFT,
    PROP_NAME,
    PROP_OPPORTUNITY_SIGNALS,
    PROP_PAPER_UPDATES,
    PROP_RISK_SIGNALS,
    PROP_RQ_UPDATES,
    PROP_RUN_ID,
    PROP_STATUS,
    PROP_TARGET_UPDATES,
    PROP_THEMES,
    PROP_WEEK_END,
    PROP_WEEK_START,
)

logger = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")


# ----------------------------------------------------------------
# Property builders (Notion API payload fragments)
# ----------------------------------------------------------------

def _title(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": text}}]}


def _rt(text: str) -> dict:
    return {"rich_text": [{"type": "text", "text": {"content": text}}]}


def _number(n: Optional[float | int]) -> dict:
    return {"number": None if n is None else float(n)}


def _select(name: str) -> dict:
    if not name:
        return {"select": None}
    return {"select": {"name": name}}


def _relation(page_ids: Sequence[str]) -> dict:
    return {"relation": [{"id": normalize_uuid(pid)} for pid in page_ids]}


def _date_iso(iso_str: str) -> dict:
    if not iso_str:
        return {"date": None}
    return {"date": {"start": iso_str}}


# ----------------------------------------------------------------
# WeeklyDigestsRepo
# ----------------------------------------------------------------

class WeeklyDigestsRepo:
    """Repo for ``WEEKLY_DIGESTS_DB``.

    Upsert key: ``"{week_id}"`` — one digest page per week.
    """

    key_property_name = "Key"
    _db_label = "WEEKLY_DIGESTS_DB"

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
        self._schema_validated = False
        self._skip_relations: set[str] = set()

    def validate_schema(self) -> None:
        """One-time schema check with sample-page fallback.

        Stores the set of missing/misconfigured relation properties so that
        property builders can safely omit them from write payloads.
        """
        db_meta = self.client.get_database(database_id=self.database_id)
        props = db_meta.get("properties", {})
        if not props:
            logger.info(
                "GET /databases returned empty properties for WEEKLY_DIGESTS_DB — "
                "inferring schema from sample page"
            )
            sample_pages = self.client.query_data_source(
                data_source_id=self.data_source_id,
                page_size=1,
                fetch_all=False,
            )
            if sample_pages:
                inferred = infer_schema_types_from_sample_page(sample_pages[0])
                db_meta = {
                    "properties": {
                        name: {"type": ptype}
                        for name, ptype in inferred.items()
                    },
                }
            else:
                logger.warning(
                    "WEEKLY_DIGESTS_DB has no pages — cannot infer schema, skipping validation"
                )
                self._schema_validated = True
                return
        missing = validate_weekly_digests_schema(db_meta)
        self._skip_relations = set(missing)
        self._schema_validated = True
        logger.info("WEEKLY_DIGESTS_DB schema validated OK")

    def ensure_schema(self) -> None:
        if not self._schema_validated:
            self.validate_schema()

    def _query_by_key(self, key: str) -> List[dict]:
        return self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={
                "property": self.key_property_name,
                "rich_text": {"equals": key},
            },
            page_size=2,
            fetch_all=False,
        )

    def upsert_row(self, *, key: str, properties: Dict[str, Any]) -> dict:
        """Idempotent write: update if key exists, create otherwise.

        Raises on API error — never silently skips.
        """
        existing = self._query_by_key(key)
        if existing:
            page_id = existing[0]["id"]
            if len(existing) > 1:
                logger.warning(
                    "[%s] Duplicate key %r found (%d pages) — updating first only",
                    self._db_label, key, len(existing),
                )
            logger.debug("[%s] Upsert UPDATE key=%s page_id=%s", self._db_label, key, page_id)
            result = self.client.update_page(page_id=page_id, properties=properties)
            logger.info(
                "[%s] SUCCESS: updated key=%s page_id=%s",
                self._db_label, key, result.get("id", page_id),
            )
            return result
        else:
            logger.debug("[%s] Upsert CREATE key=%s", self._db_label, key)
            result = self.client.create_page(
                parent_db_id=self.database_id,
                properties=properties,
            )
            logger.info(
                "[%s] SUCCESS: created key=%s page_id=%s",
                self._db_label, key, result.get("id", "?"),
            )
            return result

    def build_digest_properties(
        self,
        *,
        week_id: str,
        week_start: str,
        week_end: str,
        run_id: str,
        now_jst: datetime,
        # Optional enrichment fields — callers may pass these; ignored if
        # not mapped to a Notion property today but accepted without error.
        events_count: Optional[int] = None,
        themes_count: Optional[int] = None,
        signals_count: Optional[int] = None,
        tracker: Optional[TruncationTracker] = None,
        **kwargs: Any,
    ) -> tuple[str, Dict[str, Any]]:
        """Build Notion properties for a weekly digest page (047 bootstrap).

        Accepts additional keyword arguments from callers for forward
        compatibility.  Unknown kwargs are logged at DEBUG level and
        silently ignored.

        Returns ``(key, properties)``.
        """
        if kwargs:
            logger.debug(
                "build_digest_properties: ignoring unknown kwargs: %s",
                sorted(kwargs.keys()),
            )

        key = week_id
        title = f"Weekly Digest — {week_id}"

        props: Dict[str, Any] = {
            PROP_NAME: _title(title),
            PROP_KEY: _rt(key),
            PROP_STATUS: _select("generated"),
            PROP_GENERATED_AT: _date_iso(now_jst.isoformat(timespec="seconds")),
            PROP_RUN_ID: _rt(run_id),
            PROP_WEEK_START: _date_iso(week_start),
            PROP_WEEK_END: _date_iso(week_end),
        }

        return key, props

    def update_digest_synthesis(
        self,
        *,
        week_id: str,
        executive_summary: str = "",
        macro_shift: str = "",
        opportunity_signals: str = "",
        risk_signals: str = "",
        paper_page_ids: Optional[Sequence[str]] = None,
        theme_page_ids: Optional[Sequence[str]] = None,
        rq_update_page_ids: Optional[Sequence[str]] = None,
        target_update_page_ids: Optional[Sequence[str]] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> dict:
        """Update an existing digest row with 052's strategic synthesis.

        Finds the row by Key={week_id}, then patches it with synthesis
        fields and relation links.  Relations respect ``_skip_relations``.

        Raises
        ------
        ValueError
            If the digest row for *week_id* does not exist.
        """
        existing = self._query_by_key(week_id)
        if not existing:
            raise ValueError(
                f"Digest row for {week_id!r} not found in WEEKLY_DIGESTS_DB.  "
                f"047 should have created it via upsert."
            )
        page_id = existing[0]["id"]

        props: Dict[str, Any] = {
            PROP_STATUS: _select("synthesized"),
        }

        # Rich text synthesis fields
        if executive_summary:
            props[PROP_EXECUTIVE_SUMMARY] = _rt(notion_truncate(
                executive_summary,
                field_name=PROP_EXECUTIVE_SUMMARY,
                tracker=tracker,
            ))
        if macro_shift:
            props[PROP_MACRO_SHIFT] = _rt(notion_truncate(
                macro_shift,
                field_name=PROP_MACRO_SHIFT,
                tracker=tracker,
            ))
        if opportunity_signals:
            props[PROP_OPPORTUNITY_SIGNALS] = _rt(notion_truncate(
                opportunity_signals,
                field_name=PROP_OPPORTUNITY_SIGNALS,
                tracker=tracker,
            ))
        if risk_signals:
            props[PROP_RISK_SIGNALS] = _rt(notion_truncate(
                risk_signals,
                field_name=PROP_RISK_SIGNALS,
                tracker=tracker,
            ))

        # Relations (warn-not-fail)
        if paper_page_ids and PROP_PAPER_UPDATES not in self._skip_relations:
            try:
                props[PROP_PAPER_UPDATES] = _relation(paper_page_ids[:25])
            except (ValueError, TypeError):
                logger.debug("Skipping Paper Updates relation: invalid UUIDs")

        if theme_page_ids and PROP_THEMES not in self._skip_relations:
            try:
                props[PROP_THEMES] = _relation(theme_page_ids[:25])
            except (ValueError, TypeError):
                logger.debug("Skipping Themes relation: invalid UUIDs")

        if rq_update_page_ids and PROP_RQ_UPDATES not in self._skip_relations:
            try:
                props[PROP_RQ_UPDATES] = _relation(rq_update_page_ids[:25])
            except (ValueError, TypeError):
                logger.debug("Skipping RQ Updates relation: invalid UUIDs")

        if target_update_page_ids and PROP_TARGET_UPDATES not in self._skip_relations:
            try:
                props[PROP_TARGET_UPDATES] = _relation(target_update_page_ids[:25])
            except (ValueError, TypeError):
                logger.debug("Skipping Target Updates relation: invalid UUIDs")

        logger.info(
            "Updating digest %s with synthesis (%d properties)",
            week_id, len(props),
        )
        return self.client.update_page(page_id=page_id, properties=props)
