# src/notion/weekly_updates_repo.py
"""Repository helpers for WEEKLY_RQ_UPDATE_DB and WEEKLY_TARGET_UPDATE_DB.

Each repo:
- wraps a :class:`NotionClient` instance and a resolved ``data_source_id``
  (for content queries) plus ``database_id`` (for ``create_page``),
- validates the database schema before the first write,
- exposes an ``upsert_row(key, properties)`` method that is idempotent:
  same key → update existing page; new key → create page.

All content queries go through ``POST /data_sources/{id}/query``
(via ``NotionClient.query_data_source``).

``GET /databases/{id}`` is used **only** for one-time schema validation
before the first write — this is the only endpoint that returns property
metadata, and is explicitly allowed by the project's Notion client rules
(``client.py`` line 10).  No other ``/databases`` endpoint is called.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.notion.client import NotionClient, normalize_uuid, infer_schema_types_from_sample_page
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.weekly_rq_update_schema import (
    EXPECTED_PROPERTIES as RQ_EXPECTED,
    OPTIONAL_PROPERTIES as RQ_OPTIONAL,
    validate_weekly_rq_update_schema,
)
from src.notion.weekly_rq_update_schema import (
    UPDATE_CATEGORIES as RQ_UPDATE_CATEGORIES,
    CATEGORY_ABBREV as RQ_CATEGORY_ABBREV,
    PROP_CURRENT_APPROACH as RQ_CURRENT_APPROACH,
    PROP_EVIDENCE_COUNT as RQ_EVIDENCE_COUNT,
    PROP_EVIDENCE_DETAIL as RQ_EVIDENCE_DETAIL,
    PROP_KEY as RQ_KEY,
    PROP_NAME as RQ_NAME,
    PROP_OPEN_GAPS as RQ_OPEN_GAPS,
    PROP_OPEN_QUESTIONS as RQ_OPEN_QUESTIONS,
    PROP_PRIORITY as RQ_PRIORITY,
    PROP_RQ_RELATION as RQ_RQ_RELATION,
    PROP_STATUS as RQ_STATUS,
    PROP_TAGS as RQ_TAGS,
    PROP_UPDATE_CATEGORY as RQ_UPDATE_CATEGORY,
    PROP_UPDATE_SUMMARY as RQ_UPDATE_SUMMARY,
    PROP_WEEK as RQ_WEEK,
)
from src.notion.weekly_target_update_schema import (
    EXPECTED_PROPERTIES as TGT_EXPECTED,
    OPTIONAL_PROPERTIES as TGT_OPTIONAL,
    validate_weekly_target_update_schema,
)
from src.notion.weekly_target_update_schema import (
    PROP_ACTION as TGT_ACTION,
    PROP_ALIASES as TGT_ALIASES,
    PROP_ALREADY_TRACKED as TGT_ALREADY_TRACKED,
    PROP_CURRENT_CADENCE as TGT_CURRENT_CADENCE,
    PROP_CURRENT_PRIORITY as TGT_CURRENT_PRIORITY,
    PROP_DAYS_SINCE_LAST as TGT_DAYS_SINCE_LAST,
    PROP_ENTITY_TYPE as TGT_ENTITY_TYPE,
    PROP_EVENT_COUNT as TGT_EVENT_COUNT,
    PROP_EVIDENCE_DETAIL as TGT_EVIDENCE_DETAIL,
    PROP_FINAL_SCORE as TGT_FINAL_SCORE,
    PROP_KEY as TGT_KEY,
    PROP_KEYWORDS_STALE as TGT_KEYWORDS_STALE,
    PROP_KEYWORDS_TO_ADD as TGT_KEYWORDS_TO_ADD,
    PROP_NAME as TGT_NAME,
    PROP_NOISE_SCORE as TGT_NOISE_SCORE,
    PROP_PROPOSED_CADENCE as TGT_PROPOSED_CADENCE,
    PROP_PROPOSED_PRIORITY as TGT_PROPOSED_PRIORITY,
    PROP_REASON as TGT_REASON,
    PROP_RECENCY_SCORE as TGT_RECENCY_SCORE,
    PROP_SIGNAL_SCORE as TGT_SIGNAL_SCORE,
    PROP_SOURCE_SCRIPT as TGT_SOURCE_SCRIPT,
    PROP_SUGGESTED_KEYWORDS as TGT_SUGGESTED_KEYWORDS,
    PROP_SUGGESTED_URLS as TGT_SUGGESTED_URLS,
    PROP_TARGET_RELATION as TGT_TARGET_RELATION,
    PROP_WEEK as TGT_WEEK,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Property builders (Notion API payload fragments)
# ----------------------------------------------------------------

def _title(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": text}}]}


def _rt(text: str) -> dict:
    return {"rich_text": [{"type": "text", "text": {"content": text}}]}


def _number(n: Optional[float | int]) -> dict:
    return {"number": None if n is None else float(n)}


def _checkbox(v: bool) -> dict:
    return {"checkbox": bool(v)}


def _select(value: str) -> dict:
    return {"select": {"name": value} if value else None}


def _multi_select(values: Sequence[str]) -> dict:
    return {"multi_select": [{"name": v} for v in values if v]}


def _relation(page_ids: Sequence[str]) -> dict:
    return {"relation": [{"id": normalize_uuid(pid)} for pid in page_ids]}


# ----------------------------------------------------------------
# Base upsert mixin
# ----------------------------------------------------------------

class _UpsertMixin:
    """Shared upsert logic for weekly update repos.

    Subclasses must set:
    - ``client``: :class:`NotionClient`
    - ``database_id``: str (for ``create_page``)
    - ``data_source_id``: str (for ``query_data_source``)
    - ``key_property_name``: str (Notion property name for the Key field)
    """

    client: NotionClient
    database_id: str
    data_source_id: str
    key_property_name: str

    def _query_by_key(self, key: str) -> List[dict]:
        """Look up existing page(s) by idempotency key.

        Uses ``POST /data_sources/{id}/query`` exclusively.
        """
        return self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={
                "property": self.key_property_name,
                "rich_text": {"equals": key},
            },
            page_size=2,      # at most 1 expected; fetch 2 to detect duplicates
            fetch_all=False,
        )

    def upsert_row(self, *, key: str, properties: Dict[str, Any]) -> dict:
        """Idempotent write: update if *key* exists, create otherwise.

        Parameters
        ----------
        key:
            The composite idempotency key (e.g. ``"2026-W08::abc-123"``).
        properties:
            Full Notion properties payload (already built by caller).
            Must include the Key property.

        Returns
        -------
        dict
            The created or updated Notion page object.

        Raises
        ------
        NotionAPIError
            If the Notion API rejects the write.  Never silently skipped.
        """
        db_label = getattr(self, "_db_label", self.__class__.__name__)
        existing = self._query_by_key(key)
        if existing:
            page_id = existing[0]["id"]
            if len(existing) > 1:
                logger.warning(
                    "[%s] Duplicate key %r found (%d pages) — updating first only",
                    db_label, key, len(existing),
                )
            logger.debug("[%s] Upsert UPDATE key=%s page_id=%s", db_label, key, page_id)
            result = self.client.update_page(page_id=page_id, properties=properties)
            logger.info(
                "[%s] SUCCESS: updated key=%s page_id=%s",
                db_label, key, result.get("id", page_id),
            )
            return result
        else:
            logger.debug("[%s] Upsert CREATE key=%s", db_label, key)
            result = self.client.create_page(
                parent_db_id=self.database_id,
                properties=properties,
            )
            logger.info(
                "[%s] SUCCESS: created key=%s page_id=%s",
                db_label, key, result.get("id", "?"),
            )
            return result


# ================================================================
# WeeklyRQUpdateRepo  (written by 049)
# ================================================================

class WeeklyRQUpdateRepo(_UpsertMixin):
    """Repo for ``WEEKLY_RQ_UPDATE_DB``.

    Attributes
    ----------
    key_property_name:
        Always ``"Key"`` — the Notion property used for idempotency lookup.
    """

    key_property_name = "Key"
    _db_label = "WEEKLY_RQ_UPDATE_DB"

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
        self._skip_props: set[str] = set()

    def validate_schema(self) -> None:
        """One-time schema check with sample-page fallback.

        Uses ``GET /databases/{id}`` for schema metadata.  If properties
        are empty (Notion API 2025-09-03+), falls back to inferring schema
        from a sample page via ``data_sources`` query.

        Stores missing optional properties in ``_skip_props`` so property
        builders can omit them from write payloads.
        """
        db_meta = self.client.get_database(database_id=self.database_id)
        props = db_meta.get("properties", {})
        if not props:
            logger.info(
                "GET /databases returned empty properties for WEEKLY_RQ_UPDATE_DB — "
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
                    "WEEKLY_RQ_UPDATE_DB has no pages — cannot infer schema, skipping validation"
                )
                self._schema_validated = True
                return
        missing = validate_weekly_rq_update_schema(db_meta)
        self._skip_props = set(missing)
        self._schema_validated = True
        logger.info("WEEKLY_RQ_UPDATE_DB schema validated OK")

    def ensure_schema(self) -> None:
        """Validate schema exactly once (idempotent)."""
        if not self._schema_validated:
            self.validate_schema()

    def build_rq_properties(
        self,
        *,
        rq_record: Dict[str, Any],
        week_id: str,
        week_page_ids: Optional[Sequence[str]] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Convert a 049 rq_status record into Notion properties.

        Returns
        -------
        (key, properties):
            key: idempotency key string.
            properties: Notion API properties dict.
        """
        rq_id = str(rq_record["rq_id"])
        key = f"{week_id}::{rq_id}"

        # Build evidence detail: top 5 events then papers, each title on a line
        evidence_lines: list[str] = []
        for ev in (rq_record.get("related_events") or [])[:5]:
            evidence_lines.append(f"[event] {ev.get('title', '?')}")
        for pa in (rq_record.get("related_papers") or [])[:5]:
            evidence_lines.append(f"[paper] {pa.get('title', '?')}")
        evidence_text = "\n".join(evidence_lines)

        tags_str = ", ".join(rq_record.get("tags") or [])

        skip = self._skip_props

        # Core required properties (always present)
        props: Dict[str, Any] = {
            RQ_NAME: _title(str(rq_record.get("rq_title", ""))),
            RQ_KEY: _rt(key),
        }

        # Week — relation in actual DB
        if RQ_WEEK not in skip:
            if week_page_ids:
                props[RQ_WEEK] = _relation(week_page_ids)
            # If no page IDs available, skip (don't send empty relation)

        # Status — select in actual DB
        if RQ_STATUS not in skip:
            props[RQ_STATUS] = _select(str(rq_record.get("status", "")))

        # Priority — may not exist in DB
        if RQ_PRIORITY not in skip:
            props[RQ_PRIORITY] = _rt(str(rq_record.get("priority", "")))

        # Evidence Count — may not exist in DB
        if RQ_EVIDENCE_COUNT not in skip:
            props[RQ_EVIDENCE_COUNT] = _number(rq_record.get("evidence_count", 0))

        # Tags — may not exist in DB
        if RQ_TAGS not in skip:
            props[RQ_TAGS] = _rt(notion_truncate(
                tags_str, field_name=RQ_TAGS, tracker=tracker,
            ))

        # Evidence Detail — may not exist in DB
        if RQ_EVIDENCE_DETAIL not in skip:
            props[RQ_EVIDENCE_DETAIL] = _rt(notion_truncate(
                evidence_text,
                field_name=RQ_EVIDENCE_DETAIL,
                tracker=tracker,
            ))

        # Update Summary (rich_text — present in actual DB)
        if RQ_UPDATE_SUMMARY not in skip:
            props[RQ_UPDATE_SUMMARY] = _rt(notion_truncate(
                rq_record.get("reason", ""),
                field_name=RQ_UPDATE_SUMMARY,
                tracker=tracker,
            ))

        # Open Questions (rich_text — present in actual DB)
        if RQ_OPEN_QUESTIONS not in skip:
            props[RQ_OPEN_QUESTIONS] = _rt(notion_truncate(
                rq_record.get("proposed_text", rq_record.get("open_gaps", "")),
                field_name=RQ_OPEN_QUESTIONS,
                tracker=tracker,
            ))

        # Optional/backward-compat fields — only if present in DB
        if RQ_OPEN_GAPS not in skip:
            props[RQ_OPEN_GAPS] = _rt(notion_truncate(
                rq_record.get("open_gaps"),
                field_name=RQ_OPEN_GAPS,
                tracker=tracker,
            ))
        if RQ_CURRENT_APPROACH not in skip:
            props[RQ_CURRENT_APPROACH] = _rt(notion_truncate(
                rq_record.get("current_approach"),
                field_name=RQ_CURRENT_APPROACH,
                tracker=tracker,
            ))

        # Research Question relation — actual DB property name
        from src.notion.weekly_rq_update_schema import PROP_RESEARCH_QUESTION as RQ_RESEARCH_QUESTION
        if RQ_RESEARCH_QUESTION not in skip:
            try:
                normalize_uuid(rq_id)
                props[RQ_RESEARCH_QUESTION] = _relation([rq_id])
            except (ValueError, TypeError):
                logger.debug("Skipping Research Question relation: rq_id %r is not a UUID", rq_id)
        elif RQ_RQ_RELATION not in skip:
            # Fallback to legacy property name
            try:
                normalize_uuid(rq_id)
                props[RQ_RQ_RELATION] = _relation([rq_id])
            except (ValueError, TypeError):
                logger.debug("Skipping RQ Relation: rq_id %r is not a UUID", rq_id)

        return key, props

    def build_rq_revision_properties(
        self,
        *,
        revision: Dict[str, Any],
        week_id: str,
        revision_index: int,
        week_page_ids: Optional[Sequence[str]] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Build properties for a single categorized revision proposal.

        Each revision gets its own Notion page.  Multiple revisions per RQ
        per week are supported.

        Parameters
        ----------
        revision : dict
            Keys: category, proposed_text, reason, rq_id, rq_title, priority,
            status, tags, evidence_lines, confidence
        week_id : str
            e.g. ``"2026-W08"``
        revision_index : int
            0-based index within the RQ's revisions for this week.

        Returns ``(key, properties)``.
        """
        rq_id = str(revision["rq_id"])
        category = str(revision.get("category", ""))
        if category not in RQ_UPDATE_CATEGORIES:
            logger.warning(
                "Unexpected Update Category %r for RQ %s "
                "(expected one of %s) — writing as-is",
                category, rq_id, sorted(RQ_UPDATE_CATEGORIES),
            )
        cat_abbrev = RQ_CATEGORY_ABBREV.get(category, "unk")
        key = f"{week_id}::{rq_id}::{cat_abbrev}::{revision_index}"

        rq_title = str(revision.get("rq_title", ""))
        name = f"{rq_title}: {category} #{revision_index + 1}"
        if len(name) > 200:
            name = name[:197] + "..."

        # Evidence detail
        evidence_lines = revision.get("evidence_lines") or []
        evidence_text = "\n".join(str(line) for line in evidence_lines[:10])

        tags_str = ", ".join(revision.get("tags") or [])

        skip = self._skip_props

        # Core required properties
        props: Dict[str, Any] = {
            RQ_NAME: _title(name),
            RQ_KEY: _rt(key),
        }

        # Week — relation in actual DB
        if RQ_WEEK not in skip:
            if week_page_ids:
                props[RQ_WEEK] = _relation(week_page_ids)

        # Status — select in actual DB
        if RQ_STATUS not in skip:
            props[RQ_STATUS] = _select(str(revision.get("status", "")))

        # Priority — may not exist
        if RQ_PRIORITY not in skip:
            props[RQ_PRIORITY] = _rt(str(revision.get("priority", "")))

        # Evidence Count — may not exist
        if RQ_EVIDENCE_COUNT not in skip:
            props[RQ_EVIDENCE_COUNT] = _number(revision.get("evidence_count", 0))

        # Tags — may not exist
        if RQ_TAGS not in skip:
            props[RQ_TAGS] = _rt(notion_truncate(
                tags_str, field_name=RQ_TAGS, tracker=tracker,
            ))

        # Evidence Detail — may not exist
        if RQ_EVIDENCE_DETAIL not in skip:
            props[RQ_EVIDENCE_DETAIL] = _rt(notion_truncate(
                evidence_text,
                field_name=RQ_EVIDENCE_DETAIL,
                tracker=tracker,
            ))

        # Categorized fields — only if present in DB
        if RQ_UPDATE_CATEGORY not in skip:
            props[RQ_UPDATE_CATEGORY] = _rt(category)
        if RQ_OPEN_QUESTIONS not in skip:
            props[RQ_OPEN_QUESTIONS] = _rt(notion_truncate(
                revision.get("proposed_text", ""),
                field_name=RQ_OPEN_QUESTIONS,
                tracker=tracker,
            ))
        if RQ_UPDATE_SUMMARY not in skip:
            props[RQ_UPDATE_SUMMARY] = _rt(notion_truncate(
                revision.get("reason", ""),
                field_name=RQ_UPDATE_SUMMARY,
                tracker=tracker,
            ))

        # Research Question relation
        from src.notion.weekly_rq_update_schema import PROP_RESEARCH_QUESTION as RQ_RESEARCH_QUESTION
        if RQ_RESEARCH_QUESTION not in skip:
            try:
                normalize_uuid(rq_id)
                props[RQ_RESEARCH_QUESTION] = _relation([rq_id])
            except (ValueError, TypeError):
                logger.debug("Skipping Research Question relation: rq_id %r is not a UUID", rq_id)
        elif RQ_RQ_RELATION not in skip:
            try:
                normalize_uuid(rq_id)
                props[RQ_RQ_RELATION] = _relation([rq_id])
            except (ValueError, TypeError):
                logger.debug("Skipping RQ Relation: rq_id %r is not a UUID", rq_id)

        return key, props


# ================================================================
# WeeklyTargetUpdateRepo  (written by 050 + 051)
# ================================================================

class WeeklyTargetUpdateRepo(_UpsertMixin):
    """Repo for ``WEEKLY_TARGET_UPDATE_DB``.

    Handles rows from both 050 (target reviews) and 051 (discovery candidates).
    """

    key_property_name = "Key"
    _db_label = "WEEKLY_TARGET_UPDATE_DB"

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
        self._skip_props: set[str] = set()

    def validate_schema(self) -> None:
        """One-time schema check with sample-page fallback.

        Stores missing optional properties in ``_skip_props`` so property
        builders can omit them from write payloads.
        """
        db_meta = self.client.get_database(database_id=self.database_id)
        props = db_meta.get("properties", {})
        if not props:
            logger.info(
                "GET /databases returned empty properties for WEEKLY_TARGET_UPDATE_DB — "
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
                    "WEEKLY_TARGET_UPDATE_DB has no pages — cannot infer schema, skipping validation"
                )
                self._schema_validated = True
                return
        missing = validate_weekly_target_update_schema(db_meta)
        self._skip_props = set(missing)
        self._schema_validated = True
        logger.info("WEEKLY_TARGET_UPDATE_DB schema validated OK")

    def ensure_schema(self) -> None:
        if not self._schema_validated:
            self.validate_schema()

    # ---- 050: target review rows ----

    def build_target_review_properties(
        self,
        *,
        target_record: Dict[str, Any],
        week_id: str,
        week_page_ids: Optional[Sequence[str]] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Convert a 050 targets_review record into Notion properties.

        Returns ``(key, properties)``.
        """
        target_id = str(target_record["target_id"])
        key = f"{week_id}::{target_id}"

        # keyword suggestions
        kw_sug = target_record.get("keyword_suggestions") or {}
        kw_add_list = [
            str(k.get("keyword", ""))
            for k in (kw_sug.get("keywords_to_add") or [])
        ]
        kw_stale_list = [
            str(k.get("keyword", ""))
            for k in (kw_sug.get("keywords_stale") or [])
        ]

        skip = self._skip_props

        # Core required properties
        props: Dict[str, Any] = {
            TGT_NAME: _title(str(target_record.get("target_name", ""))),
            TGT_KEY: _rt(key),
        }

        # Week — relation in actual DB
        if TGT_WEEK not in skip:
            if week_page_ids:
                props[TGT_WEEK] = _relation(week_page_ids)

        # Source Script — rich_text
        if TGT_SOURCE_SCRIPT not in skip:
            props[TGT_SOURCE_SCRIPT] = _rt("050")

        # Entity Type — select in actual DB
        if TGT_ENTITY_TYPE not in skip:
            props[TGT_ENTITY_TYPE] = _select(str(target_record.get("target_type", "")))

        # Action — select in actual DB
        if TGT_ACTION not in skip:
            props[TGT_ACTION] = _select(str(target_record.get("action", "")))

        # Current/Proposed Priority/Cadence — may not exist
        if TGT_CURRENT_PRIORITY not in skip:
            props[TGT_CURRENT_PRIORITY] = _rt(str(target_record.get("current_priority", "")))
        if TGT_PROPOSED_PRIORITY not in skip:
            props[TGT_PROPOSED_PRIORITY] = _rt(str(target_record.get("proposed_priority", "")))
        if TGT_CURRENT_CADENCE not in skip:
            props[TGT_CURRENT_CADENCE] = _rt(str(target_record.get("current_cadence", "")))
        if TGT_PROPOSED_CADENCE not in skip:
            props[TGT_PROPOSED_CADENCE] = _rt(str(target_record.get("proposed_cadence", "")))

        # Numeric scores
        if TGT_SIGNAL_SCORE not in skip:
            props[TGT_SIGNAL_SCORE] = _number(target_record.get("signal_score"))
        if TGT_NOISE_SCORE not in skip:
            props[TGT_NOISE_SCORE] = _number(target_record.get("noise_score"))
        if TGT_EVENT_COUNT not in skip:
            props[TGT_EVENT_COUNT] = _number(target_record.get("number_of_events"))
        if TGT_DAYS_SINCE_LAST not in skip:
            props[TGT_DAYS_SINCE_LAST] = _number(target_record.get("days_since_last_event"))
        if TGT_RECENCY_SCORE not in skip:
            props[TGT_RECENCY_SCORE] = _number(target_record.get("recency_score"))
        if TGT_FINAL_SCORE not in skip:
            props[TGT_FINAL_SCORE] = _number(None)  # not applicable for 050

        # Reason — rich_text
        if TGT_REASON not in skip:
            props[TGT_REASON] = _rt(notion_truncate(
                target_record.get("reason"),
                field_name=TGT_REASON,
                tracker=tracker,
            ))

        # Aliases — multi_select in actual DB (not applicable for 050)
        if TGT_ALIASES not in skip:
            props[TGT_ALIASES] = _multi_select([])

        # Already Tracked — checkbox
        if TGT_ALREADY_TRACKED not in skip:
            props[TGT_ALREADY_TRACKED] = _checkbox(False)

        # Keywords To Add — multi_select in actual DB
        if TGT_KEYWORDS_TO_ADD not in skip:
            props[TGT_KEYWORDS_TO_ADD] = _multi_select(kw_add_list)

        # Keywords Stale — multi_select in actual DB
        if TGT_KEYWORDS_STALE not in skip:
            props[TGT_KEYWORDS_STALE] = _multi_select(kw_stale_list)

        # Evidence Detail — rich_text (not applicable for 050)
        if TGT_EVIDENCE_DETAIL not in skip:
            props[TGT_EVIDENCE_DETAIL] = _rt("")

        # Optional fields
        if TGT_SUGGESTED_KEYWORDS not in skip:
            props[TGT_SUGGESTED_KEYWORDS] = _rt(notion_truncate(
                ", ".join(target_record.get("suggested_keywords") or []),
                field_name=TGT_SUGGESTED_KEYWORDS,
                tracker=tracker,
            ))
        if TGT_SUGGESTED_URLS not in skip:
            props[TGT_SUGGESTED_URLS] = _rt(notion_truncate(
                "\n".join(target_record.get("suggested_source_urls") or []),
                field_name=TGT_SUGGESTED_URLS,
                tracker=tracker,
            ))

        # Target Relation — only if target_id is a valid UUID and property exists
        if TGT_TARGET_RELATION not in skip:
            try:
                normalize_uuid(target_id)
                props[TGT_TARGET_RELATION] = _relation([target_id])
            except (ValueError, TypeError):
                logger.debug("Skipping Target Relation: target_id %r is not a UUID", target_id)

        return key, props

    # ---- 051: discovery candidate rows ----

    def build_discovery_properties(
        self,
        *,
        candidate: Dict[str, Any],
        week_id: str,
        week_page_ids: Optional[Sequence[str]] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Convert a 051 candidates record into Notion properties.

        Idempotency key includes entity type to prevent collisions:
        ``"{week_id}::disc::{type}::{candidate_name_normalized}"``.

        Returns ``(key, properties)``.
        """
        cand_type = str(candidate.get("type", "Unknown"))
        cand_norm = str(candidate.get("candidate_name_normalized", ""))
        key = f"{week_id}::disc::{cand_type}::{cand_norm}"

        # Evidence detail: top sample event titles
        evidence = candidate.get("evidence") or {}
        ev_titles = (evidence.get("sample_event_titles") or [])[:5]
        pa_titles = (evidence.get("sample_paper_titles") or [])[:5]
        evidence_lines: list[str] = []
        for t in ev_titles:
            evidence_lines.append(f"[event] {t}")
        for t in pa_titles:
            evidence_lines.append(f"[paper] {t}")
        evidence_text = "\n".join(evidence_lines)

        aliases_list = candidate.get("aliases") or []

        skip = self._skip_props

        # Core required properties
        props: Dict[str, Any] = {
            TGT_NAME: _title(str(candidate.get("candidate_name", ""))),
            TGT_KEY: _rt(key),
        }

        # Week — relation in actual DB
        if TGT_WEEK not in skip:
            if week_page_ids:
                props[TGT_WEEK] = _relation(week_page_ids)

        # Source Script — rich_text
        if TGT_SOURCE_SCRIPT not in skip:
            props[TGT_SOURCE_SCRIPT] = _rt("051")

        # Entity Type — select in actual DB
        if TGT_ENTITY_TYPE not in skip:
            props[TGT_ENTITY_TYPE] = _select(cand_type)

        # Action — select (not applicable for 051)
        # Don't write empty selects — Notion API may reject them

        # Numeric scores
        if TGT_EVENT_COUNT not in skip:
            props[TGT_EVENT_COUNT] = _number(candidate.get("mention_count"))
        if TGT_FINAL_SCORE not in skip:
            props[TGT_FINAL_SCORE] = _number(candidate.get("final_score"))

        # Reason — rich_text
        if TGT_REASON not in skip:
            props[TGT_REASON] = _rt(notion_truncate(
                candidate.get("why_notable"),
                field_name=TGT_REASON,
                tracker=tracker,
            ))

        # Aliases — multi_select in actual DB
        if TGT_ALIASES not in skip:
            props[TGT_ALIASES] = _multi_select(aliases_list)

        # Already Tracked — checkbox
        if TGT_ALREADY_TRACKED not in skip:
            props[TGT_ALREADY_TRACKED] = _checkbox(bool(candidate.get("already_tracked", False)))

        # Evidence Detail — rich_text
        if TGT_EVIDENCE_DETAIL not in skip:
            props[TGT_EVIDENCE_DETAIL] = _rt(notion_truncate(
                evidence_text,
                field_name=TGT_EVIDENCE_DETAIL,
                tracker=tracker,
            ))

        # No Target Relation for 051 — discovery candidates have no Notion page ID

        return key, props
