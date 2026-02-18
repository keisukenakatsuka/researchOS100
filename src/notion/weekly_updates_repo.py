# src/notion/weekly_updates_repo.py
"""Repository helpers for WEEKLY_RQ_UPDATE_DB and WEEKLY_TARGET_UPDATE_DB.

Each repo:
- wraps a :class:`NotionClient` instance and a resolved ``data_source_id``
  (for content queries) plus ``database_id`` (for ``create_page``),
- validates the database schema before the first write (with sample-page
  fallback for API 2025-09-03),
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

import json
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
    PROP_CATEGORY as RQ_CATEGORY,
    PROP_CONFIDENCE as RQ_CONFIDENCE,
    PROP_DECIDED_AT as RQ_DECIDED_AT,
    PROP_EVIDENCE_EVENTS as RQ_EVIDENCE_EVENTS,
    PROP_EVIDENCE_PAPERS as RQ_EVIDENCE_PAPERS,
    PROP_KEY as RQ_KEY,
    PROP_NAME as RQ_NAME,
    PROP_OPEN_QUESTIONS as RQ_OPEN_QUESTIONS,
    PROP_RELATED_THEMES as RQ_RELATED_THEMES,
    PROP_RESEARCH_QUESTION as RQ_RESEARCH_QUESTION,
    PROP_STATUS as RQ_STATUS,
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
    PROP_CONFIDENCE as TGT_CONFIDENCE,
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
    PROP_PRIORITY as TGT_PRIORITY,
    PROP_REASON as TGT_REASON,
    PROP_RECENCY_SCORE as TGT_RECENCY_SCORE,
    PROP_SIGNAL_SCORE as TGT_SIGNAL_SCORE,
    PROP_SOURCE_SCRIPT as TGT_SOURCE_SCRIPT,
    PROP_STATUS as TGT_STATUS,
    PROP_TAGS as TGT_TAGS,
    PROP_TARGET as TGT_TARGET,
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


def _select(name: str) -> dict:
    """Select property — Notion auto-creates the option if it doesn't exist."""
    if not name:
        return {"select": None}
    return {"select": {"name": name}}


def _multi_select(names: Sequence[str]) -> dict:
    """Multi-select property — Notion auto-creates options."""
    return {"multi_select": [{"name": n} for n in names if n]}


def _relation(page_ids: Sequence[str]) -> dict:
    return {"relation": [{"id": normalize_uuid(pid)} for pid in page_ids]}


def _date_iso(iso_str: str) -> dict:
    """Date property from ISO string."""
    if not iso_str:
        return {"date": None}
    return {"date": {"start": iso_str}}


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
        """
        existing = self._query_by_key(key)
        if existing:
            page_id = existing[0]["id"]
            if len(existing) > 1:
                logger.warning(
                    "Duplicate key %r found (%d pages) — updating first only",
                    key, len(existing),
                )
            logger.debug("Upsert UPDATE key=%s page_id=%s", key, page_id)
            return self.client.update_page(page_id=page_id, properties=properties)
        else:
            logger.debug("Upsert CREATE key=%s", key)
            return self.client.create_page(
                parent_db_id=self.database_id,
                properties=properties,
            )

    def _validate_with_fallback(
        self,
        validate_fn,
        db_label: str,
    ) -> List[str]:
        """Validate schema with sample-page fallback for API 2025-09-03.

        Returns
        -------
        List[str]
            Names of relation properties that are missing or misconfigured.
            Callers should store this list and strip these properties from
            write payloads to avoid 400 errors.
        """
        db_meta = self.client.get_database(database_id=self.database_id)
        props = db_meta.get("properties", {})
        if not props:
            logger.info(
                "GET /databases returned empty properties for %s — "
                "inferring schema from sample page",
                db_label,
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
                    "%s has no pages — cannot infer schema, skipping validation",
                    db_label,
                )
                return []
        return validate_fn(db_meta)


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
        ``build_rq_properties`` can safely omit them from write payloads.
        """
        missing = self._validate_with_fallback(
            validate_weekly_rq_update_schema,
            "WEEKLY_RQ_UPDATE_DB",
        )
        self._skip_relations = set(missing)
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

        # Build update summary: evidence counts + open gaps
        evidence_lines: list[str] = []
        for ev in (rq_record.get("related_events") or [])[:5]:
            evidence_lines.append(f"[event] {ev.get('title', '?')}")
        for pa in (rq_record.get("related_papers") or [])[:5]:
            evidence_lines.append(f"[paper] {pa.get('title', '?')}")
        update_text = "\n".join(evidence_lines) if evidence_lines else ""

        tags_list = rq_record.get("tags") or []

        # Build open questions from open_gaps
        open_gaps = rq_record.get("open_gaps", "") or ""

        props: Dict[str, Any] = {
            RQ_NAME: _title(str(rq_record.get("rq_title", ""))),
            RQ_KEY: _rt(key),
            RQ_STATUS: _select("Proposed"),
            RQ_CATEGORY: _rt(str(rq_record.get("priority", ""))),
            RQ_CONFIDENCE: _number(rq_record.get("evidence_count", 0)),
            RQ_UPDATE_SUMMARY: _rt(notion_truncate(
                update_text,
                field_name=RQ_UPDATE_SUMMARY,
                tracker=tracker,
            )),
            RQ_OPEN_QUESTIONS: _rt(notion_truncate(
                open_gaps,
                field_name=RQ_OPEN_QUESTIONS,
                tracker=tracker,
            )),
        }

        # Relation: Research Question — only if configured and rq_id is a UUID
        if RQ_RESEARCH_QUESTION not in self._skip_relations:
            try:
                normalize_uuid(rq_id)
                props[RQ_RESEARCH_QUESTION] = _relation([rq_id])
            except (ValueError, TypeError):
                logger.debug("Skipping Research Question relation: rq_id %r is not a UUID", rq_id)

        # Relation: Evidence Papers
        if RQ_EVIDENCE_PAPERS not in self._skip_relations:
            paper_ids = [
                p["id"] for p in (rq_record.get("related_papers") or [])
                if p.get("id")
            ]
            if paper_ids:
                try:
                    props[RQ_EVIDENCE_PAPERS] = _relation(paper_ids[:25])
                except (ValueError, TypeError):
                    logger.debug("Skipping Evidence Papers relation: invalid UUIDs")

        # Relation: Evidence Events
        if RQ_EVIDENCE_EVENTS not in self._skip_relations:
            event_ids = [
                e["id"] for e in (rq_record.get("related_events") or [])
                if e.get("id")
            ]
            if event_ids:
                try:
                    props[RQ_EVIDENCE_EVENTS] = _relation(event_ids[:25])
                except (ValueError, TypeError):
                    logger.debug("Skipping Evidence Events relation: invalid UUIDs")

        return key, props


# ================================================================
# WeeklyTargetUpdateRepo  (written by 050 + 051)
# ================================================================
#
# DESIGN NOTE — 050/051 collision & semantic overlap
# ---------------------------------------------------
# 050 (target reviews) and 051 (discovery candidates) both write to
# WEEKLY_TARGET_UPDATE_DB but use distinct key prefixes:
#
#   050 key: "{week_id}::{target_id}"
#   051 key: "{week_id}::disc::{type}::{candidate_name_normalized}"
#
# The "disc::" infix guarantees no physical key collision between the
# two scripts, even when both run on the same week.
#
# However, there is a potential SEMANTIC overlap: a discovery candidate
# (051) may refer to the same real-world entity as an existing monitored
# target (050).  For example, 051 might discover "Sequoia Capital" as a
# new candidate while 050 already tracks "sequoia-capital" as a target.
#
# Current design intentionally keeps both rows:
#   - 050 rows carry signal/noise metrics, keyword suggestions, and a
#     relation to the Monitoring Targets DB page.
#   - 051 rows carry discovery metadata, frequency scores, and the
#     "already_tracked" flag (set by comparing against 050 output).
#
# If a discovery candidate is later "promoted" to a monitored target,
# future promotion/merge logic should:
#   1. Create the target in Monitoring Targets DB (manually or automated).
#   2. Mark the 051 row status as "Promoted" (or similar).
#   3. Ensure subsequent 050 runs pick up the new target.
#
# This separation is deliberate — it prevents data loss during the
# discovery-to-tracking transition and maintains auditability.
# ================================================================

class WeeklyTargetUpdateRepo(_UpsertMixin):
    """Repo for ``WEEKLY_TARGET_UPDATE_DB``.

    Handles rows from both 050 (target reviews) and 051 (discovery candidates).
    See design note above for collision handling rationale.
    """

    key_property_name = "Key"

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
        missing = self._validate_with_fallback(
            validate_weekly_target_update_schema,
            "WEEKLY_TARGET_UPDATE_DB",
        )
        self._skip_relations = set(missing)
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

        props: Dict[str, Any] = {
            TGT_NAME: _title(str(target_record.get("target_name", ""))),
            TGT_KEY: _rt(key),
            TGT_STATUS: _select("Proposed"),
            TGT_SOURCE_SCRIPT: _rt("050"),
            TGT_ENTITY_TYPE: _select(str(target_record.get("target_type", ""))),
            TGT_ACTION: _select(str(target_record.get("action", ""))),
            TGT_PRIORITY: _select(str(target_record.get("current_priority", ""))),
            TGT_CONFIDENCE: _number(None),
            TGT_SIGNAL_SCORE: _number(target_record.get("signal_score")),
            TGT_NOISE_SCORE: _number(target_record.get("noise_score")),
            TGT_EVENT_COUNT: _number(target_record.get("number_of_events")),
            TGT_DAYS_SINCE_LAST: _number(target_record.get("days_since_last_event")),
            TGT_RECENCY_SCORE: _number(target_record.get("recency_score")),
            TGT_FINAL_SCORE: _number(None),  # not applicable for 050
            TGT_REASON: _rt(notion_truncate(
                target_record.get("reason"),
                field_name=TGT_REASON,
                tracker=tracker,
            )),
            TGT_ALIASES: _multi_select([]),  # not applicable for 050
            TGT_ALREADY_TRACKED: _checkbox(False),  # not applicable for 050
            TGT_KEYWORDS_TO_ADD: _multi_select(kw_add_list),
            TGT_KEYWORDS_STALE: _multi_select(kw_stale_list),
            TGT_EVIDENCE_DETAIL: _rt(""),  # not applicable for 050
        }

        # Target Relation — only if configured in Notion and target_id is a valid UUID
        if TGT_TARGET not in self._skip_relations:
            try:
                normalize_uuid(target_id)
                props[TGT_TARGET] = _relation([target_id])
            except (ValueError, TypeError):
                logger.debug("Skipping Target relation: target_id %r is not a UUID", target_id)

        return key, props

    # ---- 051: discovery candidate rows ----

    def build_discovery_properties(
        self,
        *,
        candidate: Dict[str, Any],
        week_id: str,
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

        props: Dict[str, Any] = {
            TGT_NAME: _title(str(candidate.get("candidate_name", ""))),
            TGT_KEY: _rt(key),
            TGT_STATUS: _select("Proposed"),
            TGT_SOURCE_SCRIPT: _rt("051"),
            TGT_ENTITY_TYPE: _select(cand_type),
            TGT_ACTION: _select(""),  # not applicable for 051
            TGT_PRIORITY: _select(""),  # not applicable for 051
            TGT_CONFIDENCE: _number(None),
            TGT_SIGNAL_SCORE: _number(None),  # not applicable for 051
            TGT_NOISE_SCORE: _number(None),  # not applicable for 051
            TGT_EVENT_COUNT: _number(candidate.get("mention_count")),
            TGT_DAYS_SINCE_LAST: _number(None),  # not applicable for 051
            TGT_RECENCY_SCORE: _number(None),  # not applicable for 051
            TGT_FINAL_SCORE: _number(candidate.get("final_score")),
            TGT_REASON: _rt(notion_truncate(
                candidate.get("why_notable"),
                field_name=TGT_REASON,
                tracker=tracker,
            )),
            TGT_ALIASES: _multi_select(aliases_list),
            TGT_ALREADY_TRACKED: _checkbox(bool(candidate.get("already_tracked", False))),
            TGT_KEYWORDS_TO_ADD: _multi_select([]),  # not applicable for 051
            TGT_KEYWORDS_STALE: _multi_select([]),  # not applicable for 051
            TGT_EVIDENCE_DETAIL: _rt(notion_truncate(
                evidence_text,
                field_name=TGT_EVIDENCE_DETAIL,
                tracker=tracker,
            )),
            # No Target Relation for 051 — discovery candidates have no Notion page ID
        }

        return key, props
