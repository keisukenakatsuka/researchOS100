# src/notion/weekly_target_additional_repo.py
"""Repository helper for WEEKLY_TARGET_ADDITIONAL_DB (written by 051).

Follows the same upsert pattern as the other weekly repos:
- wraps a :class:`NotionClient` instance with resolved ``data_source_id``
  and ``database_id``
- validates the database schema before the first write
- exposes ``upsert_row(key, properties)`` via :class:`_UpsertMixin`

All content queries go through ``POST /data_sources/{id}/query``.
``GET /databases/{id}`` is used **only** for one-time schema validation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from src.notion.client import NotionClient, normalize_uuid, infer_schema_types_from_sample_page
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.weekly_target_additional_schema import (
    EXPECTED_PROPERTIES as ADD_EXPECTED,
    OPTIONAL_PROPERTIES as ADD_OPTIONAL,
    validate_weekly_target_additional_schema,
)
from src.notion.weekly_target_additional_schema import (
    PROP_ALIASES as ADD_ALIASES,
    PROP_CADENCE as ADD_CADENCE,
    PROP_DECISION as ADD_DECISION,
    PROP_ENTITY_TYPE as ADD_ENTITY_TYPE,
    PROP_EVIDENCE as ADD_EVIDENCE,
    PROP_FINAL_SCORE as ADD_FINAL_SCORE,
    PROP_KEY as ADD_KEY,
    PROP_MENTION_COUNT as ADD_MENTION_COUNT,
    PROP_NAME as ADD_NAME,
    PROP_PRIORITY as ADD_PRIORITY,
    PROP_SEARCH_KEYWORDS as ADD_SEARCH_KEYWORDS,
    PROP_SOURCE_SCRIPT as ADD_SOURCE_SCRIPT,
    PROP_SOURCE_TYPE as ADD_SOURCE_TYPE,
    PROP_SOURCE_URLS as ADD_SOURCE_URLS,
    PROP_STATUS as ADD_STATUS,
    PROP_WEEK as ADD_WEEK,
)
from src.notion.weekly_updates_repo import _UpsertMixin

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Property builders (same helpers as in weekly_updates_repo)
# ----------------------------------------------------------------

def _title(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": text}}]}


def _rt(text: str) -> dict:
    return {"rich_text": [{"type": "text", "text": {"content": text}}]}


def _number(n: Optional[float | int]) -> dict:
    return {"number": None if n is None else float(n)}


def _select(value: str) -> dict:
    return {"select": {"name": value} if value else None}


# ----------------------------------------------------------------
# Normalize candidate name for idempotency key
# ----------------------------------------------------------------

_NORM_RE = re.compile(r"[^a-z0-9\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+")


def _normalize_name(name: str) -> str:
    """Lowercase, strip non-alphanumeric (keep JA chars), truncate to 80 chars."""
    return _NORM_RE.sub("_", (name or "").lower().strip()).strip("_")[:80]


# ================================================================
# WeeklyTargetAdditionalRepo  (written by 051)
# ================================================================

class WeeklyTargetAdditionalRepo(_UpsertMixin):
    """Repo for ``WEEKLY_TARGET_ADDITIONAL_DB``.

    Each row is a new entity proposal discovered by 051.
    """

    key_property_name = "Key"
    _db_label = "WEEKLY_TARGET_ADDITIONAL_DB"

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
        self._use_name_as_key = False  # fallback if Key property doesn't exist

    def _query_by_key(self, key: str) -> list:
        """Override: if Key property is missing, fall back to Name-based lookup."""
        if self._use_name_as_key:
            # Extract candidate name from key for title-based lookup
            # Key format: "{week_id}::add::{type}::{name_normalized}"
            # We can't reliably reconstruct the display name from the key,
            # so just try creating (no upsert dedup without Key property).
            return []
        return super()._query_by_key(key)

    def validate_schema(self) -> None:
        """One-time schema check with sample-page fallback.

        Stores missing optional properties in ``_skip_props`` so property
        builders can omit them from write payloads.
        """
        db_meta = self.client.get_database(database_id=self.database_id)
        props = db_meta.get("properties", {})
        if not props:
            logger.info(
                "GET /databases returned empty properties for WEEKLY_TARGET_ADDITIONAL_DB — "
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
                    "WEEKLY_TARGET_ADDITIONAL_DB has no pages — cannot infer schema, skipping validation"
                )
                self._schema_validated = True
                return
        missing = validate_weekly_target_additional_schema(db_meta)
        self._skip_props = set(missing)

        # If Key property doesn't exist, we can't do key-based upsert
        if "Key" in self._skip_props:
            logger.warning(
                "WEEKLY_TARGET_ADDITIONAL_DB: 'Key' property missing — "
                "upsert deduplication disabled (create-only mode)"
            )
            self._use_name_as_key = True

        self._schema_validated = True
        logger.info("WEEKLY_TARGET_ADDITIONAL_DB schema validated OK")

    def ensure_schema(self) -> None:
        if not self._schema_validated:
            self.validate_schema()

    def build_proposal_properties(
        self,
        *,
        candidate: Dict[str, Any],
        week_id: str,
        tracker: Optional[TruncationTracker] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Convert a 051 candidate into Notion properties.

        Idempotency key: ``"{week_id}::add::{type}::{name_normalized}"``.

        Parameters
        ----------
        candidate : dict
            Expected keys: candidate_name, candidate_name_normalized, type,
            final_score, mention_count, aliases, why_notable (or evidence),
            search_keywords, source_urls, evidence (nested).
        week_id : str
            e.g. ``"2026-W08"``

        Returns ``(key, properties)``.
        """
        cand_type = str(candidate.get("type", "Unknown")).upper()
        cand_norm = str(
            candidate.get("candidate_name_normalized")
            or _normalize_name(candidate.get("candidate_name", ""))
        )
        key = f"{week_id}::add::{cand_type}::{cand_norm}"

        # Evidence text: rationale + sample titles
        evidence_parts: list[str] = []
        rationale = candidate.get("why_notable") or ""
        if rationale:
            evidence_parts.append(rationale)

        ev_data = candidate.get("evidence") or {}
        for t in (ev_data.get("sample_event_titles") or [])[:5]:
            evidence_parts.append(f"[event] {t}")
        for t in (ev_data.get("sample_paper_titles") or [])[:5]:
            evidence_parts.append(f"[paper] {t}")
        evidence_text = "\n".join(evidence_parts)

        # Search keywords and source URLs (from external search enrichment)
        search_kw = candidate.get("search_keywords") or []
        source_urls = candidate.get("source_urls") or []
        aliases = candidate.get("aliases") or []

        skip = self._skip_props

        # Core required property: Name (title) — always present
        props: Dict[str, Any] = {
            ADD_NAME: _title(str(candidate.get("candidate_name", ""))),
        }

        # Key — may not exist in DB
        if ADD_KEY not in skip:
            props[ADD_KEY] = _rt(key)

        # Week — may not exist in DB
        if ADD_WEEK not in skip:
            props[ADD_WEEK] = _rt(week_id)

        # Select properties — only if present
        if ADD_ENTITY_TYPE not in skip:
            props[ADD_ENTITY_TYPE] = _select(cand_type)
        if ADD_STATUS not in skip:
            props[ADD_STATUS] = _select("ACTIVE")
        if ADD_PRIORITY not in skip:
            props[ADD_PRIORITY] = _select(str(candidate.get("priority", "Medium")))
        if ADD_CADENCE not in skip:
            props[ADD_CADENCE] = _select("WEEKLY")
        if ADD_SOURCE_TYPE not in skip:
            props[ADD_SOURCE_TYPE] = _select(str(candidate.get("source_type", "HTML")))
        if ADD_DECISION not in skip:
            props[ADD_DECISION] = _select("Proposal")

        # Rich text properties — only if present
        if ADD_SEARCH_KEYWORDS not in skip:
            props[ADD_SEARCH_KEYWORDS] = _rt(notion_truncate(
                ", ".join(search_kw) if isinstance(search_kw, list) else str(search_kw),
                field_name=ADD_SEARCH_KEYWORDS,
                tracker=tracker,
            ))
        if ADD_SOURCE_URLS not in skip:
            props[ADD_SOURCE_URLS] = _rt(notion_truncate(
                "\n".join(source_urls) if isinstance(source_urls, list) else str(source_urls),
                field_name=ADD_SOURCE_URLS,
                tracker=tracker,
            ))

        # Evidence — may not exist in DB
        if ADD_EVIDENCE not in skip:
            props[ADD_EVIDENCE] = _rt(notion_truncate(
                evidence_text,
                field_name=ADD_EVIDENCE,
                tracker=tracker,
            ))

        # Optional fields — only include if property exists in DB
        if ADD_SOURCE_SCRIPT not in skip:
            props[ADD_SOURCE_SCRIPT] = _rt("051")
        if ADD_ALIASES not in skip:
            props[ADD_ALIASES] = _rt(notion_truncate(
                ", ".join(aliases) if isinstance(aliases, list) else str(aliases),
                field_name=ADD_ALIASES,
                tracker=tracker,
            ))
        if ADD_MENTION_COUNT not in skip:
            props[ADD_MENTION_COUNT] = _number(candidate.get("mention_count"))
        if ADD_FINAL_SCORE not in skip:
            props[ADD_FINAL_SCORE] = _number(candidate.get("final_score"))

        return key, props
