# src/notion/weekly_themes_repo.py
"""Repository helper for WEEKLY_THEMES_DB.

Written by: 048_weekly_events_digest.py

Each row is one structural theme identified from weekly events.
Upsert key format: ``"{week_id}::{theme_name_normalized}"``.

Uses sample-page fallback for schema validation (API 2025-09-03).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from src.notion.client import NotionAPIError, NotionClient, normalize_uuid, infer_schema_types_from_sample_page
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.weekly_themes_schema import (
    RELATION_PROPERTIES,
    validate_weekly_themes_schema,
    PROP_KEY,
    PROP_KEY_EVENTS,
    PROP_NAME,
    PROP_RELATED_RQS,
    PROP_RELATED_TARGETS,
    PROP_SOURCE_SCRIPT,
    PROP_SUMMARY,
    PROP_WEEK,
    PROP_WHY_IT_MATTERS,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Property builders
# ----------------------------------------------------------------

def _title(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": text}}]}


def _rt(text: str) -> dict:
    return {"rich_text": [{"type": "text", "text": {"content": text}}]}


def _relation(page_ids: Sequence[str]) -> dict:
    return {"relation": [{"id": normalize_uuid(pid)} for pid in page_ids]}


def _normalize_theme_name(name: str) -> str:
    """Normalize a theme name for use in idempotency keys."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:80]


# ----------------------------------------------------------------
# WeeklyThemesRepo
# ----------------------------------------------------------------

class WeeklyThemesRepo:
    """Repo for ``WEEKLY_THEMES_DB``.

    Upsert key: ``"{week_id}::{theme_name_normalized}"``.
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
        self._known_props: set[str] = set()  # property names found in DB

    def validate_schema(self) -> None:
        """One-time schema check with sample-page fallback."""
        db_meta = self.client.get_database(database_id=self.database_id)
        props = db_meta.get("properties", {})
        if not props:
            logger.info(
                "GET /databases returned empty properties for WEEKLY_THEMES_DB — "
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
                    "WEEKLY_THEMES_DB has no pages — cannot infer schema, skipping validation"
                )
                self._schema_validated = True
                return
        self._known_props = set(db_meta.get("properties", {}).keys())
        missing = validate_weekly_themes_schema(db_meta)
        self._skip_relations = set(missing)
        self._schema_validated = True
        logger.info(
            "WEEKLY_THEMES_DB schema validated OK (known_props=%s)",
            sorted(self._known_props),
        )

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

    @staticmethod
    def _strip_relation_props(properties: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of *properties* with all relation-type props removed."""
        return {k: v for k, v in properties.items() if k not in RELATION_PROPERTIES}

    def upsert_row(self, *, key: str, properties: Dict[str, Any]) -> dict:
        """Idempotent write: update if key exists, create otherwise.

        If Notion returns 404 ``object_not_found`` (typically because a
        relation references an inaccessible page), the write is retried
        once with all relation-type properties removed so that non-
        relation fields still persist.
        """
        existing = self._query_by_key(key)
        try:
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
        except NotionAPIError as e:
            is_object_not_found = (
                e.status_code == 404
                or e.payload.get("code") == "object_not_found"
            )
            if not is_object_not_found:
                raise

            # Retry once without relation properties
            stripped = self._strip_relation_props(properties)
            dropped = sorted(set(properties) - set(stripped))
            logger.warning(
                "Upsert key=%s got object_not_found (404) — "
                "retrying without relation properties %s",
                key, dropped,
            )
            if existing:
                page_id = existing[0]["id"]
                return self.client.update_page(page_id=page_id, properties=stripped)
            else:
                return self.client.create_page(
                    parent_db_id=self.database_id,
                    properties=stripped,
                )

    def build_theme_properties(
        self,
        *,
        theme: Dict[str, Any],
        week_id: str,
        digest_page_id: Optional[str] = None,
        tracker: Optional[TruncationTracker] = None,
        # Optional enrichment fields — callers may pass these; accepted
        # without error for forward compatibility.
        rank: Optional[int] = None,
        week_start: Optional[str] = None,
        run_id: Optional[str] = None,
        now_jst: Optional[Any] = None,
        **kwargs: Any,
    ) -> tuple[str, Dict[str, Any]]:
        """Convert an LLM theme dict into Notion properties.

        Parameters
        ----------
        theme : dict
            LLM theme dict with keys: name, summary, why_it_matters,
            key_event_page_ids, related_target_ids, related_rq_ids
        week_id : str
            e.g. ``"2026-W08"``
        digest_page_id : str | None
            Page-id of the WEEKLY_DIGESTS_DB row for this week.
            Used to write the ``Week`` relation.  If *None*, the
            ``Week`` relation is skipped with a warning.

        Accepts additional keyword arguments from callers for forward
        compatibility.  Unknown kwargs are logged at DEBUG level and
        silently ignored.

        Returns ``(key, properties)``.
        """
        if kwargs:
            logger.debug(
                "build_theme_properties: ignoring unknown kwargs: %s",
                sorted(kwargs.keys()),
            )

        theme_name = str(theme.get("name", "Untitled Theme"))
        theme_norm = _normalize_theme_name(theme_name)
        key = f"{week_id}::{theme_norm}"

        props: Dict[str, Any] = {
            PROP_NAME: _title(theme_name),
            PROP_KEY: _rt(key),
            PROP_SUMMARY: _rt(notion_truncate(
                theme.get("summary", ""),
                field_name=PROP_SUMMARY,
                tracker=tracker,
            )),
            PROP_WHY_IT_MATTERS: _rt(notion_truncate(
                theme.get("why_it_matters", ""),
                field_name=PROP_WHY_IT_MATTERS,
                tracker=tracker,
            )),
        }

        # Week — relation to WEEKLY_DIGESTS_DB (not text)
        if digest_page_id and PROP_WEEK not in self._skip_relations:
            try:
                props[PROP_WEEK] = _relation([digest_page_id])
                logger.info(
                    "Week(relation) set for week_id=%s digest_page_id=%s",
                    week_id, digest_page_id,
                )
            except (ValueError, TypeError):
                logger.debug("Skipping Week relation: invalid digest_page_id %r", digest_page_id)
        elif not digest_page_id:
            logger.warning(
                "No digest_page_id provided for week_id=%s — "
                "Week relation will be empty",
                week_id,
            )

        # Source Script — only write if the property exists in the DB
        if self._known_props and PROP_SOURCE_SCRIPT not in self._known_props:
            logger.debug(
                "Skipping %r: property does not exist in WEEKLY_THEMES_DB",
                PROP_SOURCE_SCRIPT,
            )
        else:
            props[PROP_SOURCE_SCRIPT] = _rt("048")

        # Relations (warn-not-fail, respect _skip_relations)
        event_ids = theme.get("key_event_page_ids") or []
        if event_ids and PROP_KEY_EVENTS not in self._skip_relations:
            try:
                props[PROP_KEY_EVENTS] = _relation(event_ids[:25])
            except (ValueError, TypeError):
                logger.debug("Skipping Key Events relation: invalid UUIDs")

        target_ids = theme.get("related_target_ids") or []
        if target_ids and PROP_RELATED_TARGETS not in self._skip_relations:
            try:
                props[PROP_RELATED_TARGETS] = _relation(target_ids[:25])
            except (ValueError, TypeError):
                logger.debug("Skipping Related Targets relation: invalid UUIDs")

        rq_ids = theme.get("related_rq_ids") or []
        if rq_ids and PROP_RELATED_RQS not in self._skip_relations:
            try:
                props[PROP_RELATED_RQS] = _relation(rq_ids[:25])
            except (ValueError, TypeError):
                logger.debug("Skipping Related RQs relation: invalid UUIDs")

        return key, props
