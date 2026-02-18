# src/notion/values_repo.py
"""Repository helper for ROS_Values_Codex.

Created by: 054_values_scale_setup.py (row creation via upsert)

Each row is one value domain definition (12 rows per quarter).
Upsert key format: ``"{review_quarter}:{domain_id}"``
    e.g. ``"2026-Q1:career_work"``

Supports revision updates within a quarter — incrementing the
Revision field and updating Change Notes without creating a new row.

Uses sample-page fallback for schema validation (API 2025-09-03).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from src.notion.client import NotionClient, normalize_uuid, infer_schema_types_from_sample_page
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.values_schema import (
    validate_values_codex_schema,
    PROP_BEHAVIORAL_TRANSLATION,
    PROP_CHANGE_NOTES,
    PROP_DOMAIN_KEY,
    PROP_EXAMPLE_BEHAVIORS,
    PROP_IDEMPOTENCY_KEY,
    PROP_LAST_UPDATED,
    PROP_MICRO_HABITS,
    PROP_MISALIGNMENT,
    PROP_NAME,
    PROP_REFLECTION_QUESTIONS,
    PROP_REVIEW_QUARTER,
    PROP_REVISION,
    PROP_SOURCE,
    PROP_STATUS,
    PROP_VALUE_DEFINITION,
    PROP_VERSION,
)
from src.values.schema import ValueDomain

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


def _select(name: str) -> dict:
    if not name:
        return {"select": None}
    return {"select": {"name": name}}


def _date_iso(iso_str: str) -> dict:
    if not iso_str:
        return {"date": None}
    return {"date": {"start": iso_str}}


# ----------------------------------------------------------------
# ValuesCodexRepo
# ----------------------------------------------------------------

class ValuesCodexRepo:
    """Repo for ``ROS_Values_Codex``.

    Upsert key: ``"{review_quarter}:{domain_id}"`` — one row per domain
    per review quarter. Supports revision increments within a quarter.
    """

    key_property_name = "Idempotency Key"

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
        self._skip_optional: set[str] = set()

    def validate_schema(self) -> None:
        """One-time schema check with sample-page fallback."""
        db_meta = self.client.get_database(database_id=self.database_id)
        props = db_meta.get("properties", {})
        if not props:
            logger.info(
                "GET /databases returned empty properties for ROS_Values_Codex — "
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
                    "ROS_Values_Codex has no pages — cannot infer schema, "
                    "skipping validation"
                )
                self._schema_validated = True
                return
        missing = validate_values_codex_schema(db_meta)
        self._skip_optional = set(missing)
        self._schema_validated = True
        logger.info("ROS_Values_Codex schema validated OK")

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

    def upsert_domain(
        self,
        *,
        key: str,
        properties: Dict[str, Any],
    ) -> dict:
        """Idempotent write: update if key exists, create otherwise."""
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

    def build_domain_properties(
        self,
        *,
        domain: ValueDomain,
        review_quarter: str,
        now_iso: str,
        tracker: Optional[TruncationTracker] = None,
    ) -> tuple[str, Dict[str, Any]]:
        """Build Notion properties for a single value domain row.

        Returns ``(key, properties)`` where key is the idempotency key.
        """
        key = f"{review_quarter}:{domain.domain_id}"

        # Format example behaviors as newline-separated text
        behaviors_text = "\n".join(
            f"{i+1}. {b.description} [{b.frequency_hint}]"
            if b.frequency_hint else f"{i+1}. {b.description}"
            for i, b in enumerate(domain.example_behaviors)
        )

        # Format reflection questions as newline-separated text
        questions_text = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(domain.reflection_questions)
        )

        # Format micro habits as newline-separated text
        habits_text = "\n".join(
            f"{i+1}. {h}" for i, h in enumerate(domain.micro_habits)
        )

        props: Dict[str, Any] = {
            PROP_NAME: _title(domain.domain_label),
            PROP_DOMAIN_KEY: _rt(domain.domain_id),
            PROP_REVIEW_QUARTER: _rt(review_quarter),
            PROP_IDEMPOTENCY_KEY: _rt(key),
            PROP_VALUE_DEFINITION: _rt(notion_truncate(
                domain.value_definition,
                field_name=PROP_VALUE_DEFINITION,
                tracker=tracker,
            )),
            PROP_BEHAVIORAL_TRANSLATION: _rt(notion_truncate(
                domain.behavioral_translation,
                field_name=PROP_BEHAVIORAL_TRANSLATION,
                tracker=tracker,
            )),
            PROP_EXAMPLE_BEHAVIORS: _rt(notion_truncate(
                behaviors_text,
                field_name=PROP_EXAMPLE_BEHAVIORS,
                tracker=tracker,
            )),
            PROP_MISALIGNMENT: _rt(notion_truncate(
                domain.misalignment_description,
                field_name=PROP_MISALIGNMENT,
                tracker=tracker,
            )),
            PROP_STATUS: _select("seed"),
            PROP_SOURCE: _select(domain.source),
            PROP_VERSION: _number(domain.version),
            PROP_REVISION: _number(domain.revision),
            PROP_LAST_UPDATED: _date_iso(now_iso),
        }

        # Optional fields — skip if property missing in Notion
        if questions_text and PROP_REFLECTION_QUESTIONS not in self._skip_optional:
            props[PROP_REFLECTION_QUESTIONS] = _rt(notion_truncate(
                questions_text,
                field_name=PROP_REFLECTION_QUESTIONS,
                tracker=tracker,
            ))

        if habits_text and PROP_MICRO_HABITS not in self._skip_optional:
            props[PROP_MICRO_HABITS] = _rt(notion_truncate(
                habits_text,
                field_name=PROP_MICRO_HABITS,
                tracker=tracker,
            ))

        if domain.change_notes and PROP_CHANGE_NOTES not in self._skip_optional:
            props[PROP_CHANGE_NOTES] = _rt(notion_truncate(
                domain.change_notes,
                field_name=PROP_CHANGE_NOTES,
                tracker=tracker,
            ))

        return key, props

    def build_revision_properties(
        self,
        *,
        domain: ValueDomain,
        now_iso: str,
        source: str = "Hybrid",
        change_notes: str = "",
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        """Build Notion properties for a revision update (patch only changed fields)."""
        behaviors_text = "\n".join(
            f"{i+1}. {b.description} [{b.frequency_hint}]"
            if b.frequency_hint else f"{i+1}. {b.description}"
            for i, b in enumerate(domain.example_behaviors)
        )
        questions_text = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(domain.reflection_questions)
        )
        habits_text = "\n".join(
            f"{i+1}. {h}" for i, h in enumerate(domain.micro_habits)
        )

        props: Dict[str, Any] = {
            PROP_VALUE_DEFINITION: _rt(notion_truncate(
                domain.value_definition, field_name=PROP_VALUE_DEFINITION, tracker=tracker,
            )),
            PROP_BEHAVIORAL_TRANSLATION: _rt(notion_truncate(
                domain.behavioral_translation, field_name=PROP_BEHAVIORAL_TRANSLATION, tracker=tracker,
            )),
            PROP_EXAMPLE_BEHAVIORS: _rt(notion_truncate(
                behaviors_text, field_name=PROP_EXAMPLE_BEHAVIORS, tracker=tracker,
            )),
            PROP_MISALIGNMENT: _rt(notion_truncate(
                domain.misalignment_description, field_name=PROP_MISALIGNMENT, tracker=tracker,
            )),
            PROP_SOURCE: _select(source),
            PROP_VERSION: _number(domain.version),
            PROP_REVISION: _number(domain.revision),
            PROP_LAST_UPDATED: _date_iso(now_iso),
        }

        if questions_text and PROP_REFLECTION_QUESTIONS not in self._skip_optional:
            props[PROP_REFLECTION_QUESTIONS] = _rt(notion_truncate(
                questions_text, field_name=PROP_REFLECTION_QUESTIONS, tracker=tracker,
            ))
        if habits_text and PROP_MICRO_HABITS not in self._skip_optional:
            props[PROP_MICRO_HABITS] = _rt(notion_truncate(
                habits_text, field_name=PROP_MICRO_HABITS, tracker=tracker,
            ))
        if change_notes and PROP_CHANGE_NOTES not in self._skip_optional:
            props[PROP_CHANGE_NOTES] = _rt(notion_truncate(
                change_notes, field_name=PROP_CHANGE_NOTES, tracker=tracker,
            ))

        return props

    def fetch_by_quarter(self, review_quarter: str) -> List[dict]:
        """Fetch all value domain rows for a given quarter."""
        return self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={
                "property": PROP_REVIEW_QUARTER,
                "rich_text": {"equals": review_quarter},
            },
            page_size=20,
            fetch_all=False,
        )

    def fetch_domain_page_id(self, key: str) -> Optional[str]:
        """Fetch the Notion page ID for a domain by idempotency key."""
        pages = self._query_by_key(key)
        if pages:
            return pages[0]["id"]
        return None
