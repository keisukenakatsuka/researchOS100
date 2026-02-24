# src/notion/alignment_repo.py
"""Repository helper for ROS_Alignment_Log.

Handles daily, weekly, and quarterly reflection entries using the
two-dimensional Value Evaluation Scale:
  - importance_score (1–5)
  - alignment_score  (1–5)
  - gap_score        (computed: importance - alignment)
  - significant_gap  (computed: gap >= 2)

Each row links back to a domain in ROS_Values_Codex via relation.
No idempotency key — each reflection is a new row (append-only).

Provides:
- ``validate_score_range()`` — fail-fast integer range check (1–5)
- ``validate_no_same_day_duplicate()`` — optional dedup guard
- ``AlignmentLogRepo`` — full CRUD with schema validation
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from src.notion.client import NotionClient, normalize_uuid, infer_schema_types_from_sample_page
from src.notion.truncation import TruncationTracker, notion_truncate
from src.notion.alignment_schema import (
    validate_alignment_log_schema,
    PROP_AI_SUMMARY,
    PROP_ALIGNMENT_SCORE,
    PROP_AUDIO_URL,
    PROP_DATE,
    PROP_DOMAIN,
    PROP_GAP_SCORE,
    PROP_IMPORTANCE_SCORE,
    PROP_MISALIGNMENT_NOTES,
    PROP_NAME,
    PROP_NEXT_ADJUSTMENT,
    PROP_REFLECTION_TEXT,
    PROP_REVIEW_TYPE,
    PROP_SIGNIFICANT_GAP,
    PROP_TAGS,
    PROP_TRANSCRIPT,
)
from src.values.schema import AlignmentEntry

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Score & dedup validation (fail-fast, called before Notion writes)
# ----------------------------------------------------------------

class AlignmentScoreError(ValueError):
    """Raised when a score violates range or type constraints."""


class AlignmentDuplicateError(ValueError):
    """Raised when a same-day duplicate is detected and prevention is enabled."""


def validate_score_range(score: int, name: str = "score") -> None:
    """Validate that a score is an integer in [1, 5].

    Parameters
    ----------
    score : int
        The score to validate.
    name : str
        Name for error messages (e.g. "importance_score").

    Raises
    ------
    AlignmentScoreError
        If ``score`` is not an int or is outside [1, 5].
    """
    if not isinstance(score, int):
        raise AlignmentScoreError(
            f"{name} must be int, got {type(score).__name__}: {score!r}"
        )
    if score < 1 or score > 5:
        raise AlignmentScoreError(
            f"{name} must be between 1 and 5 (inclusive), got {score}. "
            f"The Alignment Log requires concrete self-assessments."
        )


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

def _multi_select(names: Sequence[str]) -> dict:
    return {"multi_select": [{"name": n} for n in names]}

def _date_iso(iso_str: str) -> dict:
    if not iso_str:
        return {"date": None}
    return {"date": {"start": iso_str}}

def _url(href: str) -> dict:
    if not href:
        return {"url": None}
    return {"url": href}

def _checkbox(val: bool) -> dict:
    return {"checkbox": val}

def _relation(page_ids: Sequence[str]) -> dict:
    return {"relation": [{"id": normalize_uuid(pid)} for pid in page_ids]}


# ----------------------------------------------------------------
# AlignmentLogRepo
# ----------------------------------------------------------------

class AlignmentLogRepo:
    """Repo for ``ROS_Alignment_Log``.

    Each row is one reflection entry.  No idempotency key — reflections
    are append-only.

    Uses two-dimensional Value Evaluation Scale:
    - importance_score (1–5)
    - alignment_score  (1–5)
    - gap_score        (importance − alignment, computed)
    - significant_gap  (gap >= 2, computed)
    """

    def __init__(
        self,
        *,
        client: NotionClient,
        database_id: str,
        data_source_id: str,
        prevent_duplicate_same_day: bool = False,
    ):
        self.client = client
        self.database_id = normalize_uuid(database_id)
        self.data_source_id = data_source_id
        self.prevent_duplicate_same_day = prevent_duplicate_same_day
        self._schema_validated = False
        self._skip_items: set[str] = set()

    def validate_schema(self) -> None:
        """One-time schema check with sample-page fallback."""
        db_meta = self.client.get_database(database_id=self.database_id)
        props = db_meta.get("properties", {})
        if not props:
            logger.info(
                "GET /databases returned empty properties for ROS_Alignment_Log — "
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
                    "ROS_Alignment_Log has no pages — cannot infer schema, "
                    "skipping validation"
                )
                self._schema_validated = True
                return
        missing = validate_alignment_log_schema(db_meta)
        self._skip_items = set(missing)
        self._schema_validated = True
        logger.info("ROS_Alignment_Log schema validated OK")

    def ensure_schema(self) -> None:
        if not self._schema_validated:
            self.validate_schema()

    def create_entry(self, *, properties: Dict[str, Any]) -> dict:
        """Create a new reflection entry (append-only, no upsert).

        Low-level — does NOT validate scores or check duplicates.
        Prefer ``create_validated_entry()`` for production writes.
        """
        logger.debug("Creating alignment log entry")
        return self.client.create_page(
            parent_db_id=self.database_id,
            properties=properties,
        )

    def create_validated_entry(
        self,
        *,
        entry: AlignmentEntry,
        codex_page_id: Optional[str] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> dict:
        """Validate, build, and create an alignment log entry.

        Validates both importance_score and alignment_score (1–5),
        optionally checks for same-day duplicates, then writes.
        """
        validate_score_range(entry.importance_score, "importance_score")
        validate_score_range(entry.alignment_score, "alignment_score")

        if self.prevent_duplicate_same_day:
            self.validate_no_same_day_duplicate(
                domain_id=entry.domain_id,
                date_iso=entry.date_iso,
                review_type=entry.review_type,
            )

        props = self.build_entry_properties(
            entry=entry,
            codex_page_id=codex_page_id,
            tracker=tracker,
        )
        return self.create_entry(properties=props)

    def validate_no_same_day_duplicate(
        self,
        *,
        domain_id: str,
        date_iso: str,
        review_type: str,
    ) -> None:
        """Check for an existing entry with the same (domain, date, type)."""
        existing = self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={
                "and": [
                    {"property": PROP_DATE, "date": {"equals": date_iso}},
                    {"property": PROP_REVIEW_TYPE, "select": {"equals": review_type}},
                ],
            },
            page_size=50,
            fetch_all=False,
        )

        for page in existing:
            title_parts = []
            title_prop = page.get("properties", {}).get(PROP_NAME, {})
            for t in title_prop.get("title", []):
                title_parts.append(t.get("plain_text", ""))
            title_text = "".join(title_parts)
            if domain_id in title_text:
                raise AlignmentDuplicateError(
                    f"Duplicate entry found for domain={domain_id!r}, "
                    f"date={date_iso!r}, type={review_type!r}. "
                    f"Existing page: {page.get('id', 'unknown')}"
                )

    def build_entry_properties(
        self,
        *,
        entry: AlignmentEntry,
        codex_page_id: Optional[str] = None,
        tracker: Optional[TruncationTracker] = None,
    ) -> Dict[str, Any]:
        """Build Notion properties for a single alignment log entry."""
        title = f"{entry.review_type} — {entry.domain_id} — {entry.date_iso}"

        props: Dict[str, Any] = {
            PROP_NAME: _title(title),
            PROP_DATE: _date_iso(entry.date_iso),
            PROP_REVIEW_TYPE: _select(entry.review_type),
            PROP_IMPORTANCE_SCORE: _number(entry.importance_score),
            PROP_ALIGNMENT_SCORE: _number(entry.alignment_score),
        }

        if PROP_GAP_SCORE not in self._skip_items:
            props[PROP_GAP_SCORE] = _number(entry.gap_score)
        if PROP_SIGNIFICANT_GAP not in self._skip_items:
            props[PROP_SIGNIFICANT_GAP] = _checkbox(entry.significant_gap)

        if entry.reflection_text:
            props[PROP_REFLECTION_TEXT] = _rt(notion_truncate(
                entry.reflection_text, field_name=PROP_REFLECTION_TEXT, tracker=tracker,
            ))
        if entry.misalignment_notes:
            props[PROP_MISALIGNMENT_NOTES] = _rt(notion_truncate(
                entry.misalignment_notes, field_name=PROP_MISALIGNMENT_NOTES, tracker=tracker,
            ))
        if entry.next_adjustment:
            props[PROP_NEXT_ADJUSTMENT] = _rt(notion_truncate(
                entry.next_adjustment, field_name=PROP_NEXT_ADJUSTMENT, tracker=tracker,
            ))

        # Optional fields
        if entry.transcript and PROP_TRANSCRIPT not in self._skip_items:
            props[PROP_TRANSCRIPT] = _rt(notion_truncate(
                entry.transcript, field_name=PROP_TRANSCRIPT, tracker=tracker,
            ))
        if entry.audio_url and PROP_AUDIO_URL not in self._skip_items:
            props[PROP_AUDIO_URL] = _url(entry.audio_url)
        if entry.ai_summary and PROP_AI_SUMMARY not in self._skip_items:
            props[PROP_AI_SUMMARY] = _rt(notion_truncate(
                entry.ai_summary, field_name=PROP_AI_SUMMARY, tracker=tracker,
            ))
        if entry.tags and PROP_TAGS not in self._skip_items:
            props[PROP_TAGS] = _multi_select(entry.tags)

        # Domain relation
        if codex_page_id and PROP_DOMAIN not in self._skip_items:
            try:
                props[PROP_DOMAIN] = _relation([codex_page_id])
            except (ValueError, TypeError):
                logger.debug("Skipping Domain relation: invalid page_id %r", codex_page_id)

        return props

    def fetch_by_date(self, date_iso: str) -> List[dict]:
        """Fetch all alignment entries for a given date."""
        return self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={"property": PROP_DATE, "date": {"equals": date_iso}},
            page_size=50,
            fetch_all=False,
        )
