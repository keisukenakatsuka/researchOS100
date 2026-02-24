# src/notion/github_timer_repo.py
"""Repository for the GITHUB_TIMER_DB Notion database.

Reads enabled rows from the database and updates timestamp/checkbox
fields after git operations.  No upsert — only read + update.

All content queries use ``POST /data_sources/{id}/query``
(via ``NotionClient.query_data_source``).

Schema validation
-----------------
``GET /databases/{id}`` may return empty ``properties`` on newer Notion
API versions.  When that happens, we fall back to sampling one page via
``data_sources`` and inferring the schema from the page's property
types (only ``"type"`` fields are trusted in inference mode — see
``client.infer_schema_types_from_sample_page``).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Set

from src.notion.client import (
    NotionClient,
    infer_schema_types_from_sample_page,
    normalize_uuid,
)
from src.notion.properties import extract_property_value, page_to_record

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Notion property names  (must match what you create in the UI)
# ----------------------------------------------------------------

PROP_NAME = "Name"                                   # title
PROP_ENABLED = "enabled"                             # checkbox
PROP_SRC_PATH = "src_path"                           # rich_text
PROP_DST_PATH = "dst_path"                           # rich_text
PROP_INCLUDE_GLOBS = "include_globs"                 # multi_select
PROP_EXCLUDE_GLOBS = "exclude_globs"                 # multi_select
PROP_LAST_PRIVATE_RUN_AT = "last_private_run_at"     # date
PROP_LAST_CHECKED_AT = "last_checked_at"             # date
PROP_PRIVATE_COMMENT_NEEDED = "private_comment_needed"  # checkbox
PROP_LAST_PROD_RUN_AT = "last_prod_run_at"           # date
PROP_PROD_DUE_AT = "prod_due_at"                     # date
PROP_PROD_COMMENT_NEEDED = "prod_comment_needed"     # checkbox


# Expected Notion property type per property name.
EXPECTED_PROPERTIES: Dict[str, str] = {
    PROP_NAME: "title",
    PROP_ENABLED: "checkbox",
    PROP_SRC_PATH: "rich_text",
    PROP_DST_PATH: "rich_text",
    PROP_INCLUDE_GLOBS: "multi_select",
    PROP_EXCLUDE_GLOBS: "multi_select",
    PROP_LAST_PRIVATE_RUN_AT: "date",
    PROP_LAST_CHECKED_AT: "date",
    PROP_PRIVATE_COMMENT_NEEDED: "checkbox",
    PROP_LAST_PROD_RUN_AT: "date",
    PROP_PROD_DUE_AT: "date",
    PROP_PROD_COMMENT_NEEDED: "checkbox",
}

# Properties that are optional (won't fail validation if missing).
# last_checked_at is written by the sync tool on first run; it may not
# exist until the first successful update_page creates it.
OPTIONAL_PROPERTIES: Set[str] = {
    PROP_EXCLUDE_GLOBS,
    PROP_LAST_CHECKED_AT,
}

# Property names to extract when reading rows.
_READ_PROPERTIES: List[str] = [
    PROP_NAME,
    PROP_ENABLED,
    PROP_SRC_PATH,
    PROP_DST_PATH,
    PROP_INCLUDE_GLOBS,
    PROP_EXCLUDE_GLOBS,
    PROP_LAST_PRIVATE_RUN_AT,
    PROP_LAST_CHECKED_AT,
    PROP_PRIVATE_COMMENT_NEEDED,
    PROP_LAST_PROD_RUN_AT,
    PROP_PROD_DUE_AT,
    PROP_PROD_COMMENT_NEEDED,
]


# ----------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------

class GitHubTimerSchemaError(ValueError):
    """Raised when the Notion DB schema does not match expectations."""


def validate_github_timer_schema(
    db_meta: Mapping[str, Any],
    *,
    allow_missing: Set[str] | None = None,
) -> None:
    """Validate that GITHUB_TIMER_DB has the required properties.

    Parameters
    ----------
    db_meta:
        Result of ``GET /databases/{database_id}``.
    allow_missing:
        Extra property names to treat as optional beyond
        ``OPTIONAL_PROPERTIES``.

    Raises
    ------
    GitHubTimerSchemaError
        If any required property is missing or has the wrong type.
    """
    allow = set(OPTIONAL_PROPERTIES)
    if allow_missing:
        allow |= allow_missing
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise GitHubTimerSchemaError(
            "GITHUB_TIMER_DB: db_meta.properties missing or invalid"
        )

    existing = {name: p.get("type") for name, p in props.items()}
    errors: list[str] = []

    for name, expected_type in EXPECTED_PROPERTIES.items():
        if name in allow:
            continue
        if name not in existing:
            errors.append(f"  - Missing property: {name!r} (expected type: {expected_type})")
        elif existing[name] != expected_type:
            errors.append(
                f"  - Wrong type for {name!r}: "
                f"got {existing[name]!r}, expected {expected_type!r}"
            )

    if errors:
        raise GitHubTimerSchemaError(
            "GITHUB_TIMER_DB schema validation failed.\n"
            "Please create the following properties manually in Notion "
            "before running:\n"
            + "\n".join(errors)
            + "\n\nExisting properties: "
            + ", ".join(sorted(existing.keys()))
        )


# ----------------------------------------------------------------
# Property builders (Notion API payload fragments)
# ----------------------------------------------------------------

def _date_iso(dt: datetime) -> dict:
    """Build a Notion date property value from a datetime."""
    return {"date": {"start": dt.isoformat(timespec="seconds")}}


def _checkbox(v: bool) -> dict:
    return {"checkbox": bool(v)}


# ----------------------------------------------------------------
# GitHubTimerRepo
# ----------------------------------------------------------------

class GitHubTimerRepo:
    """Repository for the GitHub Timer Notion database.

    Reads enabled rows and updates timestamp/checkbox fields.
    No upsert — we only read existing rows and update specific fields.

    Parameters
    ----------
    client:
        A :class:`NotionClient` instance.
    database_id:
        The Notion database UUID (for ``GET /databases`` schema calls).
    data_source_id:
        The resolved data_source_id (for ``POST /data_sources`` queries).
    """

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

    # ---- Schema ----

    def validate_schema(self) -> None:
        """One-time schema check.

        Tries ``GET /databases/{id}`` first.  If that returns empty
        ``properties`` (common on newer Notion API versions), falls back
        to sampling one page via ``data_sources`` and inferring types.
        """
        db_meta = self.client.get_database(database_id=self.database_id)
        props = db_meta.get("properties", {})

        if not props:
            # Inference mode: sample one page via data_sources
            logger.info(
                "GET /databases returned empty properties — "
                "inferring schema from sample page"
            )
            sample_pages = self.client.query_data_source(
                data_source_id=self.data_source_id,
                page_size=1,
                fetch_all=False,
            )
            if not sample_pages:
                raise GitHubTimerSchemaError(
                    "GITHUB_TIMER_DB has no pages — cannot infer schema. "
                    "Add at least one row before running."
                )
            inferred = infer_schema_types_from_sample_page(sample_pages[0])
            # Wrap inferred types in the same shape as db_meta.properties
            # so validate_github_timer_schema can consume it
            db_meta = {
                "properties": {
                    name: {"type": ptype} for name, ptype in inferred.items()
                }
            }

        validate_github_timer_schema(db_meta)
        self._schema_validated = True
        logger.info("GITHUB_TIMER_DB schema validated OK")

    def ensure_schema(self) -> None:
        """Validate schema exactly once (idempotent)."""
        if not self._schema_validated:
            self.validate_schema()

    # ---- Read ----

    def fetch_enabled_rows(
        self,
        *,
        name_contains: str = "",
    ) -> List[Dict[str, Any]]:
        """Fetch rows with ``enabled=True`` via Notion-side filter.

        The ``enabled`` checkbox filter is applied at the
        ``query_data_source()`` level for efficiency.  Only the optional
        ``--name-contains`` substring filtering is done in Python.

        Returns
        -------
        list[dict]
            Flat dicts via :func:`page_to_record` — each dict has
            ``notion_page_id``, ``notion_url``, and all ``_READ_PROPERTIES``.
        """
        pages = self.client.query_data_source(
            data_source_id=self.data_source_id,
            filter={
                "property": PROP_ENABLED,
                "checkbox": {"equals": True},
            },
            fetch_all=True,
        )

        records = [page_to_record(p, _READ_PROPERTIES) for p in pages]

        # Optional Python-side name filter
        if name_contains:
            needle = name_contains.lower()
            records = [
                r for r in records
                if needle in (r.get(PROP_NAME) or "").lower()
            ]

        logger.info(
            "Fetched %d enabled rows%s",
            len(records),
            f" (name_contains={name_contains!r})" if name_contains else "",
        )
        return records

    # ---- Update: private phase ----

    def update_private_changed(
        self,
        *,
        page_id: str,
        now_jst: datetime,
    ) -> None:
        """Set ``last_private_run_at`` and clear ``private_comment_needed``.

        Called in two cases:
        1. First-check — the row's ``last_private_run_at`` was empty
           (page just created), so we initialise it.
        2. File changes — the row's ``src_path`` had actual changes in
           the private repo commit.
        """
        self.client.update_page(
            page_id=page_id,
            properties={
                PROP_LAST_PRIVATE_RUN_AT: _date_iso(now_jst),
                PROP_PRIVATE_COMMENT_NEEDED: _checkbox(False),
            },
        )
        logger.debug("Updated private_changed for page_id=%s", page_id)

    def update_checked(
        self,
        *,
        page_id: str,
        now_jst: datetime,
    ) -> None:
        """Sync job ran: set ``last_checked_at`` regardless of changes.

        Called for every enabled row every time the sync job runs.

        If the ``last_checked_at`` property does not yet exist in the
        Notion database, the API returns 400.  We catch that and log a
        warning instead of failing — the user can add the property later.
        """
        from src.notion.client import NotionAPIError

        try:
            self.client.update_page(
                page_id=page_id,
                properties={
                    PROP_LAST_CHECKED_AT: _date_iso(now_jst),
                },
            )
            logger.debug("Updated last_checked_at for page_id=%s", page_id)
        except NotionAPIError as e:
            if e.status_code == 400 and "not a property that exists" in str(e):
                logger.warning(
                    "Skipping last_checked_at update — property does not "
                    "exist yet in Notion. Create a 'last_checked_at' date "
                    "property to enable this feature."
                )
                # Only warn once, then suppress further identical warnings
                self.update_checked = self._update_checked_noop  # type: ignore[assignment]
            else:
                raise

    def _update_checked_noop(self, *, page_id: str, now_jst: datetime) -> None:
        """No-op replacement after first 400 for missing property."""
        pass

    # ---- Update: production phase ----

    def update_prod_run(
        self,
        *,
        page_id: str,
        now_jst: datetime,
    ) -> None:
        """Row copied to prod: set ``last_prod_run_at`` and clear comment flag.

        Called only for rows whose files were actually copied to the
        production repo.
        """
        self.client.update_page(
            page_id=page_id,
            properties={
                PROP_LAST_PROD_RUN_AT: _date_iso(now_jst),
                PROP_PROD_COMMENT_NEEDED: _checkbox(False),
            },
        )
        logger.debug("Updated prod_run for page_id=%s", page_id)
