# src/notion/daily_upsert.py
"""Shared upsert helper for the Daily Logs hub page.

All daily scripts (057–060) operate on the same Daily Logs page,
identified by the **LogDate** property.  This module provides a
single ``upsert_daily_log`` function that:

  1. Resolves the Daily Logs database from ``NOTION_Daily_Logs_ID``.
  2. Queries by ``LogDate`` for an existing page.
  3. Updates the page if found, or creates a new one.
  4. Returns a result dict with ``ok``, ``page_id``, ``page_url``, ``action``.

Each calling script only passes its own layer's properties + ``stage``,
so other layers' fields are never overwritten.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TRUNCATION_LIMIT = 2000
_TRUNCATION_SUFFIX = " ...(truncated)"


def safe_truncate(text: Optional[str], limit: int = _TRUNCATION_LIMIT) -> str:
    """Truncate text to fit Notion rich_text limit."""
    if not text:
        return ""
    text = str(text).strip()
    safe = limit - len(_TRUNCATION_SUFFIX)
    if len(text) <= safe:
        return text
    return text[:safe].rstrip() + _TRUNCATION_SUFFIX


def upsert_daily_log(
    *,
    date_iso: str,
    properties: Dict[str, Any],
    log_label: str = "daily_log",
) -> Dict[str, Any]:
    """Upsert a Daily Logs page by LogDate.

    Parameters
    ----------
    date_iso : str
        The target date (YYYY-MM-DD), used as the idempotency key
        via the ``LogDate`` Notion property.
    properties : dict
        Ready-to-send Notion property payload (from
        ``build_daily_log_properties``).
    log_label : str
        Label used in log messages (e.g., "057_raw", "058_structured").

    Returns
    -------
    dict
        On success: ``{ok: True, page_id, page_url, action, date}``
        On failure: ``{ok: False, error, date}``
    """
    try:
        from src.config import load_env, get_db_id
        load_env()

        from src.notion.client import (
            build_notion_client_from_env,
            NotionDataSourceResolver,
        )

        client = build_notion_client_from_env(
            log_requests=True, log_responses=True,
        )
        db_id = get_db_id("NOTION_Daily_Logs_ID")
        logger.info("[%s] Notion Daily Logs DB ID: %s", log_label, db_id[:8])

        resolver = NotionDataSourceResolver(client=client)
        resolved = resolver.resolve_once(
            name="daily_logs", database_id=db_id,
        )
        logger.info(
            "[%s] Resolved data_source_id: %s",
            log_label, resolved.data_source_id[:8],
        )

        # Idempotent lookup by LogDate
        existing_id: Optional[str] = None
        try:
            pages = client.query_data_source(
                data_source_id=resolved.data_source_id,
                filter={
                    "property": "LogDate",
                    "date": {"equals": date_iso},
                },
                page_size=1,
                fetch_all=False,
            )
            if pages:
                existing_id = pages[0].get("id")
        except Exception as exc:
            logger.debug(
                "[%s] LogDate lookup failed (non-fatal): %s",
                log_label, exc,
            )

        # Create or update
        if existing_id:
            logger.info(
                "[%s] Updating existing page: %s",
                log_label, existing_id[:8],
            )
            page = client.update_page(
                page_id=existing_id, properties=properties,
            )
            page_id = existing_id
            action = "updated"
        else:
            logger.info(
                "[%s] Creating new page for %s", log_label, date_iso,
            )
            page = client.create_page(
                parent_db_id=db_id, properties=properties,
            )
            page_id = page.get("id", "")
            action = "created"

        page_url = page.get(
            "url",
            f"https://www.notion.so/{page_id.replace('-', '')}",
        )
        logger.info(
            "[%s] Upsert OK: %s page_id=%s url=%s",
            log_label, action, page_id, page_url,
        )

        return {
            "ok": True,
            "page_id": page_id,
            "page_url": page_url,
            "action": action,
            "date": date_iso,
        }

    except Exception as exc:
        logger.error(
            "[%s] Upsert failed: %s", log_label, exc, exc_info=True,
        )
        return {
            "ok": False,
            "error": str(exc),
            "date": date_iso,
        }
