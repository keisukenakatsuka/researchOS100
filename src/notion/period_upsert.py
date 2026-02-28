# src/notion/period_upsert.py
"""Period resolution and Weekly/Monthly log upsert helpers.

This module provides:

1. **Period resolution** — given a date, resolve the current Weekly or Monthly
   period from PERIOD_DB (find or create).
2. **Log upsert** — find or create a WEEKLY_LOG / MONTHLY_LOG page by
   (Period relation + Log Type), then update it idempotently.

The pattern mirrors ``daily_upsert.py`` but generalizes to period-based logs.

Usage::

    from src.notion.period_upsert import (
        resolve_period,
        upsert_weekly_log,
        upsert_monthly_log,
    )

    period = resolve_period(date_iso="2026-02-27", period_type="Weekly")
    result = upsert_weekly_log(
        period_id=period["page_id"],
        log_type="Planning",
        properties=props,
        log_label="062_weekly_planning",
    )
"""

from __future__ import annotations

import logging
from calendar import monthrange
from datetime import date, datetime, timedelta
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


# ── Period date resolution ────────────────────────────────────────

def compute_weekly_period(date_iso: str) -> Dict[str, str]:
    """Compute ISO week boundaries (Mon-Sun) for a given date.

    Returns: {"name": "2026-W09", "start_date": "2026-02-23", "end_date": "2026-03-01"}
    """
    d = date.fromisoformat(date_iso)
    # ISO weekday: Mon=1, Sun=7
    monday = d - timedelta(days=d.isoweekday() - 1)
    sunday = monday + timedelta(days=6)
    iso_year, iso_week, _ = d.isocalendar()
    return {
        "name": f"{iso_year}-W{iso_week:02d}",
        "start_date": monday.isoformat(),
        "end_date": sunday.isoformat(),
    }


def compute_monthly_period(date_iso: str) -> Dict[str, str]:
    """Compute month boundaries (1st - last day) for a given date.

    Returns: {"name": "2026-02", "start_date": "2026-02-01", "end_date": "2026-02-28"}
    """
    d = date.fromisoformat(date_iso)
    first = d.replace(day=1)
    _, last_day = monthrange(d.year, d.month)
    last = d.replace(day=last_day)
    return {
        "name": f"{d.year}-{d.month:02d}",
        "start_date": first.isoformat(),
        "end_date": last.isoformat(),
    }


# ── Notion client helpers (lazy, same pattern as daily_upsert) ────

def _build_client_and_resolver():
    """Build a Notion client + resolver. Lazy import to avoid circular deps."""
    from src.config import load_env
    load_env()

    from src.notion.client import (
        build_notion_client_from_env,
        NotionDataSourceResolver,
    )
    client = build_notion_client_from_env(
        log_requests=True, log_responses=True,
    )
    resolver = NotionDataSourceResolver(client=client)
    return client, resolver


# ── Period resolution ─────────────────────────────────────────────

def resolve_period(
    *,
    date_iso: str,
    period_type: str,  # "Weekly" or "Monthly"
    log_label: str = "period",
) -> Dict[str, Any]:
    """Find or create a PERIOD_DB page for the given date and period type.

    Parameters
    ----------
    date_iso : str
        Reference date (YYYY-MM-DD).
    period_type : str
        "Weekly" or "Monthly".
    log_label : str
        Label for log messages.

    Returns
    -------
    dict
        ``{ok, page_id, page_url, name, start_date, end_date, action}``
        On failure: ``{ok: False, error}``
    """
    try:
        from src.config import get_db_id
        from src.notion.period_schema import build_period_properties

        # Compute period boundaries
        if period_type == "Weekly":
            period = compute_weekly_period(date_iso)
        elif period_type == "Monthly":
            period = compute_monthly_period(date_iso)
        else:
            raise ValueError(f"Unknown period_type: {period_type}")

        period_name = period["name"]
        start_date = period["start_date"]
        end_date = period["end_date"]

        logger.info(
            "[%s] Resolving %s period: name=%s (%s to %s)",
            log_label, period_type, period_name, start_date, end_date,
        )

        client, resolver = _build_client_and_resolver()
        db_id = get_db_id("NOTION_PERIOD_DB_ID")
        resolved = resolver.resolve_once(name="period_db", database_id=db_id)

        logger.info(
            "[%s] PERIOD_DB data_source_id: %s",
            log_label, resolved.data_source_id[:8],
        )

        # Strategy 1: Query by Period Type + Start Date + End Date
        existing_id: Optional[str] = None
        try:
            logger.info(
                "[%s] Period query strategy 1: Period Type=%s, "
                "Start Date=%s, End Date=%s",
                log_label, period_type, start_date, end_date,
            )
            pages = client.query_data_source(
                data_source_id=resolved.data_source_id,
                filter={
                    "and": [
                        {"property": "Period Type", "select": {"equals": period_type}},
                        {"property": "Start Date", "date": {"equals": start_date}},
                        {"property": "End Date", "date": {"equals": end_date}},
                    ]
                },
                page_size=1,
                fetch_all=False,
            )
            if pages:
                existing_id = pages[0].get("id")
                logger.info(
                    "[%s] Strategy 1 matched: page_id=%s",
                    log_label, existing_id[:8] if existing_id else "???",
                )
            else:
                logger.info(
                    "[%s] Strategy 1: no match (0 results)", log_label,
                )
        except Exception as exc:
            logger.warning(
                "[%s] Period lookup by dates failed (strategy 1): %s",
                log_label, exc,
            )

        # Strategy 2: Fallback — query by Period Type + Name
        if not existing_id:
            try:
                logger.info(
                    "[%s] Period query strategy 2: Period Type=%s, Name=%s",
                    log_label, period_type, period_name,
                )
                pages = client.query_data_source(
                    data_source_id=resolved.data_source_id,
                    filter={
                        "and": [
                            {"property": "Period Type", "select": {"equals": period_type}},
                            {"property": "Name", "title": {"equals": period_name}},
                        ]
                    },
                    page_size=1,
                    fetch_all=False,
                )
                if pages:
                    existing_id = pages[0].get("id")
                    logger.info(
                        "[%s] Strategy 2 matched: page_id=%s",
                        log_label, existing_id[:8] if existing_id else "???",
                    )
                else:
                    logger.info(
                        "[%s] Strategy 2: no match (0 results)", log_label,
                    )
            except Exception as exc:
                logger.warning(
                    "[%s] Period lookup by Name failed (strategy 2): %s",
                    log_label, exc,
                )

        # Strategy 3: Broader fallback — query by Period Type only,
        # then client-side filter by date or name
        if not existing_id:
            try:
                logger.info(
                    "[%s] Period query strategy 3: Period Type=%s (broad scan)",
                    log_label, period_type,
                )
                pages = client.query_data_source(
                    data_source_id=resolved.data_source_id,
                    filter={
                        "property": "Period Type",
                        "select": {"equals": period_type},
                    },
                    page_size=50,
                    fetch_all=False,
                )
                logger.info(
                    "[%s] Strategy 3: got %d pages, filtering client-side",
                    log_label, len(pages),
                )
                for page in pages:
                    props = page.get("properties", {})
                    # Check Name (title)
                    name_parts = props.get("Name", {}).get("title", [])
                    page_name = "".join(
                        p.get("plain_text", "") for p in name_parts
                    )
                    # Check Start Date / End Date
                    sd_prop = props.get("Start Date", {}).get("date") or {}
                    ed_prop = props.get("End Date", {}).get("date") or {}
                    page_start = (sd_prop.get("start") or "")[:10]
                    page_end = (ed_prop.get("start") or "")[:10]

                    if page_name == period_name:
                        existing_id = page.get("id")
                        logger.info(
                            "[%s] Strategy 3 matched by name=%s: page_id=%s "
                            "(start=%s, end=%s)",
                            log_label, page_name,
                            existing_id[:8] if existing_id else "???",
                            page_start, page_end,
                        )
                        break
                    if page_start == start_date and page_end == end_date:
                        existing_id = page.get("id")
                        logger.info(
                            "[%s] Strategy 3 matched by dates: page_id=%s "
                            "(name=%s, start=%s, end=%s)",
                            log_label,
                            existing_id[:8] if existing_id else "???",
                            page_name, page_start, page_end,
                        )
                        break

                if not existing_id:
                    logger.info(
                        "[%s] Strategy 3: no match after scanning %d pages",
                        log_label, len(pages),
                    )
            except Exception as exc:
                logger.warning(
                    "[%s] Period broad scan failed (strategy 3): %s",
                    log_label, exc,
                )

        if existing_id:
            logger.info(
                "[%s] Found existing period: %s (page_id=%s)",
                log_label, period_name, existing_id[:8],
            )
            page = client.retrieve_page(page_id=existing_id)
            return {
                "ok": True,
                "page_id": existing_id,
                "page_url": page.get(
                    "url",
                    f"https://www.notion.so/{existing_id.replace('-', '')}",
                ),
                "name": period_name,
                "start_date": start_date,
                "end_date": end_date,
                "action": "found",
            }

        # Create new period
        logger.info(
            "[%s] Creating new period: %s (%s to %s)",
            log_label, period_name, start_date, end_date,
        )
        props = build_period_properties(
            name=period_name,
            period_type=period_type,
            start_date=start_date,
            end_date=end_date,
            status="Open",
        )
        page = client.create_page(parent_db_id=db_id, properties=props)
        page_id = page.get("id", "")
        page_url = page.get(
            "url",
            f"https://www.notion.so/{page_id.replace('-', '')}",
        )
        logger.info(
            "[%s] Created period: %s page_id=%s url=%s",
            log_label, period_name, page_id[:8], page_url,
        )
        return {
            "ok": True,
            "page_id": page_id,
            "page_url": page_url,
            "name": period_name,
            "start_date": start_date,
            "end_date": end_date,
            "action": "created",
        }

    except Exception as exc:
        logger.error(
            "[%s] Period resolution failed: %s", log_label, exc, exc_info=True,
        )
        return {"ok": False, "error": str(exc)}


# ── Reverse relation backfill (PERIOD_DB ← log page) ─────────────

def _backfill_period_reverse_relation(
    *,
    period_id: str,
    log_page_id: str,
    reverse_relation_name: str,
    log_label: str,
) -> None:
    """Ensure the PERIOD_DB page's reverse relation contains *log_page_id*.

    For example, after upserting a MONTHLY_LOG page, call this with
    ``reverse_relation_name="Monthly Logs"`` so that the PERIOD_DB row
    links back to the log.  Idempotent: no-ops if already present.
    """
    try:
        from src.notion.client import normalize_uuid

        client, resolver = _build_client_and_resolver()
        norm_log_id = normalize_uuid(log_page_id)
        norm_period_id = normalize_uuid(period_id)

        logger.info(
            "[%s] Backfill reverse relation: checking '%s' on "
            "period=%s for log=%s",
            log_label, reverse_relation_name,
            norm_period_id[:8], norm_log_id[:8],
        )

        # Retrieve current PERIOD_DB page
        page = client.retrieve_page(page_id=norm_period_id)
        props = page.get("properties", {})

        rel_prop = props.get(reverse_relation_name, {})
        if rel_prop.get("type") != "relation":
            logger.warning(
                "[%s] Property '%s' not found or not a relation on "
                "PERIOD_DB page %s (type=%s). Skipping backfill.",
                log_label, reverse_relation_name,
                norm_period_id[:8], rel_prop.get("type"),
            )
            return

        # Collect existing relation IDs (normalize for comparison)
        existing_entries = rel_prop.get("relation", [])
        existing_ids: set[str] = set()
        for entry in existing_entries:
            raw_id = entry.get("id", "")
            if raw_id:
                try:
                    existing_ids.add(normalize_uuid(raw_id))
                except ValueError:
                    existing_ids.add(raw_id)

        if rel_prop.get("has_more"):
            logger.warning(
                "[%s] Relation '%s' has_more=True — partial read. "
                "Proceeding with visible entries (%d).",
                log_label, reverse_relation_name, len(existing_ids),
            )

        # Already present?
        if norm_log_id in existing_ids:
            logger.info(
                "[%s] Reverse relation '%s' already contains log=%s "
                "(total %d entries). No update needed.",
                log_label, reverse_relation_name,
                norm_log_id[:8], len(existing_ids),
            )
            return

        # Build updated relation list (existing + new)
        updated_relation = [{"id": eid} for eid in existing_ids]
        updated_relation.append({"id": norm_log_id})

        client.update_page(
            page_id=norm_period_id,
            properties={
                reverse_relation_name: {"relation": updated_relation},
            },
        )
        logger.info(
            "[%s] Backfilled reverse relation '%s': added log=%s to "
            "period=%s (now %d entries)",
            log_label, reverse_relation_name,
            norm_log_id[:8], norm_period_id[:8],
            len(updated_relation),
        )

    except Exception as exc:
        logger.warning(
            "[%s] Reverse relation backfill failed (non-fatal): %s",
            log_label, exc,
        )


# ── Month ↔ Week cross-linking ───────────────────────────────────

def _read_relation_ids(
    props: Dict[str, Any],
    relation_name: str,
) -> set[str]:
    """Read a Notion relation property into a set of normalised page IDs."""
    from src.notion.client import normalize_uuid

    rel_prop = props.get(relation_name, {})
    ids: set[str] = set()
    for entry in rel_prop.get("relation", []):
        raw_id = entry.get("id", "")
        if raw_id:
            try:
                ids.add(normalize_uuid(raw_id))
            except ValueError:
                ids.add(raw_id)
    return ids


def _append_to_relation(
    client: Any,
    page_id: str,
    relation_name: str,
    existing_ids: set[str],
    new_id: str,
    log_label: str,
) -> bool:
    """Append *new_id* to a relation if it is not already present.

    Returns True if an update was made, False if already present.
    """
    from src.notion.client import normalize_uuid

    norm_new = normalize_uuid(new_id)
    if norm_new in existing_ids:
        return False

    updated = [{"id": eid} for eid in existing_ids]
    updated.append({"id": norm_new})
    client.update_page(
        page_id=page_id,
        properties={relation_name: {"relation": updated}},
    )
    logger.info(
        "[%s] Appended %s to '%s' on page %s (now %d)",
        log_label, norm_new[:8], relation_name,
        page_id[:8] if page_id else "???", len(updated),
    )
    return True


def _get_overlapping_weekly_periods(
    *,
    month_start: str,
    month_end: str,
    log_label: str,
) -> List[Dict[str, Any]]:
    """Query PERIOD_DB for Weekly period pages that overlap [month_start, month_end].

    Overlap condition (inclusive):
        weekly_start <= month_end AND weekly_end >= month_start

    Returns the raw Notion page dicts for matching Weekly periods.
    """
    from src.config import get_db_id

    client, resolver = _build_client_and_resolver()
    db_id = get_db_id("NOTION_PERIOD_DB_ID")
    resolved = resolver.resolve_once(name="period_db", database_id=db_id)

    logger.info(
        "[%s] Querying Weekly periods overlapping %s..%s",
        log_label, month_start, month_end,
    )

    # Server-side: fetch all Weekly periods (could refine with date
    # filters, but compound AND across Start Date + End Date + Period Type
    # can be flaky with certain API versions).  We filter client-side
    # for reliability.
    try:
        pages = client.query_data_source(
            data_source_id=resolved.data_source_id,
            filter={
                "property": "Period Type",
                "select": {"equals": "Weekly"},
            },
            page_size=100,
            fetch_all=True,
        )
    except Exception as exc:
        logger.warning(
            "[%s] Failed to query weekly periods: %s", log_label, exc,
        )
        return []

    logger.info(
        "[%s] Fetched %d Weekly period rows, filtering by overlap",
        log_label, len(pages),
    )

    m_start = date.fromisoformat(month_start)
    m_end = date.fromisoformat(month_end)
    overlapping: List[Dict[str, Any]] = []

    for pg in pages:
        props = pg.get("properties", {})
        sd_prop = props.get("Start Date", {}).get("date") or {}
        ed_prop = props.get("End Date", {}).get("date") or {}
        w_start_str = (sd_prop.get("start") or "")[:10]
        w_end_str = (ed_prop.get("start") or "")[:10]
        if not w_start_str or not w_end_str:
            continue
        try:
            w_start = date.fromisoformat(w_start_str)
            w_end = date.fromisoformat(w_end_str)
        except ValueError:
            continue

        # Overlap: w_start <= m_end AND w_end >= m_start
        if w_start <= m_end and w_end >= m_start:
            name_parts = props.get("Name", {}).get("title", [])
            w_name = "".join(p.get("plain_text", "") for p in name_parts)
            pid = pg.get("id", "")
            logger.info(
                "[%s]   overlap: %s (%s..%s) page_id=%s",
                log_label, w_name, w_start_str, w_end_str,
                pid[:8] if pid else "???",
            )
            overlapping.append(pg)

    logger.info(
        "[%s] Found %d overlapping Weekly periods for %s..%s",
        log_label, len(overlapping), month_start, month_end,
    )
    return overlapping


def backfill_monthly_log_to_weekly_periods(
    *,
    monthly_log_page_id: str,
    month_start: str,
    month_end: str,
    log_label: str = "ml_to_wp",
) -> Dict[str, Any]:
    """Append a MONTHLY_LOG page to each overlapping Weekly period's
    ``Monthly Logs`` relation in PERIOD_DB.

    After 064/065 upserts a MONTHLY_LOG page, call this to ensure
    every Weekly period row that overlaps the month has the monthly log
    linked via its ``Monthly Logs`` relation property.  Idempotent.

    Parameters
    ----------
    monthly_log_page_id : str
        The MONTHLY_LOG page ID (from upsert result).
    month_start, month_end : str
        YYYY-MM-DD boundaries of the month.
    log_label : str
        Label for log messages.

    Returns
    -------
    dict
        ``{ok, weeks_found, weeks_updated}``
    """
    from src.notion.client import normalize_uuid

    logger.info(
        "[%s] === Backfill Monthly Log → Weekly periods: "
        "log=%s range=%s..%s ===",
        log_label, monthly_log_page_id[:8], month_start, month_end,
    )

    try:
        overlapping = _get_overlapping_weekly_periods(
            month_start=month_start,
            month_end=month_end,
            log_label=f"{log_label}_overlap",
        )

        if not overlapping:
            logger.info("[%s] No overlapping weekly periods found", log_label)
            return {"ok": True, "weeks_found": 0, "weeks_updated": 0}

        client, _ = _build_client_and_resolver()
        norm_log_id = normalize_uuid(monthly_log_page_id)
        updated_count = 0

        for pg in overlapping:
            pid = pg.get("id", "")
            if not pid:
                continue
            try:
                # Re-fetch to get latest relation state
                page = client.retrieve_page(page_id=pid)
                props = page.get("properties", {})
                existing = _read_relation_ids(props, "Monthly Logs")
                if _append_to_relation(
                    client, pid, "Monthly Logs", existing, norm_log_id,
                    log_label=log_label,
                ):
                    updated_count += 1
                else:
                    logger.info(
                        "[%s] Weekly %s already has Monthly Logs containing %s",
                        log_label, pid[:8], norm_log_id[:8],
                    )
            except Exception as exc:
                logger.warning(
                    "[%s] Failed to backfill Monthly Logs on weekly %s: %s",
                    log_label, pid[:8], exc,
                )

        logger.info(
            "[%s] === Backfill complete: weeks_found=%d, "
            "weeks_updated=%d ===",
            log_label, len(overlapping), updated_count,
        )

        return {
            "ok": True,
            "weeks_found": len(overlapping),
            "weeks_updated": updated_count,
        }

    except Exception as exc:
        logger.warning(
            "[%s] Backfill Monthly Log → Weekly periods failed "
            "(non-fatal): %s",
            log_label, exc,
        )
        return {"ok": False, "error": str(exc)}


def _backfill_monthly_weekly_logs(
    *,
    monthly_period_id: str,
    overlapping_weekly_pages: List[Dict[str, Any]],
    log_label: str,
) -> int:
    """Populate the Monthly period's ``Weekly Logs`` relation with WEEKLY_LOG
    pages whose ``Period`` matches any overlapping weekly period.

    Collects *all* WEEKLY_LOG page IDs (both Planning and Review) across
    the overlapping weeks, then ensures each is present in the Monthly
    period's ``Weekly Logs`` relation.  Returns the number of new entries
    added.
    """
    from src.config import get_db_id
    from src.notion.client import normalize_uuid

    if not overlapping_weekly_pages:
        logger.info("[%s] No overlapping weeks — nothing to backfill", log_label)
        return 0

    client, resolver = _build_client_and_resolver()
    norm_monthly = normalize_uuid(monthly_period_id)

    # Resolve WEEKLY_LOG data source
    wl_db_id = get_db_id("NOTION_WEEKLY_LOG_ID")
    wl_resolved = resolver.resolve_once(name="weekly_log", database_id=wl_db_id)

    # Collect all weekly log page IDs across overlapping weeks
    weekly_log_ids: set[str] = set()
    for pg in overlapping_weekly_pages:
        w_pid = pg.get("id", "")
        if not w_pid:
            continue
        try:
            pages = client.query_data_source(
                data_source_id=wl_resolved.data_source_id,
                filter={
                    "property": "Period",
                    "relation": {"contains": w_pid},
                },
                page_size=10,
                fetch_all=False,
            )
            for lp in pages:
                lid = lp.get("id", "")
                if lid:
                    weekly_log_ids.add(normalize_uuid(lid))
                    log_type_prop = lp.get("properties", {}).get("Log Type", {})
                    lt = (log_type_prop.get("select") or {}).get("name", "?")
                    logger.info(
                        "[%s]   WEEKLY_LOG %s (type=%s) via week %s",
                        log_label, lid[:8], lt, w_pid[:8],
                    )
        except Exception as exc:
            logger.warning(
                "[%s] Failed to query WEEKLY_LOG for week %s: %s",
                log_label, w_pid[:8], exc,
            )

    if not weekly_log_ids:
        logger.info("[%s] No WEEKLY_LOG pages found for overlapping weeks", log_label)
        return 0

    logger.info(
        "[%s] Found %d WEEKLY_LOG pages across overlapping weeks",
        log_label, len(weekly_log_ids),
    )

    # Read the Monthly period's current Weekly Logs relation
    try:
        monthly_page = client.retrieve_page(page_id=norm_monthly)
        m_props = monthly_page.get("properties", {})
        existing_wl = _read_relation_ids(m_props, "Weekly Logs")
    except Exception as exc:
        logger.warning(
            "[%s] Cannot read Monthly period %s: %s",
            log_label, norm_monthly[:8], exc,
        )
        return 0

    # Append missing weekly log IDs
    to_add = weekly_log_ids - existing_wl
    if not to_add:
        logger.info(
            "[%s] Monthly period already has all %d weekly logs",
            log_label, len(weekly_log_ids),
        )
        return 0

    updated_relation = [{"id": eid} for eid in existing_wl]
    for wl_id in to_add:
        updated_relation.append({"id": wl_id})

    try:
        client.update_page(
            page_id=norm_monthly,
            properties={
                "Weekly Logs": {"relation": updated_relation},
            },
        )
        logger.info(
            "[%s] Backfilled %d weekly logs onto monthly period %s "
            "(total now %d)",
            log_label, len(to_add), norm_monthly[:8],
            len(updated_relation),
        )
    except Exception as exc:
        logger.warning(
            "[%s] Failed to update Weekly Logs on monthly %s: %s",
            log_label, norm_monthly[:8], exc,
        )
        return 0

    return len(to_add)


def link_monthly_period_to_weeks(
    *,
    monthly_period_id: str,
    month_start: str,
    month_end: str,
    log_label: str = "month_week_link",
) -> Dict[str, Any]:
    """Orchestrate Month ↔ Week cross-linking for a resolved Monthly period.

    This is the public entry point that scripts (064, 065) should call
    after resolving the monthly period.  It performs:

    **Monthly period → Weekly Logs**:  Collect all WEEKLY_LOG pages whose
    ``Period`` matches one of the overlapping weekly periods, and ensure
    the Monthly period page's ``Weekly Logs`` relation contains them.

    Note: linking the MONTHLY_LOG page to each Weekly period's
    ``Monthly Logs`` relation is handled separately by
    ``backfill_monthly_log_to_weekly_periods()`` (called after the
    monthly log upsert, once the page_id is known).

    All operations are idempotent.

    Returns
    -------
    dict
        ``{ok, weeks_found, weekly_logs_added}``
    """
    logger.info(
        "[%s] === Month↔Week cross-link: monthly=%s range=%s..%s ===",
        log_label, monthly_period_id[:8], month_start, month_end,
    )

    try:
        # Find overlapping weekly periods
        overlapping = _get_overlapping_weekly_periods(
            month_start=month_start,
            month_end=month_end,
            log_label=f"{log_label}_overlap",
        )

        # Backfill Weekly Logs on monthly period
        wl_added = _backfill_monthly_weekly_logs(
            monthly_period_id=monthly_period_id,
            overlapping_weekly_pages=overlapping,
            log_label=f"{log_label}_wlogs",
        )

        logger.info(
            "[%s] === Cross-link complete: weeks_found=%d, "
            "weekly_logs_added=%d ===",
            log_label, len(overlapping), wl_added,
        )

        return {
            "ok": True,
            "weeks_found": len(overlapping),
            "weekly_logs_added": wl_added,
        }

    except Exception as exc:
        logger.warning(
            "[%s] Month↔Week cross-link failed (non-fatal): %s",
            log_label, exc,
        )
        return {"ok": False, "error": str(exc)}


# ── Generic log upsert ────────────────────────────────────────────

def _upsert_log(
    *,
    db_env_name: str,
    resolver_name: str,
    period_id: str,
    log_type: str,
    properties: Dict[str, Any],
    log_label: str,
    reverse_relation_name: str = "",
) -> Dict[str, Any]:
    """Find or create a log page by (Period + Log Type), then update.

    Parameters
    ----------
    db_env_name : str
        Env var for the target DB ID (e.g. "NOTION_WEEKLY_LOG_ID").
    resolver_name : str
        Cache key for the data source resolver.
    period_id : str
        Page ID of the resolved PERIOD_DB row.
    log_type : str
        "Planning" or "Review".
    properties : dict
        Ready-to-send Notion property payload.
    log_label : str
        Label for log messages.
    reverse_relation_name : str
        If non-empty, backfill this relation property on the PERIOD_DB page
        so it contains the log page_id (e.g. "Monthly Logs", "Weekly Logs").

    Returns
    -------
    dict
        ``{ok, page_id, page_url, action}``
    """
    try:
        from src.config import get_db_id

        client, resolver = _build_client_and_resolver()
        db_id = get_db_id(db_env_name)
        resolved = resolver.resolve_once(name=resolver_name, database_id=db_id)

        logger.info(
            "[%s] %s data_source_id: %s",
            log_label, resolver_name, resolved.data_source_id[:8],
        )

        # Query by Period relation + Log Type
        existing_id: Optional[str] = None
        try:
            logger.info(
                "[%s] Querying log: Period contains=%s, Log Type=%s",
                log_label, period_id[:8], log_type,
            )
            pages = client.query_data_source(
                data_source_id=resolved.data_source_id,
                filter={
                    "and": [
                        {"property": "Period", "relation": {"contains": period_id}},
                        {"property": "Log Type", "select": {"equals": log_type}},
                    ]
                },
                page_size=1,
                fetch_all=False,
            )
            if pages:
                existing_id = pages[0].get("id")
                logger.info(
                    "[%s] Found existing log: page_id=%s",
                    log_label, existing_id[:8] if existing_id else "???",
                )
            else:
                logger.info(
                    "[%s] No existing log found (0 results)", log_label,
                )
        except Exception as exc:
            logger.warning(
                "[%s] Log lookup failed: %s", log_label, exc,
            )

        if existing_id:
            logger.info(
                "[%s] Updating existing log: %s",
                log_label, existing_id[:8],
            )
            page = client.update_page(
                page_id=existing_id, properties=properties,
            )
            page_id = existing_id
            action = "updated"
        else:
            logger.info("[%s] Creating new log", log_label)
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
            log_label, action, page_id[:8], page_url,
        )

        # ── Backfill reverse relation on PERIOD_DB ──
        if reverse_relation_name and page_id:
            _backfill_period_reverse_relation(
                period_id=period_id,
                log_page_id=page_id,
                reverse_relation_name=reverse_relation_name,
                log_label=f"{log_label}_backfill",
            )

        return {
            "ok": True,
            "page_id": page_id,
            "page_url": page_url,
            "action": action,
        }

    except Exception as exc:
        logger.error(
            "[%s] Upsert failed: %s", log_label, exc, exc_info=True,
        )
        return {"ok": False, "error": str(exc)}


def upsert_weekly_log(
    *,
    period_id: str,
    log_type: str,
    properties: Dict[str, Any],
    log_label: str = "weekly_log",
) -> Dict[str, Any]:
    """Upsert a WEEKLY_LOG page by (Period + Log Type).

    Also backfills the ``Weekly Logs`` reverse relation on the PERIOD_DB
    page so that the period row links back to this log.
    """
    return _upsert_log(
        db_env_name="NOTION_WEEKLY_LOG_ID",
        resolver_name="weekly_log",
        period_id=period_id,
        log_type=log_type,
        properties=properties,
        log_label=log_label,
        reverse_relation_name="Weekly Logs",
    )


def upsert_monthly_log(
    *,
    period_id: str,
    log_type: str,
    properties: Dict[str, Any],
    log_label: str = "monthly_log",
) -> Dict[str, Any]:
    """Upsert a MONTHLY_LOG page by (Period + Log Type).

    Also backfills the ``Monthly Logs`` reverse relation on the PERIOD_DB
    page so that the period row links back to this log.
    """
    return _upsert_log(
        db_env_name="NOTION_MONTHLY_LOG_ID",
        resolver_name="monthly_log",
        period_id=period_id,
        log_type=log_type,
        properties=properties,
        log_label=log_label,
        reverse_relation_name="Monthly Logs",
    )


# ── Query existing log ────────────────────────────────────────────

def query_existing_log(
    *,
    db_env_name: str,
    resolver_name: str,
    period_id: str,
    log_type: str,
    log_label: str = "query_log",
) -> Optional[Dict[str, Any]]:
    """Query for an existing log page by (Period + Log Type).

    Returns the raw Notion page dict if found, else None.
    """
    try:
        from src.config import get_db_id

        client, resolver = _build_client_and_resolver()
        db_id = get_db_id(db_env_name)
        resolved = resolver.resolve_once(name=resolver_name, database_id=db_id)

        pages = client.query_data_source(
            data_source_id=resolved.data_source_id,
            filter={
                "and": [
                    {"property": "Period", "relation": {"contains": period_id}},
                    {"property": "Log Type", "select": {"equals": log_type}},
                ]
            },
            page_size=1,
            fetch_all=False,
        )
        if pages:
            logger.info(
                "[%s] Found existing log: page_id=%s",
                log_label, pages[0].get("id", "")[:8],
            )
            return pages[0]

        logger.info("[%s] No existing log found", log_label)
        return None

    except Exception as exc:
        logger.warning("[%s] Query failed: %s", log_label, exc)
        return None


def extract_rich_text(prop: Dict[str, Any]) -> str:
    """Extract plain text from a Notion rich_text property."""
    parts = prop.get("rich_text", [])
    return "".join(p.get("plain_text", "") for p in parts)


def extract_title_text(prop: Dict[str, Any]) -> str:
    """Extract plain text from a Notion title property."""
    parts = prop.get("title", [])
    return "".join(p.get("plain_text", "") for p in parts)
