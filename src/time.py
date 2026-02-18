# src/time.py
"""Week/time context utilities.

Pure functions and a frozen dataclass for computing ISO-week windows
in any timezone.  No I/O, no side effects.

Usage::

    from src.time import get_week_context, get_iso_week_context

    wk = get_week_context()                          # rolling 7-day UTC
    wk = get_iso_week_context(tz=ZoneInfo("Asia/Tokyo"))  # Mon–Sun JST

Migration note
--------------
When this package is renamed ``src/`` → ``researchos/``,
update consumer imports:  ``from researchos.time import ...``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo


# ----------------------------------------------------------------
# WeekContext dataclass
# ----------------------------------------------------------------

@dataclass(frozen=True)
class WeekContext:
    """Immutable snapshot of the current ISO-week context.

    All datetime values are **UTC-aware** (``tzinfo=timezone.utc``).

    Timezone convention
    -------------------
    - ``now_utc`` / ``week_start``: full ``datetime`` objects in UTC.
    - ``date_from_iso`` / ``date_to_iso``: ISO 8601 strings **with**
      timezone offset (e.g. ``2026-02-08T23:35:52+00:00``).  These are
      the values that should be sent to Notion date filters and recorded
      in ``run_metadata.json``.
    - ``start_date`` / ``end_date``: bare ``YYYY-MM-DD`` (UTC calendar
      day, kept for display / backward-compat).
    """
    now_utc: datetime
    week_start: datetime
    start_date: str        # YYYY-MM-DD  (display only)
    end_date: str          # YYYY-MM-DD  (display only)
    date_from_iso: str     # ISO 8601 with tz, e.g. "2026-02-08T23:35:52+00:00"
    date_to_iso: str       # ISO 8601 with tz
    iso_year: int
    iso_week: int
    week_id: str           # e.g. "2025-W07"


# ----------------------------------------------------------------
# Rolling 7-day window (UTC)
# ----------------------------------------------------------------

def get_week_context(*, reference_time: Optional[datetime] = None) -> WeekContext:
    """Build a :class:`WeekContext` for the current (or given) UTC time.

    All timestamps are explicitly UTC.  If *reference_time* is naive
    (no ``tzinfo``), it is treated as UTC.
    """
    now = reference_time or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    week_start = now - timedelta(days=7)
    iso_year, iso_week, _ = now.isocalendar()
    return WeekContext(
        now_utc=now,
        week_start=week_start,
        start_date=week_start.strftime("%Y-%m-%d"),
        end_date=now.strftime("%Y-%m-%d"),
        date_from_iso=week_start.isoformat(timespec="seconds"),
        date_to_iso=now.isoformat(timespec="seconds"),
        iso_year=iso_year,
        iso_week=iso_week,
        week_id=f"{iso_year}-W{iso_week:02d}",
    )


# ----------------------------------------------------------------
# ISO week boundaries (Mon 00:00 → Sun 23:59:59 in *tz*)
# ----------------------------------------------------------------

def get_iso_week_context(
    *,
    reference_time: Optional[datetime] = None,
    tz: ZoneInfo | timezone = timezone.utc,
) -> WeekContext:
    """Build a :class:`WeekContext` aligned to **ISO week boundaries**.

    Unlike :func:`get_week_context` (rolling 7-day window),
    this function snaps to Monday 00:00 → Sunday 23:59:59 in *tz*.

    Parameters
    ----------
    reference_time:
        Wall-clock time to derive the week from.  If ``None``, uses
        ``datetime.now(tz)``.
    tz:
        Timezone for week boundaries.  ``Asia/Tokyo`` is the primary
        use case (events digest).

    Returns a :class:`WeekContext` whose ``week_start`` / ``now_utc``
    are tz-aware (in the given *tz*), and ISO strings carry the offset.
    """
    now = reference_time or datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)

    # ISO weekday: Mon=1 … Sun=7
    iso_wd = now.isoweekday()
    week_start = (now - timedelta(days=iso_wd - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    week_end = week_start + timedelta(days=7) - timedelta(seconds=1)

    iso_year, iso_week, _ = now.isocalendar()
    week_id = f"{iso_year}-W{iso_week:02d}"

    return WeekContext(
        now_utc=now,
        week_start=week_start,
        start_date=week_start.strftime("%Y-%m-%d"),
        end_date=week_end.strftime("%Y-%m-%d"),
        date_from_iso=week_start.isoformat(timespec="seconds"),
        date_to_iso=week_end.isoformat(timespec="seconds"),
        iso_year=iso_year,
        iso_week=iso_week,
        week_id=week_id,
    )
