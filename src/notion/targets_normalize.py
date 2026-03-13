# src/notion/targets_normalize.py
"""Normalize, filter, and compute metrics for Monitoring Target pages.

Extracted from notebook 043_weekly_targets_review Cells 05/07.
Pure functions — no API calls, no pandas, no side effects.

Usage::

    from src.notion.targets_normalize import (
        normalize_targets, filter_targets, compute_target_metrics,
    )

    records = normalize_targets(raw_pages)
    active = filter_targets(records)
    enriched = compute_target_metrics(active, events, reference_time=now)
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from src.notion.properties import extract_property_value
from src.notion.targets_schema import (
    DEFAULT_TARGET_EXCLUDE_STATUSES,
    DEFAULT_TARGET_FILTER_ENABLED,
    NOISE_WEIGHTS,
    SIGNAL_WEIGHTS,
    VOLUME_CAP,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _to_date_str(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else s


def _normalize_uuid(uuid_str: str) -> str:
    """Normalise a UUID to hyphenated lowercase form."""
    if not uuid_str:
        return ""
    s = re.sub(r"[^a-f0-9]", "", uuid_str.strip().lower())
    if len(s) == 32:
        return f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"
    return uuid_str.strip().lower()


# ----------------------------------------------------------------
# Normalize a single target page
# ----------------------------------------------------------------

def normalize_target(page: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw Notion Monitoring Target page into a flat record.

    Returns ``None`` if the page lacks a ``Name`` (title).
    """
    page_id = _normalize_uuid(page.get("id", "") or "")
    ev = extract_property_value

    name = ev(page, "Name") or ""
    if not name:
        return None

    # Parse numeric fields safely
    error_count_raw = ev(page, "Error Count")
    try:
        error_count = int(error_count_raw) if error_count_raw is not None else 0
    except (ValueError, TypeError):
        error_count = 0

    consecutive_misses_raw = ev(page, "Consecutive Misses")
    try:
        consecutive_misses = int(consecutive_misses_raw) if consecutive_misses_raw is not None else 0
    except (ValueError, TypeError):
        consecutive_misses = 0

    return {
        "page_id": page_id,
        "name": name,
        "type": ev(page, "Type") or "Unknown",
        "status": ev(page, "Status") or "Active",
        "priority": ev(page, "Priority") or "Medium",
        "cadence": ev(page, "Cadence") or "Weekly",
        "enabled": bool(ev(page, "Enabled")),
        "search_keywords": ev(page, "Search Keywords") or "",
        "source_urls": ev(page, "Source URLs") or "",
        "source_type": ev(page, "Source Type") or "",
        "last_checked": _to_date_str(ev(page, "Last Checked")),
        "next_check": _to_date_str(ev(page, "Next Check")),
        "last_error": ev(page, "Last Error") or "",
        "error_count": error_count,
        "last_hit_at": _to_date_str(ev(page, "Last Hit At")),
        "consecutive_misses": consecutive_misses,
        "cadence_reason": ev(page, "Cadence Reason") or "",
        "created_by": ev(page, "Created By") or "",
        "source_session": ev(page, "Source Session") or "",
    }


def normalize_targets(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of raw Notion Target pages."""
    records: List[Dict[str, Any]] = []
    skipped = 0
    for page in pages:
        rec = normalize_target(page)
        if rec is None:
            skipped += 1
            logger.warning("Skipping target page %s: missing Name/title",
                           page.get("id", "?"))
            continue
        records.append(rec)
    logger.info("Normalised %d targets (skipped %d)", len(records), skipped)
    return records


# ----------------------------------------------------------------
# Filter
# ----------------------------------------------------------------

def filter_targets(
    targets: List[Dict[str, Any]],
    *,
    enabled: bool = DEFAULT_TARGET_FILTER_ENABLED,
    exclude_statuses: Set[str] | frozenset = DEFAULT_TARGET_EXCLUDE_STATUSES,
) -> List[Dict[str, Any]]:
    """Defensive local filter matching notebook 043 Cell 05 logic."""
    exclude_lower = {s.lower() for s in exclude_statuses}
    out = []
    for t in targets:
        if enabled and not t.get("enabled", False):
            continue
        status_l = (t.get("status") or "").strip().lower()
        if status_l in exclude_lower:
            continue
        out.append(t)
    logger.info("Filtered targets: %d → %d (enabled=%s, exclude=%s)",
                len(targets), len(out), enabled, sorted(exclude_statuses))
    return out


# ----------------------------------------------------------------
# Compute per-target metrics from events
# ----------------------------------------------------------------

def compute_target_metrics(
    targets: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    *,
    reference_time: datetime,
    volume_cap: int = VOLUME_CAP,
) -> List[Dict[str, Any]]:
    """Enrich target records with signal/noise metrics from linked events.

    Events are linked via the ``target_ids`` field (list of UUIDs) already
    present in the 048 output JSON.

    Returns a new list of dicts — original target fields plus metrics.

    Parameters
    ----------
    reference_time:
        "now" for computing days_since_last_event / recency_score.
    """
    # Build target_id → events index
    events_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        tids = ev.get("target_ids") or []
        if isinstance(tids, str):
            tids = [t.strip() for t in tids.split(",") if t.strip()]
        for tid in tids:
            events_by_target[_normalize_uuid(tid)].append(ev)

    enriched = []
    for t in targets:
        pid = _normalize_uuid(t["page_id"])
        linked = events_by_target.get(pid, [])

        n_events = len(linked)
        linked_event_ids = [e.get("page_id", e.get("notion_page_id", "")) for e in linked]

        # --- Action-needed share ---
        action_count = sum(1 for e in linked if e.get("action_needed"))
        share_action_needed = action_count / n_events if n_events else 0.0

        # --- Average confidence ---
        confidences = []
        for e in linked:
            c = e.get("confidence")
            if c is not None:
                try:
                    confidences.append(float(c))
                except (ValueError, TypeError):
                    pass
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        # --- Dedup rate ---
        dedup_keys = [e.get("dedup_key", "") for e in linked if e.get("dedup_key")]
        if dedup_keys:
            dedup_dup_rate = 1.0 - (len(set(dedup_keys)) / len(dedup_keys))
        else:
            dedup_dup_rate = 0.0

        # --- Days since last event ---
        last_event_date: Optional[str] = None
        for e in sorted(linked, key=lambda x: x.get("window_date", "") or "", reverse=True):
            d = e.get("window_date") or e.get("event_date") or e.get("detected_at")
            if d:
                last_event_date = str(d)[:10]
                break

        if last_event_date:
            try:
                last_dt = datetime.strptime(last_event_date, "%Y-%m-%d").replace(
                    tzinfo=reference_time.tzinfo or timezone.utc,
                )
                days_since = (reference_time - last_dt).total_seconds() / 86400.0
            except Exception:
                days_since = None
        else:
            days_since = None

        # --- Scores (matching Cell 07 formula) ---
        volume_score = min(n_events / volume_cap, 1.0)

        # Confidence: auto-scale if > 1.5 (handles 0..100 range)
        if avg_confidence is not None:
            conf_scaled = avg_confidence / 100.0 if avg_confidence > 1.5 else avg_confidence
            confidence_score = max(0.0, min(1.0, conf_scaled))
        else:
            confidence_score = 0.0

        recency_score = math.exp(-(days_since or 999) / 7.0)

        signal_score = (
            SIGNAL_WEIGHTS["volume"] * volume_score
            + SIGNAL_WEIGHTS["action_needed"] * share_action_needed
            + SIGNAL_WEIGHTS["confidence"] * confidence_score
        )
        signal_score = max(0.0, min(1.0, signal_score))

        low_confidence_proxy = max(0.0, min(1.0, 1.0 - confidence_score))
        noise_score = (
            NOISE_WEIGHTS["dedup_dup_rate"] * dedup_dup_rate
            + NOISE_WEIGHTS["low_confidence"] * low_confidence_proxy
        )
        noise_score = max(0.0, min(1.0, noise_score))

        enriched.append({
            **t,
            "number_of_events": n_events,
            "linked_event_ids": linked_event_ids,
            "share_action_needed": round(share_action_needed, 4),
            "avg_confidence": round(avg_confidence, 4) if avg_confidence is not None else None,
            "dedup_dup_rate": round(dedup_dup_rate, 4),
            "last_event_date": last_event_date,
            "days_since_last_event": round(days_since, 2) if days_since is not None else None,
            "volume_score": round(volume_score, 4),
            "confidence_score": round(confidence_score, 4),
            "recency_score": round(recency_score, 4),
            "signal_score": round(signal_score, 4),
            "noise_score": round(noise_score, 4),
        })

    # Sort by signal_score descending
    enriched.sort(key=lambda x: -x["signal_score"])

    with_events = sum(1 for t in enriched if t["number_of_events"] > 0)
    logger.info(
        "Target metrics: %d targets, %d with events, %d without",
        len(enriched), with_events, len(enriched) - with_events,
    )
    return enriched
