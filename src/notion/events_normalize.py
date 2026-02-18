# src/notion/events_normalize.py
"""Normalize, deduplicate, and filter Notion Event pages.

Extracted from notebook 041_weekly_events_digest Cells 05–06.
Pure functions — no API calls, no side effects.

Usage::

    from src.notion.events_normalize import normalize_event, filter_events

    records = [normalize_event(page) for page in raw_pages]
    clean = filter_events(records)
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from src.notion.events_schema import (
    DEFAULT_EXCLUDED_SOURCES,
    DEFAULT_EXCLUDED_STATUSES,
    DEFAULT_MIN_CONFIDENCE,
)
from src.notion.properties import extract_property_value

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _to_date_str(raw: Any) -> str:
    """Normalise a date value to ``YYYY-MM-DD`` or ``""``."""
    s = str(raw or "").strip()
    if not s:
        return ""
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else s


def _extract_keywords(text: str, *, max_k: int = 12) -> List[str]:
    """Simple stopword-filtered keyword extraction (fallback)."""
    _STOP = frozenset({
        "this", "that", "with", "from", "into", "over", "will", "have",
        "has", "been", "were", "their", "about", "after", "before",
        "also", "than", "then", "them", "they", "what", "when", "where",
        "which", "while", "said", "says",
    })
    tokens = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    seen: List[str] = []
    for t in tokens:
        if t in _STOP or t in seen:
            continue
        seen.append(t)
        if len(seen) >= max_k:
            break
    return seen


def _relation_ids(page: Dict[str, Any], prop_name: str) -> List[str]:
    """Extract relation IDs from a Notion page property."""
    props = page.get("properties", {})
    prop = props.get(prop_name, {})
    arr = prop.get("relation", []) or []
    return [r.get("id", "") for r in arr if isinstance(r, dict) and r.get("id")]


# ----------------------------------------------------------------
# Normalize a single event page
# ----------------------------------------------------------------

def normalize_event(page: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw Notion Event page into a flat record dict.

    Returns ``None`` if the page lacks both ``Date`` and ``Detected At``
    (required for weekly slicing).

    Uses :func:`src.notion.properties.extract_property_value` for
    standard types and adds events-specific derived fields.
    """
    page_id = page.get("id", "") or ""
    ev = extract_property_value

    title = ev(page, "Name") or "Untitled Event"
    event_date = _to_date_str(ev(page, "Date"))
    detected_at = _to_date_str(ev(page, "Detected At"))
    ingested_at = _to_date_str(ev(page, "Ingested At"))

    # Weekly slicing key: prefer Detected At, fallback to Date
    window_date = detected_at or event_date
    if not window_date:
        return None

    source_url = ev(page, "Source URL") or ""
    summary_text = ev(page, "Summary") or ""
    dedup_key = ev(page, "Dedup Key") or ""

    if not dedup_key:
        h_in = f"{title}|{window_date}|{source_url}".encode("utf-8")
        dedup_key = hashlib.sha256(h_in).hexdigest()[:16]

    keywords = _extract_keywords(f"{title} {summary_text}")

    confidence_raw = ev(page, "Confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else 0.0
    except (ValueError, TypeError):
        confidence = 0.0

    return {
        # --- compatibility keys ---
        "page_id": page_id,
        "notion_page_id": page_id,
        # --- core fields ---
        "title": title,
        "window_date": window_date,
        "detected_at": detected_at,
        "event_date": event_date,
        "target_ids": _relation_ids(page, "Target"),
        "event_type": ev(page, "Event Type") or "Unknown",
        "source": ev(page, "Source") or "Unknown",
        "source_url": source_url,
        "summary_text": summary_text,
        "confidence": confidence,
        "dedup_key": dedup_key,
        "status": ev(page, "Status") or "",
        # --- operational ---
        "run_id": ev(page, "Run ID") or "",
        "ingested_at": ingested_at,
        "action_needed": bool(ev(page, "Action Needed")),
        "related_paper_ids": _relation_ids(page, "Related Papers"),
        # --- derived ---
        "keywords": keywords,
    }


def normalize_events(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of raw Notion Event pages.

    Pages missing a usable date are silently skipped (logged as warning).
    """
    records: List[Dict[str, Any]] = []
    skipped = 0
    for page in pages:
        rec = normalize_event(page)
        if rec is None:
            skipped += 1
            logger.warning("Skipping page %s: missing both Date and Detected At",
                           page.get("id", "?"))
            continue
        records.append(rec)
    logger.info("Normalised %d events (skipped %d)", len(records), skipped)
    return records


# ----------------------------------------------------------------
# Dedup + noise filter
# ----------------------------------------------------------------

def filter_events(
    events: List[Dict[str, Any]],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    excluded_statuses: Set[str] | frozenset = DEFAULT_EXCLUDED_STATUSES,
    excluded_sources: Set[str] | frozenset = DEFAULT_EXCLUDED_SOURCES,
) -> List[Dict[str, Any]]:
    """Deduplicate and noise-filter normalised event records.

    Removal reasons are logged at INFO level.

    Parameters
    ----------
    min_confidence:
        Events with ``confidence < min_confidence`` are dropped.
    excluded_statuses:
        Case-insensitive status values to exclude.
    excluded_sources:
        Case-insensitive source values to exclude.
    """
    excluded_statuses_lower = {s.lower() for s in excluded_statuses}
    excluded_sources_lower = {s.lower() for s in excluded_sources}

    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    rm_dup = rm_conf = rm_status = rm_source = 0

    for ev in events:
        dk = ev.get("dedup_key", "")
        if dk in seen:
            rm_dup += 1
            continue

        if ev.get("confidence", 0.0) < min_confidence:
            rm_conf += 1
            continue

        status_l = (ev.get("status") or "").lower()
        if status_l and status_l in excluded_statuses_lower:
            rm_status += 1
            continue

        source_l = (ev.get("source") or "").lower()
        if source_l and source_l in excluded_sources_lower:
            rm_source += 1
            continue

        seen.add(dk)
        out.append(ev)

    total_removed = len(events) - len(out)
    logger.info(
        "Filtered events: %d → %d (removed %d: dup=%d, low_conf=%d, status=%d, source=%d)",
        len(events), len(out), total_removed,
        rm_dup, rm_conf, rm_status, rm_source,
    )

    if out:
        # Log distribution summary
        type_counts: Dict[str, int] = defaultdict(int)
        for e in out:
            type_counts[e.get("event_type", "UNKNOWN")] += 1
        logger.info("Event type distribution: %s", dict(type_counts))

    return out
