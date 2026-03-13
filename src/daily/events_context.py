# src/daily/events_context.py
"""077 Events Context Bridge — service logic.

Fetches recent events from EVENTS_DB, builds a context cache file
(data/cache/events_context/{date}.json) that recall.py can load
to provide event context to 073's planner.

Scope: EVENTS_DB only. WEEKLY_THEMES_DB deferred to Phase 2.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("077_events_context")

# -- constants ---------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONTEXT_DIR = _PROJECT_ROOT / "data" / "cache" / "events_context"
_MIN_CONFIDENCE = 0.5
_DEFAULT_WINDOW_DAYS = 30

# Common stop words for keyword extraction
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "of", "in", "to", "for", "on", "with", "at", "by", "from",
    "and", "or", "not", "no", "it", "its", "this", "that",
    "の", "は", "が", "を", "に", "で", "と", "も", "か",
    "http", "https", "www", "com",
})


# -- Keyword extraction from event text -------------------------------------


def extract_event_keywords(text: str, max_keywords: int = 8) -> List[str]:
    """Extract keywords from event text for recall matching."""
    tokens = re.split(r'[\s、。,.\-/;:?!！？「」（）()\[\]]+', text)
    seen: set[str] = set()
    keywords: List[str] = []

    for token in tokens:
        token = token.strip().lower()
        if not token or len(token) < 2:
            continue
        if token in _STOP_WORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break

    return keywords


# -- Event fetching ----------------------------------------------------------


def fetch_recent_events(
    notion_client: Any,
    events_data_source_id: str,
    *,
    days: int = _DEFAULT_WINDOW_DAYS,
    min_confidence: float = _MIN_CONFIDENCE,
) -> List[Dict[str, Any]]:
    """Fetch events from EVENTS_DB within the time window.

    Filters:
    - Detected At >= (today - days)
    - Confidence >= min_confidence (applied post-fetch)
    """
    since = (date.today() - timedelta(days=days)).isoformat()
    filt = {
        "property": "Detected At",
        "date": {"on_or_after": since},
    }

    try:
        pages = notion_client.query_data_source(
            data_source_id=events_data_source_id,
            filter=filt,
            fetch_all=True,
        )
    except Exception as e:
        logger.error("Failed to fetch events: %s", e)
        return []

    from src.notion.properties import extract_property_value as ev

    events: List[Dict[str, Any]] = []
    for page in pages:
        # Parse confidence
        conf_raw = ev(page, "Confidence")
        try:
            confidence = float(conf_raw) if conf_raw is not None else 0.0
        except (ValueError, TypeError):
            confidence = 0.0

        if confidence < min_confidence:
            continue

        name = ev(page, "Name") or ""
        summary = ev(page, "Summary") or ""
        event_date = ev(page, "Date") or ""
        if isinstance(event_date, str):
            event_date = event_date[:10]

        # Extract target info from relation (just page IDs)
        target_ids = []
        props = page.get("properties", {})
        target_prop = props.get("Target", {})
        if target_prop.get("type") == "relation":
            for rel in target_prop.get("relation", []):
                target_ids.append(rel.get("id", ""))

        events.append({
            "event_id": page.get("id", ""),
            "name": name,
            "date": event_date,
            "event_type": ev(page, "Event Type") or "Unknown",
            "source": ev(page, "Source") or "Unknown",
            "summary": summary[:500],
            "confidence": confidence,
            "target_ids": target_ids,
            "keywords": extract_event_keywords(f"{name} {summary}"),
        })

    logger.info(
        "Fetched %d events (confidence >= %.1f, last %d days)",
        len(events), min_confidence, days,
    )
    return events


# -- Context building --------------------------------------------------------


def build_events_context(
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the events context structure from raw events."""
    by_type: Dict[str, int] = {}
    for e in events:
        t = e.get("event_type", "Unknown")
        by_type[t] = by_type.get(t, 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": _DEFAULT_WINDOW_DAYS,
        "events": events,
        "event_count": len(events),
        "by_type": by_type,
    }


# -- Cache I/O --------------------------------------------------------------


def save_context_cache(
    context: Dict[str, Any],
    output_dir: Path = _CONTEXT_DIR,
) -> Path:
    """Save context to data/cache/events_context/{date}.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = output_dir / f"{today}.json"
    path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved events context cache: %s (%d events)", path.name, context["event_count"])
    return path


def load_latest_context(
    context_dir: Path = _CONTEXT_DIR,
) -> Optional[Dict[str, Any]]:
    """Load the latest events context cache file.

    Returns None if no cache file exists.
    """
    if not context_dir.exists():
        return None

    files = sorted(context_dir.glob("*.json"), reverse=True)
    if not files:
        return None

    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load context cache %s: %s", files[0], e)
        return None


# -- Batch processing --------------------------------------------------------


def run_events_context_bridge(
    *,
    notion_client: Any,
    events_data_source_id: str,
    days: int = _DEFAULT_WINDOW_DAYS,
    dry_run: bool = False,
    output_dir: Path = _CONTEXT_DIR,
) -> Dict[str, Any]:
    """Run the events context bridge pipeline.

    Returns a summary dict.
    """
    summary: Dict[str, Any] = {
        "events_fetched": 0,
        "by_type": {},
        "cache_path": None,
        "dry_run": dry_run,
    }

    events = fetch_recent_events(
        notion_client,
        events_data_source_id,
        days=days,
    )
    summary["events_fetched"] = len(events)

    context = build_events_context(events)
    summary["by_type"] = context["by_type"]

    if dry_run:
        logger.info("=== DRY-RUN: %d events would be cached ===", len(events))
        for t, count in sorted(context["by_type"].items()):
            logger.info("  %s: %d", t, count)
        return summary

    path = save_context_cache(context, output_dir=output_dir)
    summary["cache_path"] = str(path)

    return summary
