#!/usr/bin/env python
# src/scripts/048_weekly_events_digest.py
"""Weekly events digest — fetch, normalise, LLM theme clustering, output.

Responsibilities:

1. **Fetch weekly events** from Notion Events DB.
2. **Normalise + filter** events.
3. **LLM theme clustering**: cluster events into 5-12 structural themes.
4. **Write themes to WEEKLY_THEMES_DB** (upsert, Week relation to digest row).
5. **Rank top signals** by confidence.
6. **Persist local artefacts** (events.json, themes.json, themes_summary.md,
   summary.md, run_metadata.json).
7. **Upsert WEEKLY_DIGESTS_DB** row for the week.

Usage::

    # Dry-run (default)
    python -m src.scripts.048_weekly_events_digest

    # Live run
    python -m src.scripts.048_weekly_events_digest --run

    # Live run with Notion writes
    python -m src.scripts.048_weekly_events_digest --run --write
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# ----------------------------------------------------------------
# sys.path bridge (INTERIM)
# ----------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    RunMetadata,
    WeekContext,
    get_db_id,
    get_optional_db_id,
    get_iso_week_context,
    get_output_dir,
    load_env,
    setup_logging,
)
from src.notion import (
    NotionClient,
    NotionDataSourceResolver,
    build_notion_client_from_env,
)
from src.notion.events_normalize import filter_events, normalize_events
from src.notion.events_schema import (
    DEFAULT_EXCLUDED_SOURCES,
    DEFAULT_EXCLUDED_STATUSES,
    DEFAULT_MIN_CONFIDENCE,
)
from src.notion.truncation import TruncationTracker
from src.notion.weekly_digests_repo import WeeklyDigestsRepo
from src.llm.openai_client import OpenAIClient, build_openai_client_from_env

logger = logging.getLogger("048_weekly_events_digest")

SCRIPT_NAME = "048_weekly_events_digest"
JST = ZoneInfo("Asia/Tokyo")


# ================================================================
# Core pipeline steps
# ================================================================

def resolve_events_data_source(client: NotionClient, db_id: str) -> str:
    """Resolve the data_source_id for the Events database."""
    resolver = NotionDataSourceResolver(client)
    resolved = resolver.resolve_once(name="EVENTS_DB", database_id=db_id)
    logger.info(
        "Resolved data_source_id=%s for EVENTS_DB (database_id=%s)",
        resolved.data_source_id,
        resolved.database_id,
    )
    return resolved.data_source_id


def fetch_weekly_events(
    client: NotionClient,
    data_source_id: str,
    *,
    date_from: str,
    date_to: str,
    date_property: str = "Date",
) -> List[dict]:
    """Fetch events within a date window from the Events data source."""
    filt = {
        "and": [
            {"property": date_property, "date": {"on_or_after": date_from}},
            {"property": date_property, "date": {"on_or_before": date_to}},
        ],
    }
    sorts = [{"property": date_property, "direction": "descending"}]

    logger.info(
        "Fetching events where '%s' between %s and %s …",
        date_property, date_from, date_to,
    )
    pages = client.query_data_source(
        data_source_id=data_source_id, filter=filt, sorts=sorts, fetch_all=True,
    )
    logger.info("Fetched %d events.", len(pages))
    return pages


# ================================================================
# Simple theme grouping (v1 heuristic fallback)
# ================================================================

def group_by_theme(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group events by ``event_type`` as a simple v1 theme proxy."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        theme = ev.get("event_type") or "Unknown"
        groups[theme].append(ev)
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


def rank_signals(events: List[Dict[str, Any]], *, top_n: int = 10) -> List[Dict[str, Any]]:
    """Return the top-N events ranked by confidence (descending)."""
    sorted_evs = sorted(events, key=lambda e: e.get("confidence", 0.0), reverse=True)
    signals = []
    for i, ev in enumerate(sorted_evs[:top_n], 1):
        signals.append({
            "rank": i,
            "title": ev.get("title", "?"),
            "confidence": ev.get("confidence", 0.0),
            "event_type": ev.get("event_type", "?"),
            "source": ev.get("source", "?"),
            "reason": f"Confidence {ev.get('confidence', 0.0):.2f}, type={ev.get('event_type', '?')}",
        })
    return signals


# ================================================================
# LLM theme clustering from events
# ================================================================

THEME_CLUSTERING_SYSTEM_PROMPT = """\
You are a research intelligence analyst specialising in startup ecosystems, \
venture capital dynamics, and entrepreneurship policy.

Given a set of weekly events, cluster them into 5-12 **structural themes**.

Focus on:
- Structural ecosystem patterns (not news-level grouping)
- VC / startup / institutional dynamics
- Policy direction shifts
- Innovation regime changes
- Capital flow patterns

For each theme, return:
- theme: short descriptive name (2-6 words)
- summary: 1-3 sentences explaining the structural pattern
- why_it_matters: 1 sentence on implications for the research programme
- event_ids: list of event page_ids that belong to this theme
- keywords: 3-8 relevant keywords

Return a JSON object:
{
  "themes": [
    {
      "theme": "...",
      "summary": "...",
      "why_it_matters": "...",
      "event_ids": ["id1", "id2", ...],
      "keywords": ["kw1", "kw2", ...]
    }
  ]
}
"""


def _slugify(name: str) -> str:
    """Slugify a theme name for use in upsert keys."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:80]


def build_event_themes(
    llm: OpenAIClient,
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Cluster events into structural themes via OpenAI.

    Returns a list of theme dicts with keys:
        name, summary, why_it_matters, key_event_page_ids, keywords
    """
    if not events:
        return []

    # Build payload — include page IDs so LLM can reference them
    events_payload = []
    for ev in events:
        events_payload.append({
            "event_id": ev.get("notion_page_id", ""),
            "title": ev.get("title", ""),
            "event_type": ev.get("event_type", ""),
            "source": ev.get("source", ""),
            "confidence": ev.get("confidence", 0.0),
            "summary": (ev.get("summary_text") or "")[:300],
        })

    user_prompt = (
        f"Cluster these {len(events_payload)} events into structural themes.\n\n"
        + json.dumps(events_payload, indent=2, ensure_ascii=False)
    )

    result = llm.call_json(system=THEME_CLUSTERING_SYSTEM_PROMPT, user=user_prompt)
    raw_themes = result.parsed.get("themes", [])

    # Normalise into the format expected by WeeklyThemesRepo
    themes: List[Dict[str, Any]] = []
    for idx, t in enumerate(raw_themes, start=1):
        theme_name = t.get("theme") or ""
        if not theme_name.strip():
            theme_name = f"(untitled-theme-{idx})"

        themes.append({
            "name": theme_name,
            "summary": t.get("summary", ""),
            "why_it_matters": t.get("why_it_matters", ""),
            "key_event_page_ids": t.get("event_ids", []),
            "keywords": t.get("keywords", []),
        })

    logger.info("Built %d theme clusters from events", len(themes))
    return themes


# ================================================================
# Output writers
# ================================================================

def write_events_json(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str) + "\n",
    )
    logger.info("Wrote %d events to %s", len(records), path)


def write_themes_json(themes: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(themes, indent=2, ensure_ascii=False, default=str) + "\n",
    )
    logger.info("Wrote %d themes to %s", len(themes), path)


def write_themes_summary_md(themes: List[dict], wk: WeekContext, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Event Themes — {wk.week_id}\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Total themes:** {len(themes)}\n\n")
        for i, t in enumerate(themes, 1):
            name = t.get("name", "(untitled)")
            summary = t.get("summary", "")
            why = t.get("why_it_matters", "")
            kw = ", ".join(t.get("keywords", [])[:8])
            n_events = len(t.get("key_event_page_ids", []))
            f.write(f"## {i}. {name}\n\n")
            if summary:
                f.write(f"{summary}\n\n")
            if why:
                f.write(f"**Why it matters:** {why}\n\n")
            f.write(f"- Events: {n_events}\n")
            if kw:
                f.write(f"- Keywords: {kw}\n")
            f.write("\n")
        f.write(f"---\n\n*Generated by {SCRIPT_NAME}*\n")
    logger.info("Wrote themes summary to %s", path)


def write_summary_md(
    events: List[Dict[str, Any]],
    themes: Dict[str, List[Dict[str, Any]]],
    signals: List[Dict[str, Any]],
    wk: WeekContext,
    path: Path,
) -> None:
    """Write a Markdown summary grouped by theme + top signals."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Weekly Events Digest — {wk.week_id}\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- **Events (after filtering):** {len(events)}\n")
        f.write(f"- **Date window:** {wk.start_date} to {wk.end_date}\n")
        f.write(f"- **Themes:** {len(themes)}\n\n")

        # ---- Top signals ----
        if signals:
            f.write("## Top Signals\n\n")
            f.write("| # | Event | Confidence | Type | Source |\n")
            f.write("|---|-------|------------|------|--------|\n")
            for s in signals:
                title_short = s["title"][:60] + ("…" if len(s["title"]) > 60 else "")
                f.write(f"| {s['rank']} | {title_short} | {s['confidence']:.2f} | {s['event_type']} | {s['source']} |\n")
            f.write("\n")

        # ---- Themes ----
        if themes:
            f.write("## Events by Theme\n\n")
            for theme_name, theme_events in themes.items():
                f.write(f"### {theme_name} ({len(theme_events)} events)\n\n")
                for ev in theme_events[:10]:
                    title = ev.get("title", "(untitled)")
                    source = ev.get("source", "")
                    conf = ev.get("confidence", 0.0)
                    summary = (ev.get("summary_text") or "")[:200]

                    f.write(f"- **{title}**")
                    if source:
                        f.write(f" ({source})")
                    f.write(f" — confidence {conf:.2f}\n")
                    if summary:
                        f.write(f"  {summary}\n")
                f.write("\n")

        f.write("---\n\n")
        f.write(f"*Generated by {SCRIPT_NAME}*\n")

    logger.info("Wrote summary to %s", path)


# ================================================================
# CLI entry-point
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Fetch weekly events from Notion, cluster themes via LLM, write digest.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", default=False,
                       help="Execute live Notion fetch (default is dry-run).")
    mode.add_argument("--dry-run", action="store_true", default=True,
                       help="Print plan without API calls (default).")
    p.add_argument("--days", type=int, default=7,
                    help="Lookback window in days (default: 7).")
    p.add_argument("--iso-week", action="store_true", default=False,
                    help="Use ISO week boundaries (Mon–Sun JST).")
    p.add_argument("--date-property", default="Date",
                    help="Notion property for date filtering (default: 'Date').")
    p.add_argument("--output-base", default="outputs",
                    help="Base output directory (default: outputs/).")
    p.add_argument("--write", action="store_true", default=False,
                    help="Persist to Notion (default: off).")
    p.add_argument("-v", "--verbose", action="store_true",
                    help="Enable DEBUG logging.")
    return p


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    is_live = args.run
    write_enabled = args.write

    # ---- Setup ----
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    load_env()

    db_id = get_db_id("NOTION_EVENTS_DB_ID")

    # ---- Compute date window ----
    if args.iso_week:
        wk = get_iso_week_context(tz=JST)
        date_from_str = wk.start_date
        date_to_str = wk.end_date
    else:
        wk = get_iso_week_context(tz=JST)
        now_jst = wk.now_utc
        date_from_jst = now_jst - timedelta(days=args.days)
        date_from_str = date_from_jst.strftime("%Y-%m-%d")
        date_to_str = now_jst.strftime("%Y-%m-%d")

    date_from_iso = wk.date_from_iso if args.iso_week else date_from_jst.isoformat(timespec="seconds")
    date_to_iso = wk.date_to_iso if args.iso_week else now_jst.isoformat(timespec="seconds")

    out_dir = get_output_dir(SCRIPT_NAME, wk.week_id, base=args.output_base, create=is_live)

    logger.info("=== %s ===", SCRIPT_NAME)
    logger.info("Week: %s  |  Mode: %s", wk.week_id,
                "ISO-week" if args.iso_week else f"rolling {args.days} days")
    logger.info("Date window (JST): %s → %s", date_from_iso, date_to_iso)
    logger.info("Output: %s", out_dir)
    logger.info("Mode: %s  |  Write: %s", "LIVE" if is_live else "DRY-RUN",
                "ON" if write_enabled else "OFF")

    result: Dict[str, Any] = {
        "ok": False, "week_id": wk.week_id, "output_dir": str(out_dir),
        "summary": {}, "errors": [],
        "themes": [], "theme_page_ids": [],
    }

    if not is_live:
        logger.info("[DRY-RUN] Would fetch events from NOTION_EVENTS_DB_ID=%s", db_id)
        logger.info("[DRY-RUN] Would cluster themes via OpenAI")
        logger.info("[DRY-RUN] Pass --run to execute.")
        result["ok"] = True
        return result

    # ---- Live pipeline ----
    import uuid as _uuid
    now_jst_dt = datetime.now(JST)
    run_id = str(_uuid.uuid4())[:8]

    client = build_notion_client_from_env()
    llm = build_openai_client_from_env(
        cache_dir=Path(args.output_base) / "weekly" / wk.week_id / ".cache" / "llm",
    )
    ds_id = resolve_events_data_source(client, db_id)

    raw_pages = fetch_weekly_events(
        client, ds_id,
        date_from=date_from_str, date_to=date_to_str,
        date_property=args.date_property,
    )

    # Normalize + filter
    records = normalize_events(raw_pages)
    clean = filter_events(records)

    # Group by event_type (v1 heuristic, used for summary.md)
    heuristic_themes = group_by_theme(clean)
    signals = rank_signals(clean, top_n=10)

    # ---- LLM theme clustering ----
    logger.info("Clustering themes from %d events via OpenAI …", len(clean))
    themes = build_event_themes(llm, clean)
    result["themes"] = themes

    # ---- Write local artefacts ----
    write_events_json(clean, out_dir / "events.json")
    write_themes_json(themes, out_dir / "themes.json")
    write_themes_summary_md(themes, wk, out_dir / "themes_summary.md")
    write_summary_md(clean, heuristic_themes, signals, wk, out_dir / "summary.md")

    # ---- Notion writeback ----
    notion_digest_count = 0
    notion_theme_count = 0
    theme_page_ids: List[str] = []
    digest_page_id: Optional[str] = None
    trunc_tracker = TruncationTracker()

    if write_enabled:
        # 1) Ensure WEEKLY_DIGESTS_DB row
        try:
            digests_db_id = get_db_id("NOTION_WEEKLY_DIGESTS_DB_ID")
            digests_resolver = NotionDataSourceResolver(client)
            digests_resolved = digests_resolver.resolve_once(
                name="WEEKLY_DIGESTS_DB", database_id=digests_db_id,
            )
            repo = WeeklyDigestsRepo(
                client=client,
                database_id=digests_resolved.database_id,
                data_source_id=digests_resolved.data_source_id,
            )
            repo.validate_schema()

            key, props = repo.build_digest_properties(
                week_id=wk.week_id,
                week_start=wk.start_date,
                week_end=wk.end_date,
                run_id=run_id,
                now_jst=now_jst_dt,
                events_count=len(clean),
                themes_count=len(themes),
                signals_count=len(signals),
                tracker=trunc_tracker,
            )
            page = repo.upsert_row(key=key, properties=props)
            digest_page_id = page.get("id")
            notion_digest_count = 1
            logger.info(
                "Notion: digest page upserted for %s (page_id=%s)",
                wk.week_id, digest_page_id,
            )
        except Exception as e:
            logger.warning("Failed to upsert digest row: %s", e)
            result["errors"].append(f"Digest upsert: {e}")

        # 2) Write themes to WEEKLY_THEMES_DB
        themes_db_id = get_optional_db_id("NOTION_WEEKLY_THEMES_DB_ID")
        if themes_db_id and themes:
            try:
                from src.notion.weekly_themes_repo import WeeklyThemesRepo

                themes_resolver = NotionDataSourceResolver(client)
                themes_resolved = themes_resolver.resolve_once(
                    name="WEEKLY_THEMES_DB", database_id=themes_db_id,
                )
                themes_repo = WeeklyThemesRepo(
                    client=client,
                    database_id=themes_resolved.database_id,
                    data_source_id=themes_resolved.data_source_id,
                )
                themes_repo.ensure_schema()

                for t in themes:
                    key, props = themes_repo.build_theme_properties(
                        theme=t,
                        week_id=wk.week_id,
                        digest_page_id=digest_page_id,
                        tracker=trunc_tracker,
                    )
                    page = themes_repo.upsert_row(key=key, properties=props)
                    theme_page_ids.append(page.get("id", ""))
                    notion_theme_count += 1

                logger.info(
                    "Notion: %d themes upserted to WEEKLY_THEMES_DB", notion_theme_count,
                )
            except Exception as e:
                logger.warning("Failed to write themes: %s", e)
                result["errors"].append(f"Themes write: {e}")

    result["theme_page_ids"] = theme_page_ids

    # ---- Run metadata ----
    type_counts: Dict[str, int] = defaultdict(int)
    for ev in clean:
        type_counts[ev.get("event_type", "UNKNOWN")] += 1

    extra_meta: Dict[str, Any] = {
        "date_property": args.date_property,
        "data_source_id": ds_id,
        "timezone": "Asia/Tokyo",
        "mode": "iso-week" if args.iso_week else f"rolling-{args.days}d",
        "write_enabled": write_enabled,
        "themes_written": notion_theme_count,
        "theme_page_ids": theme_page_ids,
        "digest_page_id": digest_page_id,
    }
    if write_enabled:
        extra_meta["notion_digest_upserted"] = notion_digest_count
    if trunc_tracker.had_truncations:
        extra_meta["truncated_fields"] = trunc_tracker.report()

    meta = RunMetadata.build(
        notebook=SCRIPT_NAME,
        week_id=wk.week_id,
        date_from=date_from_iso,
        date_to=date_to_iso,
        counts={
            "events_fetched": len(raw_pages),
            "events_normalised": len(records),
            "events_filtered": len(clean),
            "themes_clustered": len(themes),
            "top_signals": len(signals),
            "by_type": dict(type_counts),
        },
        extra=extra_meta,
    )
    meta.save(out_dir / "run_metadata.json")

    result["ok"] = True
    result["summary"] = {
        "events_filtered": len(clean),
        "themes_clustered": len(themes),
        "top_signals": len(signals),
        "themes_written": notion_theme_count,
        "digest_page_id": digest_page_id,
    }
    logger.info(
        "=== Done: %d events, %d themes → %s ===",
        len(clean), len(themes), out_dir,
    )
    logger.info(llm.usage_summary())
    return result


if __name__ == "__main__":
    r = main()
    raise SystemExit(0 if r.get("ok") else 1)
