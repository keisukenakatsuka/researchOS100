#!/usr/bin/env python
# src/scripts/047_weekly_papers_review.py
"""Standalone weekly papers review — fetch, summarise, output.

Minimal CLI equivalent of the 040_weekly_papers_review notebook,
designed for automation (cron, CI).  Reuses shared modules under
``src/`` and produces JSON + Markdown outputs.

Usage::

    # Dry-run (default) — prints plan, no API calls
    python src/scripts/047_weekly_papers_review.py

    # Live run — fetches from Notion and writes outputs
    python src/scripts/047_weekly_papers_review.py --run

    # Custom lookback window
    python src/scripts/047_weekly_papers_review.py --run --days 14

Outputs (under outputs/weekly/<week_id>/047_weekly_papers_review/)::

    papers.json           — flat list of paper records
    summary.md            — human-readable Markdown report
    run_metadata.json     — git SHA, week_id, counts (reproducibility)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

# ----------------------------------------------------------------
# sys.path bridge (INTERIM — same pattern as notebooks)
# Ensures `from src.…` works regardless of cwd.
# TODO: replace with `pip install -e .` or stable PYTHONPATH.
# ----------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # src/scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent              # project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    RunMetadata,
    WeekContext,
    get_db_id,
    get_output_dir,
    get_week_context,
    load_env,
    setup_logging,
)
from src.notion import (
    NotionClient,
    NotionDataSourceResolver,
    build_notion_client_from_env,
)
from src.notion.papers_schema import (
    PAPERS_REQUIRED_PROPERTIES,
    PAPERS_SCORING_PROPERTIES,
    get_papers_schema,
)
from src.notion.properties import extract_property_value, page_to_record

logger = logging.getLogger("047_weekly_papers_review")

SCRIPT_NAME = "047_weekly_papers_review"


# ================================================================
# Core pipeline steps
# ================================================================

def resolve_papers_data_source(
    client: NotionClient,
    db_id: str,
) -> str:
    """Resolve the data_source_id for the Papers database.

    Uses :class:`NotionDataSourceResolver` (typed, cached) instead of
    the legacy ``client.request()`` bridge.
    """
    resolver = NotionDataSourceResolver(client)
    resolved = resolver.resolve_once(name="PAPERS_DB", database_id=db_id)
    logger.info(
        "Resolved data_source_id=%s for PAPERS_DB (database_id=%s)",
        resolved.data_source_id,
        resolved.database_id,
    )
    return resolved.data_source_id


def fetch_recent_papers(
    client: NotionClient,
    data_source_id: str,
    *,
    since: datetime,
    date_property: str = "Ingested At",
) -> List[dict]:
    """Fetch papers ingested on or after *since* from Notion.

    Uses the typed ``client.query_data_source()`` with auto-pagination
    (no raw REST calls).
    """
    filt = {
        "property": date_property,
        "date": {"on_or_after": since.isoformat()},
    }
    sorts = [{"property": date_property, "direction": "descending"}]

    logger.info(
        "Fetching papers where '%s' >= %s …",
        date_property,
        since.strftime("%Y-%m-%d"),
    )
    pages = client.query_data_source(
        data_source_id=data_source_id,
        filter=filt,
        sorts=sorts,
        fetch_all=True,
    )
    logger.info("Fetched %d papers.", len(pages))
    return pages


def normalize_papers(
    pages: List[dict],
    property_names: List[str],
) -> List[Dict[str, Any]]:
    """Convert raw Notion pages to flat dicts via :func:`page_to_record`."""
    records = [page_to_record(p, property_names) for p in pages]
    logger.info("Normalised %d paper records.", len(records))
    return records


# ================================================================
# Output writers
# ================================================================

def write_papers_json(records: List[dict], path: Path) -> None:
    """Write the paper records list to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str) + "\n",
    )
    logger.info("Wrote %d papers to %s", len(records), path)


def write_summary_md(
    records: List[dict],
    wk: WeekContext,
    path: Path,
) -> None:
    """Write a minimal Markdown summary report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Tally sources
    sources: Dict[str, int] = {}
    for r in records:
        s = r.get("Source") or "unknown"
        sources[s] = sources.get(s, 0) + 1

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Weekly Papers Fetch — {wk.week_id}\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- **Papers fetched:** {len(records)}\n")
        f.write(f"- **Date range:** {wk.start_date} to {wk.end_date}\n\n")

        if sources:
            f.write("## Sources\n\n")
            f.write("| Source | Count |\n|--------|-------|\n")
            for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
                f.write(f"| {src} | {cnt} |\n")
            f.write("\n")

        f.write("## Papers\n\n")
        for i, r in enumerate(records, 1):
            name = r.get("Name", "(untitled)")
            authors = r.get("Authors & Year", "")
            source = r.get("Source", "")
            pdf = r.get("PDF Link", "")

            f.write(f"### {i}. {name}\n\n")
            if authors:
                f.write(f"- **Authors:** {authors}\n")
            if source:
                f.write(f"- **Source:** {source}\n")
            if pdf:
                f.write(f"- **PDF:** [{pdf}]({pdf})\n")

            # Show first non-empty text field as abstract
            for field in ("Core Idea", "Findings", "Notes"):
                text = r.get(field)
                if text:
                    preview = text[:300] + ("…" if len(text) > 300 else "")
                    f.write(f"- **{field}:** {preview}\n")
                    break
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
        description="Fetch recent papers from Notion and write JSON + Markdown.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Execute live Notion fetch (default is dry-run).",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print plan without API calls (default).",
    )
    p.add_argument(
        "--days",
        type=int,
        default=7,
        help="Lookback window in days (default: 7).",
    )
    p.add_argument(
        "--date-property",
        default="Ingested At",
        help="Notion property for date filtering (default: 'Ingested At').",
    )
    p.add_argument(
        "--output-base",
        default="outputs",
        help="Base output directory (default: outputs/).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    is_live = args.run

    # ---- Setup ----
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    load_env()

    wk = get_week_context()
    db_id = get_db_id("NOTION_LIT_DB_ID")

    # Compute the query window explicitly in UTC.
    # "since" = now_utc minus N days; "until" = now_utc.
    # Both carry +00:00 so there is no JST/UTC ambiguity.
    date_to_utc: datetime = wk.now_utc                             # already tz-aware (UTC)
    date_from_utc: datetime = date_to_utc - timedelta(days=args.days)

    date_from_iso: str = date_from_utc.isoformat(timespec="seconds")  # e.g. 2026-02-08T23:35:52+00:00
    date_to_iso: str = date_to_utc.isoformat(timespec="seconds")

    out_dir = get_output_dir(
        SCRIPT_NAME,
        wk.week_id,
        base=args.output_base,
        create=is_live,
    )

    logger.info("=== %s ===", SCRIPT_NAME)
    logger.info("Week: %s  |  Lookback: %d days", wk.week_id, args.days)
    logger.info("Date window (UTC): %s → %s", date_from_iso, date_to_iso)
    logger.info("Output: %s", out_dir)
    logger.info("Mode: %s", "LIVE" if is_live else "DRY-RUN")

    if not is_live:
        logger.info("[DRY-RUN] Would fetch papers from NOTION_LIT_DB_ID=%s", db_id)
        logger.info("[DRY-RUN] Filter: '%s' >= %s", args.date_property, date_from_iso)
        logger.info("[DRY-RUN] Outputs would be written to %s/", out_dir)
        logger.info("[DRY-RUN] Pass --run to execute.")
        return 0

    # ---- Live pipeline ----
    client = build_notion_client_from_env()
    ds_id = resolve_papers_data_source(client, db_id)

    raw_pages = fetch_recent_papers(
        client,
        ds_id,
        since=date_from_utc,
        date_property=args.date_property,
    )

    property_names = list(PAPERS_REQUIRED_PROPERTIES.keys())
    records = normalize_papers(raw_pages, property_names)

    # ---- Write outputs ----
    write_papers_json(records, out_dir / "papers.json")
    write_summary_md(records, wk, out_dir / "summary.md")

    # ---- Run metadata ----
    meta = RunMetadata.build(
        notebook=SCRIPT_NAME,
        week_id=wk.week_id,
        date_from=date_from_iso,
        date_to=date_to_iso,
        counts={
            "papers_fetched": len(raw_pages),
            "papers_normalised": len(records),
            "lookback_days": args.days,
        },
        extra={
            "date_property": args.date_property,
            "data_source_id": ds_id,
        },
    )
    meta.save(out_dir / "run_metadata.json")

    logger.info("=== Done: %d papers → %s ===", len(records), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# test-marker
