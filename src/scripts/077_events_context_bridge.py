#!/usr/bin/env python
# src/scripts/077_events_context_bridge.py
"""077 Events Context Bridge — make EVENTS_DB available to 073 recall.

Pipeline:
1. Fetch recent events from EVENTS_DB (last 30 days, confidence >= 0.5)
2. Build context structure with keywords for recall matching
3. Save to data/cache/events_context/{date}.json
4. 073's planner reads this cache via recall_events_context()

Usage::

    # Dry-run: fetch and show stats without saving
    python -m src.scripts.077_events_context_bridge --dry-run

    # Custom window
    python -m src.scripts.077_events_context_bridge --days 14

    # Full run
    python -m src.scripts.077_events_context_bridge

    # Verbose logging
    python -m src.scripts.077_events_context_bridge -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env

logger = logging.getLogger("077_events_context_bridge")

SCRIPT_NAME = "077_events_context_bridge"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=f"python -m src.scripts.{SCRIPT_NAME}",
        description="Build events context cache for 073 recall.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch events and show stats without saving cache.",
    )
    p.add_argument(
        "--days",
        type=int,
        default=30,
        help="Look-back window in days (default: 30).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Environment
    load_env()

    logger.info("=== 077 Events Context Bridge ===")
    if args.dry_run:
        logger.info("Mode: DRY-RUN")
    logger.info("Window: %d days", args.days)

    # Build Notion client
    from src.config import get_db_id
    from src.notion import build_notion_client_from_env, NotionDataSourceResolver

    notion_client = build_notion_client_from_env()
    events_db_id = get_db_id("NOTION_EVENTS_DB_ID")

    resolver = NotionDataSourceResolver(notion_client)
    events_resolved = resolver.resolve_once(name="EVENTS_DB", database_id=events_db_id)
    logger.info(
        "Events DB: database_id=%s data_source_id=%s",
        events_db_id[:12], events_resolved.data_source_id[:12],
    )

    # Run pipeline
    from src.daily.events_context import run_events_context_bridge

    summary = run_events_context_bridge(
        notion_client=notion_client,
        events_data_source_id=events_resolved.data_source_id,
        days=args.days,
        dry_run=args.dry_run,
    )

    # Print summary
    logger.info("=== Summary ===")
    logger.info("  Events fetched : %d", summary["events_fetched"])
    for t, count in sorted(summary["by_type"].items()):
        logger.info("    %s: %d", t, count)
    if summary.get("cache_path"):
        logger.info("  Cache saved    : %s", summary["cache_path"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
