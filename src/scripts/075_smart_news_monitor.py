#!/usr/bin/env python
# src/scripts/075_smart_news_monitor.py
"""075 Smart News Monitor — frequency-optimized news monitoring.

Pipeline:
1. Query MONITORING_TARGETS_DB for targets due today
2. For each target:
   a. Run Google CSE + NewsAPI searches
   b. Dedup results against recent EVENTS_DB entries
   c. Write new events to EVENTS_DB
   d. Apply cadence state machine (Weekly ↔ Monthly → Paused)
   e. Update target operational fields
3. Print summary

Usage::

    # Dry-run: show due targets and simulated transitions
    python -m src.scripts.075_smart_news_monitor --dry-run

    # Process up to 5 targets
    python -m src.scripts.075_smart_news_monitor --limit 5

    # Filter by type
    python -m src.scripts.075_smart_news_monitor --type VC

    # Full run
    python -m src.scripts.075_smart_news_monitor

    # Verbose logging
    python -m src.scripts.075_smart_news_monitor -v
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

logger = logging.getLogger("075_smart_news_monitor")

SCRIPT_NAME = "075_smart_news_monitor"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=f"python -m src.scripts.{SCRIPT_NAME}",
        description="Smart news monitoring with cadence optimization.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show due targets and simulated transitions without searching or writing.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of targets to process (0 = unlimited).",
    )
    p.add_argument(
        "--type",
        type=str,
        default="",
        dest="target_type",
        help="Filter targets by type (VC, Startup, Policy, People).",
    )
    p.add_argument(
        "--migrate",
        action="store_true",
        help="One-shot: normalize cadence values and set Next Check for all Active targets.",
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

    logger.info("=== 075 Smart News Monitor ===")
    if args.dry_run:
        logger.info("Mode: DRY-RUN")

    # Build clients
    from src.config import get_db_id
    from src.notion import build_notion_client_from_env, NotionDataSourceResolver

    notion_client = build_notion_client_from_env()

    # Resolve data source IDs
    targets_db_id = get_db_id("NOTION_MONITORING_TARGETS_DB_ID")
    events_db_id = get_db_id("NOTION_EVENTS_DB_ID")

    resolver = NotionDataSourceResolver(notion_client)
    targets_resolved = resolver.resolve_once(name="MONITORING_TARGETS_DB", database_id=targets_db_id)
    events_resolved = resolver.resolve_once(name="EVENTS_DB", database_id=events_db_id)

    logger.info(
        "Targets DB: database_id=%s data_source_id=%s",
        targets_db_id[:12], targets_resolved.data_source_id[:12],
    )
    logger.info(
        "Events DB: database_id=%s data_source_id=%s",
        events_db_id[:12], events_resolved.data_source_id[:12],
    )

    # Migration mode
    if args.migrate:
        from src.daily.news_monitor import migrate_targets
        logger.info("=== Migration Mode ===")
        mig_summary = migrate_targets(
            notion_client=notion_client,
            targets_data_source_id=targets_resolved.data_source_id,
            dry_run=args.dry_run,
        )
        logger.info("=== Migration Summary ===")
        logger.info("  Total    : %d", mig_summary["total"])
        logger.info("  Migrated : %d", mig_summary["migrated"])
        logger.info("  Skipped  : %d", mig_summary["skipped"])
        logger.info("  Dry-run  : %s", mig_summary["dry_run"])
        return

    # Build search clients (only needed for non-dry-run)
    google_client = None
    news_client = None
    if not args.dry_run:
        from src.search.google_cse import build_google_cse_from_env
        from src.search.newsapi import build_newsapi_from_env
        google_client = build_google_cse_from_env()
        news_client = build_newsapi_from_env()
        logger.info("Search clients: initialized")

    # Run pipeline
    from src.daily.news_monitor import run_smart_monitor

    summary = run_smart_monitor(
        notion_client=notion_client,
        targets_data_source_id=targets_resolved.data_source_id,
        events_db_id=events_db_id,
        events_data_source_id=events_resolved.data_source_id,
        google_client=google_client,
        news_client=news_client,
        dry_run=args.dry_run,
        limit=args.limit,
        target_type=args.target_type,
    )

    # Print summary
    logger.info("=== Summary ===")
    logger.info("  Total targets  : %d", summary["total_targets"])
    logger.info("  Due today      : %d", summary["due_today"])
    if not args.dry_run:
        logger.info("  Searched       : %d", summary["searched"])
        logger.info("  Events created : %d", summary["events_created"])
        logger.info("  Promotions     : %d", summary["promotions"])
        logger.info("  Demotions      : %d", summary["demotions"])
        logger.info("  Paused         : %d", summary["paused"])
        logger.info("  Errors         : %d", summary["errors"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
