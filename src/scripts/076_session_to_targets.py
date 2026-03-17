#!/usr/bin/env python
# src/scripts/076_session_to_targets.py
"""076 Session to Targets — reflect 073 research sessions into monitoring.

Pipeline:
1. Scan recent 073 session outputs (data/deep_research_sessions/)
2. For each completed run:
   a. Load plan.json, memo.json, claims.json from data/deep_research/{run_id}/
   b. Extract monitoring target candidates via LLM
   c. Validate People candidates (full name, affiliation, not public figure)
   d. Check for duplicates against existing MONITORING_TARGETS_DB
   e. Register new targets (Active, Weekly, Created By=076_session)
3. Print summary

Usage::

    # Dry-run: extract candidates without registering
    python -m src.scripts.076_session_to_targets --dry-run

    # Process last 48 hours of sessions
    python -m src.scripts.076_session_to_targets --hours 48

    # Full run (default: last 24 hours)
    python -m src.scripts.076_session_to_targets

    # Verbose logging
    python -m src.scripts.076_session_to_targets -v
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

logger = logging.getLogger("076_session_to_targets")

SCRIPT_NAME = "076_session_to_targets"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=f"python -m src.scripts.{SCRIPT_NAME}",
        description="Extract monitoring targets from 073 research sessions.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract candidates without registering in Notion.",
    )
    p.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Look back N hours for sessions (default: 24).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of targets to register (0 = unlimited).",
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

    logger.info("=== 076 Session to Targets ===")
    if args.dry_run:
        logger.info("Mode: DRY-RUN")
    logger.info("Look-back: %d hours", args.hours)

    # Build clients
    from src.config import get_db_id
    from src.notion import build_notion_client_from_env, NotionDataSourceResolver

    notion_client = build_notion_client_from_env()
    targets_db_id = get_db_id("NOTION_MONITORING_TARGETS_DB_ID")

    resolver = NotionDataSourceResolver(notion_client)
    targets_resolved = resolver.resolve_once(
        name="MONITORING_TARGETS_DB", database_id=targets_db_id,
    )
    logger.info(
        "Targets DB: database_id=%s data_source_id=%s",
        targets_db_id[:12], targets_resolved.data_source_id[:12],
    )

    # Build LLM client
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()
    logger.info("Claude client: initialized")

    # Run pipeline
    from src.daily.session_targets import run_session_to_targets

    summary = run_session_to_targets(
        notion_client=notion_client,
        targets_db_id=targets_db_id,
        targets_data_source_id=targets_resolved.data_source_id,
        llm_client=llm_client,
        hours=args.hours,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    # Print summary
    logger.info("=== Summary ===")
    logger.info("  Sessions scanned  : %d", summary["sessions_scanned"])
    logger.info("  Runs processed    : %d", summary["runs_processed"])
    logger.info("  Candidates found  : %d", summary["candidates_extracted"])
    logger.info("  Low confidence    : %d", summary["low_confidence_skipped"])
    logger.info("  People rejected   : %d", summary["people_rejected"])
    logger.info("  Duplicates        : %d", summary["duplicates_skipped"])
    if not args.dry_run:
        logger.info("  Registered        : %d", summary["registered"])
        logger.info("  Errors            : %d", summary["errors"])
    else:
        would_register = sum(
            1 for r in summary["results"] if r.get("action") == "would_register"
        )
        logger.info("  Would register    : %d", would_register)

    # Detailed results
    for r in summary["results"]:
        action_str = r.get("action", "?")
        reason_str = f" ({r['reason']})" if r.get("reason") else ""
        logger.info(
            "  → [%s] %s (type=%s, conf=%s)%s",
            action_str.upper(),
            r.get("name", "?"),
            r.get("type", "?"),
            r.get("confidence", "?"),
            reason_str,
        )

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
