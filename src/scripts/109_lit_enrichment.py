#!/usr/bin/env python
# src/scripts/109_lit_enrichment.py
"""109 LIT Enrichment — enrich LIT DB papers with content fields.

Enriches papers that have PDF but lack content fields (Core Idea, Methods,
Datasets, Findings, Notes). Uses Claude Sonnet to extract structured fields
from PDF text, caches results locally, and writes back to Notion.

Part of the daily pipeline (078): runs after 074 (LIT Inbox), before 031.

Usage::

    # Daily: enrich new papers (INBOX + HAS_PDF + Core Idea empty)
    python -m src.scripts.109_lit_enrichment

    # Backfill: enrich existing READ/KEEP papers
    python -m src.scripts.109_lit_enrichment --backfill

    # Normalize language: re-enrich English fields to Japanese
    python -m src.scripts.109_lit_enrichment --normalize-lang --force

    # Dry-run: show eligible papers
    python -m src.scripts.109_lit_enrichment --dry-run

    # Limit processing count
    python -m src.scripts.109_lit_enrichment --backfill --limit 10
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

from src.config import load_env, is_notion_writeback_enabled

logger = logging.getLogger("109_lit_enrichment")

SCRIPT_NAME = "109_lit_enrichment"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=f"python -m src.scripts.{SCRIPT_NAME}",
        description="Enrich LIT DB papers with content fields from PDF text.",
    )
    p.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill mode: enrich existing READ/KEEP papers with empty fields.",
    )
    p.add_argument(
        "--normalize-lang",
        action="store_true",
        help="Language normalization: re-enrich papers with English content fields.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing fields (use with --normalize-lang).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show eligible papers without processing or writing.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of papers to process (0 = unlimited).",
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

    logger.info("=== 109 LIT Enrichment ===")
    if args.dry_run:
        logger.info("Mode: DRY-RUN")
    if args.backfill:
        logger.info("Mode: BACKFILL")
    if args.normalize_lang:
        logger.info("Mode: NORMALIZE-LANG")
    if args.force:
        logger.info("Mode: FORCE")

    # Notion client + data_source_id (same pattern as 074)
    from src.config import get_db_id
    from src.notion import build_notion_client_from_env, NotionDataSourceResolver

    notion_client = build_notion_client_from_env()
    db_id = get_db_id("NOTION_LIT_DB_ID")

    resolver = NotionDataSourceResolver(notion_client)
    resolved = resolver.resolve_once(name="LIT_DB", database_id=db_id)
    data_source_id = resolved.data_source_id
    logger.info("LIT DB: database_id=%s data_source_id=%s", db_id[:12], data_source_id[:12])

    # ENABLE_NOTION_WRITEBACK check
    enable_writeback = is_notion_writeback_enabled()
    if enable_writeback:
        logger.info("ENABLE_NOTION_WRITEBACK: ON")
    else:
        logger.info("ENABLE_NOTION_WRITEBACK: OFF (cache only, no Notion writes)")

    # LLM client (only needed for non-dry-run)
    llm_client = None
    if not args.dry_run:
        from src.llm.claude_client import build_claude_client_from_env
        llm_client = build_claude_client_from_env()
        logger.info("Claude client: initialized")

    # Run enrichment
    from src.daily.lit_enrichment import run_lit_enrichment

    summary = run_lit_enrichment(
        notion_client=notion_client,
        data_source_id=data_source_id,
        llm_client=llm_client,
        backfill=args.backfill,
        normalize_lang=args.normalize_lang,
        force=args.force,
        dry_run=args.dry_run,
        limit=args.limit,
        enable_writeback=enable_writeback,
    )

    # Print summary
    logger.info("=== Summary ===")
    logger.info("  Total eligible : %d", summary["total"])
    if not args.dry_run:
        logger.info("  Processed      : %d", summary["processed"])
        logger.info("  Enriched       : %d", summary["enriched"])
        logger.info("  Written        : %d", summary["written"])
        logger.info("  Skipped        : %d", summary["skipped"])
        logger.info("  Errors         : %d", summary["errors"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
