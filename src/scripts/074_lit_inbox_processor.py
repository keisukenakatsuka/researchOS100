#!/usr/bin/env python
# src/scripts/074_lit_inbox_processor.py
"""074 LIT Inbox Processor — process INBOX papers from LIT DB.

Pipeline:
1. Query LIT DB for Status=INBOX papers without a Decision
2. For each paper:
   a. Resolve PDF URL (arXiv > PDF Link > DOI/Unpaywall)
   b. Download PDF to data/downloads/lit_inbox/
   c. Extract text via src.pdf.metadata (reuse, NOT 031 pipeline)
   d. LLM relevance judgment → READ / KEEP / SKIP
   e. Update LIT DB: Decision, Decision Reason, PDF Status
3. Print summary

Usage::

    # Dry-run: show eligible papers and resolved PDF URLs, no downloads or writes
    python -m src.scripts.074_lit_inbox_processor --dry-run

    # Process up to 3 papers
    python -m src.scripts.074_lit_inbox_processor --limit 3

    # Full run
    python -m src.scripts.074_lit_inbox_processor

    # Verbose logging
    python -m src.scripts.074_lit_inbox_processor -v
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

logger = logging.getLogger("074_lit_inbox_processor")

SCRIPT_NAME = "074_lit_inbox_processor"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=f"python -m src.scripts.{SCRIPT_NAME}",
        description="Process INBOX papers from LIT DB: PDF download → LLM judgment → Decision writeback.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show eligible papers and PDF URLs without downloading or writing.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of papers to process (0 = unlimited).",
    )
    p.add_argument(
        "--no-slides",
        action="store_true",
        default=True,
        help="Skip slide generation (default: skip).",
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

    logger.info("=== 074 LIT Inbox Processor ===")
    if args.dry_run:
        logger.info("Mode: DRY-RUN")

    # Build Notion client + resolve data_source_id
    from src.config import get_db_id
    from src.notion import build_notion_client_from_env, NotionDataSourceResolver

    notion_client = build_notion_client_from_env()
    db_id = get_db_id("NOTION_LIT_DB_ID")

    resolver = NotionDataSourceResolver(notion_client)
    resolved = resolver.resolve_once(name="LIT_DB", database_id=db_id)
    data_source_id = resolved.data_source_id
    logger.info("LIT DB: database_id=%s data_source_id=%s", db_id[:12], data_source_id[:12])

    # Build LLM client (only needed for non-dry-run)
    llm_client = None
    if not args.dry_run:
        from src.llm.claude_client import build_claude_client_from_env
        llm_client = build_claude_client_from_env()
        logger.info("Claude client: initialized")

    # Run pipeline
    from src.daily.lit_inbox import run_lit_inbox

    summary = run_lit_inbox(
        notion_client=notion_client,
        data_source_id=data_source_id,
        llm_client=llm_client,
        dry_run=args.dry_run,
        limit=args.limit,
        no_slides=args.no_slides,
    )

    # Print summary
    logger.info("=== Summary ===")
    logger.info("  Total eligible : %d", summary["total"])
    if not args.dry_run:
        logger.info("  Processed      : %d", summary["processed"])
        logger.info("  READ           : %d", summary["read"])
        logger.info("  KEEP           : %d", summary["keep"])
        logger.info("  SKIP           : %d", summary["skip"])
        logger.info("  PDF downloaded : %d", summary["pdf_downloaded"])
        logger.info("  NO_PDF         : %d", summary["no_pdf"])
        logger.info("  Errors         : %d", summary["errors"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
