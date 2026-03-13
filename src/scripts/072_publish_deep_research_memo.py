#!/usr/bin/env python
# src/scripts/072_publish_deep_research_memo.py
"""072 Publish Deep Research Memo — CLI entrypoint.

Generates an evidence-based memo and optionally publishes to Notion.

Usage::

    # Dry-run (local only, no Notion writes)
    python -m src.scripts.072_publish_deep_research_memo \\
        --run-id dr_20260307_1poh0

    # With Notion writeback (requires ENABLE_NOTION_WRITEBACK=true)
    python -m src.scripts.072_publish_deep_research_memo \\
        --run-id dr_20260307_1poh0

    # Force dry-run even if env says writeback enabled
    python -m src.scripts.072_publish_deep_research_memo \\
        --run-id dr_20260307_1poh0 --dry-run

    # Verbose logging
    python -m src.scripts.072_publish_deep_research_memo \\
        --run-id dr_20260307_1poh0 -v
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
from src.deep_research.publisher import run as run_publisher

logger = logging.getLogger("072_publish_deep_research_memo")

SCRIPT_NAME = "072_publish_deep_research_memo"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Generate and publish a deep research memo.",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Run ID (e.g. dr_20260307_abc12).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run: generate locally but skip Notion writes.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
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

    logger.info("=== 072 Publish Deep Research Memo ===")
    logger.info("run_id  : %s", args.run_id)

    # Determine writeback
    enable_writeback = is_notion_writeback_enabled() and not args.dry_run
    logger.info("dry_run : %s", args.dry_run)
    logger.info("writeback: %s", enable_writeback)

    # Preflight check (when writeback is requested)
    if enable_writeback:
        from src.deep_research.publisher import preflight_check

        pf = preflight_check()
        if not pf["ok"]:
            if not pf["writeback_enabled"]:
                logger.error("ENABLE_NOTION_WRITEBACK is not 'true'. Set it or use --dry-run.")
            if not pf["notion_token"]:
                logger.error("NOTION_TOKEN is not set.")
            if pf["missing_db_ids"]:
                logger.error(
                    "Missing Notion DB env vars: %s",
                    ", ".join(pf["missing_db_ids"]),
                )
            sys.exit(1)
        logger.info("Preflight: OK")

    # Notion client (only if writing)
    notion_client = None
    if enable_writeback:
        try:
            from src.notion import build_notion_client_from_env
            notion_client = build_notion_client_from_env()
            logger.info("Notion client: connected")
        except Exception as e:
            logger.warning("Notion client unavailable (%s) — falling back to dry-run", e)
            enable_writeback = False

    # Run publisher
    result = run_publisher(
        run_id=args.run_id,
        notion_client=notion_client,
        enable_writeback=enable_writeback,
    )

    # Summary
    memo = result["memo"]
    paths = result["paths"]

    logger.info("--- Publish Summary ---")
    logger.info("  title     : %s", memo["title"])
    logger.info("  memo_id   : %s", memo["memo_id"])
    logger.info("  claims    : %d", len(memo["claim_ids"]))
    logger.info("  evidence  : %d", len(memo["evidence_ids"]))
    logger.info("  sources   : %d", len(memo["source_ids"]))
    logger.info("  memo.md   : %s", paths.get("md", ""))
    logger.info("  memo.json : %s", paths.get("json", ""))

    notion_result = result.get("notion_result")
    if notion_result:
        logger.info("  --- Notion ---")
        logger.info("  source pages : %d", len(notion_result.get("source_id_map", {})))
        logger.info("  evidence pages: %d", len(notion_result.get("evidence_id_map", {})))
        logger.info("  claim pages  : %d", len(notion_result.get("claim_id_map", {})))
        logger.info("  memo page    : %s", notion_result.get("memo_page_id", "N/A"))
        logger.info("  run page     : %s", notion_result.get("run_page_id", "N/A"))
        errors = notion_result.get("errors", [])
        if errors:
            logger.warning("  errors: %d", len(errors))
            for err in errors:
                logger.warning("    %s", err)
    else:
        logger.info("  notion     : dry-run (no writes)")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
