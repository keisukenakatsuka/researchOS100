#!/usr/bin/env python
# src/scripts/073_user_research_session.py
"""073 User Research Session — CLI entrypoint.

Accepts a free-form user question, decomposes it into research questions,
runs the Deep Research pipeline (067–072) for each, and generates a
unified answer.

By default the session runs non-interactively: question decomposition is
followed by automatic pipeline execution without user confirmation.

Usage::

    # Non-interactive (default): auto-execute after decomposition
    python -m src.scripts.073_user_research_session \
        --question "OpenAI の最近の戦略を調べて"

    # Confirm decomposed questions before execution
    python -m src.scripts.073_user_research_session \
        --question "OpenAI の最近の戦略を調べて" --confirm-plan

    # Interactive mode (prompts for question input)
    python -m src.scripts.073_user_research_session

    # Dry-run (skip Notion writeback)
    python -m src.scripts.073_user_research_session --dry-run

    # Verbose logging
    python -m src.scripts.073_user_research_session -v
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

logger = logging.getLogger("073_user_research_session")

SCRIPT_NAME = "073_user_research_session"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Run a deep research session (non-interactive by default).",
    )
    p.add_argument(
        "--question",
        default=None,
        help="Research question (if omitted, prompts interactively).",
    )
    p.add_argument(
        "--confirm-plan",
        action="store_true",
        help="Confirm decomposed questions interactively before running research.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run: skip Notion writes.",
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

    logger.info("=== 073 User Research Session ===")

    # Get question
    question = args.question
    if not question:
        print()
        question = input("ユーザー質問を入力してください:\n> ").strip()
        if not question:
            print("質問が入力されませんでした。終了します。")
            sys.exit(0)

    logger.info("question: %s", question[:120])

    # Build clients
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    from src.search.google_cse import build_google_cse_from_env
    search_client = build_google_cse_from_env()

    news_client = None
    try:
        from src.search.newsapi import build_newsapi_from_env
        news_client = build_newsapi_from_env()
        logger.info("NewsAPI: enabled")
    except Exception:
        logger.info("NewsAPI: not configured — skipping news search")

    # Notion client (optional)
    enable_writeback = is_notion_writeback_enabled() and not args.dry_run
    notion_client = None
    if enable_writeback:
        try:
            from src.notion import build_notion_client_from_env
            notion_client = build_notion_client_from_env()
            logger.info("Notion: connected (writeback enabled)")
        except Exception as e:
            logger.warning("Notion unavailable (%s) — writeback disabled", e)
            enable_writeback = False

    logger.info("writeback: %s", enable_writeback)

    # Run session
    from src.deep_research.session import run_session

    result = run_session(
        question,
        llm_client=llm_client,
        search_client=search_client,
        news_client=news_client,
        notion_client=notion_client,
        enable_writeback=enable_writeback,
        confirm_plan=args.confirm_plan,
    )

    # Final status
    logger.info("--- Session Summary ---")
    logger.info("  session_id: %s", result["session_id"])
    logger.info("  status    : %s", result["status"])
    if result.get("output_path"):
        logger.info("  output    : %s", result["output_path"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
