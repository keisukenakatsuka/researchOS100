#!/usr/bin/env python
# src/scripts/068_multi_source_collector.py
"""068 Multi-Source Collector — CLI entrypoint.

Collects web sources based on search queries from plan.json,
fetches page content, and saves sources.json.

Usage::

    # Basic usage
    python -m src.scripts.068_multi_source_collector \\
        --run-id dr_20260307_1poh0

    # Search only, skip content fetching
    python -m src.scripts.068_multi_source_collector \\
        --run-id dr_20260307_1poh0 --skip-fetch

    # Verbose logging
    python -m src.scripts.068_multi_source_collector \\
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

from src.config import load_env
from src.deep_research import save_step_output
from src.deep_research.collector import run as run_collector
from src.search.google_cse import build_google_cse_from_env

logger = logging.getLogger("068_multi_source_collector")

SCRIPT_NAME = "068_multi_source_collector"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Collect sources from plan.json search queries.",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Run ID from 067 planner (e.g. dr_20260307_abc12).",
    )
    p.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip web content fetching (search-only mode).",
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

    logger.info("=== 068 Multi-Source Collector ===")
    logger.info("run_id    : %s", args.run_id)
    logger.info("skip_fetch: %s", args.skip_fetch)

    # Build search clients
    search_client = build_google_cse_from_env()

    news_client = None
    try:
        from src.search.newsapi import build_newsapi_from_env
        news_client = build_newsapi_from_env()
        logger.info("NewsAPI: enabled")
    except Exception:
        logger.info("NewsAPI: not configured — skipping news search")

    # Run collector
    result = run_collector(
        run_id=args.run_id,
        search_client=search_client,
        news_client=news_client,
        skip_fetch=args.skip_fetch,
    )

    # Save sources.json
    path = save_step_output(args.run_id, "068", result)
    logger.info("sources.json saved: %s", path)

    # Summary
    logger.info("--- Collection Summary ---")
    logger.info("  total_sources : %d", result["total_sources"])
    status_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    total_chars = 0
    for s in result["sources"]:
        st = s["fetch_status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        st2 = s["source_type"]
        type_counts[st2] = type_counts.get(st2, 0) + 1
        total_chars += s.get("fetched_char_count", 0)
    for st, cnt in sorted(status_counts.items()):
        logger.info("  fetch_%-8s: %d", st, cnt)
    for st, cnt in sorted(type_counts.items()):
        logger.info("  type_%-8s: %d", st, cnt)
    logger.info("  total_chars   : %d", total_chars)
    logger.info("  collected_at  : %s", result["collected_at"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
