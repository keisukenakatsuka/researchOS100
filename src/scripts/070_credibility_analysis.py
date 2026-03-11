#!/usr/bin/env python
# src/scripts/070_credibility_analysis.py
"""070 Credibility Analysis — CLI entrypoint.

Reads evidence.json and sources.json, annotates each Evidence item
with confidence / confidence_reason, and saves credibility.json.

Usage::

    python -m src.scripts.070_credibility_analysis \\
        --run-id dr_20260307_1poh0

    python -m src.scripts.070_credibility_analysis \\
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
from src.deep_research.credibility import run as run_credibility

logger = logging.getLogger("070_credibility_analysis")

SCRIPT_NAME = "070_credibility_analysis"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Annotate evidence with credibility scores.",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Run ID (e.g. dr_20260307_abc12).",
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

    logger.info("=== 070 Credibility Analysis ===")
    logger.info("run_id: %s", args.run_id)

    # Run credibility
    result = run_credibility(run_id=args.run_id)

    # Save credibility.json
    path = save_step_output(args.run_id, "070", result)
    logger.info("credibility.json saved: %s", path)

    # Summary
    logger.info("--- Credibility Summary ---")
    logger.info("  total_annotated: %d", result["total_annotated"])
    dist = result.get("confidence_distribution", {})
    for level_name in ("high", "medium", "low"):
        logger.info("  %-7s: %d", level_name, dist.get(level_name, 0))
    logger.info("  generated_at   : %s", result["generated_at"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
