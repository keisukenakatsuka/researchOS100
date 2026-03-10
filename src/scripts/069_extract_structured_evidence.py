#!/usr/bin/env python
# src/scripts/069_extract_structured_evidence.py
"""069 Extract Structured Evidence — CLI entrypoint.

Reads sources.json, extracts Evidence items from fetched text,
and saves evidence.json.

Usage::

    # With LLM extraction
    python -m src.scripts.069_extract_structured_evidence \\
        --run-id dr_20260307_1poh0

    # Rule-based fallback only (no LLM)
    python -m src.scripts.069_extract_structured_evidence \\
        --run-id dr_20260307_1poh0 --no-llm

    # Verbose logging
    python -m src.scripts.069_extract_structured_evidence \\
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
from src.deep_research.extractor import run as run_extractor

logger = logging.getLogger("069_extract_structured_evidence")

SCRIPT_NAME = "069_extract_structured_evidence"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Extract structured evidence from sources.json.",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Run ID (e.g. dr_20260307_abc12).",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM extraction; use rule-based fallback only.",
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

    logger.info("=== 069 Extract Structured Evidence ===")
    logger.info("run_id : %s", args.run_id)
    logger.info("no_llm : %s", args.no_llm)

    # LLM client (optional)
    llm_client = None
    if not args.no_llm:
        try:
            from src.llm.claude_client import build_claude_client_from_env
            llm_client = build_claude_client_from_env()
            logger.info("LLM: enabled (Claude)")
        except Exception as e:
            logger.warning("LLM client unavailable (%s) — using fallback", e)

    # Run extractor
    result = run_extractor(
        run_id=args.run_id,
        llm_client=llm_client,
    )

    # Save evidence.json
    path = save_step_output(args.run_id, "069", result)
    logger.info("evidence.json saved: %s", path)

    # Summary
    logger.info("--- Extraction Summary ---")
    logger.info("  total_evidence: %d", result["total_evidence"])
    tag_counts: dict[str, int] = {}
    for ev in result["evidence"]:
        for tag in ev.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
        logger.info("  tag_%-12s: %d", tag, cnt)
    logger.info("  generated_at  : %s", result["generated_at"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
