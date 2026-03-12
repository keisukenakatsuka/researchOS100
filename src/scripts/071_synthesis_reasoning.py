#!/usr/bin/env python
# src/scripts/071_synthesis_reasoning.py
"""071 Synthesis Reasoning — CLI entrypoint.

Reads credibility.json, clusters evidence, generates Claims,
and saves claims.json.

Usage::

    # With LLM synthesis
    python -m src.scripts.071_synthesis_reasoning \\
        --run-id dr_20260307_1poh0

    # Rule-based fallback only (no LLM)
    python -m src.scripts.071_synthesis_reasoning \\
        --run-id dr_20260307_1poh0 --no-llm

    # Verbose logging
    python -m src.scripts.071_synthesis_reasoning \\
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
from src.deep_research.synthesizer import run as run_synthesizer

logger = logging.getLogger("071_synthesis_reasoning")

SCRIPT_NAME = "071_synthesis_reasoning"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Synthesize claims from annotated evidence.",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help="Run ID (e.g. dr_20260307_abc12).",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM synthesis; use rule-based fallback only.",
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

    logger.info("=== 071 Synthesis Reasoning ===")
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

    # Run synthesizer
    result = run_synthesizer(
        run_id=args.run_id,
        llm_client=llm_client,
    )

    # Save claims.json
    path = save_step_output(args.run_id, "071", result)
    logger.info("claims.json saved: %s", path)

    # Summary
    logger.info("--- Synthesis Summary ---")
    logger.info("  total_claims: %d", result["total_claims"])
    conf_counts: dict[str, int] = {}
    for cl in result["claims"]:
        c = cl["confidence"]
        conf_counts[c] = conf_counts.get(c, 0) + 1
    for c in ("high", "medium", "low"):
        logger.info("  %-7s: %d", c, conf_counts.get(c, 0))
    total_ev_links = sum(len(cl["evidence_ids"]) for cl in result["claims"])
    logger.info("  evidence links: %d", total_ev_links)
    logger.info("  generated_at  : %s", result["generated_at"])
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
