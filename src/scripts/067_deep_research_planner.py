#!/usr/bin/env python
# src/scripts/067_deep_research_planner.py
"""067 Deep Research Planner — CLI entrypoint.

Transforms a free-form research request into a structured ResearchPlan
(plan.json) that drives all downstream pipeline steps (068-072).

Usage::

    # Basic usage
    python -m src.scripts.067_deep_research_planner \\
        --request "OpenAI の最近の戦略を調べて"

    # Resume an existing run
    python -m src.scripts.067_deep_research_planner \\
        --request "OpenAI の最近の戦略を調べて" \\
        --run-id dr_20260307_abc12

    # Verbose logging
    python -m src.scripts.067_deep_research_planner \\
        --request "OpenAI の最近の戦略を調べて" -v
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
from src.deep_research import generate_run_id, save_step_output
from src.deep_research.planner import run as run_planner
from src.llm.claude_client import build_claude_client_from_env

logger = logging.getLogger("067_deep_research_planner")

SCRIPT_NAME = "067_deep_research_planner"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Generate a research plan from a free-form request.",
    )
    p.add_argument(
        "--request",
        required=True,
        help="Free-form research request (Japanese or English).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Existing run_id to resume (default: auto-generate).",
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

    # Run ID
    run_id = args.run_id or generate_run_id()
    logger.info("=== 067 Deep Research Planner ===")
    logger.info("run_id : %s", run_id)
    logger.info("request: %s", args.request[:120])

    # LLM client
    llm_client = build_claude_client_from_env()

    # Notion client for Knowledge Recall (optional)
    notion_client = None
    from src.deep_research.recall import is_recall_enabled
    if is_recall_enabled():
        try:
            from src.notion import build_notion_client_from_env
            notion_client = build_notion_client_from_env()
            logger.info("Recall: Notion client connected")
        except Exception as e:
            logger.warning("Recall: Notion client unavailable (%s) — recall disabled", e)

    # Execute planner
    plan = run_planner(args.request, llm_client, run_id=run_id, notion_client=notion_client)

    # Persist plan.json
    plan_dict = plan.to_dict()
    path = save_step_output(run_id, "067", plan_dict)
    logger.info("plan.json saved: %s", path)

    # Summary
    logger.info("--- Plan Summary ---")
    logger.info("  intent      : %s", plan.intent)
    logger.info("  created_at  : %s", plan.created_at.isoformat())
    logger.info("  targets     : %s", plan.targets)
    logger.info("  questions   : %d", len(plan.key_questions))
    logger.info("  queries     : %d", len(plan.search_queries))
    logger.info("  deliverables: %s", plan.deliverables)
    logger.info("  constraints : %s", plan.constraints)
    logger.info("  recalled_ev : %d", len(plan.recalled_evidence_ids))
    logger.info("  recalled_cl : %d", len(plan.recalled_claim_ids))
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
