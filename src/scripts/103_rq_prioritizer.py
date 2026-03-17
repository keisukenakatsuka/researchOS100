#!/usr/bin/env python
"""103 RQ Prioritizer — Block 1 portfolio management.

Assigns promote/refine/defer/merge recommendations to evaluated RQ candidates.

Exit codes:
  0: portfolio generated
  1: fatal error

Usage::

    python -m src.scripts.103_rq_prioritizer --run-id <parent_run_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env
from src.question.prioritizer import prioritize_rq_candidates, _render_markdown

logger = logging.getLogger("103_rq_prioritizer")

_QF_DATA_DIR = _PROJECT_ROOT / "data" / "question_formation"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.103_rq_prioritizer",
        description="103 RQ Prioritizer — portfolio management for RQ candidates",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    load_env()

    out_dir = _QF_DATA_DIR / f"from_{args.run_id}"
    eval_path = out_dir / "rq_evaluation.json"

    if not eval_path.exists():
        logger.error("Evaluation file not found: %s", eval_path)
        logger.error("Run 102_rq_evaluator first")
        sys.exit(1)

    logger.info("=== 103 RQ Prioritizer === parent_run=%s", args.run_id)

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = prioritize_rq_candidates(eval_path, llm_client=llm_client)

    if result.status == "generated":
        json_path = out_dir / "rq_portfolio.json"
        json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

        md_text = _render_markdown(result)
        md_path = out_dir / "rq_portfolio.md"
        md_path.write_text(md_text)

        s = result.summary
        print(f"\n{'=' * 60}")
        print(f"103 RQ Prioritizer — COMPLETED")
        print(f"{'=' * 60}")
        print(f"Total: {s.total_candidates} | promote: {s.promote} | refine: {s.refine} | defer: {s.defer} | merge: {s.merge}\n")
        for e in sorted(result.portfolio, key=lambda x: x.priority_rank):
            action_icon = {"promote": "+", "refine": "~", "defer": "-", "merge": "M"}.get(e.recommendation, "?")
            merge_info = f" → {e.merge_target_id}" if e.merge_target_id else ""
            print(f"  [{action_icon}] #{e.priority_rank} {e.title[:50]} ({e.portfolio_role}){merge_info}")

        print(f"\nOutputs: {out_dir}")
        print(f"  rq_portfolio.json")
        print(f"  rq_portfolio.md")
        sys.exit(0)
    else:
        print(f"Error: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
