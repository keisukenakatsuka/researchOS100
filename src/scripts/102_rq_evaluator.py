#!/usr/bin/env python
"""102 RQ Evaluator — Block 1 quality scoring.

Evaluates RQ candidates on 4 axes (Specificity, Testability, Novelty, Feasibility).

Exit codes:
  0: evaluation completed
  1: fatal error

Usage::

    python -m src.scripts.102_rq_evaluator --run-id <parent_run_id>
    python -m src.scripts.102_rq_evaluator --run-id <parent_run_id> --dry-run
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
from src.question.evaluator import evaluate_rq_candidates, _render_markdown

logger = logging.getLogger("102_rq_evaluator")

_QF_DATA_DIR = _PROJECT_ROOT / "data" / "question_formation"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.102_rq_evaluator",
        description="102 RQ Evaluator — 4-axis quality scoring of RQ candidates",
    )
    p.add_argument("--run-id", type=str, required=True, help="Parent research run ID")
    p.add_argument("--dry-run", action="store_true", help="Show candidates without evaluating")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    load_env()

    out_dir = _QF_DATA_DIR / f"from_{args.run_id}"
    candidates_path = out_dir / "rq_candidates.json"

    if not candidates_path.exists():
        logger.error("Candidates file not found: %s", candidates_path)
        logger.error("Run 101_rq_generator first")
        sys.exit(1)

    logger.info("=== 102 RQ Evaluator === parent_run=%s", args.run_id)

    if args.dry_run:
        data = json.loads(candidates_path.read_text())
        candidates = data.get("candidates", [])
        print(f"\nDRY RUN — {len(candidates)} candidates to evaluate:")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. [{c.get('source_type', '')}] {c.get('title', '')}")
        print(f"\nWould generate: rq_evaluation.json + rq_evaluation.md")
        sys.exit(0)

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = evaluate_rq_candidates(candidates_path, llm_client=llm_client)

    if result.status == "generated":
        json_path = out_dir / "rq_evaluation.json"
        json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

        md_text = _render_markdown(result)
        md_path = out_dir / "rq_evaluation.md"
        md_path.write_text(md_text)

        sorted_evals = sorted(result.evaluations, key=lambda e: e.composite_score, reverse=True)

        print(f"\n{'=' * 60}")
        print(f"102 RQ Evaluator — COMPLETED")
        print(f"{'=' * 60}")
        print(f"Candidates evaluated: {len(result.evaluations)}\n")
        print(f"{'Rank':<5} {'Score':<7} {'Spec':<5} {'Test':<5} {'Nov':<5} {'Feas':<5} Title")
        print(f"{'-'*5} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*30}")
        for i, e in enumerate(sorted_evals, 1):
            print(
                f"{i:<5} {e.composite_score:<7.2f} "
                f"{e.specificity.score:<5} {e.testability.score:<5} "
                f"{e.novelty.score:<5} {e.feasibility.score:<5} "
                f"{e.title[:40]}"
            )

        print(f"\nOutputs: {out_dir}")
        print(f"  rq_evaluation.json")
        print(f"  rq_evaluation.md")
        sys.exit(0)
    else:
        print(f"Error: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
