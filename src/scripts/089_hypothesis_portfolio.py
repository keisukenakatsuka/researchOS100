#!/usr/bin/env python
"""089 Hypothesis Portfolio — prioritize hypotheses for research.

Scores hypotheses on 5 axes and produces a portfolio view
for research prioritization decisions.

Usage::

    python -m src.scripts.089_hypothesis_portfolio --run-id <id>
    python -m src.scripts.089_hypothesis_portfolio --run-id <id> --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env
from src.lit_review.portfolio import build_portfolio, collect_inputs

logger = logging.getLogger("089_hypothesis_portfolio")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.089_hypothesis_portfolio",
        description="089 Hypothesis Portfolio — score and prioritize hypotheses",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--dry-run", action="store_true")
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

    run_dir = _DATA_DIR / args.run_id
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        return

    for req in ["hypotheses.json", "assumptions.json"]:
        if not (run_dir / req).exists():
            logger.error("%s not found in %s", req, args.run_id)
            return

    logger.info("=== 089 Hypothesis Portfolio === run_id=%s", args.run_id)

    if args.dry_run:
        inputs = collect_inputs(run_dir)
        print(f"\nDRY RUN — inputs:")
        print(f"  RQ: {inputs.get('rq_title', '?')}")
        print(f"  Hypotheses: {len(inputs['hypotheses'])}")
        print(f"  Assumption sets: {len(inputs['hypothesis_assumptions'])}")
        print(f"  Cross-RQ opportunities: {len(inputs['cross_rq_opportunities'])}")
        return

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = build_portfolio(run_dir, llm_client=llm_client)

    # Save
    json_path = run_dir / "hypothesis_portfolio.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved hypothesis_portfolio.json")

    md_path = run_dir / "hypothesis_portfolio.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved hypothesis_portfolio.md")

    # Summary
    by_rec = Counter(h.recommendation for h in result.scored_hypotheses)
    by_quad = Counter(h.quadrant for h in result.scored_hypotheses)

    print(f"\n{'=' * 60}")
    print(f"089 Hypothesis Portfolio — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {result.rq_title}")
    print(f"Hypotheses scored: {result.hypotheses_scored}")
    print(f"")
    print(f"Recommendations: {dict(by_rec)}")
    print(f"Quadrants: {dict(by_quad)}")
    print(f"")

    for i, h in enumerate(result.scored_hypotheses, 1):
        s = h.scores
        print(f"  {i}. [{h.recommendation:16s}] (comp={h.composite_score}) {h.statement[:55]}")
        print(f"     nov={s.get('novelty')} test={s.get('testability')} vuln={s.get('vulnerability')} "
              f"feas={s.get('feasibility')} strat={s.get('strategic_importance')} | {h.quadrant}")

    print(f"\nOutputs: {run_dir}")
    print(f"  hypothesis_portfolio.json")
    print(f"  hypothesis_portfolio.md")

    logger.info("=== 089 Done ===")


if __name__ == "__main__":
    main()
