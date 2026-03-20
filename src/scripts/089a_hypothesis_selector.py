#!/usr/bin/env python
"""089a Hypothesis Selector — select H1/H2 from candidate hypotheses.

Evaluates hypotheses on 6 axes and selects primary (H1) and
optionally secondary (H2) for the focused paper.

Usage::

    python -m src.scripts.089a_hypothesis_selector --run-id <id>
    python -m src.scripts.089a_hypothesis_selector --run-id <id> --force-single
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
from src.lit_review.selector import select_hypotheses, collect_inputs

logger = logging.getLogger("089a_hypothesis_selector")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.089a_hypothesis_selector",
        description="089a Hypothesis Selector — select H1/H2 from candidates",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--max-secondary-score", type=float, default=3.0,
                   help="Minimum composite score for H2 selection (default: 3.0)")
    p.add_argument("--force-single", action="store_true",
                   help="Force single hypothesis (H1 only, no H2)")
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

    for req in ["hypotheses.json", "hypothesis_portfolio.json"]:
        if not (run_dir / req).exists():
            logger.error("%s not found in %s", req, args.run_id)
            return

    logger.info("=== 089a Hypothesis Selector === run_id=%s", args.run_id)

    if args.dry_run:
        inputs = collect_inputs(run_dir)
        print(f"\nDRY RUN — inputs:")
        print(f"  RQ: {inputs.get('rq_title', '?')}")
        print(f"  Hypotheses: {len(inputs['hypotheses'])}")
        print(f"  Portfolio scores: {len(inputs['portfolio'])}")
        print(f"  Force single: {args.force_single}")
        return

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = select_hypotheses(
        run_dir,
        llm_client=llm_client,
        max_secondary_score=args.max_secondary_score,
        force_single=args.force_single,
    )

    # Save
    json_path = run_dir / "hypothesis_selection.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved hypothesis_selection.json")

    md_path = run_dir / "hypothesis_selection.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved hypothesis_selection.md")

    # Update manifest
    try:
        from src.lit_review.run_manifest import update_step
        update_step(run_dir, "089a_hypothesis_selector",
                    status="completed",
                    outputs=["hypothesis_selection.json", "hypothesis_selection.md"])
    except Exception as e:
        logger.warning("Manifest update failed: %s", e)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"089a Hypothesis Selector — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {result.rq_title}")
    print(f"Candidates: {result.metadata.get('total_candidates', 0)}")
    print(f"Selected: {result.metadata.get('selected_count', 0)}")
    print(f"")

    for c in result.candidates:
        marker = ""
        if c.status == "selected_primary":
            marker = " *** H1"
        elif c.status == "selected_secondary":
            marker = " **  H2"
        print(f"  {c.rank}. [{c.status:18s}] (comp={c.composite_score}) {c.hypothesis_statement[:50]}{marker}")

    print(f"\nRationale: {result.selection_rationale[:120]}")
    print(f"\nOutputs: {run_dir}")
    print(f"  hypothesis_selection.json")
    print(f"  hypothesis_selection.md")

    logger.info("=== 089a Done ===")


if __name__ == "__main__":
    main()
