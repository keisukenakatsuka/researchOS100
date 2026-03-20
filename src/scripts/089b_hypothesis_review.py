#!/usr/bin/env python
"""089b Hypothesis Review — apply human review or auto-accept selector.

Reads hypothesis_review.json (human-authored) if present.
Otherwise auto-accepts the selector output from 089a.

Usage::

    python -m src.scripts.089b_hypothesis_review --run-id <id>
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
from src.lit_review.review import apply_review

logger = logging.getLogger("089b_hypothesis_review")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.089b_hypothesis_review",
        description="089b Hypothesis Review — apply human review or auto-accept",
    )
    p.add_argument("--run-id", type=str, required=True)
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

    if not (run_dir / "hypothesis_selection.json").exists():
        logger.error("hypothesis_selection.json not found — run 089a first")
        return

    logger.info("=== 089b Hypothesis Review === run_id=%s", args.run_id)

    try:
        decision = apply_review(run_dir)
    except (ValueError, FileNotFoundError) as e:
        logger.error("089b failed: %s", e)
        print(f"\nERROR: {e}")
        print(f"If hypothesis_review.json is malformed, fix or delete it to auto-accept.")
        sys.exit(1)

    # Save
    json_path = run_dir / "hypothesis_review_decision.json"
    json_path.write_text(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved hypothesis_review_decision.json")

    # Update manifest
    try:
        from src.lit_review.run_manifest import update_step
        update_step(run_dir, "089b_hypothesis_review",
                    status="completed",
                    outputs=["hypothesis_review_decision.json"])
    except Exception as e:
        logger.warning("Manifest update failed: %s", e)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"089b Hypothesis Review — Complete")
    print(f"{'=' * 60}")
    print(f"Review source: {decision.review_source}")
    print(f"")
    print(f"H1 (Primary): {decision.primary.get('hypothesis_statement', '')[:70]}")
    print(f"  ID: {decision.primary.get('hypothesis_id', '')}")
    if decision.has_secondary:
        print(f"H2 (Secondary): {decision.secondary.get('hypothesis_statement', '')[:70]}")
        print(f"  ID: {decision.secondary.get('hypothesis_id', '')}")
    else:
        print(f"H2 (Secondary): Not selected")
    print(f"")
    if decision.notes_for_downstream:
        print(f"Notes for downstream: {decision.notes_for_downstream}")
    print(f"Non-selected: {len(decision.non_selected)}")
    print(f"\nOutputs: {run_dir}")
    print(f"  hypothesis_review_decision.json")

    logger.info("=== 089b Done ===")


if __name__ == "__main__":
    main()
