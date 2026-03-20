#!/usr/bin/env python
"""089c Focused Hypotheses — assemble canonical H1/H2 for downstream.

Produces focused_hypotheses.json from review decision,
enriched with full hypothesis data, portfolio scores, and assumptions.

Usage::

    python -m src.scripts.089c_focused_hypotheses --run-id <id>
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
from src.lit_review.focus import build_focused

logger = logging.getLogger("089c_focused_hypotheses")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.089c_focused_hypotheses",
        description="089c Focused Hypotheses — assemble canonical H1/H2",
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

    if not (run_dir / "hypothesis_review_decision.json").exists():
        logger.error("hypothesis_review_decision.json not found — run 089b first")
        return

    logger.info("=== 089c Focused Hypotheses === run_id=%s", args.run_id)

    result = build_focused(run_dir)

    # Save
    json_path = run_dir / "focused_hypotheses.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved focused_hypotheses.json")

    md_path = run_dir / "focused_hypotheses.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved focused_hypotheses.md")

    # Update manifest
    try:
        from src.lit_review.run_manifest import update_step
        update_step(run_dir, "089c_focused_hypotheses",
                    status="completed",
                    outputs=["focused_hypotheses.json", "focused_hypotheses.md"])
    except Exception as e:
        logger.warning("Manifest update failed: %s", e)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"089c Focused Hypotheses — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {result.rq_title}")
    print(f"Review source: {result.review_source}")
    print(f"")

    if result.primary:
        print(f"H1: {result.primary.get('hypothesis_statement', '')[:70]}")
        print(f"    ID: {result.primary.get('hypothesis_id', '')}")
        print(f"    Portfolio: {result.primary.get('portfolio_recommendation', '')} "
              f"(comp={result.primary.get('portfolio_composite', '')})")

    if result.has_secondary and result.secondary:
        print(f"H2: {result.secondary.get('hypothesis_statement', '')[:70]}")
        print(f"    ID: {result.secondary.get('hypothesis_id', '')}")
    else:
        print(f"H2: Not selected")

    ns = result.non_selected_summary
    print(f"\nNon-selected: {ns.get('deferred_count', 0)} deferred, {ns.get('rejected_count', 0)} rejected")

    print(f"\nOutputs: {run_dir}")
    print(f"  focused_hypotheses.json")
    print(f"  focused_hypotheses.md")

    logger.info("=== 089c Done ===")


if __name__ == "__main__":
    main()
