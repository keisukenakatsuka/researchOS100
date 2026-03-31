#!/usr/bin/env python
"""090 Validation Designer — design research validations for hypotheses.

Generates concrete research designs for high-priority hypotheses,
including identification strategy, data requirements, and next steps.

Usage::

    python -m src.scripts.090_validation_designer --run-id <id>
    python -m src.scripts.090_validation_designer --run-id <id> --max-designs 3
    python -m src.scripts.090_validation_designer --run-id <id> --dry-run
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
from src.lit_review.validation import design_validations, collect_inputs, select_target_hypotheses

logger = logging.getLogger("090_validation_designer")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.090_validation_designer",
        description="090 Validation Designer — generate research designs for hypotheses",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--max-designs", type=int, default=5, help="Max hypotheses to design for (default: 5)")
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

    for req in ["hypotheses.json", "assumptions.json", "hypothesis_portfolio.json"]:
        if not (run_dir / req).exists():
            logger.error("%s not found in %s (run 087/088/089 first)", req, args.run_id)
            return

    logger.info("=== 090 Validation Designer === run_id=%s", args.run_id)

    if args.dry_run:
        inputs = collect_inputs(run_dir)
        targets = select_target_hypotheses(inputs, max_designs=args.max_designs)
        print(f"\nDRY RUN — would design validations for {len(targets)} hypotheses:")
        for t in targets:
            print(f"  [{t['recommendation']:16s}] {t['hypothesis_statement'][:60]}")
            print(f"    assumptions: {len(t.get('assumptions', []))}, vulnerability: {t.get('overall_vulnerability', '?')}")
        return

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = design_validations(run_dir, llm_client=llm_client, max_designs=args.max_designs)

    # Save
    json_path = run_dir / "validation_designs.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved validation_designs.json")

    md_path = run_dir / "validation_designs.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved validation_designs.md")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"090 Validation Designer — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {result.rq_title}")
    print(f"Designs generated: {result.designs_generated}")
    print(f"")

    for i, d in enumerate(result.validation_designs, 1):
        risk_counts = {}
        for r in d.key_risks:
            risk_counts[r.severity] = risk_counts.get(r.severity, 0) + 1
        print(f"  {i}. [{d.recommendation}] {d.hypothesis_statement[:55]}")
        print(f"     Design: {d.design_type} | Strategy: {d.identification_strategy}")
        print(f"     Data: {', '.join(d.data_requirements.data_sources[:3])}")
        print(f"     Risks: {risk_counts} | Steps: {len(d.next_steps)}")
        print(f"")

    print(f"Outputs: {run_dir}")
    print(f"  validation_designs.json")
    print(f"  validation_designs.md")

    logger.info("=== 090 Done ===")


if __name__ == "__main__":
    main()
