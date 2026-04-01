#!/usr/bin/env python
"""091 Data Requirements — detail data needs for validation designs.

Operationalizes variables, evaluates data sources, identifies
alternatives and missing risks for each research design.

Usage::

    python -m src.scripts.091_data_requirements --run-id <id>
    python -m src.scripts.091_data_requirements --run-id <id> --max-designs 3
    python -m src.scripts.091_data_requirements --run-id <id> --dry-run
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
from src.lit_review.data_reqs import plan_data_requirements, collect_inputs

logger = logging.getLogger("091_data_requirements")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.091_data_requirements",
        description="091 Data Requirements — detail data needs for validation designs",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--max-designs", type=int, default=5)
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
    if not (run_dir / "validation_designs.json").exists():
        logger.error("validation_designs.json not found (run 090 first)")
        return

    logger.info("=== 091 Data Requirements === run_id=%s", args.run_id)

    if args.dry_run:
        inputs = collect_inputs(run_dir)
        designs = inputs["validation_designs"][:args.max_designs]
        print(f"\nDRY RUN — would detail data for {len(designs)} designs:")
        for d in designs:
            dr = d.get("data_requirements", {})
            print(f"  [{d.get('identification_strategy', '?')}] {d.get('hypothesis_statement', '')[:55]}")
            print(f"    DV: {dr.get('dependent_variable', '?')}")
            print(f"    Sources: {', '.join(dr.get('data_sources', [])[:3])}")
        return

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = plan_data_requirements(run_dir, llm_client=llm_client, max_designs=args.max_designs)

    # Save
    json_path = run_dir / "data_requirements.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved data_requirements.json")

    md_path = run_dir / "data_requirements.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved data_requirements.md")

    # Summary
    all_vars = [v for p in result.data_plans for v in p.variables]
    by_diff = Counter(v.primary_source.acquisition_difficulty for v in all_vars)
    by_risk = Counter(v.missing_risk for v in all_vars)
    by_feas = Counter(p.overall_feasibility for p in result.data_plans)

    print(f"\n{'=' * 60}")
    print(f"091 Data Requirements — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {result.rq_title}")
    print(f"Designs detailed: {result.designs_detailed}")
    print(f"Total variables: {len(all_vars)}")
    print(f"")
    print(f"Acquisition difficulty: {dict(by_diff)}")
    print(f"Missing risk: {dict(by_risk)}")
    print(f"Overall feasibility: {dict(by_feas)}")
    print(f"")

    for i, plan in enumerate(result.data_plans, 1):
        gaps = len(plan.critical_data_gaps)
        print(f"  {i}. [{plan.overall_feasibility}] {plan.hypothesis_statement[:50]}")
        print(f"     Variables: {len(plan.variables)}, Gaps: {gaps}, Steps: {len(plan.recommended_first_steps)}")

    print(f"\nOutputs: {run_dir}")
    print(f"  data_requirements.json")
    print(f"  data_requirements.md")

    logger.info("=== 091 Done ===")


if __name__ == "__main__":
    main()
