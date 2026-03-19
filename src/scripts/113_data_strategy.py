# src/scripts/113_data_strategy.py
"""Data Strategy Planner — build a phased data acquisition roadmap.

Reads dataset_registry.json, data_requirements.json, and
hypothesis_portfolio.json from a lit_review run. Generates a data
acquisition strategy with gap analysis and phased roadmap. Outputs:
  - data_strategy.json (structured data)
  - data_strategy.md   (human-readable report)

Usage:
    python -m src.scripts.113_data_strategy --run-id <run_id>
    python -m src.scripts.113_data_strategy --run-id <run_id> --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_env

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIT_DATA_DIR = PROJECT_ROOT / "data" / "lit_review"


def _load_json(path: Path, label: str) -> dict | None:
    if not path.exists():
        logger.warning("%s not found at %s", label, path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", label, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="113 Data Strategy Planner")
    parser.add_argument("--run-id", required=True, help="Lit review run ID")
    parser.add_argument("--run-dir", help="Override run directory path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing LLM calls")
    args = parser.parse_args()

    # Setup
    load_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_id = args.run_id
    run_dir = Path(args.run_dir) if args.run_dir else LIT_DATA_DIR / run_id

    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    # Load required inputs
    registry = _load_json(run_dir / "dataset_registry.json", "dataset_registry.json")
    if not registry:
        logger.error("dataset_registry.json is required (run 111 first)")
        sys.exit(1)

    data_reqs = _load_json(run_dir / "data_requirements.json", "data_requirements.json")
    if not data_reqs:
        logger.error("data_requirements.json is required (run 091 first)")
        sys.exit(1)

    portfolio = _load_json(run_dir / "hypothesis_portfolio.json", "hypothesis_portfolio.json")
    if not portfolio:
        logger.error("hypothesis_portfolio.json is required (run 089 first)")
        sys.exit(1)

    n_datasets = len(registry.get("datasets", []))
    n_plans = len(data_reqs.get("data_plans", []))
    n_hyps = len(portfolio.get("scored_hypotheses", []))

    logger.info("Loaded: %d datasets, %d data plans, %d hypotheses", n_datasets, n_plans, n_hyps)

    if args.dry_run:
        summary = registry.get("summary", {})
        print(f"\n=== Data Strategy Planner — DRY RUN ===")
        print(f"Run ID:        {run_id}")
        print(f"Datasets:      {n_datasets} (open={summary.get('open', 0)}, commercial={summary.get('commercial', 0)})")
        print(f"Data plans:    {n_plans}")
        print(f"Hypotheses:    {n_hyps}")
        print(f"\nLLM calls:     1 (single batch)")
        print(f"\nTo execute: remove --dry-run")
        return

    # Build LLM client
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    from src.validation.data_strategy import DataStrategy, plan_strategy

    created_at = datetime.now(timezone.utc).isoformat()

    # Generate strategy (single LLM call)
    logger.info("Generating data strategy...")
    raw_strategy = plan_strategy(registry, data_reqs, portfolio, llm_client)

    if not raw_strategy:
        logger.error("Strategy generation failed")
        sys.exit(1)

    # Build result
    result = DataStrategy(
        run_id=run_id,
        created_at=created_at,
        gap_analysis=raw_strategy.get("gap_analysis", {}),
        acquisition_roadmap=raw_strategy.get("acquisition_roadmap", {}),
        recommendations=raw_strategy.get("recommendations", []),
    )

    # Write outputs
    result_path = run_dir / "data_strategy.json"
    result_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s", result_path)

    report_path = run_dir / "data_strategy.md"
    report_path.write_text(result.to_markdown(), encoding="utf-8")
    logger.info("Wrote %s", report_path)

    # Print summary
    ga = result.gap_analysis
    gaps = ga.get("critical_gaps", [])
    covered = ga.get("covered", [])
    coverage = ga.get("coverage_rate", 0)
    roadmap = result.acquisition_roadmap

    print(f"\n=== Data Strategy Complete ===")
    print(f"Run ID:        {run_id}")
    print(f"Coverage:      {coverage:.0%} ({len(covered)} covered, {len(gaps)} gaps)")
    print(f"Gaps:          {len(gaps)} critical")

    for phase_key in ["phase_1_immediate", "phase_2_short_term", "phase_3_medium_term", "phase_4_long_term"]:
        phase = roadmap.get(phase_key, {})
        ds_count = len(phase.get("datasets", []))
        cost = phase.get("estimated_cost", "-")
        print(f"  {phase_key}: {ds_count} datasets, cost={cost}")

    print(f"Recommendations: {len(result.recommendations)}")
    print(f"\nOutputs:")
    print(f"  {result_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
