# src/scripts/112_model_prototyper.py
"""Model Prototyper — generate baseline implementation blueprints.

Reads hypotheses, portfolio, method selection, and optionally dataset registry
from a lit_review run. Generates implementation blueprints for high-priority
hypotheses. Outputs:
  - model_blueprints.json (structured data)
  - model_blueprints.md   (human-readable report)

Usage:
    python -m src.scripts.112_model_prototyper --run-id <run_id>
    python -m src.scripts.112_model_prototyper --run-id <run_id> --max-blueprints 5
    python -m src.scripts.112_model_prototyper --run-id <run_id> --dry-run
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
    """Load a JSON file, returning None with warning if not found."""
    if not path.exists():
        logger.warning("%s not found at %s", label, path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", label, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="112 Model Prototyper")
    parser.add_argument("--run-id", required=True, help="Lit review run ID")
    parser.add_argument("--run-dir", help="Override run directory path")
    parser.add_argument("--max-blueprints", type=int, default=3,
                        help="Max hypotheses to generate blueprints for (default: 3)")
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
    portfolio = _load_json(run_dir / "hypothesis_portfolio.json", "hypothesis_portfolio.json")
    if not portfolio:
        logger.error("hypothesis_portfolio.json is required")
        sys.exit(1)

    hypotheses_data = _load_json(run_dir / "hypotheses.json", "hypotheses.json")
    method_data = _load_json(run_dir / "method_selection.json", "method_selection.json")

    # Load optional dataset registry (111 output)
    registry_data = _load_json(run_dir / "dataset_registry.json", "dataset_registry.json")

    # Select targets
    from src.validation.model_blueprint import select_targets, find_method_for_hypothesis
    targets = select_targets(portfolio, max_count=args.max_blueprints)

    if not targets:
        logger.error("No high_priority or promising hypotheses found in portfolio")
        sys.exit(1)

    # Build method selections lookup
    method_selections = []
    if method_data:
        method_selections = method_data.get("method_selections", [])
        logger.info("Loaded %d method selections", len(method_selections))

    # Build available datasets list
    available_datasets = []
    if registry_data:
        available_datasets = registry_data.get("datasets", [])
        logger.info("Loaded %d datasets from registry", len(available_datasets))
    else:
        logger.info("No dataset registry — blueprints will be generated without dataset info")

    if args.dry_run:
        print(f"\n=== Model Prototyper — DRY RUN ===")
        print(f"Run ID:            {run_id}")
        print(f"Max blueprints:    {args.max_blueprints}")
        print(f"Method selections: {len(method_selections)}")
        print(f"Available datasets: {len(available_datasets)} ({'from registry' if registry_data else 'none'})")
        print(f"\nTarget hypotheses ({len(targets)}):")
        for t in targets:
            hyp_id = t.get("hypothesis_id", "")
            stmt = t.get("statement", "")[:70]
            rec = t.get("recommendation", "")
            score = t.get("composite_score", 0)
            ms = find_method_for_hypothesis(hyp_id, method_selections)
            method = ms.get("primary_method", "none")[:50] if ms else "no method selection"
            print(f"  [{rec}] {hyp_id} (score={score})")
            print(f"    {stmt}...")
            print(f"    method: {method}")
        print(f"\nTo execute: remove --dry-run")
        return

    # Build LLM client
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    from src.validation.model_blueprint import (
        ModelBlueprints,
        generate_blueprint,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    blueprints = []

    for i, target in enumerate(targets):
        hyp_id = target.get("hypothesis_id", "")
        logger.info(
            "[%d/%d] Generating blueprint for %s (%s)",
            i + 1, len(targets), hyp_id, target.get("recommendation", ""),
        )

        # Find matching method selection
        ms = find_method_for_hypothesis(hyp_id, method_selections)

        bp = generate_blueprint(
            hypothesis=target,
            method_selection=ms,
            available_datasets=available_datasets,
            llm_client=llm_client,
        )
        blueprints.append(bp)

    result = ModelBlueprints(
        run_id=run_id,
        created_at=created_at,
        blueprints=blueprints,
    )

    # Write outputs
    result_path = run_dir / "model_blueprints.json"
    result_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s", result_path)

    report_path = run_dir / "model_blueprints.md"
    report_path.write_text(result.to_markdown(), encoding="utf-8")
    logger.info("Wrote %s", report_path)

    # Print summary
    print(f"\n=== Model Prototyper Complete ===")
    print(f"Run ID:      {run_id}")
    print(f"Blueprints:  {len(blueprints)}")
    for bp in blueprints:
        print(f"  [{bp.estimated_complexity}] {bp.hypothesis_id}: {bp.recommended_method}")
    print(f"\nOutputs:")
    print(f"  {result_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
