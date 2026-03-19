# src/scripts/111_dataset_registry.py
"""Dataset Registry Builder — identify and assess datasets from literature.

Reads evidence.json (and optionally data_requirements.json) from a
lit_review run, extracts dataset mentions, assesses availability,
and outputs:
  - dataset_registry.json (structured data)
  - dataset_registry.md   (human-readable report)

Usage:
    python -m src.scripts.111_dataset_registry --run-id <run_id>
    python -m src.scripts.111_dataset_registry --run-id <run_id> --skip-search
    python -m src.scripts.111_dataset_registry --run-id <run_id> --dry-run
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


def main() -> None:
    parser = argparse.ArgumentParser(description="111 Dataset Registry Builder")
    parser.add_argument("--run-id", required=True, help="Lit review run ID")
    parser.add_argument("--run-dir", help="Override run directory path")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip Web search for availability assessment (LLM knowledge only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing LLM calls")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Datasets per assessment LLM call (default: 5)")
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

    # Load evidence.json
    evidence_path = run_dir / "evidence.json"
    if not evidence_path.exists():
        logger.error("evidence.json not found at %s", evidence_path)
        sys.exit(1)

    evidence_data = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_items = evidence_data.get("evidence_items", [])
    if not evidence_items:
        logger.error("No evidence items found in evidence.json")
        sys.exit(1)

    logger.info("Loaded %d evidence items from %s", len(evidence_items), evidence_path)

    # Load data_requirements.json (optional — 091 seed)
    data_reqs_path = run_dir / "data_requirements.json"
    data_requirements = None
    if data_reqs_path.exists():
        try:
            data_requirements = json.loads(data_reqs_path.read_text(encoding="utf-8"))
            n_plans = len(data_requirements.get("data_plans", []))
            logger.info("Loaded data_requirements.json (%d plans)", n_plans)
        except Exception as e:
            logger.warning("Failed to load data_requirements.json: %s", e)
    else:
        logger.info("No data_requirements.json found — proceeding without 091 seeds")

    if args.dry_run:
        print(f"\n=== Dataset Registry Builder — DRY RUN ===")
        print(f"Run ID:           {run_id}")
        print(f"Evidence items:   {len(evidence_items)}")
        print(f"091 seeds:        {'Yes' if data_requirements else 'No'}")
        if data_requirements:
            # Count unique sources from 091
            from src.validation.dataset_registry import _extract_091_seeds
            seeds = _extract_091_seeds(data_requirements)
            print(f"091 seed sources: {len(seeds)}")
        print(f"Skip search:      {args.skip_search}")
        print(f"\nTo execute: remove --dry-run")
        return

    # Build LLM client
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    # Import modules
    from src.validation.dataset_registry import (
        DatasetRegistry,
        extract_dataset_mentions,
        assess_and_enrich,
    )

    created_at = datetime.now(timezone.utc).isoformat()

    # Step 1: Extract dataset mentions
    logger.info("Step 1: Extracting dataset mentions...")
    datasets = extract_dataset_mentions(evidence_items, data_requirements, llm_client)
    logger.info("Extracted %d unique datasets", len(datasets))

    # Step 2+3: Assess availability and discover alternatives
    logger.info("Step 2: Assessing availability...")
    datasets = assess_and_enrich(datasets, llm_client, batch_size=args.batch_size)

    # Resolve paper_ids from candidate_papers
    _resolve_paper_ids(datasets, run_dir)

    # Build registry
    registry = DatasetRegistry(
        run_id=run_id,
        created_at=created_at,
        datasets=datasets,
    )

    # Identify open data shortcuts
    open_ds = [d for d in datasets if d.availability_status == "open"]
    if open_ds:
        registry.open_data_shortcuts = [
            {
                "name": d.name,
                "url": d.access_url,
                "relevance": d.description,
                "dataset_ids": [d.dataset_id],
            }
            for d in open_ds if d.access_url
        ]

    # Write outputs
    result_path = run_dir / "dataset_registry.json"
    result_path.write_text(
        json.dumps(registry.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s", result_path)

    report_path = run_dir / "dataset_registry.md"
    report_path.write_text(registry.to_markdown(), encoding="utf-8")
    logger.info("Wrote %s", report_path)

    # Print summary
    s = registry.summary
    print(f"\n=== Dataset Registry Complete ===")
    print(f"Run ID:      {run_id}")
    print(f"Total:       {s['total_datasets']}")
    print(f"Open:        {s['open']}")
    print(f"Restricted:  {s['restricted']}")
    print(f"Commercial:  {s['commercial']}")
    print(f"Unavailable: {s['unavailable']}")
    print(f"\nOutputs:")
    print(f"  {result_path}")
    print(f"  {report_path}")


def _resolve_paper_ids(datasets: list, run_dir: Path) -> None:
    """Resolve paper_title → paper_id from candidate_papers.json."""
    from src.validation.grounding import load_candidate_papers
    candidates = load_candidate_papers(run_dir)
    if not candidates:
        return

    title_to_id = {p.get("title", "").strip().lower(): p.get("paper_id") for p in candidates}

    for ds in datasets:
        for ref in ds.mentioned_in_papers:
            if ref.get("paper_id"):
                continue
            title = ref.get("paper_title", "").strip().lower()
            if title in title_to_id:
                ref["paper_id"] = title_to_id[title]


if __name__ == "__main__":
    main()
