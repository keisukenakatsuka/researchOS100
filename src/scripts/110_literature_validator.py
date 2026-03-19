# src/scripts/110_literature_validator.py
"""Literature Validator — validate Evidence items against source papers.

Reads evidence.json from a lit_review run and checks each Evidence item
against the original paper text for alignment accuracy. Outputs:
  - validation_result.json (structured data)
  - validation_report.md   (human-readable report)

Usage:
    python -m src.scripts.110_literature_validator --run-id <run_id>
    python -m src.scripts.110_literature_validator --run-id <run_id> --pass2-all
    python -m src.scripts.110_literature_validator --run-id <run_id> --dry-run
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
    parser = argparse.ArgumentParser(description="110 Literature Validator")
    parser.add_argument("--run-id", required=True, help="Lit review run ID")
    parser.add_argument("--run-dir", help="Override run directory path")
    parser.add_argument("--pass2-all", action="store_true",
                        help="Run Pass 2 on ALL evidence (not just uncertain/contradiction)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing LLM calls")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Evidence items per LLM batch call (default: 5)")
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

    # Load candidate_papers for paper_id resolution and text retrieval
    from src.validation.grounding import load_candidate_papers, group_evidence_by_paper
    candidate_papers = load_candidate_papers(run_dir)
    logger.info("Loaded %d candidate papers", len(candidate_papers))

    # Group evidence by paper
    paper_groups = group_evidence_by_paper(evidence_items)
    logger.info("Evidence grouped into %d papers", len(paper_groups))

    if args.dry_run:
        print(f"\n=== Literature Validator — DRY RUN ===")
        print(f"Run ID:          {run_id}")
        print(f"Evidence items:  {len(evidence_items)}")
        print(f"Papers:          {len(paper_groups)}")
        print(f"Pass 2 mode:     {'all' if args.pass2_all else 'uncertain/contradiction only'}")
        print(f"Batch size:      {args.batch_size}")
        print(f"\nPapers to process:")
        for title, items in paper_groups.items():
            print(f"  [{len(items):3d} evidence] {title[:70]}")
        print(f"\nTo execute: remove --dry-run")
        return

    # Build LLM client
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    # Import validation modules
    from src.validation.ids import evidence_id, resolve_paper_id
    from src.validation.grounding import retrieve_paper_text
    from src.validation.validator import (
        ValidationItem, ValidationResult, Pass1Result,
        align_evidence_batch, check_consistency,
        assign_validation_status, needs_pass2,
    )

    validated_at = datetime.now(timezone.utc).isoformat()
    validation_items: list[ValidationItem] = []
    batch_size = args.batch_size

    # Process each paper group
    for paper_idx, (paper_title, items) in enumerate(paper_groups.items()):
        logger.info(
            "[%d/%d] Processing paper: '%s' (%d evidence items)",
            paper_idx + 1, len(paper_groups), paper_title[:60], len(items),
        )

        # Retrieve paper text
        paper_text, text_source = retrieve_paper_text(paper_title, candidate_papers)
        logger.info("  Text source: %s (%d chars)", text_source, len(paper_text))

        # Process in batches
        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start:batch_start + batch_size]

            # Pass 1: Alignment
            pass1_results = align_evidence_batch(batch, paper_text, llm_client)

            # Process each item in batch
            for ev_item, p1 in zip(batch, pass1_results):
                ev_id = evidence_id(
                    ev_item.get("claim_or_point", ""),
                    ev_item.get("paper_title", ""),
                )
                paper_id = resolve_paper_id(ev_item, candidate_papers)

                # Pass 2: Consistency check (conditional)
                p2 = None
                if args.pass2_all or needs_pass2(p1, text_source):
                    p2 = check_consistency(ev_item, paper_text, p1, llm_client)

                # Assign final status
                status, score, needs_review = assign_validation_status(p1, p2, text_source)

                validation_items.append(ValidationItem(
                    evidence_id=ev_id,
                    claim_or_point=ev_item.get("claim_or_point", ""),
                    paper_id=paper_id,
                    paper_title=paper_title,
                    text_source_used=text_source,
                    pass1=p1,
                    pass2=p2,
                    final_status=status,
                    final_alignment_score=score,
                    needs_human_review=needs_review,
                ))

    # Build result
    result = ValidationResult(
        run_id=run_id,
        source_evidence_file=str(evidence_path),
        validated_at=validated_at,
        model=_model_name(),
        validations=validation_items,
    )

    # Write outputs
    result_path = run_dir / "validation_result.json"
    result_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s", result_path)

    report_path = run_dir / "validation_report.md"
    report_path.write_text(result.to_markdown(), encoding="utf-8")
    logger.info("Wrote %s", report_path)

    # Print summary
    s = result.summary
    print(f"\n=== Literature Validation Complete ===")
    print(f"Run ID:        {run_id}")
    print(f"Total:         {s['total']}")
    print(f"Verified:      {s['verified']}")
    print(f"Uncertain:     {s['uncertain']}")
    print(f"Contradiction: {s['contradiction']}")
    print(f"Unverifiable:  {s['unverifiable']}")
    print(f"Coverage:      {s['coverage']:.1%}")
    print(f"\nOutputs:")
    print(f"  {result_path}")
    print(f"  {report_path}")


def _model_name() -> str:
    from src.validation.validator import _MODEL
    return _MODEL


if __name__ == "__main__":
    main()
