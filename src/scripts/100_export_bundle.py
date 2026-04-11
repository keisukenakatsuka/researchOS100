#!/usr/bin/env python
"""100 Export Bundle — Block 6 final assembly.

Assembles all drafts into an exportable research bundle.

Exit codes:
  0: bundle exported
  1: fatal error

Usage::

    python -m src.scripts.100_export_bundle --run-id <id>
    python -m src.scripts.100_export_bundle --run-id <id> --dry-run
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
from src.lit_review.exporter import export_bundle, check_quality_gate, _SECTIONS, _BUNDLE_FILES
from src.lit_review.run_manifest import update_step

logger = logging.getLogger("100_export_bundle")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"
_SCRIPT_ID = "100_export_bundle"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.100_export_bundle",
        description="100 Export Bundle — assemble research output bundle",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--dry-run", action="store_true", help="Show available files without exporting")
    p.add_argument("--skip-quality-gate", action="store_true",
                   help="Skip quality gate check (export regardless of score)")
    p.add_argument("--min-quality-score", type=float, default=7.0,
                   help="Minimum quality score to pass gate (default: 7.0)")
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
        sys.exit(1)

    logger.info("=== 100 Export Bundle === run_id=%s", args.run_id)

    if args.dry_run:
        print(f"\nDRY RUN — available files:")
        print(f"\n  Draft sections:")
        for sid, fname in _SECTIONS:
            exists = (run_dir / fname).exists()
            status = "\u2713" if exists else "\u2717"
            print(f"    {status} {fname}")

        print(f"\n  Supporting files:")
        for fname in _BUNDLE_FILES:
            exists = (run_dir / fname).exists()
            status = "\u2713" if exists else "\u2717"
            print(f"    {status} {fname}")

        print(f"\n  Would generate:")
        print(f"    paper_draft.md")
        print(f"    export_bundle.json")
        sys.exit(0)

    # Quality gate check
    if not args.skip_quality_gate:
        gate = check_quality_gate(run_dir, min_score=args.min_quality_score)
        if not gate.passed:
            # Write gate result
            gate_path = run_dir / "quality_gate_result.json"
            gate_path.write_text(json.dumps(gate.to_dict(), ensure_ascii=False, indent=2))

            print(f"\n{'=' * 60}")
            print(f"100 Export Bundle — BLOCKED by Quality Gate")
            print(f"{'=' * 60}")
            print(f"Score: {gate.score:.1f} / {gate.min_required:.1f} required")
            print(f"\nBlocking issues ({len(gate.blocking_issues)}):")
            for issue in gate.blocking_issues:
                print(f"  - [{issue.get('section', '')}] {issue.get('issue', '')}")
            print(f"\nSuggestions:")
            for s in gate.suggestions:
                print(f"  - {s}")
            print(f"\nTo bypass: --skip-quality-gate")
            print(f"To adjust: --min-quality-score {gate.score:.0f}")

            update_step(run_dir, _SCRIPT_ID, status="failed",
                        error=f"Quality gate: {gate.score:.1f} < {gate.min_required:.1f}")
            sys.exit(1)
        else:
            logger.info("Quality gate passed: %.1f >= %.1f", gate.score, gate.min_required)

    # Update manifest: running
    update_step(run_dir, _SCRIPT_ID, status="running")

    result = export_bundle(run_dir)

    if result.status == "generated":
        update_step(
            run_dir, _SCRIPT_ID,
            status="completed",
            outputs=["paper_draft.md", "export_bundle.json"],
        )

        meta = result.metadata
        print(f"\n{'=' * 60}")
        print(f"100 Export Bundle — COMPLETED")
        print(f"{'=' * 60}")
        print(f"RQ: {meta.get('rq_title', '')[:60]}...")
        print(f"\nPaper: {result.paper_word_count} words, {meta.get('citation_count', 0)} citations")
        print(f"Quality score: {meta.get('quality_score', 0)}/10")
        print(f"Pipeline version: {meta.get('pipeline_version', '')}")

        print(f"\nSections:")
        for sid, wc in meta.get("section_word_counts", {}).items():
            icon = "\u2713" if sid in result.sections_included else "\u2717"
            print(f"  {icon} {sid}: {wc} words")

        if result.sections_missing:
            print(f"\nMissing: {result.sections_missing}")

        print(f"\nBundle: {run_dir}")
        print(f"  paper_draft.md")
        print(f"  export_bundle.json")
        for fname in _BUNDLE_FILES:
            if (run_dir / fname).exists():
                print(f"  {fname}")
        sys.exit(0)
    else:
        update_step(run_dir, _SCRIPT_ID, status="failed", error=result.error)
        print(f"Error: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
