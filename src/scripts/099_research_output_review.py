#!/usr/bin/env python
"""099 Research Output Review — Block 6 cross-section review.

Reviews all generated drafts for quality, consistency, and L2 readiness.

Exit codes:
  0: review completed
  1: fatal error

Usage::

    python -m src.scripts.099_research_output_review --run-id <id>
    python -m src.scripts.099_research_output_review --run-id <id> --dry-run
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
from src.lit_review.reviewer import run_review, render_review_markdown
from src.lit_review.run_manifest import update_step

logger = logging.getLogger("099_research_output_review")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"
_SCRIPT_ID = "099_research_output_review"

_DRAFT_FILES = {
    "introduction": "draft_introduction.md",
    "literature_review": "draft_literature_review.md",
    "hypotheses": "draft_hypotheses.md",
    "methods": "draft_methods.md",
}


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.099_research_output_review",
        description="099 Research Output Review — cross-section quality review",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--dry-run", action="store_true", help="Show available drafts without reviewing")
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

    logger.info("=== 099 Research Output Review === run_id=%s", args.run_id)

    if args.dry_run:
        print(f"\nDRY RUN — available drafts:")
        available = 0
        for sid, fname in _DRAFT_FILES.items():
            exists = (run_dir / fname).exists()
            status = "\u2713" if exists else "\u2717"
            print(f"  {status} {fname}")
            if exists:
                available += 1

        outline = (run_dir / "paper_outline.json").exists()
        outline_icon = "\u2713" if outline else "\u2717"
        print(f"\n  paper_outline.json: {outline_icon}")
        print(f"\n  Drafts available: {available}/4")
        if available >= 3:
            print(f"  Would generate: review_report.json + review_report.md")
        else:
            print(f"  Need at least 3 drafts (intro + hypotheses + methods)")
        sys.exit(0)

    # Check minimum drafts
    available_count = sum(1 for f in _DRAFT_FILES.values() if (run_dir / f).exists())
    required = ["draft_introduction.md", "draft_hypotheses.md", "draft_methods.md"]
    missing_required = [f for f in required if not (run_dir / f).exists()]
    if missing_required:
        logger.error("Missing required drafts: %s", missing_required)
        logger.error("Run 095, 096, 097 first")
        sys.exit(1)

    # Update manifest: running
    update_step(run_dir, _SCRIPT_ID, status="running")

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = run_review(run_dir, llm_client=llm_client)

    if result.status == "generated":
        # Save JSON
        json_path = run_dir / "review_report.json"
        json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

        # Save Markdown
        md_text = render_review_markdown(result)
        md_path = run_dir / "review_report.md"
        md_path.write_text(md_text)

        update_step(
            run_dir, _SCRIPT_ID,
            status="completed",
            outputs=["review_report.json", "review_report.md"],
        )

        # Summary
        print(f"\n{'=' * 60}")
        print(f"099 Research Output Review — COMPLETED")
        print(f"{'=' * 60}")
        print(f"RQ: {result.rq_title[:60]}...")

        # Section summary
        total = 0
        for s in result.sections:
            if s.available:
                icon = "\u2713" if s.meets_target else "\u26a0"
                print(f"  {icon} {s.section_id}: {s.word_count} words (ratio={s.word_ratio:.2f}, {s.citation_count} cites)")
                total += s.word_count
            else:
                print(f"  \u2717 {s.section_id}: MISSING")
        print(f"\n  Total: {total} words")

        # Cross-section checks
        print(f"\nCross-section checks:")
        for c in result.cross_section_checks:
            icon = "\u2713" if c.passed else "\u2717"
            print(f"  {icon} {c.check_name}")

        # L2 assessment
        l2_pass = sum(1 for a in result.l2_assessment if a.passed)
        l2_total = len(result.l2_assessment)
        print(f"\nL2 Working Draft: {'PASSED' if result.l2_passed else 'NOT YET'} ({l2_pass}/{l2_total} criteria)")
        for a in result.l2_assessment:
            icon = "\u2713" if a.passed else "\u2717"
            print(f"  {icon} {a.criterion}")

        print(f"\nOverall Quality Score: {result.overall_quality_score}/10")
        print(f"\nOutputs: {run_dir}")
        print(f"  review_report.json")
        print(f"  review_report.md")
        sys.exit(0)
    else:
        update_step(run_dir, _SCRIPT_ID, status="failed", error=result.error)
        print(f"Error: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
