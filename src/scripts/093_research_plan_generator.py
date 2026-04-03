#!/usr/bin/env python
"""093 Research Plan Generator — Block 6 research plan.

Generates research_plan.md from Block 2–5 pipeline outputs.

As of v1 (094–100 split), this script generates ONLY the research plan.
Outline generation → 094_paper_outline_generator
Section drafts    → 095–098
Review            → 099_research_output_review
Export            → 100_export_bundle

Exit codes:
  0: research plan generated
  1: fatal error

Usage::

    python -m src.scripts.093_research_plan_generator --run-id <id>
    python -m src.scripts.093_research_plan_generator --run-id <id> --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env
from src.lit_review.research_output import (
    generate_research_plan,
    collect_all_inputs,
)
from src.lit_review.run_manifest import update_step

logger = logging.getLogger("093_research_plan_generator")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"
_SCRIPT_ID = "093_research_plan_generator"

# Deprecated options that have moved to 094–098
_DEPRECATED_OPTIONS = {
    "--plan-only": "093 now generates plan only by default. This flag is no longer needed.",
    "--outline-only": "Outline generation has moved to 094_paper_outline_generator.",
    "--drafts-only": "Section drafts have moved to 095–098.",
    "--section": "Section drafts have moved to 095–098.",
}


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.093_research_plan_generator",
        description="093 Research Plan Generator — generates research_plan.md",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--dry-run", action="store_true", help="Show available inputs without generating")
    p.add_argument("-v", "--verbose", action="store_true")

    # Keep deprecated flags to emit helpful warnings instead of argparse errors
    p.add_argument("--plan-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--outline-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--drafts-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--section", type=str, default=None, help=argparse.SUPPRESS)

    return p.parse_args()


def _check_deprecated(args) -> bool:
    """Warn about deprecated options. Returns True if any were used."""
    used = False
    for flag, msg in _DEPRECATED_OPTIONS.items():
        attr = flag.lstrip("-").replace("-", "_")
        val = getattr(args, attr, None)
        if val:
            used = True
            warnings.warn(
                f"DeprecationWarning: {flag} is deprecated. {msg}",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning("DEPRECATED: %s — %s", flag, msg)
    return used


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    load_env()

    # Check deprecated options
    deprecated_used = _check_deprecated(args)

    # Block execution if outline/drafts/section requested (moved to 094–098)
    if args.outline_only or args.drafts_only or args.section:
        print(f"\n{'=' * 60}")
        print(f"093 — DEPRECATED OPTIONS USED")
        print(f"{'=' * 60}")
        if args.outline_only:
            print(f"\n  --outline-only has moved to 094:")
            print(f"    python -m src.scripts.094_paper_outline_generator --run-id {args.run_id}")
        if args.drafts_only or args.section:
            print(f"\n  Section drafts have moved to 095–098:")
            print(f"    python -m src.scripts.095_introduction_drafter --run-id {args.run_id}")
            print(f"    python -m src.scripts.096_hypotheses_drafter --run-id {args.run_id}")
            print(f"    python -m src.scripts.097_methods_drafter --run-id {args.run_id}")
            print(f"    python -m src.scripts.098_literature_review_drafter --run-id {args.run_id}")
        sys.exit(1)

    run_dir = _DATA_DIR / args.run_id
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    # Check minimum required inputs
    required = ["rq_context.json", "lit_review.json", "hypotheses.json"]
    missing = [f for f in required if not (run_dir / f).exists()]
    if missing:
        logger.error("Missing required inputs: %s", missing)
        logger.error("Run 079 --full-pipeline and 087 first")
        sys.exit(1)

    logger.info("=== 093 Research Plan Generator === run_id=%s", args.run_id)

    if args.dry_run:
        inputs = collect_all_inputs(run_dir)
        print(f"\nDRY RUN — available inputs:")
        print(f"  RQ: {inputs['summary']['rq_title']}")
        for fname, avail in inputs["available"].items():
            status = "\u2713" if avail else "\u2717"
            print(f"  {status} {fname}")
        print(f"\nWould generate: research_plan.md")
        sys.exit(0)

    # Update manifest: running
    update_step(run_dir, _SCRIPT_ID, status="running")

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    inputs = collect_all_inputs(run_dir)
    plan_text = generate_research_plan(inputs, llm_client=llm_client)

    if plan_text:
        (run_dir / "research_plan.md").write_text(plan_text)
        word_count = len(plan_text.split())
        logger.info("Saved research_plan.md (%d words)", word_count)

        update_step(
            run_dir, _SCRIPT_ID,
            status="completed",
            outputs=["research_plan.md"],
        )

        print(f"\n{'=' * 60}")
        print(f"093 Research Plan Generator — COMPLETED")
        print(f"{'=' * 60}")
        print(f"RQ: {inputs['summary']['rq_title']}")
        print(f"Words: {word_count}")
        print(f"\nOutput: {run_dir / 'research_plan.md'}")
        print(f"\nNext step:")
        print(f"  python -m src.scripts.094_paper_outline_generator --run-id {args.run_id}")
        sys.exit(0)
    else:
        update_step(
            run_dir, _SCRIPT_ID,
            status="failed",
            error="LLM returned empty response",
        )
        logger.error("Failed to generate research plan")
        sys.exit(1)


if __name__ == "__main__":
    main()
