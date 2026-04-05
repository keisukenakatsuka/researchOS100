#!/usr/bin/env python
"""095 Introduction Drafter — Block 6 section draft.

Generates draft_introduction.md following the outline from 094.

Exit codes:
  0: draft generated
  1: fatal error

Usage::

    python -m src.scripts.095_introduction_drafter --run-id <id>
    python -m src.scripts.095_introduction_drafter --run-id <id> --dry-run
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
from src.lit_review.drafters.base import DraftStatus
from src.lit_review.drafters.introduction import IntroductionDrafter
from src.lit_review.run_manifest import update_step

logger = logging.getLogger("095_introduction_drafter")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"
_SCRIPT_ID = "095_introduction_drafter"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.095_introduction_drafter",
        description="095 Introduction Drafter — generates draft_introduction.md",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--dry-run", action="store_true", help="Show available inputs without generating")
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

    logger.info("=== 095 Introduction Drafter === run_id=%s", args.run_id)

    drafter = IntroductionDrafter()

    if args.dry_run:
        # Check required inputs
        missing = []
        for fname in drafter.required_inputs():
            if fname == "paper_outline.json":
                path = run_dir / fname
            else:
                path = run_dir / fname
            exists = path.exists()
            status = "\u2713" if exists else "\u2717"
            print(f"  {status} {fname}")
            if not exists:
                missing.append(fname)

        # Show outline spec if available
        outline_path = run_dir / "paper_outline.json"
        if outline_path.exists():
            outline = json.loads(outline_path.read_text())
            for section in outline.get("sections", []):
                if section.get("section_id") == "introduction":
                    print(f"\n  Outline spec:")
                    print(f"    target_words: {section.get('target_words', 'N/A')}")
                    flow = section.get("argument_flow", [])
                    for i, step in enumerate(flow, 1):
                        print(f"    {i}. {step}")

        if missing:
            print(f"\n  Missing required inputs: {missing}")
            print(f"  Run upstream steps first.")
        else:
            print(f"\n  Would generate: draft_introduction.md")
        sys.exit(0)

    # Update manifest: running
    update_step(run_dir, _SCRIPT_ID, status="running")

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = drafter.generate(run_dir, llm_client=llm_client)

    # Update manifest
    if result.status == DraftStatus.GENERATED:
        update_step(
            run_dir, _SCRIPT_ID,
            status="completed",
            outputs=["draft_introduction.md"],
        )
    else:
        update_step(
            run_dir, _SCRIPT_ID,
            status="failed",
            error=result.error,
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"095 Introduction Drafter — {result.status.value.upper()}")
    print(f"{'=' * 60}")

    if result.status == DraftStatus.GENERATED:
        diag = result.diagnostics
        print(f"Words: {result.word_count}")
        print(f"Target: {diag.target_words}")
        print(f"Ratio: {diag.word_ratio:.2f}")
        print(f"Meets target: {diag.meets_target}")
        print(f"Citations: {diag.has_citations}")

        if diag.warnings:
            print(f"\nWarnings:")
            for w in diag.warnings:
                print(f"  \u26a0 {w}")

        if result.metadata:
            print(f"\nLLM: in={result.metadata.get('input_tokens', 0)} out={result.metadata.get('output_tokens', 0)} tokens")

        print(f"\nOutput: {run_dir / 'draft_introduction.md'}")
        print(f"\nNext steps:")
        print(f"  python -m src.scripts.096_hypotheses_drafter --run-id {args.run_id}")
        print(f"  python -m src.scripts.097_methods_drafter --run-id {args.run_id}")
        print(f"  python -m src.scripts.098_literature_review_drafter --run-id {args.run_id}")
        sys.exit(0)
    else:
        print(f"Error: {result.error}")
        print(f"Retryable: {result.retryable}")
        if result.diagnostics.warnings:
            for w in result.diagnostics.warnings:
                print(f"  \u26a0 {w}")
        sys.exit(1)


if __name__ == "__main__":
    main()
