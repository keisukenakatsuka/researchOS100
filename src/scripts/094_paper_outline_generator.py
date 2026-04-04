#!/usr/bin/env python
"""094 Paper Outline Generator — Block 6 outline generation.

Generates a structured paper outline (paper_outline.json + paper_outline.md)
from Block 2–5 pipeline outputs.  The JSON outline serves as the shared
reference for all section drafters (095–098).

Exit codes:
  0: outline generated successfully
  1: fatal error (missing inputs, LLM failure, etc.)

Usage::

    python -m src.scripts.094_paper_outline_generator --run-id <id>
    python -m src.scripts.094_paper_outline_generator --run-id <id> --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env
from src.lit_review.outline import generate_outline, _load_inputs
from src.lit_review.run_manifest import update_step

logger = logging.getLogger("094_paper_outline_generator")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"
_SCRIPT_ID = "094_paper_outline_generator"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.094_paper_outline_generator",
        description="094 Paper Outline Generator — structured outline for section drafters",
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

    logger.info("=== 094 Paper Outline Generator === run_id=%s", args.run_id)

    if args.dry_run:
        try:
            inputs = _load_inputs(run_dir)
            print(f"\nDRY RUN — available inputs:")
            rq = inputs.get("rq_context.json", {})
            print(f"  RQ: {rq.get('title', '(unknown)')}")
            for fname in inputs:
                print(f"  ✓ {fname}")
        except (FileNotFoundError, ValueError) as e:
            print(f"\n  ✗ {e}")
        sys.exit(0)

    # Update manifest: running
    update_step(run_dir, _SCRIPT_ID, status="running")

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = generate_outline(run_dir, llm_client=llm_client)

    # Update manifest
    if result.status == "generated":
        update_step(
            run_dir, _SCRIPT_ID,
            status="completed",
            outputs=["paper_outline.json", "paper_outline.md"],
        )
    else:
        update_step(
            run_dir, _SCRIPT_ID,
            status="failed",
            error=result.error,
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"094 Paper Outline Generator — {result.status.upper()}")
    print(f"{'=' * 60}")

    if result.status == "generated":
        outline = result.outline
        print(f"Sections: {len(outline.get('sections', []))}")
        print(f"Total target: {outline.get('total_target_words', 0):,} words")
        print(f"Outline MD: {result.word_count} words")
        print()
        for s in outline.get("sections", []):
            sid = s.get("section_id", "")
            target = s.get("target_words", 0)
            flow_count = len(s.get("argument_flow", []))
            refs_count = len(s.get("key_references", []))
            print(f"  {sid}: {target:,} words, {flow_count} steps, {refs_count} refs")

        warnings = result.metadata.get("warnings", [])
        if warnings:
            print(f"\nWarnings:")
            for w in warnings:
                print(f"  ⚠ {w}")

        print(f"\nOutputs: {run_dir}")
        print(f"  paper_outline.json")
        print(f"  paper_outline.md")
        sys.exit(0)
    else:
        print(f"Error: {result.error}")
        print(f"Retryable: {result.retryable}")
        sys.exit(1)


if __name__ == "__main__":
    main()
