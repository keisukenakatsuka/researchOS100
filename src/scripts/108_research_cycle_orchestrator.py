#!/usr/bin/env python
"""108 Research Cycle Orchestrator — end-to-end pipeline runner.

Executes the full Research Cycle (079-106) from a single RQ input.
Supports resume, dry-run, and selective execution.

Usage::

    # New run from RQ text
    python -m src.scripts.108_research_cycle_orchestrator \\
        --rq-text "CVCはなぜ期待通りに機能しないのか"

    # Resume existing run
    python -m src.scripts.108_research_cycle_orchestrator \\
        --run-id 20260318_143022_a1b2c3d4

    # Dry-run (show execution plan only)
    python -m src.scripts.108_research_cycle_orchestrator \\
        --rq-text "..." --dry-run

    # Stop after hypothesis phase
    python -m src.scripts.108_research_cycle_orchestrator \\
        --rq-text "..." --stop-after 092

    # Include optional steps (080, 084)
    python -m src.scripts.108_research_cycle_orchestrator \\
        --rq-text "..." --include-optional

Exit codes:
    0: all steps completed
    1: fatal error (step failure without --continue-on-error)
    2: partial success (some steps failed with --continue-on-error)
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

from src.orchestrator.research_cycle import (
    PHASE_LABELS,
    PIPELINE_STEPS,
    LIT_DATA_DIR,
    build_execution_plan,
    run_pipeline,
    print_summary,
)

logger = logging.getLogger("108_research_cycle_orchestrator")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.108_research_cycle_orchestrator",
        description="Research Cycle Orchestrator — run the full pipeline (079-106) from a single RQ.",
    )

    # Input mode (mutually exclusive)
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--rq-text", type=str,
        help="Research Question text (starts new run)",
    )
    input_group.add_argument(
        "--rq-id", type=str,
        help="Notion RQ page ID (starts new run)",
    )
    input_group.add_argument(
        "--run-id", type=str,
        help="Existing run ID to resume",
    )

    # Pipeline control
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show execution plan without running",
    )
    p.add_argument(
        "--include-optional", action="store_true",
        help="Include optional steps (080 Gap Filler, 084 Writeback)",
    )
    p.add_argument(
        "--stop-after", type=str, default=None,
        help="Stop after this step (e.g., 092)",
    )
    p.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated steps to skip (e.g., 098,106)",
    )
    p.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue pipeline on step failure",
    )

    # 079 parameters
    p.add_argument("--min-score", type=int, default=65, help="079 min relevance score (default: 65)")
    p.add_argument("--max-papers", type=int, default=20, help="079 max papers (default: 20)")

    # General
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")

    return p.parse_args()


def _print_plan(plan, *, rq_text="", rq_id="", run_id="", skipped_optional=None):
    """Display the execution plan."""
    sep = "=" * 55

    mode = "DRY RUN" if not run_id else "DRY RUN (Resume)"
    print(sep)
    print(f"  Research Cycle Orchestrator — {mode}" if "DRY" in mode
          else "  Research Cycle Orchestrator")

    if rq_text:
        # Truncate long RQs for display
        display_rq = rq_text if len(rq_text) <= 60 else rq_text[:57] + "..."
        print(f"  RQ: {display_rq}")
    if run_id:
        print(f"  Run ID: {run_id}")
        # Try to load RQ title from existing run
        rq_path = LIT_DATA_DIR / run_id / "rq_context.json"
        if rq_path.exists():
            rq_data = json.loads(rq_path.read_text())
            title = rq_data.get("title", "")
            if title:
                print(f"  RQ: {title}")

    print(sep)
    print(f"\nExecution Plan ({len(plan)} steps):\n")

    current_phase = ""
    for i, step in enumerate(plan, 1):
        if step.phase != current_phase:
            current_phase = step.phase
            phase_label = PHASE_LABELS.get(current_phase, current_phase)
            print(f"  {phase_label}")
        print(f"    [{i:2d}] {step.name}  {step.description}")

    if skipped_optional:
        print(f"\n  Skipped (optional):")
        for step in skipped_optional:
            print(f"    {step.name}  {step.description}")

    print()


def main() -> None:
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse --skip
    skip_steps = [s.strip() for s in args.skip.split(",") if s.strip()] if args.skip else []

    # Build execution plan
    plan = build_execution_plan(
        include_optional=args.include_optional,
        skip_steps=skip_steps,
        stop_after=args.stop_after,
    )

    if not plan:
        print("No steps to execute.")
        sys.exit(0)

    # Identify skipped optional steps for display
    skipped_optional = [s for s in PIPELINE_STEPS if s.optional and not args.include_optional]

    # Validate --run-id exists
    if args.run_id:
        run_dir = LIT_DATA_DIR / args.run_id
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)

    # Dry-run: show plan and exit
    if args.dry_run:
        _print_plan(
            plan,
            rq_text=args.rq_text or "",
            rq_id=args.rq_id or "",
            run_id=args.run_id or "",
            skipped_optional=skipped_optional,
        )
        print("To execute: remove --dry-run")
        sys.exit(0)

    # Print header
    sep = "=" * 55
    print(sep)
    print("  Research Cycle Orchestrator")
    rq_display = args.rq_text or args.rq_id or ""
    if args.run_id:
        print(f"  Resuming: {args.run_id}")
        rq_path = LIT_DATA_DIR / args.run_id / "rq_context.json"
        if rq_path.exists():
            rq_data = json.loads(rq_path.read_text())
            rq_display = rq_data.get("title", "")
    if rq_display:
        display_rq = rq_display if len(rq_display) <= 60 else rq_display[:57] + "..."
        print(f"  RQ: {display_rq}")
    print(sep)

    # Execute pipeline
    result = run_pipeline(
        plan,
        run_id=args.run_id or "",
        rq_text=args.rq_text or "",
        rq_id=args.rq_id or "",
        min_score=args.min_score,
        max_papers=args.max_papers,
        continue_on_error=args.continue_on_error,
    )

    # Summary
    print_summary(result)

    # Exit code
    if result.success:
        sys.exit(0)
    elif args.continue_on_error and result.completed > 0:
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
