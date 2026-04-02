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
    PHASE_0_STEPS,
    PHASE_2B_STEPS,
    PHASE_3B_STEPS,
    LIT_DATA_DIR,
    build_execution_plan,
    build_execution_plan_v2,
    run_pipeline,
    print_summary,
    GateResult,
    check_phase_0_gate,
    check_phase_2_gate,
    print_gate_result,
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

    # === Phase 0: Data Source Audit (v2) ===
    p.add_argument(
        "--data-dir", type=str, default=None,
        help="CB Insights CSV directory. Enables v2 pipeline (Phase 0/2b/3b). "
             "Omit for v1-compatible run.",
    )
    p.add_argument(
        "--gvc-names", type=str, default=None,
        help="Comma-separated GVC names for co-investment detection",
    )
    p.add_argument(
        "--dv-candidates", type=str, default=None,
        help="Comma-separated DV candidate names (default: all)",
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
        "--stop-after-phase", type=int, default=None,
        help="Stop after phase N (0=audit, 1=hypothesis, 2=data, 3=output+empirical)",
    )
    p.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated steps to skip (e.g., 098,106)",
    )
    p.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue pipeline on step failure",
    )
    p.add_argument(
        "--skip-empirical", action="store_true",
        help="Skip Phase 3b (regression analysis)",
    )

    # Delivery
    p.add_argument(
        "--send-kindle", action="store_true",
        help="Send review bundle to Kindle after completion",
    )

    # 079 parameters
    p.add_argument("--min-score", type=int, default=65, help="079 min relevance score (default: 65)")
    p.add_argument("--max-papers", type=int, default=20, help="079 max papers (default: 20)")

    # Timeout
    p.add_argument("--step-timeout", type=int, default=2400,
                    help="Per-step timeout in seconds (default: 2400)")

    # General
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")

    return p.parse_args()


def _print_plan(plan, *, rq_text="", rq_id="", run_id="", data_dir="",
                skipped_optional=None, is_v2=False):
    """Display the execution plan."""
    sep = "=" * 55

    mode = "DRY RUN" if not run_id else "DRY RUN (Resume)"
    version = " (v2)" if is_v2 else ""
    print(sep)
    print(f"  Research Cycle Orchestrator — {mode}{version}" if "DRY" in mode
          else f"  Research Cycle Orchestrator{version}")

    if rq_text:
        display_rq = rq_text if len(rq_text) <= 60 else rq_text[:57] + "..."
        print(f"  RQ: {display_rq}")
    if run_id:
        print(f"  Run ID: {run_id}")
        rq_path = LIT_DATA_DIR / run_id / "rq_context.json"
        if rq_path.exists():
            rq_data = json.loads(rq_path.read_text())
            title = rq_data.get("title", "")
            if title:
                print(f"  RQ: {title}")
    if data_dir:
        csv_count = len(list(Path(data_dir).glob("*.csv")))
        print(f"  Data: {data_dir} ({csv_count} CSV files)")

    print(sep)
    print(f"\nExecution Plan ({len(plan)} steps):\n")

    current_phase = ""
    gate_phases = {"data_audit": "Go/No-Go Gate 0",
                   "hypothesis": "Go/No-Go Gate 1",
                   "data_build": "Go/No-Go Gate 2"}
    for i, step in enumerate(plan, 1):
        if step.phase != current_phase:
            # Print gate marker for previous phase
            if current_phase in gate_phases:
                print(f"    --- {gate_phases[current_phase]} ---\n")
            current_phase = step.phase
            phase_label = PHASE_LABELS.get(current_phase, current_phase)
            print(f"  {phase_label}")
        print(f"    [{i:2d}] {step.name:<6s} {step.description}")

    # Final gate marker
    if current_phase in gate_phases:
        print(f"    --- {gate_phases[current_phase]} ---")

    if skipped_optional:
        print(f"\n  Skipped (optional):")
        for step in skipped_optional:
            print(f"    {step.name:<6s} {step.description}")

    print()


def _import_module(module_path: str):
    """Import a module by dotted path (handles numeric-prefixed names)."""
    import importlib
    return importlib.import_module(module_path)


def _run_phase_0(data_dir: str, gvc_names, dv_candidates):
    """Execute Phase 0 in-process (library calls, not subprocess)."""
    mod_125 = _import_module("src.scripts.125_data_source_audit")
    mod_128 = _import_module("src.scripts.128_export_validator")

    audit_result = mod_125.run_audit(data_dir, dv_candidates=dv_candidates, gvc_names=gvc_names)
    validator_result = mod_128.run_validation(data_dir, gvc_names=gvc_names)

    gate = check_phase_0_gate(audit_result, validator_result)
    print_gate_result(gate)

    return gate, audit_result, validator_result


def _run_phase_2b(data_dir: str, gvc_names, run_id: str):
    """Execute Phase 2b in-process."""
    mod_128 = _import_module("src.scripts.128_export_validator")
    run_validation = mod_128.run_validation
    import subprocess

    # 128: Full validation
    print(f"\n  [Phase 2b] Export validation (full data) ...", end="", flush=True)
    val_result = run_validation(data_dir, gvc_names=gvc_names)
    if val_result.has_errors:
        print(" FAILED")
        for e in val_result.errors:
            print(f"    -> {e}")
        return None, None
    print(f" ok ({val_result.files_validated} files, {val_result.total_rows} rows)")

    # 130: Dataset build (subprocess — it writes CSV output)
    print(f"  [Phase 2b] Build deal dataset ...", end="", flush=True)
    cmd = [sys.executable, "-m", "src.scripts.130_build_deal_dataset"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(_PROJECT_ROOT), timeout=300)
    if proc.returncode != 0:
        print(f" FAILED (exit {proc.returncode})")
        stderr_last = (proc.stderr or "").strip().split("\n")[-1:]
        if stderr_last:
            print(f"    -> {stderr_last[0][:200]}")
        return val_result, None

    # Extract dataset stats from 130 output
    dataset_path = Path(data_dir) / "deal_dataset.csv"
    if not dataset_path.exists():
        dataset_path = _PROJECT_ROOT / "data" / "cb_insights" / "deal_dataset.csv"

    dataset_stats = {"treatment": 0, "control": 0}
    if dataset_path.exists():
        import csv as csv_mod
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                if row.get("treatment") == "1":
                    dataset_stats["treatment"] += 1
                else:
                    dataset_stats["control"] += 1

    print(f" ok (T={dataset_stats['treatment']}, C={dataset_stats['control']})")

    gate = check_phase_2_gate(dataset_stats)
    print_gate_result(gate)

    return val_result, gate


def _run_phase_3b(data_dir: str, dvs, run_id: str):
    """Execute Phase 3b in-process."""
    mod_132 = _import_module("src.scripts.132_regression_runner")
    run_regression = mod_132.run_regression

    dataset_path = Path(data_dir) / "deal_dataset.csv"
    if not dataset_path.exists():
        dataset_path = _PROJECT_ROOT / "data" / "cb_insights" / "deal_dataset.csv"

    if not dataset_path.exists():
        print(f"\n  [Phase 3b] deal_dataset.csv not found. Skipping.")
        return None

    print(f"\n  [Phase 3b] Running regression analysis ...")
    reg_result = run_regression(str(dataset_path), dvs=dvs)

    # Save results JSON alongside run
    if run_id:
        run_dir = LIT_DATA_DIR / run_id
        if run_dir.exists():
            results_path = run_dir / "regression_results.json"
            results_path.write_text(json.dumps(reg_result.to_dict(), indent=2, ensure_ascii=False))
            print(f"  Results saved to: {results_path}")

    return reg_result


def _run_kindle_delivery(run_id: str):
    """Run 120 review bundle + Kindle delivery."""
    import subprocess
    print(f"\n  [Delivery] Generating review bundle and sending to Kindle ...")
    cmd = [sys.executable, "-m", "src.scripts.120_review_bundle_kindle",
           "--run-id", run_id, "--send"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(_PROJECT_ROOT), timeout=300)
    if proc.returncode == 0:
        print(f"  Kindle delivery complete.")
    else:
        print(f"  Kindle delivery failed (exit {proc.returncode})")


def main() -> None:
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse comma-separated args
    skip_steps = [s.strip() for s in args.skip.split(",") if s.strip()] if args.skip else []
    gvc_names = [x.strip() for x in args.gvc_names.split(",")] if args.gvc_names else None
    dv_candidates = [x.strip() for x in args.dv_candidates.split(",")] if args.dv_candidates else None

    is_v2 = bool(args.data_dir)

    # Build execution plan (v2 if --data-dir, v1 otherwise)
    if is_v2:
        plan = build_execution_plan_v2(
            include_optional=args.include_optional,
            skip_steps=skip_steps,
            stop_after=args.stop_after,
            stop_after_phase=args.stop_after_phase,
            has_data_dir=True,
            skip_empirical=args.skip_empirical,
        )
    else:
        plan = build_execution_plan(
            include_optional=args.include_optional,
            skip_steps=skip_steps,
            stop_after=args.stop_after,
        )

    if not plan:
        print("No steps to execute.")
        sys.exit(0)

    # Identify skipped optional for display
    all_steps = PIPELINE_STEPS + (PHASE_0_STEPS + PHASE_2B_STEPS + PHASE_3B_STEPS if is_v2 else [])
    skipped_optional = [s for s in all_steps if s.optional and not args.include_optional and s not in plan]

    # Validate --run-id exists
    if args.run_id:
        run_dir = LIT_DATA_DIR / args.run_id
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)

    # Validate --data-dir exists
    if args.data_dir and not Path(args.data_dir).is_dir():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Dry-run: show plan and exit
    if args.dry_run:
        _print_plan(
            plan,
            rq_text=args.rq_text or "",
            rq_id=args.rq_id or "",
            run_id=args.run_id or "",
            data_dir=args.data_dir or "",
            skipped_optional=skipped_optional,
            is_v2=is_v2,
        )
        print("To execute: remove --dry-run")
        sys.exit(0)

    # Print header
    sep = "=" * 55
    version = " (v2)" if is_v2 else ""
    print(sep)
    print(f"  Research Cycle Orchestrator{version}")
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
    if args.data_dir:
        print(f"  Data: {args.data_dir}")
    print(sep)

    # ===== Phase 0: Data Source Audit (v2 only) =====
    if is_v2 and any(s.phase == "data_audit" for s in plan):
        gate_0, audit_result, validator_result = _run_phase_0(
            args.data_dir, gvc_names, dv_candidates)

        if not gate_0.passed:
            if not args.continue_on_error:
                sys.exit(3)
            print("  Continuing despite Phase 0 failure (--continue-on-error)")

        # Remove Phase 0 steps from plan (already executed in-process)
        plan = [s for s in plan if s.phase != "data_audit"]

        if args.stop_after_phase == 0:
            print("\n  Stopped after Phase 0 (--stop-after-phase 0)")
            sys.exit(0)

    # ===== Split plan by phase for mixed in-process / subprocess execution =====
    phase2b_plan = [s for s in plan if s.phase == "data_build"]
    phase3b_plan = [s for s in plan if s.phase == "empirical"]
    pre_2b_plan = [s for s in plan if s.phase in ("lit_review", "hypothesis", "deep_literature")]
    post_2b_plan = [s for s in plan if s.phase in ("output", "next_rq", "visualization")]

    result = None
    run_id = args.run_id or ""

    # ===== Phase 1 + 2a: Literature & Hypothesis =====
    if pre_2b_plan:
        result = run_pipeline(
            pre_2b_plan,
            run_id=run_id,
            rq_text=args.rq_text or "",
            rq_id=args.rq_id or "",
            min_score=args.min_score,
            max_papers=args.max_papers,
            continue_on_error=args.continue_on_error,
            step_timeout=args.step_timeout,
        )

        if not result.success and not args.continue_on_error:
            print_summary(result)
            sys.exit(1)

        run_id = result.run_id or run_id

    # ===== Phase 2b: Data Collection & Build (v2 only) =====
    if is_v2 and phase2b_plan:
        # Check if data is available
        data_path = Path(args.data_dir)
        csv_files = [f for f in data_path.glob("*.csv") if f.name != "deal_dataset.csv"]
        deal_dataset = data_path / "deal_dataset.csv"

        if deal_dataset.exists():
            print(f"\n  deal_dataset.csv found -> skipping Phase 2b build")
        elif len(csv_files) >= 2:
            val_result, gate_2 = _run_phase_2b(args.data_dir, gvc_names, run_id)

            if gate_2 and not gate_2.passed:
                if not args.continue_on_error:
                    if result:
                        print_summary(result)
                    sys.exit(4)
                print("  Continuing despite Phase 2 failure (--continue-on-error)")

            if val_result and not val_result.passed:
                if not args.continue_on_error:
                    if result:
                        print_summary(result)
                    sys.exit(4)
        else:
            # No data yet — pause and instruct
            sep2 = "=" * 55
            print(f"\n{sep2}")
            print(f"  Phase 2: Data Collection Required")
            print(sep2)
            print(f"\n  Place CB Insights CSV exports in: {args.data_dir}/")
            print(f"  Expected files: {{country}}_gvc.csv, {{country}}_pvc.csv")
            if run_id:
                print(f"  Then re-run with: --run-id {run_id} --data-dir {args.data_dir}")
            print()
            sys.exit(0)  # Clean exit

    # ===== Phase 3a + 4 + 5: Research Output, Next-RQ, Visualization =====
    if post_2b_plan:
        result_post = run_pipeline(
            post_2b_plan,
            run_id=run_id,
            rq_text=args.rq_text or "",
            rq_id=args.rq_id or "",
            min_score=args.min_score,
            max_papers=args.max_papers,
            continue_on_error=args.continue_on_error,
            step_timeout=args.step_timeout,
        )

        # Merge results
        if result is None:
            result = result_post
        else:
            result.steps.extend(result_post.steps)
            result.total_duration += result_post.total_duration

        if not result_post.success and not args.continue_on_error:
            print_summary(result)
            sys.exit(1)

    # ===== Phase 3b: Empirical Analysis (v2 only) =====
    if is_v2 and phase3b_plan:
        dvs = dv_candidates if dv_candidates else ["follow_on", "exit", "round_progression"]
        reg_result = _run_phase_3b(args.data_dir, dvs, run_id)

        if reg_result and reg_result.all_failed:
            print("  Phase 3b: All regressions failed.")
            if not args.continue_on_error:
                if result:
                    print_summary(result)
                sys.exit(5)

    # ===== Kindle Delivery (optional) =====
    if args.send_kindle and run_id:
        _run_kindle_delivery(run_id)

    # Summary
    if result:
        print_summary(result)

    # Exit code
    if result is None:
        sys.exit(0)
    elif result.success:
        sys.exit(0)
    elif args.continue_on_error and result.completed > 0:
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
