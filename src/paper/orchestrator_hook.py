"""Hook for 108 orchestrator — Paper-aware integration.

All functions are safe to call from the pipeline: errors are caught and
logged as warnings, never interrupting the run. If paper_id is None or
empty, all functions are no-ops.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Phase → Paper stage mapping
_PHASE_TO_STAGE = {
    "Phase 0": "data_feasibility",
    "Phase 2": "data_collection",
}

# Next stage after a gate passes
_GATE_NEXT_STAGE = {
    "Phase 0": "hypothesis",
    "Phase 2": "analysis",
}


def _safe(fn):
    """Decorator: catch all exceptions, log as warning, return None."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning("Paper hook failed (%s): %s", fn.__name__, e)
            print(f"  [paper] WARNING: {fn.__name__} failed: {e}")
            return None
    return wrapper


@_safe
def on_run_start(paper_id: Optional[str], run_id: str) -> None:
    """Called when a run_id is determined (after step 079). Links run to paper."""
    if not paper_id or not run_id:
        return
    from src.paper.registry import link_run
    link_run(paper_id, run_id)
    print(f"  [paper] Linked run {run_id} to {paper_id}")


@_safe
def on_gate_pass(
    paper_id: Optional[str],
    gate_result: Any,
    run_id: str = "",
) -> None:
    """Called when a gate check passes. Records stage transition."""
    if not paper_id:
        return

    from src.paper.stage import transition_stage
    from src.paper.overview import regenerate_status

    phase = gate_result.phase  # "Phase 0", "Phase 2"
    next_stage = _GATE_NEXT_STAGE.get(phase)
    if not next_stage:
        logger.info("No stage mapping for phase '%s', skipping", phase)
        return

    # Build a summary of gate checks for the entry reason
    check_parts = []
    for c in gate_result.checks:
        if c.passed:
            check_parts.append(f"{c.name}={c.value}")
    checks_summary = ", ".join(check_parts[:3])

    transition_stage(
        paper_id,
        next_stage,
        entry_reason=f"{phase} Gate passed: {checks_summary}",
        exit_reason=f"{phase} Gate passed",
        gate_result={
            "passed": True,
            "phase": phase,
            "checks": [
                {"name": c.name, "passed": c.passed, "value": c.value,
                 "threshold": c.threshold}
                for c in gate_result.checks
            ],
        },
        run_id=run_id,
        source="orchestrator",
    )
    regenerate_status(paper_id)
    print(f"  [paper] Stage → {next_stage} (Gate {phase} ✅)")


@_safe
def on_gate_fail(
    paper_id: Optional[str],
    gate_result: Any,
) -> None:
    """Called when a gate check fails. Sets status to blocked and creates tasks."""
    if not paper_id:
        return

    from src.paper.stage import change_status
    from src.paper.task import add_task
    from src.paper.overview import regenerate_status

    phase = gate_result.phase
    failed_checks = [c for c in gate_result.checks if not c.passed]

    # Set status to blocked
    reasons = ", ".join(c.name for c in failed_checks)
    change_status(
        paper_id,
        "blocked",
        reason=f"{phase} Gate failed: {reasons}",
        source="orchestrator",
    )

    # Create tasks for each failed check
    for c in failed_checks:
        action = c.message if c.message else f"Fix {c.name} (current: {c.value}, required: {c.threshold})"
        add_task(
            paper_id,
            content=f"[{phase}] {c.name}: {c.value} (required: {c.threshold})",
            source="gate_action",
            owner="human",
            priority="high",
            next_action=action,
            linked_stage=_PHASE_TO_STAGE.get(phase, ""),
        )

    regenerate_status(paper_id)
    n = len(failed_checks)
    print(f"  [paper] Status → blocked ({phase} Gate ❌, {n} task(s) created)")


@_safe
def on_pipeline_complete(
    paper_id: Optional[str],
    success: bool,
    run_id: str = "",
) -> None:
    """Called when the pipeline finishes. Updates status."""
    if not paper_id:
        return

    from src.paper.stage import change_status
    from src.paper.overview import regenerate_status

    if success:
        change_status(
            paper_id,
            "completed",
            reason=f"Pipeline completed successfully (run: {run_id})",
            source="orchestrator",
        )
        print(f"  [paper] Status → completed")
    else:
        change_status(
            paper_id,
            "blocked",
            reason=f"Pipeline failed (run: {run_id}). Check step errors.",
            source="orchestrator",
        )
        print(f"  [paper] Status → blocked (pipeline failed)")

    regenerate_status(paper_id)


@_safe
def on_data_pause(
    paper_id: Optional[str],
    data_dir: str,
    run_id: str = "",
) -> None:
    """Called when pipeline pauses for user data collection."""
    if not paper_id:
        return

    from src.paper.stage import transition_stage, change_status
    from src.paper.task import add_task
    from src.paper.overview import regenerate_status

    transition_stage(
        paper_id,
        "data_collection",
        entry_reason="Pipeline paused: waiting for CB Insights export",
        source="orchestrator",
    )
    change_status(
        paper_id,
        "blocked",
        reason=f"CB Insights CSV export needed in {data_dir}",
        source="orchestrator",
    )
    add_task(
        paper_id,
        content=f"Place CB Insights CSV exports in {data_dir}",
        source="gate_action",
        owner="human",
        priority="critical",
        next_action=f"Export GVC/PVC CSVs to {data_dir}, then re-run with --run-id {run_id}",
        linked_stage="data_collection",
    )
    regenerate_status(paper_id)
    print(f"  [paper] Stage → data_collection (blocked: awaiting export)")
