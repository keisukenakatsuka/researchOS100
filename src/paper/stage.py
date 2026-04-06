"""Stage transition and status change management."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.paper.models import (
    STAGES,
    STATUSES,
    SOURCES,
    TERMINAL_STAGES,
    StageTransition,
    StatusChange,
    _now_iso,
)
from src.paper.registry import _paper_dir, _paper_json, get_paper, _sync_registry_entry

logger = logging.getLogger(__name__)


def _stages_path(paper_id: str) -> Path:
    return _paper_dir(paper_id) / "stages.jsonl"


def _append_record(paper_id: str, record: Dict[str, Any]) -> None:
    """Append a JSONL record to stages.jsonl."""
    path = _stages_path(paper_id)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _check_open_tasks(paper_id: str) -> List[str]:
    """Return list of open task descriptions in the current stage."""
    tasks_path = _paper_dir(paper_id) / "tasks.json"
    if not tasks_path.exists():
        return []
    data = json.loads(tasks_path.read_text())
    paper = get_paper(paper_id)
    warnings = []
    for t in data.get("tasks", []):
        if (
            t.get("status") in ("open", "in_progress", "blocked")
            and t.get("linked_stage") == paper.current_stage
        ):
            warnings.append(f"  {t['id']}: {t['content']} ({t['status']})")
    return warnings


# ---------------------------------------------------------------------------
# Stage transition
# ---------------------------------------------------------------------------


def transition_stage(
    paper_id: str,
    to_stage: str,
    entry_reason: str,
    *,
    exit_reason: Optional[str] = None,
    gate_result: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    effective_at: Optional[str] = None,
    source: str = "cli",
) -> Tuple[StageTransition, List[str]]:
    """Transition paper to a new stage. Returns (record, warnings).

    Args:
        effective_at: For migration/bootstrap only. If set, source should
            be "migration". In normal operation, leave as None.
        source: "cli" (normal), "orchestrator" (108 auto), "migration" (bootstrap).
    """
    if to_stage not in STAGES:
        raise ValueError(f"Invalid stage '{to_stage}'. Must be one of {STAGES}")
    if source not in SOURCES:
        raise ValueError(f"Invalid source '{source}'. Must be one of {SOURCES}")

    paper = get_paper(paper_id)

    if source != "migration" and paper.current_stage in TERMINAL_STAGES:
        raise ValueError(
            f"Paper '{paper_id}' is in terminal stage '{paper.current_stage}'. "
            f"Cannot transition to '{to_stage}'."
        )

    # Determine from_stage/from_status
    if source == "migration":
        # For migration, infer from_stage from the last record in stages.jsonl
        history = get_stage_history(paper_id)
        if history:
            last = history[-1]
            d = last.to_dict()
            if d["type"] == "stage_transition":
                from_stage = d["to_stage"]
                from_status = d["to_status"]
            else:
                from_stage = d.get("stage", paper.current_stage)
                from_status = d.get("to_status", paper.current_status)
        else:
            from_stage = None
            from_status = None
    else:
        from_stage = paper.current_stage
        from_status = paper.current_status

    # Warn about open tasks (non-blocking, skip for migration)
    warnings = []
    if source != "migration":
        task_warnings = _check_open_tasks(paper_id)
        if task_warnings:
            warnings.append(
                f"WARNING: {len(task_warnings)} open task(s) in stage "
                f"'{paper.current_stage}':"
            )
            warnings.extend(task_warnings)

    # Warn if skipping stages (non-blocking, skip for migration)
    if source != "migration" and from_stage in STAGES and to_stage in STAGES:
        from_idx = STAGES.index(from_stage)
        to_idx = STAGES.index(to_stage)
        if to_idx > from_idx + 1:
            skipped = STAGES[from_idx + 1 : to_idx]
            warnings.append(
                f"NOTE: Skipping stage(s): {', '.join(skipped)}"
            )

    record = StageTransition(
        from_stage=from_stage,
        from_status=from_status,
        to_stage=to_stage,
        to_status="active",
        entry_reason=entry_reason,
        exit_reason=exit_reason,
        gate_result=gate_result,
        run_id=run_id,
        source=source,
        effective_at=effective_at or "",
        recorded_at="",
    )
    # __post_init__ fills defaults

    _append_record(paper_id, record.to_dict())

    # Update paper.json — migration records do NOT change current state
    if source != "migration":
        paper.current_stage = to_stage
        paper.current_status = "active"
        paper.updated_at = _now_iso()
        paper.save(_paper_json(paper_id))
        _sync_registry_entry(paper)

    logger.info(
        "Paper '%s': %s → %s (source=%s)",
        paper_id,
        record.from_stage,
        to_stage,
        source,
    )
    return record, warnings


# ---------------------------------------------------------------------------
# Status change
# ---------------------------------------------------------------------------


def change_status(
    paper_id: str,
    to_status: str,
    reason: str,
    *,
    effective_at: Optional[str] = None,
    source: str = "cli",
) -> StatusChange:
    """Change the status within the current stage."""
    if to_status not in STATUSES:
        raise ValueError(f"Invalid status '{to_status}'. Must be one of {STATUSES}")

    paper = get_paper(paper_id)

    record = StatusChange(
        stage=paper.current_stage,
        from_status=paper.current_status,
        to_status=to_status,
        reason=reason,
        source=source,
        effective_at=effective_at or "",
        recorded_at="",
    )

    _append_record(paper_id, record.to_dict())

    paper.current_status = to_status
    paper.updated_at = _now_iso()
    paper.save(_paper_json(paper_id))
    _sync_registry_entry(paper)

    logger.info(
        "Paper '%s' [%s]: %s → %s",
        paper_id,
        paper.current_stage,
        record.from_status,
        to_status,
    )
    return record


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def get_stage_history(
    paper_id: str,
) -> List[Union[StageTransition, StatusChange]]:
    """Read all stage/status records from stages.jsonl."""
    path = _stages_path(paper_id)
    if not path.exists():
        return []

    records: List[Union[StageTransition, StatusChange]] = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        rtype = data.get("type")
        if rtype == "stage_transition":
            records.append(StageTransition(**data))
        elif rtype == "status_change":
            records.append(StatusChange(**data))
        else:
            logger.warning("Unknown record type in stages.jsonl: %s", rtype)
    return records


def get_current(paper_id: str) -> Tuple[str, str]:
    """Return (current_stage, current_status) from paper.json."""
    paper = get_paper(paper_id)
    return paper.current_stage, paper.current_status
