"""Task management — CRUD with dependency resolution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.paper.models import (
    TASK_OWNERS,
    TASK_PRIORITIES,
    TASK_SOURCES,
    TASK_STATUSES,
    Task,
    _now_iso,
)
from src.paper.registry import (
    PAPERS_DIR,
    _paper_dir,
    get_paper,
    _sync_registry_entry,
    _paper_json,
)

logger = logging.getLogger(__name__)


def _tasks_path(paper_id: str) -> Path:
    return _paper_dir(paper_id) / "tasks.json"


def _read_tasks(paper_id: str) -> List[Dict[str, Any]]:
    path = _tasks_path(paper_id)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("tasks", [])


def _write_tasks(paper_id: str, tasks: List[Dict[str, Any]]) -> None:
    path = _tasks_path(paper_id)
    path.write_text(
        json.dumps({"tasks": tasks}, indent=2, ensure_ascii=False) + "\n"
    )
    # Sync registry to update open_tasks count
    paper = get_paper(paper_id)
    _sync_registry_entry(paper)


def _next_id(tasks: List[Dict[str, Any]]) -> str:
    """Auto-increment task ID: t001, t002, ..."""
    if not tasks:
        return "t001"
    max_num = max(int(t["id"][1:]) for t in tasks)
    return f"t{max_num + 1:03d}"


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def add_task(
    paper_id: str,
    content: str,
    source: str,
    owner: str,
    priority: str,
    *,
    linked_stage: str = "",
    linked_decision: Optional[str] = None,
    next_action: str = "",
    depends_on: Optional[List[str]] = None,
    due: Optional[str] = None,
    blocked_reason: str = "",
) -> Task:
    """Add a task. Auto-detects blocked status from depends_on or blocked_reason."""
    if source not in TASK_SOURCES:
        raise ValueError(f"Invalid source '{source}'. Must be one of {sorted(TASK_SOURCES)}")
    if owner not in TASK_OWNERS:
        raise ValueError(f"Invalid owner '{owner}'. Must be one of {sorted(TASK_OWNERS)}")
    if priority not in TASK_PRIORITIES:
        raise ValueError(f"Invalid priority '{priority}'. Must be one of {sorted(TASK_PRIORITIES)}")

    paper = get_paper(paper_id)
    if not linked_stage:
        linked_stage = paper.current_stage

    tasks = _read_tasks(paper_id)
    task_id = _next_id(tasks)
    deps = depends_on or []

    # Determine initial status
    status = "open"
    if blocked_reason:
        status = "blocked"
    elif deps:
        # Check if any dependency is not done
        done_ids = {t["id"] for t in tasks if t.get("status") == "done"}
        if not all(d in done_ids for d in deps):
            status = "blocked"
            if not blocked_reason:
                pending = [d for d in deps if d not in done_ids]
                blocked_reason = f"Waiting on: {', '.join(pending)}"

    task = Task(
        id=task_id,
        paper_id=paper_id,
        content=content,
        source=source,
        owner=owner,
        priority=priority,
        status=status,
        blocked_reason=blocked_reason,
        linked_stage=linked_stage,
        linked_decision=linked_decision,
        next_action=next_action,
        depends_on=deps,
        due=due,
    )

    tasks.append(task.to_dict())
    _write_tasks(paper_id, tasks)
    logger.info("Task %s added to paper '%s'", task_id, paper_id)
    return task


def update_task(
    paper_id: str,
    task_id: str,
    **fields: Any,
) -> Task:
    """Update task fields. Triggers dependency resolution."""
    allowed = {
        "status", "priority", "next_action", "owner", "due",
        "blocked_reason", "content", "linked_decision", "completed_at",
    }
    bad = set(fields.keys()) - allowed
    if bad:
        raise ValueError(f"Cannot update field(s): {bad}. Allowed: {sorted(allowed)}")

    tasks = _read_tasks(paper_id)
    found = None
    for t in tasks:
        if t["id"] == task_id:
            found = t
            break
    if found is None:
        raise ValueError(f"Task '{task_id}' not found in paper '{paper_id}'")

    # If manually setting blocked with a reason
    if fields.get("status") == "blocked" and "blocked_reason" not in fields:
        if not found.get("blocked_reason"):
            raise ValueError(
                "Setting status to 'blocked' requires --blocked-reason "
                "(or use depends_on for dependency-based blocking)"
            )

    # If unblocking, clear blocked_reason
    if fields.get("status") in ("open", "in_progress") and found.get("blocked_reason"):
        if "blocked_reason" not in fields:
            fields["blocked_reason"] = ""

    found.update(fields)
    _write_tasks(paper_id, tasks)

    # If a task was completed, resolve dependents
    if fields.get("status") == "done":
        _resolve_blocked(paper_id, task_id)

    # If a task was un-completed, re-block dependents
    if fields.get("status") in ("open", "in_progress", "blocked"):
        _recheck_dependents(paper_id, task_id)

    return Task(**found)


def complete_task(paper_id: str, task_id: str) -> Task:
    """Mark task as done and resolve dependents."""
    return update_task(
        paper_id,
        task_id,
        status="done",
        completed_at=_now_iso(),
        blocked_reason="",
    )


def get_tasks(
    paper_id: str,
    *,
    status_filter: Optional[str] = None,
) -> List[Task]:
    """Get tasks for a paper, optionally filtered by status."""
    tasks = _read_tasks(paper_id)
    if status_filter:
        tasks = [t for t in tasks if t.get("status") == status_filter]
    return [Task(**t) for t in tasks]


def get_open_tasks(paper_id: str) -> List[Task]:
    """Get tasks that are not done or wontfix."""
    tasks = _read_tasks(paper_id)
    active = [
        t for t in tasks if t.get("status") in ("open", "in_progress", "blocked")
    ]
    return [Task(**t) for t in active]


def get_all_open_tasks() -> List[Task]:
    """Get open tasks across all papers."""
    result = []
    if not PAPERS_DIR.exists():
        return result
    for d in sorted(PAPERS_DIR.iterdir()):
        if d.is_dir() and (d / "paper.json").exists():
            paper_id = d.name
            result.extend(get_open_tasks(paper_id))
    return result


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------


def _resolve_blocked(paper_id: str, completed_task_id: str) -> int:
    """Unblock tasks whose depends_on are now all done. Returns count unblocked."""
    tasks = _read_tasks(paper_id)
    done_ids = {t["id"] for t in tasks if t.get("status") == "done"}
    unblocked = 0

    for t in tasks:
        if t.get("status") != "blocked":
            continue
        deps = t.get("depends_on", [])
        if not deps:
            continue
        # Only unblock if ALL deps are done AND the block was dependency-derived
        if all(d in done_ids for d in deps):
            # Check if the block reason is dependency-based (not manual)
            reason = t.get("blocked_reason", "")
            if reason.startswith("Waiting on:") or not reason:
                t["status"] = "open"
                t["blocked_reason"] = ""
                unblocked += 1
                logger.info(
                    "Task %s unblocked (dependency %s completed)",
                    t["id"],
                    completed_task_id,
                )

    if unblocked:
        _write_tasks(paper_id, tasks)
    return unblocked


def _recheck_dependents(paper_id: str, reverted_task_id: str) -> int:
    """Re-block tasks that depend on a task no longer done. Returns count re-blocked."""
    tasks = _read_tasks(paper_id)
    done_ids = {t["id"] for t in tasks if t.get("status") == "done"}
    reblocked = 0

    for t in tasks:
        if t.get("status") in ("done", "wontfix"):
            continue
        deps = t.get("depends_on", [])
        if not deps:
            continue
        pending = [d for d in deps if d not in done_ids]
        if pending and t.get("status") != "blocked":
            t["status"] = "blocked"
            t["blocked_reason"] = f"Waiting on: {', '.join(pending)}"
            reblocked += 1
        elif pending and t.get("status") == "blocked":
            # Update blocked_reason to reflect current pending list
            t["blocked_reason"] = f"Waiting on: {', '.join(pending)}"

    if reblocked:
        _write_tasks(paper_id, tasks)
    return reblocked
