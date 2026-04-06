"""Data models for Paper Registry.

Minimal required schema for Paper management:
- Paper: paper_id, title, paper_type, current_stage, current_status, created_at, updated_at
- StageTransition: stage movement with entry/exit reasons, gate results, provenance
- StatusChange: status change within a stage
- Decision: research decision with rejected alternatives
- Task: actionable item with owner, dependencies, linkage

All timestamps are ISO 8601 with timezone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Stage & Status enums
# ---------------------------------------------------------------------------

STAGES = [
    "idea",
    "rq_formation",
    "literature_review",
    "data_feasibility",
    "hypothesis",
    "data_collection",
    "analysis",
    "draft",
    "revision",
    "submission",
    "under_review",
    "accepted",
    "rejected",
    "abandoned",
]

TERMINAL_STAGES = {"accepted", "rejected", "abandoned"}

STATUSES = {"active", "blocked", "paused", "completed"}

# Source of a record — who/what created it
SOURCES = {"cli", "orchestrator", "migration"}

# Paper types
PAPER_TYPES = {"empirical", "theoretical", "review", "mixed"}

DECISION_TYPES = {
    "hypothesis_selection",
    "variable_change",
    "data_source_change",
    "method_change",
    "scope_change",
    "review_response",
    "gate_decision",
    "stage_transition",
    "other",
}

TASK_SOURCES = {"gate_action", "review_feedback", "self_note", "pipeline_output"}
TASK_OWNERS = {"human", "claude", "pipeline"}
TASK_STATUSES = {"open", "in_progress", "done", "blocked", "wontfix"}
TASK_PRIORITIES = {"critical", "high", "medium", "low"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """Core paper metadata — the minimal required schema.

    Required fields: paper_id, title, paper_type, current_stage,
    current_status, created_at, updated_at.
    """

    paper_id: str
    title: str
    paper_type: str  # empirical, theoretical, review, mixed
    rq: str = ""
    target_journal: str = ""
    authors: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    current_stage: str = "idea"
    current_status: str = "active"
    run_ids: List[str] = field(default_factory=list)
    data_dir: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        now = _now_iso()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> Paper:
        data = json.loads(path.read_text())
        return cls(**data)


# ---------------------------------------------------------------------------
# Stage records (stages.jsonl)
# ---------------------------------------------------------------------------

@dataclass
class StageTransition:
    """Records a stage-to-stage movement."""

    to_stage: str
    to_status: str
    entry_reason: str
    effective_at: str = ""
    recorded_at: str = ""
    source: str = "cli"  # cli | orchestrator | migration
    from_stage: Optional[str] = None
    from_status: Optional[str] = None
    exit_reason: Optional[str] = None
    gate_result: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None
    type: str = "stage_transition"

    def __post_init__(self) -> None:
        now = _now_iso()
        if not self.recorded_at:
            self.recorded_at = now
        if not self.effective_at:
            self.effective_at = self.recorded_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatusChange:
    """Records a status change within the current stage."""

    stage: str
    from_status: str
    to_status: str
    reason: str
    effective_at: str = ""
    recorded_at: str = ""
    source: str = "cli"
    type: str = "status_change"

    def __post_init__(self) -> None:
        now = _now_iso()
        if not self.recorded_at:
            self.recorded_at = now
        if not self.effective_at:
            self.effective_at = self.recorded_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Decision (decisions.jsonl)
# ---------------------------------------------------------------------------

@dataclass
class RejectedAlternative:
    option: str
    rejection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    """A research decision with rationale and rejected alternatives.

    Minimal required fields: id, decision, reason.
    All other fields are optional to keep decisions easy to write.
    """

    id: str
    decision: str
    reason: str
    # Optional — enriches the record but not required to create one
    stage: str = ""  # auto-filled from paper.current_stage if empty
    decision_type: str = "other"  # from DECISION_TYPES
    rejected_alternatives: List[RejectedAlternative] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)
    effective_at: str = ""
    recorded_at: str = ""
    source: str = "cli"

    def __post_init__(self) -> None:
        now = _now_iso()
        if not self.recorded_at:
            self.recorded_at = now
        if not self.effective_at:
            self.effective_at = self.recorded_at

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Task (tasks.json)
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """An actionable item tied to a paper."""

    id: str
    paper_id: str
    content: str
    source: str  # gate_action | review_feedback | self_note | pipeline_output
    owner: str  # human | claude | pipeline
    priority: str  # critical | high | medium | low
    status: str = "open"
    blocked_reason: str = ""  # why blocked — for manual blocks (not depends_on derived)
    linked_stage: str = ""
    linked_decision: Optional[str] = None
    next_action: str = ""
    depends_on: List[str] = field(default_factory=list)
    due: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Registry index (_registry.json)
# ---------------------------------------------------------------------------

@dataclass
class RegistryEntry:
    """Lightweight index entry for fast listing."""

    paper_id: str
    title: str
    paper_type: str
    current_stage: str
    current_status: str
    open_tasks: int = 0
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
