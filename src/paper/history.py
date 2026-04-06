"""Unified timeline generation — merges stages, decisions, and key task events."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from src.paper.registry import get_paper
from src.paper.stage import get_stage_history
from src.paper.decision import get_decisions
from src.paper.task import get_tasks

logger = logging.getLogger(__name__)


@dataclass
class TimelineEntry:
    """A single event in the paper timeline."""

    effective_at: str  # ISO timestamp for sorting
    event_type: str  # stage | status | decision | task_done
    summary: str  # one-liner
    details: List[str]  # additional lines (indented)


def get_timeline(paper_id: str) -> List[TimelineEntry]:
    """Build a unified timeline from stages, decisions, and completed tasks."""
    entries: List[TimelineEntry] = []

    # Stage transitions and status changes
    for r in get_stage_history(paper_id):
        d = r.to_dict()
        ts = d.get("effective_at", d.get("recorded_at", ""))
        source_tag = " [migration]" if d.get("source") == "migration" else ""

        if d["type"] == "stage_transition":
            # Skip init bootstrap record
            if (d.get("entry_reason") or "").startswith("Initialized at stage"):
                continue
            from_s = d.get("from_stage") or "—"
            to_s = d["to_stage"]
            summary = f"[stage] {from_s} → {to_s} ({d['to_status']}){source_tag}"
            details = []
            if d.get("entry_reason"):
                details.append(f"Entry: {d['entry_reason']}")
            if d.get("exit_reason"):
                details.append(f"Exit: {d['exit_reason']}")
            if d.get("gate_result"):
                gate = d["gate_result"]
                icon = "✅" if gate.get("passed") else "❌"
                details.append(f"Gate {icon} {gate.get('phase', '')}")
            entries.append(TimelineEntry(ts, "stage", summary, details))

        elif d["type"] == "status_change":
            summary = (
                f"[status] {d['stage']}: "
                f"{d['from_status']} → {d['to_status']}{source_tag}"
            )
            details = []
            if d.get("reason"):
                details.append(f"Reason: {d['reason']}")
            entries.append(TimelineEntry(ts, "status", summary, details))

    # Decisions
    for d in get_decisions(paper_id):
        ts = d.effective_at or d.recorded_at
        type_str = f" ({d.decision_type})" if d.decision_type != "other" else ""
        source_tag = " [migration]" if d.source == "migration" else ""
        summary = f"[decision:{d.id}]{type_str}{source_tag}"
        details = [f"✅ {d.decision}", f"Reason: {d.reason}"]
        for ra in d.rejected_alternatives:
            reason = f" — {ra.rejection_reason}" if ra.rejection_reason else ""
            details.append(f"❌ {ra.option}{reason}")
        entries.append(TimelineEntry(ts, "decision", summary, details))

    # Completed tasks only (open/blocked tasks are noise in timeline)
    for t in get_tasks(paper_id, status_filter="done"):
        if t.completed_at:
            summary = f"[task:{t.id}] ✅ {t.content[:60]}"
            entries.append(TimelineEntry(t.completed_at, "task_done", summary, []))

    # Sort by effective_at
    entries.sort(key=lambda e: e.effective_at)
    return entries


def format_timeline(paper_id: str, entries: List[TimelineEntry]) -> str:
    """Format timeline for CLI display."""
    paper = get_paper(paper_id)
    lines = [
        "",
        "═" * 65,
        f"  {paper.paper_id} — Timeline",
        "═" * 65,
        "",
    ]

    prev_date = ""
    for e in entries:
        date = e.effective_at[:10] if e.effective_at else "?"
        time = e.effective_at[11:16] if len(e.effective_at) > 16 else ""

        # Group by date — show date header only on change
        if date != prev_date:
            if prev_date:
                lines.append("")
            lines.append(f"  {date}")
            prev_date = date

        lines.append(f"    {time}  {e.summary}")
        for detail in e.details:
            lines.append(f"          {detail}")

    lines.append("")
    lines.append("═" * 65)
    return "\n".join(lines)
