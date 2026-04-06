"""Cross-paper dashboard display."""

from __future__ import annotations

import json
from datetime import datetime

from src.paper.models import _now_iso
from src.paper.registry import PAPERS_DIR, list_papers, _paper_dir


def show_dashboard() -> str:
    """Generate dashboard output for all papers. Returns formatted string."""
    entries = list_papers()

    today = _now_iso()[:10]
    lines = [
        "",
        "═" * 75,
        f"  Research Papers Dashboard — {today}",
        "═" * 75,
        "",
    ]

    if not entries:
        lines.append("  No papers found.")
        lines.append("")
        lines.append("═" * 75)
        return "\n".join(lines)

    lines.append(
        f"  {'Paper':<28s} {'Stage':<18s} {'Status':<10s} "
        f"{'Tasks':<8s} {'Last Activity'}"
    )
    lines.append(f"  {'─' * 70}")

    for e in entries:
        tasks_str = f"{e.open_tasks} open" if e.open_tasks else "—"
        updated = e.updated_at[:10] if e.updated_at else ""

        # Get last activity description
        last_activity = _get_last_activity(e.paper_id)
        activity_str = f"{updated} {last_activity}" if last_activity else updated

        lines.append(
            f"  {e.paper_id:<28s} {e.current_stage:<18s} {e.current_status:<10s} "
            f"{tasks_str:<8s} {activity_str}"
        )

    lines.append("")
    lines.append("═" * 75)
    return "\n".join(lines)


def _get_last_activity(paper_id: str) -> str:
    """Get a short description of the most recent activity."""
    d = _paper_dir(paper_id)

    latest_ts = ""
    latest_desc = ""

    # Check stages.jsonl last line
    stages_path = d / "stages.jsonl"
    if stages_path.exists():
        text = stages_path.read_text().strip()
        if text:
            last_line = text.splitlines()[-1]
            data = json.loads(last_line)
            ts = data.get("effective_at", "")
            if ts > latest_ts:
                latest_ts = ts
                if data.get("type") == "stage_transition":
                    latest_desc = f"→ {data['to_stage']}"
                else:
                    latest_desc = f"{data.get('to_status', '')}"

    # Check decisions.jsonl last line
    decisions_path = d / "decisions.jsonl"
    if decisions_path.exists():
        text = decisions_path.read_text().strip()
        if text:
            last_line = text.splitlines()[-1]
            data = json.loads(last_line)
            ts = data.get("effective_at", "")
            if ts > latest_ts:
                latest_ts = ts
                dtype = data.get("decision_type", "")
                latest_desc = dtype if dtype != "other" else "decision"

    return latest_desc
