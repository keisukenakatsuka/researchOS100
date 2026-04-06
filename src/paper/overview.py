"""Generate current_status.md — a human-readable summary of paper state."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.paper.models import _now_iso
from src.paper.registry import _paper_dir, get_paper
from src.paper.stage import get_stage_history
from src.paper.decision import get_decisions
from src.paper.task import get_open_tasks

logger = logging.getLogger(__name__)


def regenerate_status(paper_id: str) -> Path:
    """Generate current_status.md from paper state. Returns file path."""
    paper = get_paper(paper_id)
    stages = get_stage_history(paper_id)
    decisions = get_decisions(paper_id)
    open_tasks = get_open_tasks(paper_id)

    lines: List[str] = []

    # --- Header ---
    lines.append(f"# {paper.paper_id} — Current Status")
    lines.append("")
    lines.append(f"> Last updated: {_now_iso()}")
    lines.append("")

    # --- Where We Are ---
    lines.append("## Where We Are")
    lines.append("")
    lines.append(f"**Stage:** {paper.current_stage} | **Status:** {paper.current_status}")
    lines.append(f"**Paper:** {paper.title}")
    if paper.target_journal:
        lines.append(f"**Target:** {paper.target_journal}")
    lines.append("")

    # --- What's Happening (rule-based summary) ---
    lines.append("## What's Happening")
    lines.append("")
    summary = _generate_summary(paper, open_tasks, decisions)
    lines.append(summary)
    lines.append("")

    # --- Open Tasks ---
    if open_tasks:
        prio_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        open_tasks.sort(key=lambda t: (prio_order.get(t.priority, 9), t.id))

        lines.append(f"## Open Tasks ({len(open_tasks)})")
        lines.append("")
        lines.append("| # | Task | Owner | Priority | Blocked? |")
        lines.append("|---|------|-------|----------|----------|")
        for t in open_tasks:
            blocked = ""
            if t.status == "blocked":
                if t.depends_on:
                    blocked = f"← {', '.join(t.depends_on)}"
                elif t.blocked_reason:
                    reason = t.blocked_reason[:50]
                    blocked = reason
            lines.append(
                f"| {t.id} | {t.content[:60]} | {t.owner} | {t.priority} | {blocked} |"
            )
        lines.append("")

    # --- Next Actions ---
    actionable = [
        t for t in open_tasks if t.status != "blocked" and t.next_action
    ]
    if actionable:
        lines.append("## Next Actions")
        lines.append("")
        for i, t in enumerate(actionable[:5], 1):
            lines.append(f"{i}. **{t.id}** ({t.owner}): {t.next_action}")
        lines.append("")

    # --- Recent Decisions (last 3) ---
    if decisions:
        recent = decisions[-3:]
        lines.append("## Recent Decisions")
        lines.append("")
        for d in reversed(recent):
            ts = d.effective_at[:10] if d.effective_at else "?"
            type_str = f" ({d.decision_type})" if d.decision_type != "other" else ""
            lines.append(f"- **{d.id}** ({ts}){type_str}: {d.decision}")
            for ra in d.rejected_alternatives[:3]:
                reason = f" — {ra.rejection_reason}" if ra.rejection_reason else ""
                lines.append(f"  - Rejected: {ra.option}{reason}")
        lines.append("")

    # --- Key Milestones ---
    milestones = _extract_milestones(stages)
    if milestones:
        lines.append("## Key Milestones")
        lines.append("")
        for ts, desc in milestones:
            lines.append(f"- {ts}: {desc}")
        lines.append("")

    # --- Linked Runs ---
    if paper.run_ids:
        lines.append("## Linked Runs")
        lines.append("")
        for rid in paper.run_ids:
            lines.append(f"- `{rid}`")
        lines.append("")

    # Write
    path = _paper_dir(paper_id) / "current_status.md"
    path.write_text("\n".join(lines))
    logger.info("Regenerated current_status.md for '%s'", paper_id)
    return path


def _generate_summary(paper, open_tasks, decisions) -> str:
    """Rule-based 'What's Happening' — 2-3 sentences max."""
    parts = []

    # Stage + status context
    stage_desc = {
        "idea": "テーマ検討中",
        "rq_formation": "RQ を明確化中",
        "literature_review": "文献レビュー実行中",
        "data_feasibility": "データソースの実現可能性を検証中",
        "hypothesis": "仮説の生成・評価中",
        "data_collection": "データ収集中",
        "analysis": "実証分析の実行中",
        "draft": "論文を執筆中",
        "revision": "論文を改訂中",
        "submission": "投稿準備中",
        "under_review": "査読結果を待機中",
        "accepted": "採択済み",
        "rejected": "不採択",
        "abandoned": "中止",
    }
    base = stage_desc.get(paper.current_stage, paper.current_stage)

    if paper.current_status == "blocked":
        parts.append(f"{base}（ブロック中）。")
    elif paper.current_status == "paused":
        parts.append(f"{base}（一時停止中）。")
    else:
        parts.append(f"{base}。")

    # Notes override (user-written context)
    if paper.notes:
        parts.append(paper.notes)
        return " ".join(parts)

    # Critical/high tasks
    urgent = [t for t in open_tasks if t.priority in ("critical", "high") and t.status != "blocked"]
    if urgent:
        names = ", ".join(t.content[:40] for t in urgent[:2])
        parts.append(f"直近の作業: {names}。")

    # Blocked tasks
    blocked = [t for t in open_tasks if t.status == "blocked"]
    if blocked:
        reasons = set()
        for t in blocked:
            if t.blocked_reason and not t.blocked_reason.startswith("Waiting on:"):
                reasons.add(t.blocked_reason[:40])
            elif t.depends_on:
                reasons.add(f"{', '.join(t.depends_on)} の完了待ち")
        if reasons:
            parts.append(f"ブロッカー: {'; '.join(reasons)}。")

    # Latest decision
    if decisions:
        latest = decisions[-1]
        parts.append(f"直近の判断: {latest.decision[:50]}。")

    return " ".join(parts)


def _extract_milestones(stages) -> list[tuple[str, str]]:
    """Extract key milestones from stage history (stage transitions only)."""
    milestones = []
    for r in stages:
        d = r.to_dict()
        if d["type"] != "stage_transition":
            continue
        ts = d.get("effective_at", "")[:10]
        to_stage = d["to_stage"]
        reason = d.get("entry_reason", "")

        # Skip init bootstrap record
        if reason.startswith("Initialized at stage"):
            continue

        # Gate results
        gate = d.get("gate_result")
        if gate:
            icon = "✅" if gate.get("passed") else "❌"
            milestones.append((ts, f"{to_stage} Gate {icon}"))
        else:
            short_reason = f" — {reason[:60]}" if reason else ""
            milestones.append((ts, f"{to_stage}{short_reason}"))

    # Sort chronologically and deduplicate init record
    milestones.sort(key=lambda m: m[0])
    return milestones
