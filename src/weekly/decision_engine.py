# src/weekly/decision_engine.py
"""Rule-based weekly decision & summary engine.

Pure functions that aggregate outputs from 047–051 into cohesive
weekly intelligence products: RQ progress assessments, next-week
action plans, worldview narratives, and executive summaries.

No I/O — receives plain dicts/lists, returns plain dicts/strings.
May import safe ``src`` modules (constants, schemas) but never
imports Notion client, repos, or anything that performs I/O.

All field access uses ``.get(key, default)`` for robustness against
upstream schema changes.

Usage::

    from src.weekly.decision_engine import (
        assess_rq_progress,
        build_next_week_actions,
        draft_worldview_sections,
        compile_weekly_summary,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------

# Progress-level thresholds
_MEANINGFUL_EVIDENCE_MIN = 3
_STRUCTURAL_EVIDENCE_MIN = 6
_HIGH_RQ_RELEVANCE = 0.7

# Priority ordering (higher index = higher priority)
_PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]

# Target action categories from 050
_DROP_ACTIONS = {"disable"}
_DEPRIORITIZE_ACTIONS = {"deprioritize"}
_UPGRADE_ACTIONS = {"upgrade"}


# ----------------------------------------------------------------
# 1. RQ Progress Assessment
# ----------------------------------------------------------------

def assess_rq_progress(
    rq_statuses: List[dict],
    papers: Optional[List[dict]] = None,
    events: Optional[List[dict]] = None,
) -> List[dict]:
    """Assess each RQ's weekly progress with evidence-based reasoning.

    Parameters
    ----------
    rq_statuses:
        Records from ``049_weekly_rq_status/rq_status.json``.
    papers:
        Records from ``047_weekly_papers_review/papers.json`` (optional).
    events:
        Records from ``048_weekly_events_digest/events.json`` (optional).

    Returns
    -------
    list[dict]
        Per-RQ assessment with progress_level, evidence_summary, and
        recommended_actions.
    """
    papers = papers or []
    events = events or []

    # Build a quick lookup: rq_id → related paper titles (from 049 linkage)
    # Papers list is the raw 047 output; RQ linkage is in rq_status records.
    assessments: List[dict] = []

    for rq in rq_statuses:
        rq_id = rq.get("rq_id", "")
        rq_title = rq.get("rq_title", "Untitled RQ")
        priority = rq.get("priority", "Medium")
        evidence_count = rq.get("evidence_count", 0)
        related_papers_list = rq.get("related_papers", [])
        related_events_list = rq.get("related_events", [])
        open_gaps = rq.get("open_gaps", "")
        current_approach = rq.get("current_approach", "")

        # Count evidence types available
        has_papers = len(related_papers_list) > 0
        has_events = len(related_events_list) > 0

        # Check for high-relevance papers (from 049 linkage or 047 data)
        has_high_relevance_paper = False
        for p in related_papers_list:
            # 049 links papers as dicts or strings; handle both
            if isinstance(p, dict):
                relevance = p.get("RQ Relevance", p.get("score", 0.0))
                if isinstance(relevance, (int, float)) and relevance >= _HIGH_RQ_RELEVANCE:
                    has_high_relevance_paper = True
                    break

        # Determine progress level
        progress_level = _classify_progress(
            evidence_count=evidence_count,
            has_papers=has_papers,
            has_events=has_events,
            has_high_relevance_paper=has_high_relevance_paper,
        )

        # Build evidence summary
        evidence_summary = _build_evidence_summary(
            related_papers_list, related_events_list, evidence_count,
        )

        # Build reason string
        reason = _build_progress_reason(
            progress_level, evidence_count, has_papers, has_events,
            has_high_relevance_paper,
        )

        # Recommend next-week actions
        recommended_actions = _recommend_rq_actions(
            progress_level, priority, open_gaps, current_approach,
        )

        assessments.append({
            "rq_id": rq_id,
            "rq_title": rq_title,
            "priority": priority,
            "progress_level": progress_level,
            "evidence_count": evidence_count,
            "evidence_summary": evidence_summary,
            "open_gaps": open_gaps,
            "recommended_actions": recommended_actions,
            "reason": reason,
        })

    return assessments


def _classify_progress(
    *,
    evidence_count: int,
    has_papers: bool,
    has_events: bool,
    has_high_relevance_paper: bool,
) -> str:
    """Classify progress level based on evidence counts and types."""
    if evidence_count <= 0:
        return "none"
    if evidence_count >= _STRUCTURAL_EVIDENCE_MIN:
        return "structural"
    if has_papers and has_events:
        # Cross-type evidence → structural even with fewer items
        return "structural"
    if evidence_count >= _MEANINGFUL_EVIDENCE_MIN:
        return "meaningful"
    if has_high_relevance_paper:
        return "meaningful"
    return "minor"


def _build_evidence_summary(
    related_papers: List[Any],
    related_events: List[Any],
    evidence_count: int,
) -> str:
    """Build a 1–2 sentence summary of linked evidence."""
    parts: List[str] = []
    n_papers = len(related_papers)
    n_events = len(related_events)

    if evidence_count == 0:
        return "No evidence linked this week."

    if n_events > 0:
        # Collect up to 2 event titles for illustration
        event_titles: List[str] = []
        for ev in related_events[:2]:
            title = ev.get("title", "") if isinstance(ev, dict) else str(ev)
            if title:
                event_titles.append(title)
        if event_titles:
            parts.append(
                f"{n_events} event(s) linked"
                + (f", e.g. \"{event_titles[0][:80]}\"" if event_titles else "")
            )
        else:
            parts.append(f"{n_events} event(s) linked")

    if n_papers > 0:
        parts.append(f"{n_papers} paper(s) linked")

    return "; ".join(parts) + "."


def _build_progress_reason(
    progress_level: str,
    evidence_count: int,
    has_papers: bool,
    has_events: bool,
    has_high_relevance_paper: bool,
) -> str:
    """Explain why this progress level was assigned."""
    if progress_level == "none":
        return "No evidence linked to this RQ this week."
    if progress_level == "structural":
        if has_papers and has_events:
            return (
                f"{evidence_count} evidence items with cross-type coverage "
                f"(both papers and events)."
            )
        return f"{evidence_count} evidence items (≥{_STRUCTURAL_EVIDENCE_MIN} threshold)."
    if progress_level == "meaningful":
        if has_high_relevance_paper:
            return (
                f"{evidence_count} evidence item(s); includes ≥1 paper "
                f"with RQ Relevance ≥ {_HIGH_RQ_RELEVANCE}."
            )
        return f"{evidence_count} evidence items (≥{_MEANINGFUL_EVIDENCE_MIN} threshold)."
    # minor
    return f"{evidence_count} evidence item(s); no high-relevance paper found."


def _recommend_rq_actions(
    progress_level: str,
    priority: str,
    open_gaps: str,
    current_approach: str,
) -> List[str]:
    """Generate concrete next-week actions for this RQ."""
    actions: List[str] = []

    if progress_level == "none":
        actions.append("Review monitoring targets — ensure relevant keywords cover this RQ.")
        if _priority_rank(priority) >= _priority_rank("High"):
            actions.append("Consider adding dedicated search terms for this high-priority RQ.")
    elif progress_level == "minor":
        actions.append("Look for additional sources or broaden search scope.")
    elif progress_level in ("meaningful", "structural"):
        actions.append("Synthesize evidence and update RQ status/notes.")

    if open_gaps and open_gaps not in ("TBD", ""):
        actions.append(f"Address gap: {open_gaps[:120]}")

    return actions


def _priority_rank(priority: str) -> int:
    """Return numeric rank for a priority string (higher = more urgent)."""
    try:
        return _PRIORITY_ORDER.index(priority)
    except ValueError:
        return 1  # default to Medium


# ----------------------------------------------------------------
# 2. Next-Week Action Plan
# ----------------------------------------------------------------

def build_next_week_actions(
    rq_assessments: List[dict],
    target_reviews: Optional[List[dict]] = None,
    candidates: Optional[List[dict]] = None,
    *,
    candidate_top_n: int = 5,
) -> dict:
    """Build an aggregated next-week action plan.

    Parameters
    ----------
    rq_assessments:
        Output of :func:`assess_rq_progress`.
    target_reviews:
        Records from ``050_weekly_targets_review/targets_review.json``.
    candidates:
        Records from ``051_weekly_discovery_expansion/candidates.json``.
    candidate_top_n:
        How many top discovery candidates to include.

    Returns
    -------
    dict
        Structured action plan with priority_rqs, target_changes,
        new_candidates, and monitoring_health.
    """
    target_reviews = target_reviews or []
    candidates = candidates or []

    # Priority RQs: sorted by urgency (none-progress + high-priority first)
    priority_rqs = sorted(
        rq_assessments,
        key=lambda rq: (
            # Primary: lower progress → higher urgency
            {"none": 0, "minor": 1, "meaningful": 2, "structural": 3}.get(
                rq.get("progress_level", "none"), 0
            ),
            # Secondary: higher priority → higher urgency (negate rank)
            -_priority_rank(rq.get("priority", "Medium")),
        ),
    )

    # Target changes from 050
    drop = []
    deprioritize = []
    upgrade = []
    tune_keywords = []

    for t in target_reviews:
        action = t.get("action", "keep").lower()
        summary = {
            "target_id": t.get("target_id", ""),
            "target_name": t.get("target_name", ""),
            "target_type": t.get("target_type", ""),
            "reason": t.get("reason", ""),
            "signal_score": t.get("signal_score"),
            "noise_score": t.get("noise_score"),
        }

        if action in _DROP_ACTIONS:
            drop.append(summary)
        elif action in _DEPRIORITIZE_ACTIONS:
            deprioritize.append(summary)
        elif action in _UPGRADE_ACTIONS:
            upgrade.append(summary)

        # Keyword tuning: include if there are suggestions
        ks = t.get("keyword_suggestions", {})
        if isinstance(ks, dict):
            adds = ks.get("keywords_to_add", [])
            stale = ks.get("keywords_stale", [])
            excludes = ks.get("keywords_to_exclude", [])
            if adds or stale or excludes:
                tune_keywords.append({
                    "target_id": t.get("target_id", ""),
                    "target_name": t.get("target_name", ""),
                    "keywords_to_add": len(adds),
                    "keywords_stale": len(stale),
                    "keywords_to_exclude": len(excludes),
                })

    # New candidates from 051 (not already tracked, sorted by score)
    new_candidates = [
        {
            "candidate_name": c.get("candidate_name", ""),
            "type": c.get("type", "Unknown"),
            "final_score": c.get("final_score", 0.0),
            "mention_count": c.get("mention_count", 0),
            "why_notable": c.get("why_notable", ""),
            "already_tracked": c.get("already_tracked", False),
        }
        for c in sorted(
            candidates,
            key=lambda x: x.get("final_score", 0.0),
            reverse=True,
        )
        if not c.get("already_tracked", False)
    ][:candidate_top_n]

    # Monitoring health
    monitoring_health = _compute_monitoring_health(target_reviews)

    return {
        "priority_rqs": [
            {
                "rq_id": rq.get("rq_id", ""),
                "rq_title": rq.get("rq_title", ""),
                "priority": rq.get("priority", ""),
                "progress_level": rq.get("progress_level", "none"),
                "recommended_actions": rq.get("recommended_actions", []),
            }
            for rq in priority_rqs
        ],
        "target_changes": {
            "drop": drop,
            "deprioritize": deprioritize,
            "upgrade": upgrade,
            "tune_keywords": tune_keywords,
        },
        "new_candidates": new_candidates,
        "monitoring_health": monitoring_health,
    }


def _compute_monitoring_health(target_reviews: List[dict]) -> dict:
    """Compute aggregate monitoring health metrics."""
    if not target_reviews:
        return {
            "total_targets": 0,
            "active_events_this_week": 0,
            "targets_with_zero_events": 0,
            "avg_signal_score": 0.0,
            "avg_noise_score": 0.0,
        }

    total = len(target_reviews)
    total_events = sum(t.get("number_of_events", 0) for t in target_reviews)
    zero_event_targets = sum(
        1 for t in target_reviews if t.get("number_of_events", 0) == 0
    )

    signal_scores = [
        t.get("signal_score", 0.0)
        for t in target_reviews
        if isinstance(t.get("signal_score"), (int, float))
    ]
    noise_scores = [
        t.get("noise_score", 0.0)
        for t in target_reviews
        if isinstance(t.get("noise_score"), (int, float))
    ]

    return {
        "total_targets": total,
        "active_events_this_week": total_events,
        "targets_with_zero_events": zero_event_targets,
        "avg_signal_score": round(
            sum(signal_scores) / len(signal_scores), 3
        ) if signal_scores else 0.0,
        "avg_noise_score": round(
            sum(noise_scores) / len(noise_scores), 3
        ) if noise_scores else 0.0,
    }


# ----------------------------------------------------------------
# 3. Worldview Narrative
# ----------------------------------------------------------------

def draft_worldview_sections(
    rq_assessments: List[dict],
    target_reviews: Optional[List[dict]] = None,
    candidates: Optional[List[dict]] = None,
    events: Optional[List[dict]] = None,
    *,
    week_id: str = "",
) -> str:
    """Generate a structured weekly worldview narrative (Markdown).

    Parameters
    ----------
    rq_assessments:
        Output of :func:`assess_rq_progress`.
    target_reviews:
        Records from 050 (optional).
    candidates:
        Records from 051 (optional).
    events:
        Records from 048 (optional).
    week_id:
        ISO week string for the header (e.g. ``"2026-W08"``).

    Returns
    -------
    str
        Full Markdown content for ``weekly_worldview.md``.
    """
    target_reviews = target_reviews or []
    candidates = candidates or []
    events = events or []
    sections: List[str] = []

    # --- Section 1: Week Overview ---
    sections.append(_section_week_overview(
        week_id, rq_assessments, target_reviews, events, candidates,
    ))

    # --- Section 2: Research Progress ---
    sections.append(_section_research_progress(rq_assessments))

    # --- Section 3: Monitoring Landscape ---
    sections.append(_section_monitoring_landscape(target_reviews))

    # --- Section 4: Emerging Signals ---
    sections.append(_section_emerging_signals(candidates))

    # --- Section 5: Gaps & Blind Spots ---
    sections.append(_section_gaps_and_blindspots(rq_assessments, target_reviews))

    return "\n\n".join(sections) + "\n"


def _section_week_overview(
    week_id: str,
    rq_assessments: List[dict],
    target_reviews: List[dict],
    events: List[dict],
    candidates: List[dict],
) -> str:
    """Section 1: high-level stats."""
    lines = [f"# Weekly Worldview — {week_id or 'Current Week'}", ""]
    lines.append("## 1. Week Overview")
    lines.append("")
    lines.append(f"- **Research Questions tracked**: {len(rq_assessments)}")
    lines.append(f"- **Monitoring targets reviewed**: {len(target_reviews)}")
    lines.append(f"- **Events ingested**: {len(events)}")
    lines.append(f"- **Discovery candidates found**: {len(candidates)}")

    # Progress distribution
    levels = {}
    for rq in rq_assessments:
        lvl = rq.get("progress_level", "none")
        levels[lvl] = levels.get(lvl, 0) + 1
    if levels:
        lines.append("")
        lines.append("**RQ progress distribution:**")
        for lvl in ("structural", "meaningful", "minor", "none"):
            if lvl in levels:
                lines.append(f"- {lvl}: {levels[lvl]}")

    return "\n".join(lines)


def _section_research_progress(rq_assessments: List[dict]) -> str:
    """Section 2: per-RQ progress with evidence highlights."""
    lines = ["## 2. Research Progress", ""]

    if not rq_assessments:
        lines.append("_No RQ status data available._")
        return "\n".join(lines)

    # Group by progress level for narrative structure
    by_level: Dict[str, List[dict]] = {}
    for rq in rq_assessments:
        lvl = rq.get("progress_level", "none")
        by_level.setdefault(lvl, []).append(rq)

    for lvl in ("structural", "meaningful", "minor", "none"):
        rqs = by_level.get(lvl, [])
        if not rqs:
            continue
        lines.append(f"### {lvl.capitalize()} progress ({len(rqs)} RQ{'s' if len(rqs) != 1 else ''})")
        lines.append("")
        for rq in rqs:
            title = rq.get("rq_title", "Untitled")
            priority = rq.get("priority", "")
            evidence_summary = rq.get("evidence_summary", "")
            lines.append(f"- **{title}** [{priority}]")
            if evidence_summary:
                lines.append(f"  - {evidence_summary}")
            reason = rq.get("reason", "")
            if reason:
                lines.append(f"  - _{reason}_")
        lines.append("")

    return "\n".join(lines)


def _section_monitoring_landscape(target_reviews: List[dict]) -> str:
    """Section 3: target health and notable changes."""
    lines = ["## 3. Monitoring Landscape", ""]

    if not target_reviews:
        lines.append("_No target review data available._")
        return "\n".join(lines)

    health = _compute_monitoring_health(target_reviews)
    lines.append(
        f"Reviewed **{health['total_targets']}** targets with "
        f"**{health['active_events_this_week']}** total events this week."
    )
    lines.append(
        f"Average signal score: {health['avg_signal_score']:.2f} | "
        f"Average noise score: {health['avg_noise_score']:.2f}"
    )
    if health["targets_with_zero_events"] > 0:
        lines.append(
            f"⚠ **{health['targets_with_zero_events']}** target(s) had zero events."
        )
    lines.append("")

    # Notable changes
    changes = {"disable": [], "deprioritize": [], "upgrade": []}
    for t in target_reviews:
        action = t.get("action", "keep").lower()
        if action in changes:
            changes[action].append(t)

    if any(changes.values()):
        lines.append("### Proposed changes")
        lines.append("")
        for action_type, targets in changes.items():
            if targets:
                lines.append(f"**{action_type.capitalize()}** ({len(targets)}):")
                for t in targets[:5]:
                    name = t.get("target_name", "Unknown")
                    reason = t.get("reason", "")
                    lines.append(f"- {name}: {reason[:100]}")
                lines.append("")

    return "\n".join(lines)


def _section_emerging_signals(candidates: List[dict]) -> str:
    """Section 4: top discovery candidates."""
    lines = ["## 4. Emerging Signals", ""]

    if not candidates:
        lines.append("_No discovery candidates available._")
        return "\n".join(lines)

    # Top candidates not already tracked
    new = [c for c in candidates if not c.get("already_tracked", False)]
    new_sorted = sorted(new, key=lambda c: c.get("final_score", 0.0), reverse=True)

    if not new_sorted:
        lines.append("_All discovered candidates are already tracked._")
        return "\n".join(lines)

    lines.append(f"**{len(new_sorted)}** new candidate(s) not yet tracked:")
    lines.append("")
    for c in new_sorted[:10]:
        name = c.get("candidate_name", "Unknown")
        ctype = c.get("type", "Unknown")
        score = c.get("final_score", 0.0)
        mentions = c.get("mention_count", 0)
        why = c.get("why_notable", "")
        lines.append(f"- **{name}** ({ctype}, score={score:.2f}, {mentions} mentions)")
        if why:
            lines.append(f"  - {why[:150]}")

    return "\n".join(lines)


def _section_gaps_and_blindspots(
    rq_assessments: List[dict],
    target_reviews: List[dict],
) -> str:
    """Section 5: RQs with no progress, targets with zero events."""
    lines = ["## 5. Gaps & Blind Spots", ""]

    # RQs with no progress
    no_progress = [
        rq for rq in rq_assessments
        if rq.get("progress_level") == "none"
    ]
    if no_progress:
        lines.append(f"### RQs with no evidence this week ({len(no_progress)})")
        lines.append("")
        for rq in no_progress:
            title = rq.get("rq_title", "Untitled")
            priority = rq.get("priority", "")
            lines.append(f"- **{title}** [{priority}]")
        lines.append("")
    else:
        lines.append("All RQs have at least some evidence this week.")
        lines.append("")

    # Targets with zero events
    zero_targets = [
        t for t in target_reviews
        if t.get("number_of_events", 0) == 0
    ]
    if zero_targets:
        lines.append(
            f"### Targets with zero events ({len(zero_targets)})"
        )
        lines.append("")
        for t in zero_targets[:10]:
            name = t.get("target_name", "Unknown")
            ttype = t.get("target_type", "")
            days = t.get("days_since_last_event")
            detail = f" (last event {days:.0f}d ago)" if isinstance(days, (int, float)) else ""
            lines.append(f"- {name} [{ttype}]{detail}")
        if len(zero_targets) > 10:
            lines.append(f"- ... and {len(zero_targets) - 10} more")
        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------
# 4. Weekly Summary (executive-level)
# ----------------------------------------------------------------

def compile_weekly_summary(
    rq_assessments: List[dict],
    next_actions: dict,
    *,
    week_id: str = "",
    papers_count: int = 0,
    events_count: int = 0,
    targets_count: int = 0,
    candidates_count: int = 0,
) -> str:
    """Generate an executive-level weekly summary (Markdown).

    Parameters
    ----------
    rq_assessments:
        Output of :func:`assess_rq_progress`.
    next_actions:
        Output of :func:`build_next_week_actions`.
    week_id:
        ISO week string.
    papers_count, events_count, targets_count, candidates_count:
        Raw counts from upstream file loading.

    Returns
    -------
    str
        Markdown content for ``weekly_summary.md``.
    """
    lines: List[str] = []

    lines.append(f"# Weekly Summary — {week_id or 'Current Week'}")
    lines.append("")

    # TL;DR
    lines.append("## TL;DR")
    lines.append("")
    tldr = _build_tldr(rq_assessments, next_actions)
    for bullet in tldr:
        lines.append(f"- {bullet}")
    lines.append("")

    # Key Findings
    lines.append("## Key Findings")
    lines.append("")
    findings = _extract_key_findings(rq_assessments)
    if findings:
        for f in findings:
            lines.append(f"- {f}")
    else:
        lines.append("_No significant findings this week._")
    lines.append("")

    # Decisions Made
    lines.append("## Decisions Made")
    lines.append("")
    tc = next_actions.get("target_changes", {})
    drop_count = len(tc.get("drop", []))
    depr_count = len(tc.get("deprioritize", []))
    upgr_count = len(tc.get("upgrade", []))
    tune_count = len(tc.get("tune_keywords", []))

    if any([drop_count, depr_count, upgr_count, tune_count]):
        if drop_count:
            lines.append(f"- **Drop**: {drop_count} target(s)")
        if depr_count:
            lines.append(f"- **Deprioritize**: {depr_count} target(s)")
        if upgr_count:
            lines.append(f"- **Upgrade**: {upgr_count} target(s)")
        if tune_count:
            lines.append(f"- **Keyword tuning**: {tune_count} target(s)")
    else:
        lines.append("_No target changes proposed this week._")
    lines.append("")

    # Next Week Focus
    lines.append("## Next Week Focus")
    lines.append("")
    priority_rqs = next_actions.get("priority_rqs", [])
    top_focus = [
        rq for rq in priority_rqs
        if rq.get("progress_level") in ("none", "minor")
    ][:3]
    if top_focus:
        for rq in top_focus:
            title = rq.get("rq_title", "")
            lvl = rq.get("progress_level", "none")
            actions = rq.get("recommended_actions", [])
            lines.append(f"- **{title}** ({lvl} progress)")
            for a in actions[:2]:
                lines.append(f"  - {a}")
    else:
        lines.append("_All RQs have meaningful progress — continue current approach._")

    new_cands = next_actions.get("new_candidates", [])
    if new_cands:
        lines.append("")
        lines.append(f"**New candidates to evaluate**: {len(new_cands)}")
        for c in new_cands[:3]:
            lines.append(
                f"- {c.get('candidate_name', '?')} "
                f"({c.get('type', '?')}, score={c.get('final_score', 0):.2f})"
            )
    lines.append("")

    # Numbers
    lines.append("## Numbers")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Papers reviewed | {papers_count} |")
    lines.append(f"| Events ingested | {events_count} |")
    lines.append(f"| Targets reviewed | {targets_count} |")
    lines.append(f"| Discovery candidates | {candidates_count} |")
    lines.append(f"| RQs assessed | {len(rq_assessments)} |")

    health = next_actions.get("monitoring_health", {})
    avg_signal = health.get("avg_signal_score", 0.0)
    avg_noise = health.get("avg_noise_score", 0.0)
    if avg_signal or avg_noise:
        lines.append(f"| Avg signal score | {avg_signal:.2f} |")
        lines.append(f"| Avg noise score | {avg_noise:.2f} |")
    lines.append("")

    return "\n".join(lines) + "\n"


def _build_tldr(
    rq_assessments: List[dict],
    next_actions: dict,
) -> List[str]:
    """Generate 3 TL;DR bullets."""
    bullets: List[str] = []

    # Bullet 1: progress summary
    levels = {}
    for rq in rq_assessments:
        lvl = rq.get("progress_level", "none")
        levels[lvl] = levels.get(lvl, 0) + 1
    structural = levels.get("structural", 0)
    meaningful = levels.get("meaningful", 0)
    minor = levels.get("minor", 0)
    none_count = levels.get("none", 0)
    total = len(rq_assessments)

    if structural + meaningful > 0:
        bullets.append(
            f"{structural + meaningful}/{total} RQs saw meaningful+ progress this week."
        )
    else:
        bullets.append(f"Limited progress across {total} tracked RQs this week.")

    # Bullet 2: monitoring health
    health = next_actions.get("monitoring_health", {})
    total_targets = health.get("total_targets", 0)
    total_events = health.get("active_events_this_week", 0)
    if total_targets > 0:
        bullets.append(
            f"Monitoring: {total_events} events across {total_targets} targets "
            f"(avg signal={health.get('avg_signal_score', 0):.2f})."
        )
    else:
        bullets.append("No monitoring target data available.")

    # Bullet 3: action items
    tc = next_actions.get("target_changes", {})
    new_cands = next_actions.get("new_candidates", [])
    action_count = (
        len(tc.get("drop", [])) +
        len(tc.get("deprioritize", [])) +
        len(tc.get("upgrade", []))
    )
    if action_count > 0 or new_cands:
        parts = []
        if action_count:
            parts.append(f"{action_count} target change(s)")
        if new_cands:
            parts.append(f"{len(new_cands)} new candidate(s) to evaluate")
        bullets.append("Action items: " + ", ".join(parts) + ".")
    else:
        bullets.append("No urgent action items for next week.")

    return bullets


def _extract_key_findings(rq_assessments: List[dict]) -> List[str]:
    """Extract the most noteworthy findings from RQ assessments."""
    findings: List[str] = []

    # Pick RQs with structural or meaningful progress
    for rq in rq_assessments:
        if rq.get("progress_level") in ("structural", "meaningful"):
            title = rq.get("rq_title", "Untitled")
            summary = rq.get("evidence_summary", "")
            if summary:
                findings.append(f"**{title}**: {summary}")

    return findings[:5]  # cap at 5
