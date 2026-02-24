# src/notion/daily_schema.py
"""Notion property builders for the Daily System databases.

Centralizes the EXACT property names required by the four Daily DBs
(Daily Logs, Today Commits, Meeting Briefs, Follow-ups) and provides
builder functions that produce Notion API property payloads.

Property names MUST match the Notion database schemas exactly.

Architecture
------------
Daily Logs is the single daily hub page, progressively enriched:
  057 → raw      (Raw Close Log, Satisfaction, Energy Level, Value Domains)
  058 → structured (Structured Summary, Friction/Blockers, Open Questions, ...)
  059 → prepared   (Prep Notes, Meeting Briefs relation, ...)
  060 → committed  (Final Top 3, Schedule / Time Blocks, Commits relation)

Each layer only sets fields it owns.  The ``Stage`` select tracks which
layer has been applied most recently.

IMPORTANT: The date property in Daily Logs is named **LogDate** (not Date).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Notion property value builders ────────────────────────────────

def _title(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": text}}]}


def _rich_text(text: str) -> dict:
    if not text:
        return {"rich_text": []}
    return {"rich_text": [{"type": "text", "text": {"content": text}}]}


def _number(val: Optional[float | int]) -> dict:
    return {"number": val}


def _select(name: str) -> dict:
    if not name:
        return {"select": None}
    return {"select": {"name": name}}


def _multi_select(names: List[str]) -> dict:
    return {"multi_select": [{"name": n} for n in names if n]}


def _date(iso_str: str) -> dict:
    if not iso_str:
        return {"date": None}
    return {"date": {"start": iso_str}}


def _relation(page_ids: List[str]) -> dict:
    return {"relation": [{"id": pid} for pid in page_ids if pid]}


# ── Daily Logs properties ─────────────────────────────────────────
# Layer ownership:
#   057 (raw):        Title, LogDate, Raw Close Log, Satisfaction,
#                     Energy Level, Value Domains, Stage=raw
#   058 (structured): Structured Summary, Friction / Blockers,
#                     Open Questions, Provisional Top 3,
#                     Value Domains (refined), Stage=structured
#   059 (prepared):   Prep Notes, Meeting Briefs (rel),
#                     Follow-ups (rel), Time Budget,
#                     Tomorrow Plan (raw), Stage=prepared
#   060 (committed):  Final Top 3, Schedule / Time Blocks,
#                     Commits (rel), Stage=committed

def build_daily_log_properties(
    *,
    title: str,
    date: str,
    # 057 raw layer
    raw_close_log: str = "",
    satisfaction: Optional[int] = None,
    energy_level: str = "",
    value_domains: Optional[List[str]] = None,
    # 058 structured layer
    structured_summary: str = "",
    friction_blockers: str = "",
    open_questions: str = "",
    provisional_top3: str = "",
    # 059 prepared layer
    prep_notes: str = "",
    tomorrow_plan_raw: str = "",
    time_budget_hrs: Optional[float] = None,
    meeting_brief_ids: Optional[List[str]] = None,
    follow_up_ids: Optional[List[str]] = None,
    # 060 committed layer
    final_top3: str = "",
    schedule_time_blocks: str = "",
    commit_ids: Optional[List[str]] = None,
    # meta
    stage: str = "",
    publish_status: str = "Draft",
) -> Dict[str, Any]:
    """Build Notion properties for a Daily Logs page.

    Only set properties that are non-empty / non-None so that each
    pipeline layer can call this with just its own fields and leave
    the rest untouched.
    """
    props: Dict[str, Any] = {
        "Title": _title(title),
        "LogDate": _date(date),
        "Publish Status": _select(publish_status),
    }
    # Stage
    if stage:
        props["Stage"] = _select(stage)

    # ── 057 raw fields ──
    if raw_close_log:
        props["Raw Close Log"] = _rich_text(raw_close_log)
    if satisfaction is not None:
        props["Satisfaction"] = _number(satisfaction)
    if energy_level:
        props["Energy Level"] = _select(energy_level)
    if value_domains:
        props["Value Domains"] = _multi_select(value_domains)

    # ── 058 structured fields ──
    if structured_summary:
        props["Structured Summary"] = _rich_text(structured_summary)
    if friction_blockers:
        props["Friction / Blockers"] = _rich_text(friction_blockers)
    if open_questions:
        props["Open Questions"] = _rich_text(open_questions)
    if provisional_top3:
        props["Provisional Top 3"] = _rich_text(provisional_top3)

    # ── 059 prepared fields ──
    if prep_notes:
        props["Prep Notes"] = _rich_text(prep_notes)
    if tomorrow_plan_raw:
        props["Tomorrow Plan (raw)"] = _rich_text(tomorrow_plan_raw)
    if time_budget_hrs is not None:
        props["Time Budget (hrs)"] = _number(time_budget_hrs)
    if meeting_brief_ids:
        props["Meeting Briefs"] = _relation(meeting_brief_ids)
    if follow_up_ids:
        props["Follow-ups"] = _relation(follow_up_ids)

    # ── 060 committed fields ──
    if final_top3:
        props["Final Top 3"] = _rich_text(final_top3)
    if schedule_time_blocks:
        props["Schedule / Time Blocks"] = _rich_text(schedule_time_blocks)
    if commit_ids:
        props["Commits"] = _relation(commit_ids)

    return props


# ── Today Commits properties ─────────────────────────────────────

def build_commit_properties(
    *,
    title: str,
    date: str,
    daily_log_id: str = "",
    rank: Optional[int] = None,
    status: str = "Planned",
    why: str = "",
    definition_of_done: str = "",
    planned_time_block: str = "",
    estimated_minutes: Optional[int] = None,
    order: Optional[int] = None,
    value_domains: Optional[List[str]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Build Notion properties for a Today Commits page."""
    props: Dict[str, Any] = {
        "Title": _title(title),
        "Date": _date(date),
        "Status": _select(status),
    }
    if daily_log_id:
        props["Daily Log"] = _relation([daily_log_id])
    if rank is not None:
        props["Rank"] = _number(rank)
    if why:
        props["Why"] = _rich_text(why)
    if definition_of_done:
        props["Definition of Done"] = _rich_text(definition_of_done)
    if planned_time_block:
        props["Planned Time Block"] = _rich_text(planned_time_block)
    if estimated_minutes is not None:
        props["Estimated Minutes"] = _number(estimated_minutes)
    if order is not None:
        props["Order"] = _number(order)
    if value_domains:
        props["Value Domains"] = _multi_select(value_domains)
    if notes:
        props["Notes"] = _rich_text(notes)
    return props


# ── Meeting Briefs properties ─────────────────────────────────────

def build_meeting_brief_properties(
    *,
    title: str,
    date: str,
    daily_log_id: str = "",
    people: Optional[List[str]] = None,
    purpose: str = "",
    context: str = "",
    key_questions: str = "",
    desired_outcomes: str = "",
    prep_checklist: str = "",
    links_materials: str = "",
    status: str = "Draft",
    created_by: str = "Auto",
) -> Dict[str, Any]:
    """Build Notion properties for a Meeting Briefs page."""
    props: Dict[str, Any] = {
        "Title": _title(title),
        "Date": _date(date),
        "Status": _select(status),
        "Created By": _select(created_by),
    }
    if daily_log_id:
        props["Daily Log"] = _relation([daily_log_id])
    if people:
        props["People"] = _multi_select(people)
    if purpose:
        props["Purpose"] = _rich_text(purpose)
    if context:
        props["Context"] = _rich_text(context)
    if key_questions:
        props["Key Questions"] = _rich_text(key_questions)
    if desired_outcomes:
        props["Desired Outcomes"] = _rich_text(desired_outcomes)
    if prep_checklist:
        props["Prep Checklist"] = _rich_text(prep_checklist)
    if links_materials:
        props["Links / Materials"] = _rich_text(links_materials)
    return props


# ── Follow-ups properties ────────────────────────────────────────

def build_follow_up_properties(
    *,
    title: str,
    date: str,
    due_date: str = "",
    daily_log_id: str = "",
    source: str = "Research",
    status: str = "Open",
    priority: str = "P1",
    assignee: str = "",
    draft_message: str = "",
    next_action: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    """Build Notion properties for a Follow-ups page."""
    props: Dict[str, Any] = {
        "Title": _title(title),
        "Date": _date(date),
        "Source": _select(source),
        "Status": _select(status),
        "Priority": _select(priority),
    }
    if due_date:
        props["Due Date"] = _date(due_date)
    if daily_log_id:
        props["Daily Log"] = _relation([daily_log_id])
    if assignee:
        props["Assignee"] = _select(assignee)
    if draft_message:
        props["Draft Message"] = _rich_text(draft_message)
    if next_action:
        props["Next Action"] = _rich_text(next_action)
    if notes:
        props["Notes"] = _rich_text(notes)
    return props
