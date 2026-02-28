# src/notion/period_schema.py
"""Notion property builders for the Period / Weekly Log / Monthly Log databases.

Centralizes the EXACT property names required by:
  - PERIOD_DB (Weekly / Monthly period rows)
  - WEEKLY_LOG (062/063 target)
  - MONTHLY_LOG (064/065 target)

Property names MUST match the Notion database schemas exactly.

Architecture
------------
Period rows are shared parent records that link to Daily, Weekly, and Monthly logs.
Each Weekly/Monthly log page is linked to exactly one Period row.

  062 → Weekly Planning   (Big 3, Value Links, Success Criteria, Execution Plan)
  063 → Weekly Review     (Wins 3, Improvements 3, Value Alignment Score, Adjustment)
  064 → Monthly Planning  (Big 3, Value Links, Breakdown to Weeks, Strategic Rationale)
  065 → Monthly Review    (Success 3, Improvements 3, Value Adjustment, Structural Lessons)
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


def _date(iso_str: str) -> dict:
    if not iso_str:
        return {"date": None}
    return {"date": {"start": iso_str}}


def _relation(page_ids: List[str]) -> dict:
    return {"relation": [{"id": pid} for pid in page_ids if pid]}


def _checkbox(val: bool) -> dict:
    return {"checkbox": val}


# ── PERIOD_DB properties ─────────────────────────────────────────

def build_period_properties(
    *,
    name: str,
    period_type: str,       # "Weekly" or "Monthly"
    start_date: str,        # YYYY-MM-DD
    end_date: str,          # YYYY-MM-DD
    status: str = "Open",   # "Open" or "Closed"
) -> Dict[str, Any]:
    """Build Notion properties for a PERIOD_DB page."""
    return {
        "Name": _title(name),
        "Period Type": _select(period_type),
        "Start Date": _date(start_date),
        "End Date": _date(end_date),
        "Status": _select(status),
    }


# ── WEEKLY_LOG properties ────────────────────────────────────────
# Layer ownership:
#   062 (planning): Title, Period, Log Type=Planning, Big 3, Value Links,
#                   Success Criteria, Execution Plan, Confidence,
#                   Voice Transcript, LLM Summary
#   063 (review):   Title, Period, Log Type=Review, Wins 3, Improvements 3,
#                   Value Alignment Score, Adjustment Proposal,
#                   Voice Transcript, LLM Summary, Evidence Alignment Logs

def build_weekly_log_properties(
    *,
    title: str,
    period_id: str = "",
    log_type: str = "",             # "Planning" or "Review"
    # 062 planning fields
    big_3: str = "",
    value_link_ids: Optional[List[str]] = None,
    success_criteria: str = "",
    execution_plan: str = "",
    confidence: Optional[float] = None,
    # 063 review fields
    wins_3: str = "",
    improvements_3: str = "",
    value_alignment_score: Optional[float] = None,
    adjustment_proposal: str = "",
    # shared
    voice_transcript: str = "",
    llm_summary: str = "",
    evidence_log_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build Notion properties for a WEEKLY_LOG page.

    Only set properties that are non-empty / non-None so that each
    script layer can call this with just its own fields.
    """
    props: Dict[str, Any] = {
        "Title": _title(title),
    }

    if period_id:
        props["Period"] = _relation([period_id])
    if log_type:
        props["Log Type"] = _select(log_type)

    # ── 062 planning fields ──
    if big_3:
        props["Big 3"] = _rich_text(big_3)
    if value_link_ids:
        props["Value Links"] = _relation(value_link_ids)
    if success_criteria:
        props["Success Criteria"] = _rich_text(success_criteria)
    if execution_plan:
        props["Execution Plan"] = _rich_text(execution_plan)
    if confidence is not None:
        props["Confidence"] = _number(confidence)

    # ── 063 review fields ──
    if wins_3:
        props["Wins 3"] = _rich_text(wins_3)
    if improvements_3:
        props["Improvements 3"] = _rich_text(improvements_3)
    if value_alignment_score is not None:
        props["Value Alignment Score"] = _number(value_alignment_score)
    if adjustment_proposal:
        props["Adjustment Proposal"] = _rich_text(adjustment_proposal)

    # ── shared fields ──
    if voice_transcript:
        props["Voice Transcript"] = _rich_text(voice_transcript)
    if llm_summary:
        props["LLM Summary"] = _rich_text(llm_summary)
    if evidence_log_ids:
        props["Evidence Alignment Logs"] = _relation(evidence_log_ids)

    return props


# ── MONTHLY_LOG properties ───────────────────────────────────────
# Layer ownership:
#   064 (planning): Title, Period, Log Type=Planning, Big 3, Value Links,
#                   Breakdown to Weeks, Strategic Rationale, Risks,
#                   Confidence, Voice Transcript, LLM Summary
#   065 (review):   Title, Period, Log Type=Review, Success 3, Improvements 3,
#                   Value Adjustment Needed, Value Adjustment Proposal,
#                   Structural Lessons, Voice Transcript, LLM Summary,
#                   Evidence Alignment Logs

def build_monthly_log_properties(
    *,
    title: str,
    period_id: str = "",
    log_type: str = "",             # "Planning" or "Review"
    # 064 planning fields
    big_3: str = "",
    value_link_ids: Optional[List[str]] = None,
    breakdown_week_ids: Optional[List[str]] = None,
    strategic_rationale: str = "",
    risks: str = "",
    confidence: Optional[float] = None,
    # 065 review fields
    success_3: str = "",
    improvements_3: str = "",
    value_adjustment_needed: Optional[bool] = None,
    value_adjustment_proposal: str = "",
    structural_lessons: str = "",
    # shared
    voice_transcript: str = "",
    llm_summary: str = "",
    evidence_log_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build Notion properties for a MONTHLY_LOG page.

    Only set properties that are non-empty / non-None so that each
    script layer can call this with just its own fields.
    """
    props: Dict[str, Any] = {
        "Title": _title(title),
    }

    if period_id:
        props["Period"] = _relation([period_id])
    if log_type:
        props["Log Type"] = _select(log_type)

    # ── 064 planning fields ──
    if big_3:
        props["Big 3"] = _rich_text(big_3)
    if value_link_ids:
        props["Value Links"] = _relation(value_link_ids)
    if breakdown_week_ids:
        props["Breakdown to Weeks"] = _relation(breakdown_week_ids)
    if strategic_rationale:
        props["Strategic Rationale"] = _rich_text(strategic_rationale)
    if risks:
        props["Risks"] = _rich_text(risks)
    if confidence is not None:
        props["Confidence"] = _number(confidence)

    # ── 065 review fields ──
    if success_3:
        props["Success 3"] = _rich_text(success_3)
    if improvements_3:
        props["Improvements 3"] = _rich_text(improvements_3)
    if value_adjustment_needed is not None:
        props["Value Adjustment Needed"] = _checkbox(value_adjustment_needed)
    if value_adjustment_proposal:
        props["Value Adjustment Proposal"] = _rich_text(value_adjustment_proposal)
    if structural_lessons:
        props["Structural Lessons"] = _rich_text(structural_lessons)

    # ── shared fields ──
    if voice_transcript:
        props["Voice Transcript"] = _rich_text(voice_transcript)
    if llm_summary:
        props["LLM Summary"] = _rich_text(llm_summary)
    if evidence_log_ids:
        props["Evidence Alignment Logs"] = _relation(evidence_log_ids)

    return props
