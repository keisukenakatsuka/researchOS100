#!/usr/bin/env python
# src/scripts/061_notion_publish_morning_view.py
"""Daily Operational Console — Web UI for the 057-060 pipeline.

Provides a local FastAPI web interface that:

1. Lists available dates from local pipeline outputs (057-060)
2. Displays structured results for a selected date:
   - 057 raw close log
   - 058 structured summary, provisional top 3, friction, open questions
   - 059 meeting briefs
   - 060 morning commit (final top 3, schedule)
3. Queries Notion for Daily Log and Meeting Brief status
4. Allows triggering 057/058/059/060 from the UI
5. Publishes to Notion (Daily Logs only — no Today_Commits or Follow-ups)

Only two Notion databases are used:
  - NOTION_Daily_Logs_ID
  - NOTION_Meeting_Briefs_ID

Usage::

    # Launch Web UI (opens browser)
    python -m src.scripts.061_notion_publish_morning_view

    # Without auto-open browser
    python -m src.scripts.061_notion_publish_morning_view --no-browser

    # Custom port
    python -m src.scripts.061_notion_publish_morning_view --port 8061

    # Debug logging
    python -m src.scripts.061_notion_publish_morning_view -v
"""

# NOTE: Do NOT use ``from __future__ import annotations`` here.
# FastAPI + Pydantic v2 need *runtime* type objects for endpoint
# parameter resolution.

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    load_env,
    setup_logging,
    get_optional_db_id,
)
from src.daily.models import (
    CloseRawInput,
    CloseStructured,
    MorningCommit,
    NextDayPrep,
)
from src.daily.io import (
    CLOSE_RAW_DIR,
    CLOSE_STRUCTURED_DIR,
    MORNING_COMMIT_DIR,
    NEXT_DAY_PREP_DIR,
    load_json,
)

logger = logging.getLogger("061_morning_view")

SCRIPT_NAME = "061_notion_publish_morning_view"
JST = ZoneInfo("Asia/Tokyo")
_UI_PORT = 8061

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING — local pipeline outputs
# ═══════════════════════════════════════════════════════════════════

def _available_dates() -> List[str]:
    """Scan local data dirs and return sorted unique dates (newest first)."""
    dates = set()
    for root in [CLOSE_RAW_DIR, CLOSE_STRUCTURED_DIR, NEXT_DAY_PREP_DIR, MORNING_COMMIT_DIR]:
        if root.is_dir():
            for entry in root.iterdir():
                if entry.is_dir() and _DATE_RE.match(entry.name):
                    dates.add(entry.name)
    return sorted(dates, reverse=True)


def _sidebar_dates() -> List[Dict[str, Any]]:
    """Return past 7 days from today (inclusive), newest first.

    Each entry contains the date string, whether it is today, and
    whether local pipeline data exists for that date.
    """
    today = datetime.now(tz=JST).date()
    existing = set(_available_dates())
    result = []
    for i in range(7):
        d = today - timedelta(days=i)
        d_str = d.isoformat()
        result.append({
            "date": d_str,
            "is_today": i == 0,
            "has_data": d_str in existing,
        })
    logger.info(
        "Sidebar dates: today=%s, range=%s..%s, with_data=%d/7",
        today.isoformat(),
        result[0]["date"],
        result[-1]["date"],
        sum(1 for r in result if r["has_data"]),
    )
    return result


def _load_pipeline_data(date_iso: str) -> Dict[str, Any]:
    """Load all available local pipeline outputs for a date."""
    logger.info("Loading pipeline data for date=%s", date_iso)
    data: Dict[str, Any] = {"date": date_iso}

    # 057
    raw_path = CLOSE_RAW_DIR / date_iso / "close_raw.json"
    if raw_path.exists():
        logger.info("  057 close_raw: %s", raw_path)
        raw = CloseRawInput.from_dict(load_json(raw_path))
        data["close_raw"] = {
            "raw_text": raw.raw_text,
            "satisfaction": raw.satisfaction,
            "energy_level": raw.energy_level,
            "input_mode": raw.input_mode,
        }
        logger.info("  057 raw_text: %d chars, mode=%s", len(raw.raw_text), raw.input_mode)
    else:
        logger.info("  057 close_raw: NOT FOUND at %s", raw_path)

    # 058
    struct_path = CLOSE_STRUCTURED_DIR / date_iso / "close_structured.json"
    if struct_path.exists():
        logger.info("  058 close_structured: %s", struct_path)
        s = CloseStructured.from_dict(load_json(struct_path))
        data["close_structured"] = {
            "structured_summary": s.structured_summary,
            "provisional_top3": s.provisional_top3,
            "friction_blockers": s.friction_blockers,
            "open_questions": s.open_questions,
            "value_domains": s.value_domains,
            "items_count": len(s.items),
            "stage": s.stage,
        }
        logger.info("  058 stage=%s, items=%d, top3=%d", s.stage, len(s.items), len(s.provisional_top3))
    else:
        logger.info("  058 close_structured: NOT FOUND at %s", struct_path)

    # 059
    prep_path = NEXT_DAY_PREP_DIR / date_iso / "next_day_prep.json"
    if prep_path.exists():
        logger.info("  059 next_day_prep: %s", prep_path)
        p = NextDayPrep.from_dict(load_json(prep_path))
        data["next_day_prep"] = {
            "meeting_briefs": [
                {
                    "title": b.title,
                    "date": b.date,
                    "people": b.people,
                    "purpose": b.purpose,
                    "context": b.context[:300] + "..." if len(b.context) > 300 else b.context,
                    "key_questions": b.key_questions,
                    "status": b.status,
                }
                for b in p.meeting_briefs
            ],
            "follow_ups_count": len(p.follow_ups),
            "monitoring_count": len(p.monitoring_suggestions),
        }
        logger.info("  059 briefs=%d, follow_ups=%d", len(p.meeting_briefs), len(p.follow_ups))
    else:
        logger.info("  059 next_day_prep: NOT FOUND at %s", prep_path)

    # 060
    commit_path = MORNING_COMMIT_DIR / date_iso / "morning_commit.json"
    if commit_path.exists():
        logger.info("  060 morning_commit: %s", commit_path)
        raw_commit = load_json(commit_path)
        m = MorningCommit.from_dict(raw_commit)
        data["morning_commit"] = {
            "final_top3": m.final_top3,
            "energy_level": m.energy_level,
            "time_budget_hrs": m.time_budget_hrs,
            "source_date": raw_commit.get("source_date", ""),
            "commits": [
                {
                    "rank": c.rank,
                    "title": c.title,
                    "status": c.status,
                    "planned_time_block": c.planned_time_block,
                    "estimated_minutes": c.estimated_minutes,
                }
                for c in m.commits
            ],
        }
        logger.info(
            "  060 source_date=%s, commits=%d",
            raw_commit.get("source_date", ""), len(m.commits),
        )
    else:
        logger.info("  060 morning_commit: NOT FOUND at %s", commit_path)

    return data


# ═══════════════════════════════════════════════════════════════════
# NOTION QUERIES — Daily Logs + Meeting Briefs only
# ═══════════════════════════════════════════════════════════════════

def _extract_rich_text(prop: Dict[str, Any]) -> str:
    """Extract plain text from a Notion rich_text property."""
    parts = prop.get("rich_text", [])
    return "".join(p.get("plain_text", "") for p in parts)


def _query_notion_daily_log(date_iso: str) -> Optional[Dict[str, Any]]:
    """Query Notion for the Daily Log page for a given date."""
    try:
        db_id = get_optional_db_id("NOTION_Daily_Logs_ID")
        if not db_id:
            logger.info("Notion Daily Log query skipped: NOTION_Daily_Logs_ID not set")
            return None

        logger.info("Querying Notion Daily Log: db=%s date=%s", db_id[:8], date_iso)

        from src.notion.client import (
            build_notion_client_from_env,
            NotionDataSourceResolver,
        )
        client = build_notion_client_from_env(log_requests=False, log_responses=False)
        resolver = NotionDataSourceResolver(client=client)
        resolved = resolver.resolve_once(name="daily_logs", database_id=db_id)

        pages = client.query_data_source(
            data_source_id=resolved.data_source_id,
            filter={"property": "LogDate", "date": {"equals": date_iso}},
            page_size=1,
            fetch_all=False,
        )
        if not pages:
            logger.info("Notion Daily Log: no page found for %s", date_iso)
            return None

        page = pages[0]
        props = page.get("properties", {})
        logger.info(
            "Notion Daily Log found: page_id=%s",
            page.get("id", "")[:8],
        )

        # Extract key fields
        stage_prop = props.get("Stage", {}).get("select") or {}
        publish_prop = props.get("Publish Status", {}).get("select") or {}

        # Raw data fields for canonical display (057/058 layer)
        raw_close_log = _extract_rich_text(props.get("Raw Close Log", {}))
        satisfaction_val = props.get("Satisfaction", {}).get("number")
        energy_level_val = (
            props.get("Energy Level", {}).get("select") or {}
        ).get("name", "")
        structured_summary = _extract_rich_text(
            props.get("Structured Summary", {}),
        )

        logger.info(
            "Notion Daily Log fields: raw_close_log=%d chars, "
            "satisfaction=%s, energy=%s, structured_summary=%d chars",
            len(raw_close_log),
            satisfaction_val,
            energy_level_val,
            len(structured_summary),
        )

        return {
            "page_id": page.get("id", ""),
            "url": page.get("url", ""),
            "stage": stage_prop.get("name", ""),
            "publish_status": publish_prop.get("name", ""),
            "raw_close_log": raw_close_log,
            "satisfaction": satisfaction_val,
            "energy_level": energy_level_val,
            "structured_summary": structured_summary,
        }
    except Exception as e:
        logger.warning("Notion Daily Log query failed: %s", e)
        return None


def _query_notion_meeting_briefs(date_iso: str) -> List[Dict[str, Any]]:
    """Query Notion for Meeting Briefs linked to a given date."""
    try:
        db_id = get_optional_db_id("NOTION_Meeting_Briefs_ID")
        if not db_id:
            logger.info("Notion Meeting Briefs query skipped: NOTION_Meeting_Briefs_ID not set")
            return []

        logger.info("Querying Notion Meeting Briefs: db=%s date=%s", db_id[:8], date_iso)

        from src.notion.client import (
            build_notion_client_from_env,
            NotionDataSourceResolver,
        )
        client = build_notion_client_from_env(log_requests=False, log_responses=False)
        resolver = NotionDataSourceResolver(client=client)
        resolved = resolver.resolve_once(name="meeting_briefs", database_id=db_id)

        pages = client.query_data_source(
            data_source_id=resolved.data_source_id,
            filter={"property": "Date", "date": {"equals": date_iso}},
            page_size=20,
            fetch_all=False,
        )

        results = []
        for page in pages:
            props = page.get("properties", {})
            title_parts = props.get("Title", {}).get("title", [])
            title = title_parts[0].get("plain_text", "") if title_parts else ""
            status_prop = props.get("Status", {}).get("select") or {}

            results.append({
                "page_id": page.get("id", ""),
                "url": page.get("url", ""),
                "title": title,
                "status": status_prop.get("name", ""),
            })
        logger.info("Notion Meeting Briefs: found %d briefs for %s", len(results), date_iso)
        return results
    except Exception as e:
        logger.warning("Notion Meeting Briefs query failed: %s", e)
        return []


# ═══════════════════════════════════════════════════════════════════
# RESEARCH QUESTIONS — 073-based question-driven research
# ═══════════════════════════════════════════════════════════════════

_RESEARCH_QUESTIONS_SYSTEM = (
    "You are a research assistant. Based on the user's daily reflection/close log, "
    "generate research questions that would be useful for their upcoming meetings, "
    "projects, or areas of interest mentioned in the log.\n"
    "Return a JSON array of question strings. Each question should be specific enough "
    "for a research pipeline to investigate (e.g. person names, company names, topics).\n"
    "Generate 2-5 questions. Output ONLY the JSON array, no explanation."
)


def _generate_research_questions(date_iso: str) -> Dict[str, Any]:
    """Generate research questions from 058 structured summary using LLM."""
    try:
        struct_path = CLOSE_STRUCTURED_DIR / date_iso / "close_structured.json"
        if not struct_path.exists():
            return {"ok": False, "error": f"No 058 output for {date_iso}. Run 058 first."}

        structured = CloseStructured.from_dict(load_json(struct_path))

        # Build context from structured summary
        parts = []
        if structured.structured_summary:
            parts.append(f"Summary: {structured.structured_summary}")
        if structured.provisional_top3:
            parts.append(f"Top 3: {', '.join(structured.provisional_top3)}")
        if structured.friction_blockers:
            parts.append(f"Friction: {', '.join(structured.friction_blockers)}")
        if structured.open_questions:
            parts.append(f"Open Questions: {', '.join(structured.open_questions)}")
        if structured.research_candidates:
            parts.append(f"Research Candidates: {', '.join(structured.research_candidates)}")

        context = "\n".join(parts)
        if not context.strip():
            return {"ok": True, "questions": []}

        from src.llm.router import build_router_from_env, TASK_REASONING
        router = build_router_from_env()
        result = router.call(
            task_type=TASK_REASONING,
            system=_RESEARCH_QUESTIONS_SYSTEM,
            user=f"Date: {date_iso}\n\n{context}",
            model_override="gpt-4o",
            temperature_override=0.5,
        )

        parsed = result.parsed
        if isinstance(parsed, dict) and "questions" in parsed:
            questions = parsed["questions"]
        elif isinstance(parsed, list):
            questions = parsed
        else:
            questions = []
        questions = [str(q) for q in questions if q]
        logger.info("Generated %d research questions for %s", len(questions), date_iso)
        return {"ok": True, "questions": questions}
    except Exception as e:
        logger.error("Research question generation failed: %s", e)
        return {"ok": False, "error": str(e)}


def _run_research_question(question: str, date_iso: str) -> Dict[str, Any]:
    """Execute a single research question via 073 internal functions.

    Uses run_single_pipeline() from session.py to run the full 067-072
    pipeline, then generates a final answer and saves the session.
    """
    from src.deep_research import generate_run_id
    from src.deep_research.session import (
        generate_session_id,
        run_single_pipeline,
        aggregate_results,
        generate_final_answer,
        save_session,
    )
    from src.llm.claude_client import build_claude_client_from_env
    from src.search.google_cse import build_google_cse_from_env

    llm_client = build_claude_client_from_env()
    search_client = build_google_cse_from_env()

    # Notion client (optional)
    notion_client = None
    enable_writeback = os.environ.get("ENABLE_NOTION_WRITEBACK", "").lower() == "true"
    if enable_writeback:
        try:
            from src.notion.client import build_notion_client_from_env
            notion_client = build_notion_client_from_env()
        except Exception as e:
            logger.warning("Notion client unavailable: %s", e)

    session_id = generate_session_id()
    run_id = generate_run_id()
    created_at = datetime.now(JST).isoformat()

    logger.info(
        "Research: executing question='%s' session=%s run=%s",
        question[:60], session_id, run_id,
    )

    # Run pipeline
    run_result = run_single_pipeline(
        question=question,
        run_id=run_id,
        llm_client=llm_client,
        search_client=search_client,
        notion_client=notion_client,
        enable_writeback=enable_writeback,
    )

    run_results = [run_result]
    aggregated = aggregate_results(run_results)
    intent = run_result.get("intent", "general_research")
    framework_id = run_result.get("framework_id", "")

    # Generate final answer
    final_answer = generate_final_answer(
        original_question=question,
        aggregated=aggregated,
        run_results=run_results,
        llm_client=llm_client,
        intent=intent,
        framework_id=framework_id,
    )

    # Save session locally
    status = "completed" if run_result["status"] == "completed" else "partial"
    session_dir = save_session(
        session_id=session_id,
        user_question=question,
        decomposed_questions=[question],
        run_results=run_results,
        final_answer=final_answer,
        status=status,
        created_at=created_at,
    )

    logger.info(
        "Research complete: session=%s status=%s dir=%s",
        session_id, status, session_dir,
    )

    return {
        "ok": True,
        "session_id": session_id,
        "run_id": run_id,
        "status": status,
        "question": question,
        "memo_title": run_result.get("memo_title", ""),
        "memo_summary": run_result.get("memo_summary", ""),
        "answer_preview": final_answer[:200] if final_answer else "",
        "answer_full": final_answer or "",
        "sources_count": run_result.get("sources_count", 0),
        "evidence_count": run_result.get("evidence_count", 0),
        "claims_count": run_result.get("claims_count", 0),
        "output_path": str(session_dir),
    }


# ═══════════════════════════════════════════════════════════════════
# 057 INTERACTIVE — browser-based close log input
# ═══════════════════════════════════════════════════════════════════

def _upsert_057_to_notion(
    *,
    date_iso: str,
    raw_text: str,
    satisfaction: Optional[int] = None,
    energy_level: str = "",
    value_domains: Optional[List[str]] = None,
    log_label: str = "057_raw",
) -> Dict[str, Any]:
    """Upsert 057 raw layer fields to the Notion Daily Logs page.

    Creates the page if it doesn't exist; updates if it does.
    Returns the upsert result dict from ``upsert_daily_log``.
    """
    try:
        from src.notion.daily_schema import build_daily_log_properties
        from src.notion.daily_upsert import upsert_daily_log, safe_truncate

        truncated_text = safe_truncate(raw_text)
        logger.info(
            "[%s] Notion upsert: date=%s, raw_text=%d chars "
            "(truncated=%d), satisfaction=%s, energy=%s, "
            "value_domains=%s",
            log_label,
            date_iso,
            len(raw_text),
            len(truncated_text),
            satisfaction,
            energy_level,
            value_domains or [],
        )

        props = build_daily_log_properties(
            title=f"Daily Log {date_iso}",
            date=date_iso,
            raw_close_log=truncated_text,
            satisfaction=satisfaction,
            energy_level=energy_level,
            value_domains=value_domains or [],
            stage="raw",
        )

        result = upsert_daily_log(
            date_iso=date_iso,
            properties=props,
            log_label=log_label,
        )
        if result.get("ok"):
            logger.info(
                "[%s] Notion upsert OK: %s page_id=%s url=%s",
                log_label,
                result.get("action"),
                result.get("page_id", "")[:8],
                result.get("page_url", ""),
            )
        else:
            logger.error(
                "[%s] Notion upsert FAILED: %s",
                log_label,
                result.get("error", "unknown"),
            )
        return result
    except Exception as e:
        logger.error("[%s] Notion upsert exception: %s", log_label, e)
        return {"ok": False, "error": str(e)}


def _submit_close_raw_interactive(
    *,
    date_iso: str,
    raw_text: str,
    satisfaction: Optional[int] = None,
    energy_level: Optional[str] = None,
) -> Dict[str, Any]:
    """Save a raw close log submitted from the browser UI.

    Creates a CloseRawInput and writes it directly to the data directory,
    bypassing the 057 subprocess entirely.
    """
    try:
        import dataclasses

        raw_input = CloseRawInput(
            date=date_iso,
            raw_text=raw_text,
            satisfaction=int(satisfaction) if satisfaction else None,
            energy_level=energy_level or None,
            timestamp=datetime.now(JST).isoformat(),
            input_mode="browser",
        )

        out_dir = CLOSE_RAW_DIR / date_iso
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "close_raw.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(raw_input), f, ensure_ascii=False, indent=2)

        logger.info("057 browser submit saved: %s (%d chars)", out_path, len(raw_text))

        # -- Upsert to Notion Daily Logs --
        notion_result = _upsert_057_to_notion(
            date_iso=date_iso,
            raw_text=raw_text,
            satisfaction=raw_input.satisfaction,
            energy_level=raw_input.energy_level or "",
            value_domains=[],
            log_label="057_browser",
        )
        return {
            "ok": True,
            "path": str(out_path),
            "notion": notion_result,
        }
    except Exception as e:
        logger.error("057 browser submit failed: %s", e)
        return {"ok": False, "error": str(e)}


def _submit_wizard_interactive(
    *,
    date_iso: str,
    sections: List[Dict[str, Any]],
    satisfaction: Optional[int] = None,
    energy_level: Optional[str] = None,
    value_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Save a wizard-based close log from the browser UI.

    Assembles the structured sections into the canonical ## title / text
    format and writes a CloseRawInput to disk.
    """
    import dataclasses

    # Assemble markdown text from sections
    parts: List[str] = []
    section_meta: List[Dict[str, Any]] = []
    for sec in sections:
        key = sec.get("key", "")
        title = sec.get("title_ja", key)
        text = sec.get("text", "").strip()
        skipped = not text

        if skipped:
            parts.append(f"## {title}\n(skipped)\n")
        else:
            parts.append(f"## {title}\n{text}\n")

        section_meta.append({
            "key": key,
            "title_ja": title,
            "transcript_chars": len(text),
            "skipped": skipped,
        })

    assembled_text = "\n".join(parts).strip()

    try:
        raw_input = CloseRawInput(
            date=date_iso,
            raw_text=assembled_text,
            satisfaction=int(satisfaction) if satisfaction else None,
            energy_level=energy_level or None,
            timestamp=datetime.now(JST).isoformat(),
            input_mode="browser_wizard",
            sections=section_meta,
            value_domains=value_domains or [],
        )

        out_dir = CLOSE_RAW_DIR / date_iso
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "close_raw.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(raw_input), f, ensure_ascii=False, indent=2)

        # Also save the assembled transcript as a standalone .txt file
        transcript_path = out_dir / "close_raw_transcript.txt"
        transcript_path.write_text(assembled_text, encoding="utf-8")

        logger.info(
            "057 wizard submit saved: %s (%d chars, %d sections)",
            out_path, len(assembled_text), len(sections),
        )

        # -- Upsert to Notion Daily Logs --
        notion_result = _upsert_057_to_notion(
            date_iso=date_iso,
            raw_text=assembled_text,
            satisfaction=raw_input.satisfaction,
            energy_level=raw_input.energy_level or "",
            value_domains=raw_input.value_domains or [],
            log_label="057_wizard",
        )
        return {
            "ok": True,
            "path": str(out_path),
            "notion": notion_result,
        }
    except Exception as e:
        logger.error("057 wizard submit failed: %s", e)
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# 060 INTERACTIVE — browser-based morning commit
# ═══════════════════════════════════════════════════════════════════

def _execute_morning_commit_interactive(
    *,
    date_iso: str,
    final_top3: List[str],
    energy_level: str = "Medium",
    time_budget_hrs: float = 6.0,
) -> Dict[str, Any]:
    """Execute morning commit with user-provided inputs from the browser.

    Tries LLM-based schedule generation, falls back to a simple
    time-blocked schedule.  Saves result to the morning_commit directory.
    """
    logger.info(
        "Morning commit: date=%s top3=%s energy=%s budget=%.1f",
        date_iso, final_top3, energy_level, time_budget_hrs,
    )

    commits: List[Dict[str, Any]] = []

    # Try LLM for schedule generation
    try:
        from src.llm.router import build_router_from_env, TASK_REASONING
        router = build_router_from_env()

        user_prompt = (
            f"Today: {date_iso}\n"
            f"Energy: {energy_level}\n"
            f"Available time: {time_budget_hrs} hours\n"
            f"Top 3 priorities:\n"
        )
        for i, t in enumerate(final_top3, 1):
            user_prompt += f"  {i}. {t}\n"
        user_prompt += (
            "\nCreate a realistic daily schedule. Return a JSON array:\n"
            '[{"rank": 1, "title": "task title", '
            '"planned_time_block": "09:00-11:00", '
            '"estimated_minutes": 120, "why": "why this matters", '
            '"definition_of_done": "what done looks like"}]\n'
            "Include all priorities, plus breaks if appropriate.\n"
        )

        result = router.call(
            task_type=TASK_REASONING,
            system="You are a schedule planning assistant. Return valid JSON array only.",
            user=user_prompt,
            model_override="gpt-4o",
            temperature_override=0.3,
        )
        raw_commits = result.parsed
        if isinstance(raw_commits, list):
            for c in raw_commits:
                commits.append({
                    "title": c.get("title", ""),
                    "rank": c.get("rank", 0),
                    "status": "Planned",
                    "why": c.get("why", ""),
                    "definition_of_done": c.get("definition_of_done", ""),
                    "planned_time_block": c.get("planned_time_block", ""),
                    "estimated_minutes": c.get("estimated_minutes", 0),
                    "order": c.get("rank", 0),
                    "value_domains": [],
                    "notes": "",
                })
            logger.info("LLM schedule: %d commit items", len(commits))
    except Exception as e:
        logger.warning("LLM schedule generation failed, using simple schedule: %s", e)

    # Fallback: simple time-blocked schedule from top3
    if not commits:
        mins_each = int((time_budget_hrs * 60) / max(len(final_top3), 1))
        for i, t in enumerate(final_top3):
            commits.append({
                "title": t,
                "rank": i + 1,
                "status": "Planned",
                "why": "",
                "definition_of_done": "",
                "planned_time_block": f"Block {i + 1}",
                "estimated_minutes": mins_each,
                "order": i + 1,
                "value_domains": [],
                "notes": "",
            })

    commit_data = {
        "date": date_iso,
        "energy_level": energy_level,
        "time_budget_hrs": time_budget_hrs,
        "final_top3": final_top3,
        "source_date": date_iso,
        "commits": commits,
    }

    # Save to disk
    try:
        out_dir = MORNING_COMMIT_DIR / date_iso
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "morning_commit.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(commit_data, f, ensure_ascii=False, indent=2)

        logger.info("Morning commit saved: %s", out_path)
        return {"ok": True, "path": str(out_path)}
    except Exception as e:
        logger.error("Morning commit save failed: %s", e)
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# SCRIPT EXECUTION — run 057/058/059/060 as subprocesses
# ═══════════════════════════════════════════════════════════════════

# Track running jobs: script_key -> {process, output, status}
_jobs: Dict[str, Dict[str, Any]] = {}


def _run_script(script_num: str, date_iso: str, extra_args: List[str] = None) -> str:
    """Launch a pipeline script as a subprocess. Returns a job key."""
    script_map = {
        "057": "src.scripts.057_daily_close_input",
        "058": "src.scripts.058_daily_close_structuring",
        "059": "src.scripts.059_next_day_preparation",
        "060": "src.scripts.060_morning_commit",
    }
    module = script_map.get(script_num)
    if not module:
        raise ValueError(f"Unknown script: {script_num}")

    job_key = f"{script_num}:{date_iso}"

    # Don't start if already running
    if job_key in _jobs and _jobs[job_key]["status"] == "running":
        return job_key

    cmd = [sys.executable, "-m", module, "--date", date_iso]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Launching %s: %s", job_key, " ".join(cmd))

    def _run():
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(_PROJECT_ROOT),
            )
            _jobs[job_key]["output"] = result.stdout + result.stderr
            _jobs[job_key]["returncode"] = result.returncode
            _jobs[job_key]["status"] = "done" if result.returncode == 0 else "error"
            logger.info(
                "Script %s finished: returncode=%d",
                job_key, result.returncode,
            )
        except subprocess.TimeoutExpired:
            _jobs[job_key]["output"] = "Timeout (600s)"
            _jobs[job_key]["status"] = "error"
            _jobs[job_key]["returncode"] = -1
        except Exception as e:
            _jobs[job_key]["output"] = str(e)
            _jobs[job_key]["status"] = "error"
            _jobs[job_key]["returncode"] = -1

    _jobs[job_key] = {"status": "running", "output": "", "returncode": None}
    threading.Thread(target=_run, daemon=True).start()
    return job_key


# ═══════════════════════════════════════════════════════════════════
# 060 RUN — generate morning commit proposals with fallback reflection
# ═══════════════════════════════════════════════════════════════════

def _resolve_reflection_source(commit_date: str) -> Optional[str]:
    """Find the most recent reflection (058 close_structured) before *commit_date*.

    Strategy:
      1. Try yesterday (commit_date - 1 day).
      2. If missing, scan local close_structured dirs for the most recent
         date strictly before commit_date.
      3. If still missing, query Notion Daily Logs for the latest LogDate
         before commit_date (same logic as 060's _resolve_source_date_from_notion).

    Returns the date string or None.
    """
    from datetime import date as date_cls

    commit_d = date_cls.fromisoformat(commit_date)
    yesterday = (commit_d - timedelta(days=1)).isoformat()

    logger.info(
        "[060-run] Resolving reflection source: commit_date=%s, trying yesterday=%s",
        commit_date, yesterday,
    )

    # 1. Try yesterday locally
    yesterday_path = CLOSE_STRUCTURED_DIR / yesterday / "close_structured.json"
    if yesterday_path.exists():
        logger.info("[060-run] Found yesterday's reflection: %s", yesterday)
        return yesterday

    logger.info("[060-run] Yesterday %s not found locally, trying fallback...", yesterday)

    # 2. Scan local close_structured dirs for most recent before commit_date
    if CLOSE_STRUCTURED_DIR.is_dir():
        candidates = []
        for entry in CLOSE_STRUCTURED_DIR.iterdir():
            if entry.is_dir() and _DATE_RE.match(entry.name):
                if entry.name < commit_date and (entry / "close_structured.json").exists():
                    candidates.append(entry.name)
        if candidates:
            candidates.sort(reverse=True)
            fallback = candidates[0]
            logger.info(
                "[060-run] Fallback: found local reflection for %s "
                "(scanned %d candidates)",
                fallback, len(candidates),
            )
            return fallback

    logger.info("[060-run] No local reflection found, querying Notion...")

    # 3. Query Notion for most recent Daily Log before commit_date
    try:
        db_id = get_optional_db_id("NOTION_Daily_Logs_ID")
        if db_id:
            from src.notion.client import (
                build_notion_client_from_env,
                NotionDataSourceResolver,
            )
            client = build_notion_client_from_env(log_requests=False, log_responses=False)
            resolver = NotionDataSourceResolver(client=client)
            resolved = resolver.resolve_once(name="daily_logs", database_id=db_id)
            pages = client.query_data_source(
                data_source_id=resolved.data_source_id,
                filter={"property": "LogDate", "date": {"before": commit_date}},
                sorts=[{"property": "LogDate", "direction": "descending"}],
                page_size=1,
                fetch_all=False,
            )
            if pages:
                log_date_prop = pages[0].get("properties", {}).get("LogDate", {})
                date_obj = log_date_prop.get("date") or {}
                source = (date_obj.get("start") or "")[:10]
                if source:
                    logger.info("[060-run] Fallback from Notion: %s", source)
                    return source
    except Exception as e:
        logger.warning("[060-run] Notion fallback query failed: %s", e)

    logger.warning("[060-run] No reflection source found for commit_date=%s", commit_date)
    return None


def _run_060_generate(commit_date: str) -> Dict[str, Any]:
    """Generate morning commit proposals for *commit_date*.

    1. Resolve reflection source (yesterday or fallback).
    2. Load 058 structured data from that source.
    3. Extract top 3 items.
    4. Generate proposals.
    5. Write to local JSON + Notion.

    Returns result dict with status + generated data.
    """
    logger.info("[060-run] === Starting 060 generation for commit_date=%s ===", commit_date)

    # Step 1: Resolve source date
    source_date = _resolve_reflection_source(commit_date)
    if not source_date:
        msg = f"No reflection data found before {commit_date}. Run 058 first."
        logger.error("[060-run] %s", msg)
        return {"ok": False, "error": msg}

    logger.info(
        "[060-run] Using reflection source: source_date=%s (commit_date=%s)",
        source_date, commit_date,
    )

    # Step 2: Load 058 structured output
    struct_path = CLOSE_STRUCTURED_DIR / source_date / "close_structured.json"
    structured = None
    provisional_top3: List[str] = []

    if struct_path.exists():
        structured = CloseStructured.from_dict(load_json(struct_path))
        provisional_top3 = structured.provisional_top3[:3]
        logger.info(
            "[060-run] Loaded 058 for %s: provisional_top3=%s",
            source_date,
            [t[:40] for t in provisional_top3],
        )
    else:
        logger.warning("[060-run] No local 058 for %s, trying Notion...", source_date)
        # Try to get from Notion
        notion_log = _query_notion_daily_log(source_date)
        if notion_log and notion_log.get("structured_summary"):
            # Extract provisional top 3 from Notion's Provisional Top 3 field
            try:
                db_id = get_optional_db_id("NOTION_Daily_Logs_ID")
                if db_id:
                    from src.notion.client import (
                        build_notion_client_from_env,
                        NotionDataSourceResolver,
                    )
                    client = build_notion_client_from_env(log_requests=False, log_responses=False)
                    resolver = NotionDataSourceResolver(client=client)
                    resolved = resolver.resolve_once(name="daily_logs", database_id=db_id)
                    pages = client.query_data_source(
                        data_source_id=resolved.data_source_id,
                        filter={"property": "LogDate", "date": {"equals": source_date}},
                        page_size=1,
                        fetch_all=False,
                    )
                    if pages:
                        props = pages[0].get("properties", {})
                        top3_text = _extract_rich_text(props.get("Provisional Top 3", {}))
                        if top3_text:
                            # Parse numbered list: "1. item\n2. item\n3. item"
                            import re as _re
                            lines = [
                                _re.sub(r"^\d+\.\s*", "", line).strip()
                                for line in top3_text.strip().split("\n")
                                if line.strip()
                            ]
                            provisional_top3 = lines[:3]
                            logger.info(
                                "[060-run] Extracted top 3 from Notion for %s: %s",
                                source_date, provisional_top3,
                            )
            except Exception as e:
                logger.warning("[060-run] Notion top3 extraction failed: %s", e)

    # Also try 059 follow-ups as fallback for top3
    if not provisional_top3:
        prep_path = NEXT_DAY_PREP_DIR / source_date / "next_day_prep.json"
        if prep_path.exists():
            prep = NextDayPrep.from_dict(load_json(prep_path))
            if prep.follow_ups:
                provisional_top3 = [f.title for f in prep.follow_ups[:3]]
                logger.info(
                    "[060-run] Using 059 follow-ups as top3: %s",
                    provisional_top3,
                )

    if not provisional_top3:
        msg = f"Source date {source_date} has no provisional top 3 (058) or follow-ups (059)."
        logger.error("[060-run] %s", msg)
        return {"ok": False, "error": msg}

    logger.info(
        "[060-run] Top 3 items for generation:\n  1. %s\n  2. %s\n  3. %s",
        provisional_top3[0] if len(provisional_top3) > 0 else "(none)",
        provisional_top3[1] if len(provisional_top3) > 1 else "(none)",
        provisional_top3[2] if len(provisional_top3) > 2 else "(none)",
    )

    # Step 3: Generate proposals via 060 pipeline (non-interactive)
    logger.info("[060-run] Starting proposal generation...")

    energy_level = "Medium"
    time_budget_hrs = 6.0
    final_top3 = provisional_top3[:3]

    # Use the existing interactive commit function for proposal generation
    result = _execute_morning_commit_interactive(
        date_iso=commit_date,
        final_top3=final_top3,
        energy_level=energy_level,
        time_budget_hrs=time_budget_hrs,
    )

    if not result.get("ok"):
        logger.error("[060-run] Generation failed: %s", result.get("error"))
        return result

    logger.info("[060-run] Proposals generated, saving to disk...")

    # Step 4: Patch the saved file with source_date
    commit_path = MORNING_COMMIT_DIR / commit_date / "morning_commit.json"
    if commit_path.exists():
        commit_data = load_json(commit_path)
        commit_data["source_date"] = source_date
        from src.daily.io import save_json as _save_json
        _save_json(commit_path, commit_data)
        logger.info("[060-run] Patched source_date=%s into %s", source_date, commit_path)

    # Step 5: Upsert to Notion
    logger.info("[060-run] Upserting to Notion Daily Log for %s...", commit_date)
    try:
        from src.notion.daily_schema import build_daily_log_properties
        from src.notion.daily_upsert import safe_truncate, upsert_daily_log

        top3_lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(final_top3))
        if source_date != commit_date:
            top3_lines = f"[Source Close Date: {source_date}]\n{top3_lines}"

        notion_props = build_daily_log_properties(
            title=f"Daily Log {commit_date}",
            date=commit_date,
            final_top3=safe_truncate(top3_lines),
            stage="committed",
        )
        notion_result = upsert_daily_log(
            date_iso=commit_date,
            properties=notion_props,
            log_label="060_run_webui",
        )
        logger.info(
            "[060-run] Notion upsert: ok=%s action=%s page_id=%s",
            notion_result.get("ok"),
            notion_result.get("action"),
            notion_result.get("page_id", "")[:8],
        )
    except Exception as e:
        logger.warning("[060-run] Notion upsert failed (non-fatal): %s", e)
        notion_result = {"ok": False, "error": str(e)}

    logger.info("[060-run] === 060 generation complete for %s ===", commit_date)

    return {
        "ok": True,
        "commit_date": commit_date,
        "source_date": source_date,
        "final_top3": final_top3,
        "notion": notion_result,
    }


# ═══════════════════════════════════════════════════════════════════
# WEB UI — FastAPI server + HTML frontend
# ═══════════════════════════════════════════════════════════════════

def _build_app():
    """Create the FastAPI application."""
    import uvicorn
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="061 Daily Operational Console")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _build_html()

    @app.get("/api/dates")
    async def get_dates():
        sidebar = _sidebar_dates()
        # Also include legacy flat list for backwards compat
        dates = _available_dates()
        return {"dates": dates, "sidebar": sidebar}

    @app.get("/api/data/{date_iso}")
    async def get_data(date_iso: str):
        if not _DATE_RE.match(date_iso):
            return JSONResponse({"error": "Invalid date"}, status_code=400)
        logger.info("Loading pipeline data for %s", date_iso)
        data = _load_pipeline_data(date_iso)
        return data

    @app.get("/api/notion/{date_iso}")
    async def get_notion(date_iso: str):
        if not _DATE_RE.match(date_iso):
            return JSONResponse({"error": "Invalid date"}, status_code=400)
        logger.info("Querying Notion for %s", date_iso)
        daily_log = _query_notion_daily_log(date_iso)
        meeting_briefs = _query_notion_meeting_briefs(date_iso)
        return {
            "daily_log": daily_log,
            "meeting_briefs": meeting_briefs,
        }

    @app.post("/api/run/{script_num}/{date_iso}")
    async def run_script(script_num: str, date_iso: str):
        if script_num not in ("057", "058", "059", "060"):
            return JSONResponse({"error": "Invalid script"}, status_code=400)
        if not _DATE_RE.match(date_iso):
            return JSONResponse({"error": "Invalid date"}, status_code=400)

        # Script-specific flags for non-interactive / headless execution
        extra = []
        if script_num == "057":
            extra = ["--non-interactive"]
        elif script_num == "059":
            extra = ["--no-ui"]
        elif script_num == "060":
            extra = ["--non-interactive"]

        logger.info(
            "Running script %s for %s (extra=%s)",
            script_num, date_iso, extra,
        )
        job_key = _run_script(script_num, date_iso, extra_args=extra)
        return {"job_key": job_key, "status": "running"}

    @app.get("/api/job/{job_key}")
    async def get_job(job_key: str):
        job = _jobs.get(job_key)
        if not job:
            return JSONResponse({"error": "Job not found"}, status_code=404)
        resp: Dict[str, Any] = {
            "job_key": job_key,
            "status": job["status"],
            "output": job["output"][-2000:] if job["output"] else "",
            "returncode": job["returncode"],
        }
        # Include structured result if present (used by 059 synthesize)
        if "result" in job:
            resp["result"] = job["result"]
        return resp

    # ── Research Questions — 073-based endpoints ──

    @app.post("/api/research/generate-questions/{date_iso}")
    async def generate_questions(date_iso: str):
        """Generate research questions from 058 data."""
        if not _DATE_RE.match(date_iso):
            return JSONResponse({"error": "Invalid date"}, status_code=400)
        logger.info("Research: generating questions for %s", date_iso)
        return _generate_research_questions(date_iso)

    @app.post("/api/research/run")
    async def run_research(request_data: dict):
        """Run a single research question via 073 pipeline (background job)."""
        question = request_data.get("question", "").strip()
        date_iso = request_data.get("date_iso", "")
        if not question:
            return JSONResponse({"error": "No question provided"}, status_code=400)

        job_key = f"research:{hash(question) % 100000}:{date_iso}"
        if job_key in _jobs and _jobs[job_key]["status"] == "running":
            return {"job_key": job_key, "status": "running"}

        logger.info("Research: starting job for question='%s'", question[:60])

        def _run():
            try:
                result = _run_research_question(question, date_iso)
                _jobs[job_key]["result"] = result
                _jobs[job_key]["status"] = "done"
                _jobs[job_key]["returncode"] = 0
                _jobs[job_key]["output"] = (
                    f"Research complete: {result.get('memo_title', '')}"
                )
            except Exception as exc:
                _jobs[job_key]["output"] = str(exc)
                _jobs[job_key]["status"] = "error"
                _jobs[job_key]["returncode"] = -1
                _jobs[job_key]["result"] = {"ok": False, "error": str(exc)}

        _jobs[job_key] = {
            "status": "running",
            "output": "",
            "returncode": None,
        }
        threading.Thread(target=_run, daemon=True).start()
        return {"job_key": job_key, "status": "running"}

    # -- 057 Browser-based close log submission --

    @app.post("/api/057/submit")
    async def submit_close_raw(request_data: dict):
        """Save a raw close log from browser input."""
        date_iso = request_data.get("date", "")
        if not _DATE_RE.match(date_iso):
            return JSONResponse({"error": "Invalid date"}, status_code=400)

        raw_text = request_data.get("raw_text", "").strip()
        if not raw_text:
            return JSONResponse({"error": "Raw text is required"}, status_code=400)

        logger.info("057 browser submit: date=%s, %d chars", date_iso, len(raw_text))

        result = _submit_close_raw_interactive(
            date_iso=date_iso,
            raw_text=raw_text,
            satisfaction=request_data.get("satisfaction"),
            energy_level=request_data.get("energy_level"),
        )
        return result

    # -- 057 Wizard-based structured submission --

    @app.post("/api/057/wizard-submit")
    async def wizard_submit(request_data: dict):
        """Save a wizard-based close log from structured sections."""
        date_iso = request_data.get("date", "")
        if not _DATE_RE.match(date_iso):
            return JSONResponse({"error": "Invalid date"}, status_code=400)

        sections = request_data.get("sections", [])
        if not sections:
            return JSONResponse(
                {"error": "At least one section is required"}, status_code=400,
            )

        logger.info(
            "057 wizard submit: date=%s, %d sections",
            date_iso, len(sections),
        )

        result = _submit_wizard_interactive(
            date_iso=date_iso,
            sections=sections,
            satisfaction=request_data.get("satisfaction"),
            energy_level=request_data.get("energy_level"),
            value_domains=request_data.get("value_domains"),
        )
        return result

    # -- 057 Audio upload + transcription per section --

    @app.post("/api/057/audio-upload")
    async def audio_upload(
        audio: UploadFile = File(...),
        section_key: str = Form(...),
        date_iso: str = Form(...),
    ):
        """Upload and transcribe audio for one wizard section."""
        content = await audio.read()
        if not content:
            return JSONResponse({"error": "Empty audio"}, status_code=400)
        if not _DATE_RE.match(date_iso):
            return JSONResponse({"error": "Invalid date"}, status_code=400)

        # Determine extension from content type
        ext = ".webm"
        if audio.content_type:
            if "wav" in audio.content_type:
                ext = ".wav"
            elif "mp4" in audio.content_type or "m4a" in audio.content_type:
                ext = ".m4a"
            elif "ogg" in audio.content_type:
                ext = ".ogg"

        out_dir = CLOSE_RAW_DIR / date_iso
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / f"close_raw_audio_{section_key}{ext}"
        audio_path.write_bytes(content)
        logger.info(
            "057 audio saved: %s (%d bytes, type=%s)",
            audio_path, len(content), audio.content_type,
        )

        # Transcribe
        transcript = ""
        try:
            from src.daily.audio import transcribe_audio
            transcript = transcribe_audio(str(audio_path), language="ja")
            logger.info(
                "057 audio transcribed: section=%s, %d chars",
                section_key, len(transcript),
            )
        except Exception as e:
            logger.warning("Transcription failed for %s: %s", section_key, e)
            return JSONResponse(
                {"ok": False, "error": f"Transcription failed: {e}"},
                status_code=500,
            )

        return {
            "ok": True,
            "transcript": transcript,
            "section_key": section_key,
        }

    # -- 060 Browser-based morning commit --

    @app.post("/api/060/commit")
    async def commit_morning(request_data: dict):
        """Execute morning commit with user-provided inputs."""
        date_iso = request_data.get("date", "")
        if not _DATE_RE.match(date_iso):
            return JSONResponse({"error": "Invalid date"}, status_code=400)

        final_top3 = request_data.get("final_top3", [])
        if not final_top3:
            return JSONResponse(
                {"error": "At least one priority required"}, status_code=400,
            )

        logger.info("060 interactive commit: date=%s, top3=%s", date_iso, final_top3)

        result = _execute_morning_commit_interactive(
            date_iso=date_iso,
            final_top3=final_top3,
            energy_level=request_data.get("energy_level", "Medium"),
            time_budget_hrs=float(request_data.get("time_budget_hrs", 6.0)),
        )
        return result

    # -- 060 Run: auto-generate proposals from reflection fallback --

    @app.post("/api/060/run/{date_iso}")
    async def run_060_generate(date_iso: str):
        """Generate morning commit proposals for date using reflection fallback."""
        if not _DATE_RE.match(date_iso):
            return JSONResponse({"error": "Invalid date"}, status_code=400)

        logger.info("060 run: generating proposals for %s", date_iso)

        # Run in background thread for long-running LLM calls
        job_key = f"060run:{date_iso}"
        if job_key in _jobs and _jobs[job_key]["status"] == "running":
            return {"job_key": job_key, "status": "running"}

        def _run_gen():
            try:
                result = _run_060_generate(date_iso)
                _jobs[job_key]["result"] = result
                _jobs[job_key]["status"] = "done" if result.get("ok") else "error"
                _jobs[job_key]["returncode"] = 0 if result.get("ok") else -1
                _jobs[job_key]["output"] = (
                    f"Generated proposals for {date_iso} "
                    f"(source: {result.get('source_date', '?')})"
                    if result.get("ok")
                    else f"Failed: {result.get('error', 'unknown')}"
                )
            except Exception as exc:
                _jobs[job_key]["output"] = str(exc)
                _jobs[job_key]["status"] = "error"
                _jobs[job_key]["returncode"] = -1
                _jobs[job_key]["result"] = {"ok": False, "error": str(exc)}

        _jobs[job_key] = {"status": "running", "output": "", "returncode": None}
        threading.Thread(target=_run_gen, daemon=True).start()
        return {"job_key": job_key, "status": "running"}

    # ── Section refresh endpoint (IMP-2) ──

    @app.post("/api/refresh-section/{section}")
    async def refresh_section(section: str, request_data: dict):
        """Reload a single section's data from local files."""
        if section not in ("057", "058", "059", "060"):
            return JSONResponse({"error": "Invalid section"}, status_code=400)
        date_iso = request_data.get("date", "")
        if not _DATE_RE.match(date_iso):
            return JSONResponse({"error": "Invalid date"}, status_code=400)

        logger.info("Refreshing section %s for %s", section, date_iso)
        section_data = None

        if section == "057":
            raw_path = CLOSE_RAW_DIR / date_iso / "close_raw.json"
            if raw_path.exists():
                raw = CloseRawInput.from_dict(load_json(raw_path))
                section_data = {
                    "raw_text": raw.raw_text,
                    "satisfaction": raw.satisfaction,
                    "energy_level": raw.energy_level,
                    "input_mode": raw.input_mode,
                }
        elif section == "058":
            struct_path = CLOSE_STRUCTURED_DIR / date_iso / "close_structured.json"
            if struct_path.exists():
                s = CloseStructured.from_dict(load_json(struct_path))
                section_data = {
                    "structured_summary": s.structured_summary,
                    "provisional_top3": s.provisional_top3,
                    "friction_blockers": s.friction_blockers,
                    "open_questions": s.open_questions,
                    "value_domains": s.value_domains,
                    "items_count": len(s.items),
                    "stage": s.stage,
                }
        elif section == "059":
            prep_path = NEXT_DAY_PREP_DIR / date_iso / "next_day_prep.json"
            if prep_path.exists():
                p = NextDayPrep.from_dict(load_json(prep_path))
                section_data = {
                    "meeting_briefs": [
                        {
                            "title": b.title,
                            "date": b.date,
                            "people": b.people,
                            "purpose": b.purpose,
                            "context": b.context[:300] + "..." if len(b.context) > 300 else b.context,
                            "key_questions": b.key_questions,
                            "status": b.status,
                        }
                        for b in p.meeting_briefs
                    ],
                    "follow_ups_count": len(p.follow_ups),
                    "monitoring_count": len(p.monitoring_suggestions),
                }
        elif section == "060":
            commit_path = MORNING_COMMIT_DIR / date_iso / "morning_commit.json"
            if commit_path.exists():
                raw_commit = load_json(commit_path)
                m = MorningCommit.from_dict(raw_commit)
                section_data = {
                    "final_top3": m.final_top3,
                    "energy_level": m.energy_level,
                    "time_budget_hrs": m.time_budget_hrs,
                    "source_date": raw_commit.get("source_date", ""),
                    "commits": [
                        {
                            "rank": c.rank,
                            "title": c.title,
                            "status": c.status,
                            "planned_time_block": c.planned_time_block,
                            "estimated_minutes": c.estimated_minutes,
                        }
                        for c in m.commits
                    ],
                }

        return {"section": section, "data": section_data}

    return app


def _build_html() -> str:
    """Build the single-page HTML/JS frontend."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>061 Daily Console</title>
<style>
:root {
  --bg: #ffffff; --surface: #f8f9fa; --surface2: #e9ecef;
  --border: #dee2e6; --text: #212529; --text2: #6c757d;
  --accent: #7c3aed; --accent-light: #6d28d9;
  --green: #10b981; --red: #ef4444; --orange: #f59e0b;
  --blue: #3b82f6; --cyan: #06b6d4;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg); color: var(--text);
  min-height: 100vh; display: flex;
}
/* Left panel — date list */
.sidebar {
  width: 220px; min-width: 220px;
  background: var(--surface); border-right: 1px solid var(--border);
  display: flex; flex-direction: column; height: 100vh;
  position: sticky; top: 0;
}
.sidebar h2 {
  padding: 16px; font-size: 15px; color: var(--accent-light);
  border-bottom: 1px solid var(--border);
}
.date-list {
  flex: 1; overflow-y: auto; padding: 8px;
}
.date-item {
  padding: 10px 12px; border-radius: 8px; cursor: pointer;
  font-size: 14px; color: var(--text2); margin-bottom: 2px;
  display: flex; align-items: center; gap: 8px;
}
.date-item:hover { background: var(--surface2); color: var(--text); }
.date-item.active { background: var(--accent); color: #fff; }
.date-item .indicators {
  display: flex; gap: 3px; margin-left: auto;
}
.dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--border);
}
.dot.has-data { background: var(--green); }

/* Right panel — detail */
.main {
  flex: 1; padding: 24px; overflow-y: auto; height: 100vh;
}
.main h1 { font-size: 22px; margin-bottom: 16px; }
.empty-state {
  text-align: center; padding: 60px 20px; color: var(--text2);
}
.section {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px; margin-bottom: 16px;
}
.section h3 {
  font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;
  color: var(--accent-light); margin-bottom: 12px;
  display: flex; align-items: center; gap: 8px;
}
.section h3 .badge {
  font-size: 11px; padding: 2px 8px; border-radius: 10px;
  text-transform: none; letter-spacing: 0;
}
.badge-green { background: rgba(16,185,129,0.15); color: var(--green); }
.badge-orange { background: rgba(245,158,11,0.15); color: var(--orange); }
.badge-blue { background: rgba(59,130,246,0.15); color: var(--blue); }
.badge-red { background: rgba(239,68,68,0.15); color: var(--red); }
.section pre {
  font-size: 13px; line-height: 1.6; color: var(--text);
  white-space: pre-wrap; word-break: break-word;
}
.section .item-list { list-style: none; }
.section .item-list li {
  padding: 6px 0; border-bottom: 1px solid var(--border);
  font-size: 13px;
}
.section .item-list li:last-child { border-bottom: none; }
.notion-link {
  color: var(--cyan); text-decoration: none; font-size: 12px;
}
.notion-link:hover { text-decoration: underline; }

/* Action buttons */
.actions-bar {
  display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px;
}
.btn {
  padding: 8px 18px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface2); color: var(--text); cursor: pointer;
  font-size: 13px; font-weight: 500; transition: all 0.15s;
}
.btn:hover { border-color: var(--accent); color: var(--accent-light); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn.running { border-color: var(--orange); color: var(--orange); }

/* Notion status bar */
.notion-bar {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;
  font-size: 13px; display: flex; align-items: center; gap: 16px;
  flex-wrap: wrap;
}
.notion-bar .label { color: var(--text2); }
.notion-bar .value { color: var(--text); font-weight: 500; }

/* Job output */
.job-output {
  background: #111318; border: 1px solid var(--border);
  border-radius: 8px; padding: 12px; margin-top: 12px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 12px; max-height: 200px; overflow-y: auto;
  white-space: pre-wrap; color: var(--text2); display: none;
}
.job-output.visible { display: block; }

/* 059 meeting cards */
.meeting-card {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px; margin-bottom: 10px;
  transition: opacity 0.2s;
}
.meeting-card.removed { opacity: 0.35; }
.meeting-card .card-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 10px;
}
.meeting-card .card-title { font-weight: 600; font-size: 14px; }
.meeting-card .card-toggle { display: flex; gap: 6px; align-items: center; }
.meeting-card .card-toggle button {
  padding: 3px 10px; font-size: 11px; border-radius: 6px;
  border: 1px solid var(--border); background: var(--surface);
  color: var(--text2); cursor: pointer;
}
.meeting-card .card-toggle button.active-keep {
  background: rgba(16,185,129,0.15); color: var(--green);
  border-color: var(--green);
}
.meeting-card .card-toggle button.active-remove {
  background: rgba(239,68,68,0.15); color: var(--red);
  border-color: var(--red);
}
.meeting-card .card-fields {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px;
}
.meeting-card .card-fields label {
  font-size: 11px; color: var(--text2); display: block; margin-bottom: 3px;
}
.meeting-card .card-fields input,
.meeting-card .card-fields textarea {
  width: 100%; padding: 6px 8px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 6px;
  color: var(--text); font-size: 12px; box-sizing: border-box;
}
.meeting-card .card-fields textarea { resize: vertical; }
.meeting-card .card-fields .full-width { grid-column: 1 / -1; }
/* Dynamic org/people groups */
.meeting-card .field-group {
  grid-column: 1 / -1; margin-top: 4px;
}
.meeting-card .field-group-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 6px;
}
.meeting-card .field-group-header label {
  font-size: 12px; font-weight: 600; color: var(--text2); margin: 0;
}
.meeting-card .field-group-header .required-star { color: var(--red); }
.meeting-card .btn-add-row {
  padding: 2px 8px; font-size: 10px; border-radius: 5px;
  border: 1px dashed var(--border); background: transparent;
  color: var(--text2); cursor: pointer; transition: all 0.15s;
}
.meeting-card .btn-add-row:hover {
  border-color: var(--accent); color: var(--accent);
}
.meeting-card .row-entry {
  display: grid; grid-template-columns: 1fr 1fr 28px; gap: 6px;
  align-items: end; margin-bottom: 6px;
}
.meeting-card .row-entry.three-col {
  grid-template-columns: 1fr 1fr 28px;
}
.meeting-card .btn-remove-row {
  width: 24px; height: 28px; padding: 0; border: none;
  background: transparent; color: var(--text2); cursor: pointer;
  font-size: 14px; line-height: 28px; text-align: center;
  border-radius: 4px; transition: all 0.15s;
}
.meeting-card .btn-remove-row:hover {
  background: rgba(239,68,68,0.12); color: var(--red);
}
.meeting-card .btn-remove-row:disabled {
  opacity: 0.25; cursor: default;
}
.meeting-card .validation-error {
  font-size: 11px; color: var(--red); margin-top: 2px; display: none;
}

/* Form inputs */
input[type="text"], input[type="number"], textarea, select {
  font-family: inherit;
  outline: none;
}
input[type="text"]:focus, input[type="number"]:focus,
textarea:focus, select:focus {
  border-color: var(--accent) !important;
}
select {
  -webkit-appearance: none;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%239399b2' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  padding-right: 28px !important;
}
</style>
</head>
<body>
<div class="sidebar">
  <h2>Daily Console</h2>
  <div style="padding:4px 16px 8px;font-size:11px;color:var(--text2);border-bottom:1px solid var(--border)">Past 1 week from today</div>
  <div class="date-list" id="date-list"></div>
</div>
<div class="main" id="main">
  <div class="empty-state">
    <p style="font-size:18px; margin-bottom:8px;">Select a date to view pipeline data</p>
    <p>Dates with local outputs appear in the left panel.</p>
  </div>
</div>

<script>
let currentDate = null;
let pollInterval = null;

async function loadDates() {
  const resp = await fetch('/api/dates');
  const d = await resp.json();
  const list = document.getElementById('date-list');
  list.innerHTML = '';
  var todayDate = null;
  // Use sidebar (past 7 days) if available
  var entries = d.sidebar || d.dates.map(function(dt) { return {date: dt, is_today: false, has_data: true}; });
  entries.forEach(function(entry) {
    var dt = typeof entry === 'string' ? entry : entry.date;
    var isToday = entry.is_today || false;
    var hasData = entry.has_data !== false;
    var el = document.createElement('div');
    el.className = 'date-item';
    if (isToday) {
      todayDate = dt;
      el.innerHTML = '<span style="font-weight:600">' + dt + '</span> <span style="font-size:11px;color:var(--accent-light);margin-left:4px">Today</span>';
    } else {
      el.textContent = dt;
    }
    // Data indicator dot
    var dot = document.createElement('span');
    dot.className = 'dot' + (hasData ? ' has-data' : '');
    dot.style.marginLeft = 'auto';
    el.appendChild(dot);
    el.setAttribute('data-date', dt);
    el.addEventListener('click', function() { selectDate(dt); });
    list.appendChild(el);
  });
  // Auto-select today
  if (todayDate) {
    selectDate(todayDate);
  }
}

async function selectDate(dt) {
  currentDate = dt;
  // Highlight active
  document.querySelectorAll('.date-item').forEach(function(el) {
    el.classList.toggle('active', el.getAttribute('data-date') === dt);
  });
  await loadDateView(dt);
}

async function loadDateView(dt) {
  const main = document.getElementById('main');
  main.innerHTML = '<div class="empty-state">Loading...</div>';

  // Fetch local data and Notion status in parallel
  const [dataResp, notionResp] = await Promise.all([
    fetch('/api/data/' + dt),
    fetch('/api/notion/' + dt).catch(function() { return {ok: false}; }),
  ]);
  const data = await dataResp.json();
  let notion = null;
  if (notionResp.ok) {
    notion = await notionResp.json();
  }

  renderDateView(dt, data, notion);
}

var _cachedNotion = null;

function renderDateView(dt, data, notion) {
  const main = document.getElementById('main');
  _cachedNotion = notion;
  main.innerHTML = '<h1>Daily Log: ' + dt + '</h1>'
    + '<div class="job-output" id="job-output"></div>'
    + '<div id="notion-bar-container"></div>'
    + '<div id="section-060-container"></div>'
    + '<div id="section-057-container"></div>'
    + '<div id="section-058-container"></div>'
    + '<div id="section-research-container"></div>';
  renderNotionBar(notion);
  renderSection060(data, notion);
  renderSection057(data, notion);
  renderSection058(data);
  renderSectionResearch(data);
}

function renderNotionBar(notion) {
  var container = document.getElementById('notion-bar-container');
  if (!container) return;
  if (!notion) { container.innerHTML = ''; return; }
  var html = '<div class="notion-bar">';
  if (notion.daily_log) {
    var dl = notion.daily_log;
    html += '<span class="label">Notion:</span>';
    html += '<span class="value">' + (dl.stage || '(no stage)') + '</span>';
    html += '<span class="label">Publish:</span>';
    html += '<span class="value">' + (dl.publish_status || 'Draft') + '</span>';
    if (dl.url) {
      html += '<a class="notion-link" href="' + dl.url + '" target="_blank">Open in Notion</a>';
    }
  } else {
    html += '<span class="label">Notion:</span>';
    html += '<span class="value" style="color:var(--text2)">No Daily Log page</span>';
  }
  if (notion.meeting_briefs && notion.meeting_briefs.length > 0) {
    html += '<span class="label" style="margin-left:auto">Briefs in Notion:</span>';
    html += '<span class="value">' + notion.meeting_briefs.length + '</span>';
  }
  html += '</div>';
  container.innerHTML = html;
}

function renderSection060(data, notion) {
  var container = document.getElementById('section-060-container');
  if (!container) return;
  var html = '<div class="section" id="section-060"><h3>060 Morning Commit';
  if (data.morning_commit) {
    html += ' <span class="badge badge-blue">committed</span>';
  } else {
    html += ' <span class="badge badge-orange">pending</span>';
    html += ' <button class="btn" id="btn-run-060" style="font-size:12px;padding:5px 16px;margin-left:12px;background:var(--accent);color:#fff;border-color:var(--accent)">Run</button>';
  }
  html += '</h3>';
  if (!data.morning_commit) {
    html += '<p style="font-size:13px;color:var(--text2);margin-bottom:10px">';
    html += 'Click <strong>Run</strong> to auto-generate morning commit proposals from the most recent reflection.';
    html += '</p>';
    html += '<div id="run-060-status" style="font-size:12px;color:var(--text2);margin-bottom:10px"></div>';
  }
  if (data.morning_commit) {
    var mc = data.morning_commit;
    html += '<div style="font-size:12px;color:var(--text2);margin-bottom:8px">';
    html += 'Energy: ' + mc.energy_level + ' | Time: ' + mc.time_budget_hrs + 'h';
    if (mc.source_date) html += ' | Source: ' + mc.source_date;
    html += '</div>';
    if (mc.final_top3 && mc.final_top3.length > 0) {
      html += '<h3 style="margin-top:8px">Final Top 3</h3><ul class="item-list">';
      mc.final_top3.forEach(function(t, i) { html += '<li><strong>' + (i+1) + '.</strong> ' + escHtml(t) + '</li>'; });
      html += '</ul>';
    }
    if (mc.commits && mc.commits.length > 0) {
      html += '<h3 style="margin-top:14px">Schedule</h3><ul class="item-list">';
      mc.commits.forEach(function(c) {
        html += '<li>' + c.rank + '. [' + escHtml(c.planned_time_block) + '] ' + escHtml(c.title) + ' (' + c.estimated_minutes + 'min)</li>';
      });
      html += '</ul>';
    }
  }
  // Interactive morning commit form
  var prefillTop3 = ['', '', ''];
  if (data.morning_commit && data.morning_commit.final_top3) {
    prefillTop3 = data.morning_commit.final_top3.slice(0, 3);
  } else if (data.close_structured && data.close_structured.provisional_top3) {
    prefillTop3 = data.close_structured.provisional_top3.slice(0, 3);
  }
  while (prefillTop3.length < 3) prefillTop3.push('');
  var prefillEnergy = 'Medium';
  if (data.morning_commit) prefillEnergy = data.morning_commit.energy_level || 'Medium';
  var prefillTime = '6.0';
  if (data.morning_commit) prefillTime = String(data.morning_commit.time_budget_hrs || 6.0);
  html += '<div style="border-top:1px solid var(--border);padding-top:14px;margin-top:14px">';
  html += '<h3 style="margin-bottom:10px">' + (data.morning_commit ? 'Re-commit Morning Plan' : 'Create Morning Commit') + '</h3>';
  html += '<div style="margin-bottom:10px"><label style="font-size:12px;color:var(--text2);display:block;margin-bottom:6px">Top 3 Priorities</label>';
  for (var ti = 0; ti < 3; ti++) {
    html += '<input type="text" id="commit-top3-' + ti + '" value="' + escAttr(prefillTop3[ti]) + '" placeholder="Priority ' + (ti+1) + '" style="width:100%;padding:8px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:13px;margin-bottom:6px">';
  }
  html += '</div>';
  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">';
  html += '<div><label style="font-size:12px;color:var(--text2);display:block;margin-bottom:4px">Energy Level</label>';
  html += '<select id="commit-energy" style="width:100%;padding:8px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:13px">';
  html += '<option value="Low"' + (prefillEnergy === 'Low' ? ' selected' : '') + '>Low</option>';
  html += '<option value="Medium"' + (prefillEnergy === 'Medium' ? ' selected' : '') + '>Medium</option>';
  html += '<option value="High"' + (prefillEnergy === 'High' ? ' selected' : '') + '>High</option>';
  html += '</select></div>';
  html += '<div><label style="font-size:12px;color:var(--text2);display:block;margin-bottom:4px">Time Budget (hours)</label>';
  html += '<input type="number" id="commit-time" value="' + prefillTime + '" step="0.5" min="0" max="16" style="width:100%;padding:8px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:13px"></div>';
  html += '</div>';
  html += '<button class="btn" id="btn-commit-060">Confirm Morning Commit</button>';
  html += '<div id="commit-status" style="margin-top:8px;font-size:12px;color:var(--text2)"></div>';
  html += '</div>';
  html += '</div>';
  container.innerHTML = html;
  // Attach handlers
  var run060Btn = document.getElementById('btn-run-060');
  if (run060Btn) {
    run060Btn.addEventListener('click', function() { run060Generate(currentDate); });
  }
  var commitBtn = document.getElementById('btn-commit-060');
  if (commitBtn) {
    commitBtn.addEventListener('click', function() { commitMorning(currentDate); });
  }
}

function renderSection057(data, notion) {
  var container = document.getElementById('section-057-container');
  if (!container) return;
  var html = '<div class="section" id="section-057"><h3>057 Raw Close Log';
  // Determine source: Notion page exists = canonical, else local fallback
  var raw057 = null;
  var raw057source = '';
  if (notion && notion.daily_log) {
    // Notion page exists - use it as canonical source
    raw057 = {
      raw_text: notion.daily_log.raw_close_log || '',
      satisfaction: notion.daily_log.satisfaction,
      energy_level: notion.daily_log.energy_level || '',
      input_mode: 'notion',
    };
    raw057source = 'notion';
  } else if (data.close_raw) {
    raw057 = data.close_raw;
    raw057source = 'local';
  }
  if (raw057) {
    html += ' <span class="badge badge-green">' + raw057source + '</span>';
  } else {
    html += ' <span class="badge badge-red">missing</span>';
  }
  html += '</h3>';
  if (raw057 && raw057.raw_text) {
    html += '<div style="font-size:12px;color:var(--text2);margin-bottom:6px">';
    if (raw057.satisfaction) html += 'Satisfaction: ' + raw057.satisfaction + '/5 | ';
    html += 'Energy: ' + (raw057.energy_level || 'n/a') + ' | Source: ' + raw057source;
    html += '</div>';
    html += '<pre>' + escHtml(raw057.raw_text) + '</pre>';
  } else if (raw057source === 'notion') {
    html += '<pre style="color:var(--text2)">Notion page exists but Raw Close Log is empty.</pre>';
  } else {
    html += '<pre style="color:var(--text2)">No 057 output for this date.</pre>';
  }

  // -- Guided Close Log Wizard --
  html += '<div style="border-top:1px solid var(--border);padding-top:14px;margin-top:14px">';
  html += '<h3 style="margin-bottom:12px">Guided Close Log</h3>';

  // Wizard sections (theme-based prompts)
  var wizSections = [
    {key: 'done', title: '\u4ECA\u65E5\u3084\u3063\u305F\u3053\u3068', instr: '\u4ECA\u65E5\u53D6\u308A\u7D44\u3093\u3060\u30BF\u30B9\u30AF\u3084\u6210\u679C\u3092\u8A71\u3057\u3066\u304F\u3060\u3055\u3044\u3002'},
    {key: 'friction', title: '\u8A70\u307E\u308A\u3084\u9055\u548C\u611F', instr: '\u8A70\u307E\u3063\u305F\u3053\u3068\u3001\u9055\u548C\u611F\u304C\u3042\u3063\u305F\u3053\u3068\u3001\u6C17\u306B\u306A\u3063\u305F\u6469\u64E6\u3092\u8A71\u3057\u3066\u304F\u3060\u3055\u3044\u3002'},
    {key: 'tomorrow', title: '\u660E\u65E5\u306E\u4E88\u5B9A', instr: '\u660E\u65E5\u3084\u308B\u3053\u3068\u3001\u4E88\u5B9A\u3057\u3066\u3044\u308B\u3053\u3068\u3092\u8A71\u3057\u3066\u304F\u3060\u3055\u3044\u3002'},
    {key: 'mind', title: '\u6C17\u306B\u306A\u3063\u3066\u3044\u308B\u3053\u3068', instr: '\u982D\u306E\u4E2D\u306B\u3042\u308B\u3053\u3068\u3001\u6C17\u304C\u304B\u308A\u306A\u3053\u3068\u3092\u81EA\u7531\u306B\u8A71\u3057\u3066\u304F\u3060\u3055\u3044\u3002'},
  ];

  wizSections.forEach(function(sec) {
    html += '<div style="margin-bottom:14px;background:var(--surface2);border-radius:8px;padding:12px">';
    html += '<div style="margin-bottom:8px">';
    html += '<span style="font-weight:600;font-size:13px">' + sec.title + '</span>';
    html += '<div style="font-size:11px;color:var(--text2);margin-top:2px">' + sec.instr + '</div>';
    html += '</div>';
    html += '<textarea id="wiz-' + sec.key + '" rows="3" style="width:100%;padding:8px;background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:13px;resize:vertical"></textarea>';
    html += '<div style="display:flex;gap:8px;margin-top:6px;align-items:center">';
    html += '<button class="btn" style="font-size:12px;padding:4px 12px" id="wiz-rec-' + sec.key + '">Record</button>';
    html += '<span id="wiz-status-' + sec.key + '" style="font-size:11px;color:var(--text2)"></span>';
    html += '</div>';
    html += '</div>';
  });

  // Satisfaction + Energy selectors
  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">';
  html += '<div><label style="font-size:12px;color:var(--text2);display:block;margin-bottom:4px">Satisfaction (1-5)</label>';
  html += '<select id="wiz-satisfaction" style="width:100%;padding:8px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:13px">';
  html += '<option value="">--</option>';
  for (var si = 1; si <= 5; si++) { html += '<option value="' + si + '">' + si + '</option>'; }
  html += '</select></div>';
  html += '<div><label style="font-size:12px;color:var(--text2);display:block;margin-bottom:4px">Energy Level</label>';
  html += '<select id="wiz-energy" style="width:100%;padding:8px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:13px">';
  html += '<option value="">--</option>';
  html += '<option value="Low">Low</option>';
  html += '<option value="Medium">Medium</option>';
  html += '<option value="High">High</option>';
  html += '</select></div>';
  html += '</div>';

  // Value domains multi-select
  html += '<div style="margin-bottom:14px"><label style="font-size:12px;color:var(--text2);display:block;margin-bottom:6px">Value Domains</label>';
  html += '<div id="wiz-values" style="display:flex;flex-wrap:wrap;gap:6px">';
  var valueDomains = [
    {en:'Health',ja:'\u5065\u5EB7'},{en:'Family',ja:'\u5BB6\u65CF'},{en:'Work',ja:'\u4ED5\u4E8B'},
    {en:'Learning',ja:'\u5B66\u3073'},{en:'Creativity',ja:'\u5275\u9020'},{en:'Connection',ja:'\u3064\u306A\u304C\u308A'},
    {en:'Adventure',ja:'\u5192\u967A'},{en:'Freedom',ja:'\u81EA\u7531'},{en:'Contribution',ja:'\u8CA2\u732E'},
    {en:'Growth',ja:'\u6210\u9577'},{en:'Integrity',ja:'\u8AA0\u5B9F'},{en:'Gratitude',ja:'\u611F\u8B1D'},
  ];
  valueDomains.forEach(function(vd) {
    html += '<label style="display:flex;align-items:center;gap:4px;font-size:12px;padding:4px 8px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;cursor:pointer">';
    html += '<input type="checkbox" class="wiz-value-check" value="' + vd.en + '" style="accent-color:var(--accent)">';
    html += vd.en + ' (' + vd.ja + ')';
    html += '</label>';
  });
  html += '</div></div>';

  // Submit button
  html += '<button class="btn" id="btn-wizard-submit">Submit Close Log</button>';
  html += '<div id="wizard-status" style="margin-top:8px;font-size:12px;color:var(--text2)"></div>';
  html += '</div>';
  html += '</div>';
  container.innerHTML = html;
  // Attach wizard handlers
  ['done', 'friction', 'tomorrow', 'mind'].forEach(function(key) {
    var recBtn = document.getElementById('wiz-rec-' + key);
    if (recBtn) {
      recBtn.addEventListener('click', function() { toggleWizRecord(key); });
    }
  });
  var wizSubmitBtn = document.getElementById('btn-wizard-submit');
  if (wizSubmitBtn) {
    wizSubmitBtn.addEventListener('click', function() { submitWizard(currentDate); });
  }
}

function renderSection058(data) {
  var container = document.getElementById('section-058-container');
  if (!container) return;
  var html = '<div class="section" id="section-058"><h3>058 Structured Summary';
  if (data.close_structured) {
    html += ' <span class="badge badge-green">' + data.close_structured.stage + '</span>';
  } else {
    html += ' <span class="badge badge-red">missing</span>';
  }
  html += ' <button class="btn" id="btn-run-058" style="font-size:11px;padding:3px 10px;margin-left:8px">Run 058</button>';
  html += '</h3>';
  if (data.close_structured) {
    var cs = data.close_structured;
    html += '<pre>' + escHtml(cs.structured_summary || '(empty)') + '</pre>';
    if (cs.provisional_top3 && cs.provisional_top3.length > 0) {
      html += '<h3 style="margin-top:14px">Provisional Top 3</h3><ul class="item-list">';
      cs.provisional_top3.forEach(function(t, i) { html += '<li>' + (i+1) + '. ' + escHtml(t) + '</li>'; });
      html += '</ul>';
    }
    if (cs.friction_blockers && cs.friction_blockers.length > 0) {
      html += '<h3 style="margin-top:14px">Friction / Blockers</h3><ul class="item-list">';
      cs.friction_blockers.forEach(function(f) { html += '<li>' + escHtml(f) + '</li>'; });
      html += '</ul>';
    }
    if (cs.open_questions && cs.open_questions.length > 0) {
      html += '<h3 style="margin-top:14px">Open Questions</h3><ul class="item-list">';
      cs.open_questions.forEach(function(q) { html += '<li>' + escHtml(q) + '</li>'; });
      html += '</ul>';
    }
  } else {
    html += '<pre style="color:var(--text2)">No 058 output for this date.</pre>';
  }
  html += '</div>';
  container.innerHTML = html;
  // Attach handler
  var btn058 = document.getElementById('btn-run-058');
  if (btn058) {
    btn058.addEventListener('click', function() { runScript('058'); });
  }
}

function renderSectionResearch(data) {
  var container = document.getElementById('section-research-container');
  if (!container) return;
  var html = '<div class="section" id="section-research"><h3>Research Questions</h3>';
  html += '<p style="font-size:13px;color:var(--text2);margin-bottom:10px">';
  html += '\u632F\u308A\u8FD4\u308A\u304B\u3089\u8ABF\u67FB\u8CEA\u554F\u3092\u751F\u6210\u3057\u3001073 Deep Research \u3067\u5B9F\u884C\u3057\u307E\u3059\u3002';
  html += '</p>';

  // Questions container
  html += '<div id="research-questions"></div>';

  // Add question button
  html += '<button class="btn" id="btn-add-question" style="margin-top:8px;font-size:12px" onclick="addResearchQuestion()">';
  html += '+ \u8CEA\u554F\u3092\u8FFD\u52A0</button>';

  // Action buttons
  html += '<div style="margin-top:14px;display:flex;gap:10px;align-items:center">';
  if (!data.close_structured) {
    html += '<button class="btn" disabled>058 \u304C\u5FC5\u8981\u3067\u3059</button>';
  } else {
    html += '<button class="btn" id="btn-generate-questions" onclick="generateResearchQuestions(currentDate)">\u8CEA\u554F\u3092\u751F\u6210</button>';
  }
  html += '<button class="btn" id="btn-run-research" onclick="runAllResearch(currentDate)">\u8ABF\u67FB\u3092\u5B9F\u884C</button>';
  html += '<div id="research-status" style="font-size:12px;color:var(--text2)"></div>';
  html += '</div>';

  // Results container
  html += '<div id="research-results" style="margin-top:14px"></div>';

  html += '</div>'; // end section-research
  container.innerHTML = html;
}

async function refreshSection(section, date) {
  try {
    var resp = await fetch('/api/refresh-section/' + section, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({date: date}),
    });
    var d = await resp.json();
    var sectionData = {};
    if (section === '057') {
      sectionData.close_raw = d.data;
      renderSection057(sectionData, _cachedNotion);
    } else if (section === '058') {
      sectionData.close_structured = d.data;
      renderSection058(sectionData);
    } else if (section === '060') {
      sectionData.morning_commit = d.data;
      renderSection060(sectionData, _cachedNotion);
    }
  } catch(e) {
    console.error('refreshSection error:', e);
  }
}

function escHtml(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function escAttr(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function runScript(num) {
  if (!currentDate) return;
  const btn = document.getElementById('btn-' + num);
  if (btn) { btn.disabled = true; btn.className = 'btn running'; btn.textContent = num + ' running...'; }

  const outputEl = document.getElementById('job-output');
  if (outputEl) { outputEl.className = 'job-output visible'; outputEl.textContent = 'Starting ' + num + '...'; }

  try {
    const resp = await fetch('/api/run/' + num + '/' + currentDate, {method: 'POST'});
    const d = await resp.json();
    if (d.job_key) {
      pollJob(d.job_key, num);
    }
  } catch(e) {
    if (outputEl) outputEl.textContent = 'Error: ' + e.message;
    if (btn) { btn.disabled = false; btn.className = 'btn'; btn.textContent = 'Run ' + num; }
  }
}

function pollJob(jobKey, num) {
  const outputEl = document.getElementById('job-output');
  const btn = document.getElementById('btn-' + num);
  if (pollInterval) clearInterval(pollInterval);

  pollInterval = setInterval(async function() {
    try {
      const resp = await fetch('/api/job/' + encodeURIComponent(jobKey));
      const d = await resp.json();
      if (outputEl) outputEl.textContent = d.output || '(waiting...)';
      if (d.status !== 'running') {
        clearInterval(pollInterval);
        pollInterval = null;
        if (btn) {
          btn.disabled = false;
          btn.className = 'btn';
          btn.textContent = 'Run ' + num;
        }
        if (d.status === 'done') {
          // Refresh only the affected section
          await refreshSection(num, currentDate);
        }
      }
    } catch(e) {
      // ignore polling errors
    }
  }, 2000);
}

// -- Research Questions Workflow --

var _researchQuestions = [];
var _researchResults = {};
var _researchJobKeys = {};

function renderResearchQuestions() {
  var container = document.getElementById('research-questions');
  if (!container) return;
  var html = '';
  _researchQuestions.forEach(function(q, i) {
    html += '<div style="display:flex;gap:8px;align-items:start;margin-bottom:8px">';
    html += '<textarea id="rq-' + i + '" style="flex:1;min-height:36px;padding:8px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--text);font-size:13px;font-family:inherit;resize:vertical">' + escHtml(q) + '</textarea>';
    html += '<button class="btn" style="font-size:11px;padding:4px 10px;color:var(--red);border-color:var(--red)" onclick="removeResearchQuestion(' + i + ')">&times;</button>';
    html += '</div>';
  });
  container.innerHTML = html;
}

function addResearchQuestion() {
  _researchQuestions.push('');
  renderResearchQuestions();
  var el = document.getElementById('rq-' + (_researchQuestions.length - 1));
  if (el) el.focus();
}

function removeResearchQuestion(idx) {
  _researchQuestions.splice(idx, 1);
  renderResearchQuestions();
}

function syncResearchQuestions() {
  _researchQuestions.forEach(function(q, i) {
    var el = document.getElementById('rq-' + i);
    if (el) _researchQuestions[i] = el.value.trim();
  });
}

async function generateResearchQuestions(dt) {
  var btn = document.getElementById('btn-generate-questions');
  var status = document.getElementById('research-status');
  if (btn) { btn.disabled = true; btn.textContent = '\u751F\u6210\u4E2D...'; }
  status.textContent = '058 \u304B\u3089\u8CEA\u554F\u3092\u751F\u6210\u4E2D...';
  status.style.color = 'var(--text2)';

  try {
    var resp = await fetch('/api/research/generate-questions/' + dt, {method: 'POST'});
    var d = await resp.json();
    if (d.ok && d.questions) {
      _researchQuestions = d.questions;
      renderResearchQuestions();
      status.textContent = d.questions.length + ' \u8CEA\u554F\u3092\u751F\u6210\u3057\u307E\u3057\u305F\u3002\u7DE8\u96C6\u3057\u3066\u304B\u3089\u300C\u8ABF\u67FB\u3092\u5B9F\u884C\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002';
      status.style.color = 'var(--green)';
    } else {
      status.textContent = 'Error: ' + (d.error || 'No questions generated');
      status.style.color = 'var(--red)';
    }
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
    status.style.color = 'var(--red)';
  }
  if (btn) { btn.disabled = false; btn.textContent = '\u8CEA\u554F\u3092\u751F\u6210'; }
}

async function runAllResearch(dt) {
  syncResearchQuestions();
  var questions = _researchQuestions.filter(function(q) { return q.length > 0; });
  if (questions.length === 0) {
    var status = document.getElementById('research-status');
    status.textContent = '\u8CEA\u554F\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\u3002';
    status.style.color = 'var(--red)';
    return;
  }

  var btn = document.getElementById('btn-run-research');
  var statusEl = document.getElementById('research-status');
  if (btn) { btn.disabled = true; btn.textContent = '\u5B9F\u884C\u4E2D...'; }
  statusEl.textContent = questions.length + ' \u8CEA\u554F\u3092\u5B9F\u884C\u4E2D...';
  statusEl.style.color = 'var(--text2)';

  _researchResults = {};
  _researchJobKeys = {};
  renderResearchResults(questions);

  for (var i = 0; i < questions.length; i++) {
    var q = questions[i];
    try {
      var resp = await fetch('/api/research/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: q, date_iso: dt}),
      });
      var d = await resp.json();
      if (d.job_key) {
        _researchJobKeys[i] = d.job_key;
        pollResearchJob(i, d.job_key, questions);
      }
    } catch(e) {
      _researchResults[i] = {ok: false, error: e.message, question: q};
      renderResearchResults(questions);
    }
  }
}

function pollResearchJob(idx, jobKey, questions) {
  var interval = setInterval(async function() {
    try {
      var resp = await fetch('/api/job/' + encodeURIComponent(jobKey));
      var d = await resp.json();
      if (d.status !== 'running') {
        clearInterval(interval);
        if (d.status === 'done' && d.result) {
          _researchResults[idx] = d.result;
        } else {
          _researchResults[idx] = {ok: false, error: d.output || 'Failed', question: questions[idx]};
        }
        renderResearchResults(questions);
        checkAllResearchDone(questions);
      }
    } catch(e) { /* ignore */ }
  }, 3000);
}

function checkAllResearchDone(questions) {
  var done = true;
  for (var i = 0; i < questions.length; i++) {
    if (!_researchResults[i]) { done = false; break; }
  }
  if (done) {
    var btn = document.getElementById('btn-run-research');
    var statusEl = document.getElementById('research-status');
    var okCount = 0;
    for (var j = 0; j < questions.length; j++) {
      if (_researchResults[j] && _researchResults[j].ok) okCount++;
    }
    if (btn) { btn.disabled = false; btn.textContent = '\u8ABF\u67FB\u3092\u5B9F\u884C'; }
    statusEl.textContent = okCount + '/' + questions.length + ' \u5B8C\u4E86';
    statusEl.style.color = okCount === questions.length ? 'var(--green)' : 'var(--orange)';
  }
}

function simpleMarkdown(text) {
  var s = escHtml(text);
  s = s.replace(/^### (.+)$/gm, '<h4 style="margin:10px 0 4px;font-size:13px">$1</h4>');
  s = s.replace(/^## (.+)$/gm, '<h3 style="margin:14px 0 6px;font-size:14px">$1</h3>');
  s = s.replace(/^# (.+)$/gm, '<h2 style="margin:16px 0 8px;font-size:16px">$1</h2>');
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/^- (.+)$/gm, '<li style="margin-left:16px;list-style:disc">$1</li>');
  s = s.replace(/^\d+\. (.+)$/gm, '<li style="margin-left:16px;list-style:decimal">$1</li>');
  s = s.replace(/\n/g, '<br>');
  return s;
}

function toggleAnswer(idx) {
  var preview = document.getElementById('answer-preview-' + idx);
  var full = document.getElementById('answer-full-' + idx);
  if (preview.style.display !== 'none') {
    preview.style.display = 'none';
    full.style.display = 'block';
  } else {
    preview.style.display = 'block';
    full.style.display = 'none';
  }
}

function renderResearchResults(questions) {
  var container = document.getElementById('research-results');
  if (!container) return;
  var html = '';
  questions.forEach(function(q, i) {
    var r = _researchResults[i];
    html += '<div style="background:var(--surface2);border-radius:8px;padding:12px;margin-bottom:8px">';
    if (!r) {
      html += '<div style="display:flex;align-items:center;gap:8px">';
      html += '<span style="color:var(--orange)">\u23F3</span>';
      html += '<span style="font-size:13px">' + escHtml(q) + '</span>';
      html += '<span style="font-size:11px;color:var(--text2)">\u5B9F\u884C\u4E2D...</span>';
      html += '</div>';
    } else if (r.ok) {
      html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">';
      html += '<span style="color:var(--green)">\u2713</span>';
      html += '<span style="font-size:13px;font-weight:600">' + escHtml(q) + '</span>';
      html += '</div>';
      if (r.memo_title) html += '<div style="font-size:12px;color:var(--text2)">Memo: ' + escHtml(r.memo_title) + '</div>';
      // Answer: preview + show more toggle
      if (r.answer_full) {
        var hasMore = r.answer_full.length > 200;
        html += '<div id="answer-preview-' + i + '" style="margin-top:6px">';
        html += '<pre style="white-space:pre-wrap;font-size:12px;color:var(--text);margin:0;font-family:inherit;line-height:1.6">' + escHtml(r.answer_preview || r.answer_full.substring(0, 200)) + (hasMore ? '...' : '') + '</pre>';
        if (hasMore) html += '<button class="btn" style="font-size:11px;padding:2px 10px;margin-top:4px" onclick="toggleAnswer(' + i + ')">Show more</button>';
        html += '</div>';
        if (hasMore) {
          html += '<div id="answer-full-' + i + '" style="display:none;margin-top:6px">';
          html += '<div style="max-height:600px;overflow-y:auto;font-size:12px;line-height:1.7;color:var(--text)">' + simpleMarkdown(r.answer_full) + '</div>';
          html += '<button class="btn" style="font-size:11px;padding:2px 10px;margin-top:6px" onclick="toggleAnswer(' + i + ')">\u6298\u308A\u305F\u305F\u3080</button>';
          html += '</div>';
        }
      }
      html += '<div style="font-size:11px;color:var(--text2);margin-top:4px">Sources: ' + (r.sources_count||0) + ' | Evidence: ' + (r.evidence_count||0) + ' | Claims: ' + (r.claims_count||0) + '</div>';
    } else {
      html += '<div style="display:flex;align-items:center;gap:8px">';
      html += '<span style="color:var(--red)">\u2717</span>';
      html += '<span style="font-size:13px">' + escHtml(q) + '</span>';
      html += '</div>';
      html += '<div style="font-size:12px;color:var(--red)">' + escHtml(r.error || 'Unknown error') + '</div>';
    }
    html += '</div>';
  });
  container.innerHTML = html;
}

// -- 057 Wizard: Audio Recording per Section --

var _wizRecorders = {};
var _wizTitleMap = {
  done: '\u4ECA\u65E5\u3084\u3063\u305F\u3053\u3068',
  friction: '\u8A70\u307E\u308A\u3084\u9055\u548C\u611F',
  tomorrow: '\u660E\u65E5\u306E\u4E88\u5B9A',
  mind: '\u6C17\u306B\u306A\u3063\u3066\u3044\u308B\u3053\u3068',
};

async function toggleWizRecord(sectionKey) {
  var btn = document.getElementById('wiz-rec-' + sectionKey);
  var statusEl = document.getElementById('wiz-status-' + sectionKey);
  var textarea = document.getElementById('wiz-' + sectionKey);

  // If already recording, stop
  if (_wizRecorders[sectionKey]) {
    _wizRecorders[sectionKey].stop();
    btn.textContent = 'Record';
    btn.className = 'btn';
    btn.style.cssText = 'font-size:12px;padding:4px 12px';
    statusEl.textContent = 'Transcribing...';
    statusEl.style.color = 'var(--text2)';
    return;
  }

  try {
    var stream = await navigator.mediaDevices.getUserMedia({audio: true});
    var recorder = new MediaRecorder(stream);
    var chunks = [];

    recorder.ondataavailable = function(e) { if (e.data.size > 0) chunks.push(e.data); };

    recorder.onstop = async function() {
      stream.getTracks().forEach(function(t) { t.stop(); });
      _wizRecorders[sectionKey] = null;

      var blob = new Blob(chunks, {type: 'audio/webm'});
      statusEl.textContent = 'Uploading & transcribing...';
      statusEl.style.color = 'var(--text2)';

      var formData = new FormData();
      formData.append('audio', blob, 'recording.webm');
      formData.append('section_key', sectionKey);
      formData.append('date_iso', currentDate);

      try {
        var resp = await fetch('/api/057/audio-upload', {method: 'POST', body: formData});
        var d = await resp.json();
        if (d.ok && d.transcript) {
          var existing = textarea.value.trim();
          textarea.value = existing ? (existing + '\\n' + d.transcript) : d.transcript;
          statusEl.textContent = 'Transcribed (' + d.transcript.length + ' chars)';
          statusEl.style.color = 'var(--green)';
        } else {
          statusEl.textContent = 'Error: ' + (d.error || 'Unknown');
          statusEl.style.color = 'var(--red)';
        }
      } catch(e) {
        statusEl.textContent = 'Upload error: ' + e.message;
        statusEl.style.color = 'var(--red)';
      }
    };

    recorder.start();
    _wizRecorders[sectionKey] = recorder;
    btn.textContent = 'Stop';
    btn.className = 'btn running';
    btn.style.cssText = 'font-size:12px;padding:4px 12px;border-color:var(--orange);color:var(--orange)';
    statusEl.textContent = 'Recording...';
    statusEl.style.color = 'var(--orange)';
  } catch(e) {
    statusEl.textContent = 'Mic error: ' + e.message;
    statusEl.style.color = 'var(--red)';
  }
}

// -- 057 Wizard: Submit all sections --

async function submitWizard(dt) {
  var sectionKeys = ['done', 'friction', 'tomorrow', 'mind'];
  var sections = [];
  sectionKeys.forEach(function(key) {
    var textarea = document.getElementById('wiz-' + key);
    sections.push({
      key: key,
      title_ja: _wizTitleMap[key] || key,
      text: textarea ? textarea.value.trim() : '',
    });
  });

  var satisfaction = document.getElementById('wiz-satisfaction').value;
  var energy = document.getElementById('wiz-energy').value;

  var valueDomainsList = [];
  document.querySelectorAll('.wiz-value-check:checked').forEach(function(cb) {
    valueDomainsList.push(cb.value);
  });

  var statusEl = document.getElementById('wizard-status');
  var btn = document.getElementById('btn-wizard-submit');

  // Validate: at least one section has content
  var hasContent = sections.some(function(s) { return s.text.length > 0; });
  if (!hasContent) {
    statusEl.textContent = 'At least one section must have content.';
    statusEl.style.color = 'var(--red)';
    return;
  }

  if (btn) { btn.disabled = true; btn.textContent = 'Submitting...'; btn.className = 'btn running'; }
  statusEl.textContent = 'Saving guided close log...';
  statusEl.style.color = 'var(--text2)';

  try {
    var resp = await fetch('/api/057/wizard-submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        date: dt,
        sections: sections,
        satisfaction: satisfaction ? parseInt(satisfaction) : null,
        energy_level: energy || null,
        value_domains: valueDomainsList,
      }),
    });
    var d = await resp.json();
    if (d.ok) {
      statusEl.textContent = 'Close log saved!';
      statusEl.style.color = 'var(--green)';
      setTimeout(function() { refreshSection('057', currentDate); }, 800);
    } else {
      statusEl.textContent = 'Error: ' + (d.error || 'Unknown');
      statusEl.style.color = 'var(--red)';
    }
  } catch(e) {
    statusEl.textContent = 'Error: ' + e.message;
    statusEl.style.color = 'var(--red)';
  }
  if (btn) { btn.disabled = false; btn.textContent = 'Submit Close Log'; btn.className = 'btn'; }
}

// -- 060 Run: Auto-generate proposals --

async function run060Generate(dt) {
  var btn = document.getElementById('btn-run-060');
  var statusEl = document.getElementById('run-060-status');
  if (btn) { btn.disabled = true; btn.textContent = 'Running...'; btn.className = 'btn running'; }
  if (statusEl) { statusEl.textContent = 'Resolving reflection source and generating proposals...'; statusEl.style.color = 'var(--text2)'; }

  try {
    var resp = await fetch('/api/060/run/' + dt, {method: 'POST'});
    var d = await resp.json();
    if (d.job_key) {
      poll060RunJob(d.job_key, dt);
    } else if (d.ok) {
      if (statusEl) { statusEl.textContent = 'Proposals generated! Refreshing...'; statusEl.style.color = 'var(--green)'; }
      setTimeout(function() { refreshSection('060', currentDate); }, 800);
    } else {
      if (statusEl) { statusEl.textContent = 'Error: ' + (d.error || 'Generation failed'); statusEl.style.color = 'var(--red)'; }
      if (btn) { btn.disabled = false; btn.textContent = 'Run'; btn.className = 'btn'; btn.style.cssText = 'font-size:12px;padding:5px 16px;margin-left:12px;background:var(--accent);color:#fff;border-color:var(--accent)'; }
    }
  } catch(e) {
    if (statusEl) { statusEl.textContent = 'Error: ' + e.message; statusEl.style.color = 'var(--red)'; }
    if (btn) { btn.disabled = false; btn.textContent = 'Run'; btn.className = 'btn'; btn.style.cssText = 'font-size:12px;padding:5px 16px;margin-left:12px;background:var(--accent);color:#fff;border-color:var(--accent)'; }
  }
}

function poll060RunJob(jobKey, dt) {
  var statusEl = document.getElementById('run-060-status');
  var btn = document.getElementById('btn-run-060');
  var interval = setInterval(async function() {
    try {
      var resp = await fetch('/api/job/' + encodeURIComponent(jobKey));
      var d = await resp.json();
      if (d.output && statusEl) statusEl.textContent = d.output;
      if (d.status !== 'running') {
        clearInterval(interval);
        if (d.status === 'done' && d.result && d.result.ok) {
          if (statusEl) {
            statusEl.textContent = 'Proposals generated! Source: ' + (d.result.source_date || '?') + '. Refreshing...';
            statusEl.style.color = 'var(--green)';
          }
          setTimeout(function() { refreshSection('060', currentDate); }, 800);
        } else {
          var errMsg = (d.result && d.result.error) ? d.result.error : (d.output || 'Unknown error');
          if (statusEl) { statusEl.textContent = 'Failed: ' + errMsg; statusEl.style.color = 'var(--red)'; }
          if (btn) { btn.disabled = false; btn.textContent = 'Run'; btn.className = 'btn'; btn.style.cssText = 'font-size:12px;padding:5px 16px;margin-left:12px;background:var(--accent);color:#fff;border-color:var(--accent)'; }
        }
      }
    } catch(e) { /* ignore poll errors */ }
  }, 2000);
}

// -- 060 Browser Morning Commit --

async function commitMorning(dt) {
  var top3 = [];
  for (var i = 0; i < 3; i++) {
    var v = document.getElementById('commit-top3-' + i).value.trim();
    if (v) top3.push(v);
  }
  var energy = document.getElementById('commit-energy').value;
  var timeHrs = parseFloat(document.getElementById('commit-time').value) || 6.0;
  var statusEl = document.getElementById('commit-status');
  var btn = document.getElementById('btn-commit-060');

  if (top3.length === 0) { statusEl.textContent = 'At least one priority is required.'; statusEl.style.color = 'var(--red)'; return; }

  if (btn) { btn.disabled = true; btn.textContent = 'Committing...'; btn.className = 'btn running'; }
  statusEl.textContent = 'Generating morning commit (may take a moment)...';
  statusEl.style.color = 'var(--text2)';

  try {
    var resp = await fetch('/api/060/commit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        date: dt,
        final_top3: top3,
        energy_level: energy,
        time_budget_hrs: timeHrs,
      }),
    });
    var d = await resp.json();
    if (d.ok) {
      statusEl.textContent = 'Morning commit saved!';
      statusEl.style.color = 'var(--green)';
      setTimeout(function() { refreshSection('060', currentDate); }, 800);
    } else {
      statusEl.textContent = 'Error: ' + (d.error || 'Unknown');
      statusEl.style.color = 'var(--red)';
    }
  } catch(e) {
    statusEl.textContent = 'Error: ' + e.message;
    statusEl.style.color = 'var(--red)';
  }
  if (btn) { btn.disabled = false; btn.textContent = 'Confirm Morning Commit'; btn.className = 'btn'; }
}

// Init
loadDates();
</script>
</body>
</html>'''


# ═══════════════════════════════════════════════════════════════════
# SERVER LAUNCHER
# ═══════════════════════════════════════════════════════════════════

def run_server(
    *,
    port: int = _UI_PORT,
    verbose: bool = False,
    no_browser: bool = False,
) -> None:
    """Launch the Web UI server."""
    load_env()
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    logger.info("Starting %s on port %d", SCRIPT_NAME, port)

    # Log which Notion databases are configured
    daily_logs_id = get_optional_db_id("NOTION_Daily_Logs_ID")
    meeting_briefs_id = get_optional_db_id("NOTION_Meeting_Briefs_ID")
    logger.info(
        "Notion databases: Daily_Logs=%s, Meeting_Briefs=%s",
        daily_logs_id[:8] + "..." if daily_logs_id else "(not set)",
        meeting_briefs_id[:8] + "..." if meeting_briefs_id else "(not set)",
    )

    # Log local data directories
    for label, path in [
        ("close_raw", CLOSE_RAW_DIR),
        ("close_structured", CLOSE_STRUCTURED_DIR),
        ("next_day_prep", NEXT_DAY_PREP_DIR),
        ("morning_commit", MORNING_COMMIT_DIR),
    ]:
        logger.info("Data dir [%s]: %s (exists=%s)", label, path, path.is_dir())

    import uvicorn

    app = _build_app()

    url = f"http://localhost:{port}"
    if not no_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    logger.info("Daily Console available at %s", url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="061 Daily Operational Console — Web UI for pipeline management",
    )
    parser.add_argument("--port", type=int, default=_UI_PORT,
                        help=f"Server port (default: {_UI_PORT})")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")
    args = parser.parse_args()

    run_server(
        port=args.port,
        verbose=args.verbose,
        no_browser=args.no_browser,
    )


if __name__ == "__main__":
    main()
