#!/usr/bin/env python
# src/scripts/065_monthly_review.py
"""Monthly Review — month-end summary via voice/text input.

Captures:
  - Success 3 (top 3 successes this month)
  - Improvements 3 (top 3 areas for improvement)
  - Value adjustment needed (yes/no)
  - Value adjustment proposal (if needed)
  - Structural lessons

Mirrors the 061 UX: show existing Notion content, accept voice/text input,
write structured fields back to Notion MONTHLY_LOG with Log Type = Review.

Usage::

    # Interactive (text prompt)
    python -m src.scripts.065_monthly_review

    # With voice input (wizard)
    python -m src.scripts.065_monthly_review --record-wizard

    # Specific date
    python -m src.scripts.065_monthly_review --date 2026-02-27

    # Debug logging
    python -m src.scripts.065_monthly_review -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env, setup_logging, RunMetadata, get_iso_week_context

logger = logging.getLogger("065_monthly_review")

SCRIPT_NAME = "065_monthly_review"
JST = ZoneInfo("Asia/Tokyo")


# ── Display existing Notion content ──────────────────────────────

def _show_existing_content(page: Optional[Dict[str, Any]]) -> None:
    """Display existing MONTHLY_LOG Review content from Notion."""
    if not page:
        print("\n  (No existing Monthly Review found for this period.)\n")
        return

    from src.notion.period_upsert import extract_rich_text, extract_title_text

    props = page.get("properties", {})
    title = extract_title_text(props.get("Title", {}))
    success_3 = extract_rich_text(props.get("Success 3", {}))
    improvements_3 = extract_rich_text(props.get("Improvements 3", {}))
    value_adj_needed = props.get("Value Adjustment Needed", {}).get("checkbox")
    value_adj_proposal = extract_rich_text(props.get("Value Adjustment Proposal", {}))
    structural_lessons = extract_rich_text(props.get("Structural Lessons", {}))
    voice_transcript = extract_rich_text(props.get("Voice Transcript", {}))
    llm_summary = extract_rich_text(props.get("LLM Summary", {}))

    print(f"\n{'='*60}")
    print(f"  Existing Monthly Review: {title}")
    print(f"{'='*60}")
    if success_3:
        print(f"\n  Success 3:\n    {success_3}")
    if improvements_3:
        print(f"\n  Improvements 3:\n    {improvements_3}")
    if value_adj_needed is not None:
        print(f"\n  Value Adjustment Needed: {'Yes' if value_adj_needed else 'No'}")
    if value_adj_proposal:
        print(f"\n  Value Adjustment Proposal:\n    {value_adj_proposal}")
    if structural_lessons:
        print(f"\n  Structural Lessons:\n    {structural_lessons}")
    if llm_summary:
        print(f"\n  LLM Summary:\n    {llm_summary[:300]}")
    if voice_transcript:
        print(f"\n  Voice Transcript: ({len(voice_transcript)} chars)")
    print(f"{'='*60}\n")


# ── Display existing Planning content for reference ──────────────

def _show_planning_context(page: Optional[Dict[str, Any]]) -> None:
    """Display the existing Monthly Planning for context during review."""
    if not page:
        return

    from src.notion.period_upsert import extract_rich_text, extract_title_text

    props = page.get("properties", {})
    title = extract_title_text(props.get("Title", {}))
    big_3 = extract_rich_text(props.get("Big 3", {}))
    strategic_rationale = extract_rich_text(props.get("Strategic Rationale", {}))

    if big_3 or strategic_rationale:
        print(f"\n{'─'*60}")
        print(f"  Reference — Monthly Planning: {title}")
        print(f"{'─'*60}")
        if big_3:
            print(f"\n  Big 3:\n    {big_3}")
        if strategic_rationale:
            print(f"\n  Strategic Rationale:\n    {strategic_rationale}")
        print(f"{'─'*60}\n")


# ── Input collection ─────────────────────────────────────────────

def _read_interactive(prompt: str) -> str:
    """Read multi-line text from stdin until EOF or empty line."""
    print(f"\n{prompt} (press Ctrl-D or enter empty line to finish):\n")
    lines = []
    try:
        while True:
            line = input()
            if line.strip() == "" and lines:
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()


def _collect_voice_input(
    *,
    record_wizard: bool = False,
    record_seconds: int = 120,
    language: str = "ja",
) -> str:
    """Collect input via browser-based voice recording."""
    import tempfile
    from src.daily.audio import (
        check_browser_dependencies,
        launch_wizard_recorder,
        launch_browser_recorder,
        get_recorder_transcript,
    )
    check_browser_dependencies()

    if record_wizard:
        section_defs = [
            {"key": "success_3", "title_ja": "Success 3 (今月の3大成果)", "title_en": "Top 3 successes this month"},
            {"key": "improvements_3", "title_ja": "今月うまくいかなかったこと3つ", "title_en": "Top 3 things that didn't go well this month"},
            {"key": "value_adjustment", "title_ja": "来月チャレンジしたいこと3つ", "title_en": "Top 3 challenges for next month"},
            {"key": "structural_lessons", "title_ja": "自分の行動・意思決定のパターンからの学び3つ", "title_en": "Top 3 learnings from behavioral/decision patterns"},
        ]
        out_dir = tempfile.mkdtemp(prefix="065_wizard_")
        section_results = launch_wizard_recorder(
            out_dir=out_dir,
            sections=section_defs,
            seconds=record_seconds,
            language=language,
            port=8065,
        )
        # launch_wizard_recorder returns List[Dict] with transcripts
        # already populated — no need to re-transcribe.
        parts = []
        for sec in section_results:
            transcript = sec.get("transcript", "")
            if transcript and not sec.get("skipped"):
                title = sec.get("title_ja", sec.get("key", ""))
                parts.append(f"## {title}\n{transcript}")
        return "\n\n".join(parts)
    else:
        out_path = tempfile.mktemp(prefix="065_audio_", suffix=".webm")
        launch_browser_recorder(
            output_path=out_path,
            seconds=record_seconds,
            language=language,
            port=8065,
        )
        # launch_browser_recorder transcribes internally;
        # retrieve via get_recorder_transcript().
        return get_recorder_transcript()


def _summarize_with_llm(transcript: str, date_iso: str, period_name: str) -> Dict[str, Any]:
    """Use LLM to extract structured monthly review fields from transcript."""
    try:
        from src.llm.router import build_router_from_env, TASK_REASONING

        router = build_router_from_env()
        system = (
            "You are a monthly review assistant. Extract structured review "
            "fields from the user's voice transcript. Return valid JSON with:\n"
            '{"success_3": "1. ... 2. ... 3. ...", '
            '"improvements_3": "1. ... 2. ... 3. ...", '
            '"value_adjustment_needed": true, '
            '"value_adjustment_proposal": "...", '
            '"structural_lessons": "..."}\n'
            "Keep the original language. Be concise but complete."
        )
        user = (
            f"Date: {date_iso}\nPeriod: {period_name}\n\n"
            f"Voice transcript:\n{transcript}"
        )
        result = router.call(
            task_type=TASK_REASONING,
            system=system,
            user=user,
            model_override="gpt-4o",
            temperature_override=0.3,
        )
        parsed = result.parsed
        if isinstance(parsed, dict):
            return {
                "success_3": str(parsed.get("success_3", "")),
                "improvements_3": str(parsed.get("improvements_3", "")),
                "value_adjustment_needed": bool(parsed.get("value_adjustment_needed", False)),
                "value_adjustment_proposal": str(parsed.get("value_adjustment_proposal", "")),
                "structural_lessons": str(parsed.get("structural_lessons", "")),
                "llm_summary": json.dumps(parsed, ensure_ascii=False),
            }
    except Exception as e:
        logger.warning("LLM summarization failed: %s", e)

    return {}


# ── Main pipeline ─────────────────────────────────────────────────

def run_pipeline(
    *,
    date_override: Optional[str] = None,
    text: Optional[str] = None,
    record: bool = False,
    record_wizard: bool = False,
    record_seconds: int = 120,
    language: str = "ja",
    no_llm: bool = False,
    non_interactive: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the monthly review pipeline."""
    load_env()
    setup_logging(logging.DEBUG if verbose else logging.INFO)

    now_jst = datetime.now(JST)
    date_iso = date_override or now_jst.strftime("%Y-%m-%d")
    wk = get_iso_week_context(tz=JST)

    logger.info("065 Monthly Review: date=%s", date_iso)

    # ── 1. Resolve period ──
    from src.notion.period_upsert import (
        resolve_period,
        query_existing_log,
        upsert_monthly_log,
        safe_truncate,
        link_monthly_period_to_weeks,
        backfill_monthly_log_to_weekly_periods,
    )
    from src.notion.period_schema import build_monthly_log_properties

    period = resolve_period(
        date_iso=date_iso,
        period_type="Monthly",
        log_label="065",
    )
    if not period.get("ok"):
        logger.error("Period resolution failed: %s", period.get("error"))
        return {"ok": False, "error": f"Period resolution failed: {period.get('error')}"}

    period_id = period["page_id"]
    period_name = period["name"]
    logger.info(
        "Resolved period: %s (page_id=%s, %s)",
        period_name, period_id[:8], period.get("action"),
    )

    # ── 1b. Cross-link monthly period ↔ overlapping weeks ──
    cross_link = link_monthly_period_to_weeks(
        monthly_period_id=period_id,
        month_start=period["start_date"],
        month_end=period["end_date"],
        log_label="065_cross_link",
    )
    logger.info(
        "Month↔Week cross-link: ok=%s weeks_found=%s weekly_logs_added=%s",
        cross_link.get("ok"),
        cross_link.get("weeks_found"),
        cross_link.get("weekly_logs_added"),
    )

    # ── 2. Show Planning context ──
    planning = query_existing_log(
        db_env_name="NOTION_MONTHLY_LOG_ID",
        resolver_name="monthly_log",
        period_id=period_id,
        log_type="Planning",
        log_label="065_planning_ref",
    )
    _show_planning_context(planning)

    # ── 3. Query existing review ──
    existing = query_existing_log(
        db_env_name="NOTION_MONTHLY_LOG_ID",
        resolver_name="monthly_log",
        period_id=period_id,
        log_type="Review",
        log_label="065_query",
    )
    _show_existing_content(existing)

    # ── 4. Collect input ──
    raw_text = ""
    if text:
        raw_text = text
        input_mode = "text"
    elif record_wizard:
        raw_text = _collect_voice_input(
            record_wizard=True,
            record_seconds=record_seconds,
            language=language,
        )
        input_mode = "voice_wizard"
    elif record:
        raw_text = _collect_voice_input(
            record_wizard=False,
            record_seconds=record_seconds,
            language=language,
        )
        input_mode = "voice"
    elif not non_interactive:
        raw_text = _read_interactive(
            "Enter your monthly review notes "
            "(Success 3, Improvements 3, value adjustments, structural lessons)"
        )
        input_mode = "interactive"
    else:
        logger.info("Non-interactive mode with no input — skipping")
        input_mode = "none"

    if not raw_text and input_mode != "none":
        logger.warning("No input collected — aborting")
        return {"ok": False, "error": "No input collected"}

    logger.info("Input collected: %d chars (mode=%s)", len(raw_text), input_mode)

    # ── 5. LLM summarization ──
    llm_fields: Dict[str, Any] = {}
    if raw_text and not no_llm:
        llm_fields = _summarize_with_llm(raw_text, date_iso, period_name)
        logger.info("LLM fields: %s", list(llm_fields.keys()))

    # ── 6. Build properties ──
    success_3 = llm_fields.get("success_3", raw_text) if llm_fields else raw_text
    improvements_3 = llm_fields.get("improvements_3", "")
    value_adj_needed = llm_fields.get("value_adjustment_needed")
    value_adj_proposal = llm_fields.get("value_adjustment_proposal", "")
    structural_lessons = llm_fields.get("structural_lessons", "")
    llm_summary = llm_fields.get("llm_summary", "")

    title = f"Monthly Review {period_name}"

    props = build_monthly_log_properties(
        title=title,
        period_id=period_id,
        log_type="Review",
        success_3=safe_truncate(success_3),
        improvements_3=safe_truncate(improvements_3),
        value_adjustment_needed=value_adj_needed,
        value_adjustment_proposal=safe_truncate(value_adj_proposal),
        structural_lessons=safe_truncate(structural_lessons),
        voice_transcript=safe_truncate(raw_text),
        llm_summary=safe_truncate(llm_summary),
    )

    # ── 7. Upsert to Notion ──
    notion_result = upsert_monthly_log(
        period_id=period_id,
        log_type="Review",
        properties=props,
        log_label="065_monthly_review",
    )

    # ── 7b. Backfill Monthly Log → overlapping Weekly periods ──
    if notion_result.get("ok") and notion_result.get("page_id"):
        ml_backfill = backfill_monthly_log_to_weekly_periods(
            monthly_log_page_id=notion_result["page_id"],
            month_start=period["start_date"],
            month_end=period["end_date"],
            log_label="065_ml_backfill",
        )
        logger.info(
            "Monthly Log → Weekly periods backfill: ok=%s "
            "weeks_found=%s weeks_updated=%s",
            ml_backfill.get("ok"),
            ml_backfill.get("weeks_found"),
            ml_backfill.get("weeks_updated"),
        )

    # ── 8. Run metadata ──
    meta = RunMetadata.build(
        notebook=SCRIPT_NAME,
        week_id=wk.week_id,
        counts={
            "input_chars": len(raw_text),
            "input_mode": input_mode,
            "period_name": period_name,
            "llm_used": bool(llm_fields),
        },
    )

    result = {
        "ok": True,
        "date": date_iso,
        "period": period_name,
        "input_mode": input_mode,
        "input_chars": len(raw_text),
        "llm_used": bool(llm_fields),
        "notion": notion_result,
    }

    print(f"\nMonthly Review saved for {period_name}")
    print(f"  Input: {input_mode} | {len(raw_text)} chars")
    if notion_result.get("ok"):
        print(f"  Notion: {notion_result['action']} -> {notion_result.get('page_url', '')}")
    else:
        print(f"  Notion: FAILED -> {notion_result.get('error', 'unknown')}")

    return result


# ── Web UI configuration ──────────────────────────────────────────

def _monthly_review_pre_hook(period, notion_result):
    """Cross-link monthly period to overlapping weekly periods."""
    from src.notion.period_upsert import link_monthly_period_to_weeks
    link_monthly_period_to_weeks(
        monthly_period_id=period["page_id"],
        month_start=period["start_date"],
        month_end=period["end_date"],
        log_label="065_cross_link",
    )


def _monthly_review_post_hook(period, notion_result):
    """Backfill monthly log to overlapping weekly periods."""
    from src.notion.period_upsert import backfill_monthly_log_to_weekly_periods
    if notion_result.get("ok") and notion_result.get("page_id"):
        backfill_monthly_log_to_weekly_periods(
            monthly_log_page_id=notion_result["page_id"],
            month_start=period["start_date"],
            month_end=period["end_date"],
            log_label="065_ml_backfill",
        )


def _build_web_ui_config():
    """Build PeriodWebUIConfig for 065 Monthly Review."""
    from src.app.period_web_ui import PeriodWebUIConfig, SectionDef, FieldMapping
    from src.notion.period_schema import build_monthly_log_properties
    from src.notion.period_upsert import upsert_monthly_log

    return PeriodWebUIConfig(
        script_id="065",
        script_name=SCRIPT_NAME,
        title="Monthly Review",
        port=8065,
        period_type="Monthly",
        log_type="Review",
        db_env_name="NOTION_MONTHLY_LOG_ID",
        resolver_name="monthly_log",
        log_label="065_monthly_review",
        sections=[
            SectionDef(key="success_3", title_ja="Success 3 (今月の3大成果)", title_en="Top 3 successes this month"),
            SectionDef(key="improvements_3", title_ja="今月うまくいかなかったこと3つ", title_en="Top 3 things that didn't go well this month"),
            SectionDef(key="value_adjustment", title_ja="来月チャレンジしたいこと3つ", title_en="Top 3 challenges for next month"),
            SectionDef(key="structural_lessons", title_ja="自分の行動・意思決定のパターンからの学び3つ", title_en="Top 3 learnings from behavioral/decision patterns"),
        ],
        field_mappings=[
            FieldMapping(section_key="success_3", notion_property="Success 3", llm_json_key="success_3", property_kwarg="success_3", is_primary=True),
            FieldMapping(section_key="improvements_3", notion_property="Improvements 3", llm_json_key="improvements_3", property_kwarg="improvements_3"),
            FieldMapping(section_key="value_adjustment", notion_property="Value Adjustment Proposal", llm_json_key="value_adjustment_proposal", property_kwarg="value_adjustment_proposal"),
            FieldMapping(section_key="structural_lessons", notion_property="Structural Lessons", llm_json_key="structural_lessons", property_kwarg="structural_lessons"),
        ],
        score_fields=[
            SectionDef(key="value_adjustment_needed", title_ja="Value Adjustment Needed", title_en="Value adjustment needed", input_type="checkbox"),
        ],
        has_planning_context=True,
        planning_context_fields=["Big 3", "Strategic Rationale"],
        llm_system_role="You are a monthly review assistant.",
        llm_json_schema='{"success_3": "1. ... 2. ... 3. ...", "improvements_3": "今月うまくいかなかったこと 1. ... 2. ... 3. ...", "value_adjustment_needed": true, "value_adjustment_proposal": "来月チャレンジしたいこと 1. ... 2. ... 3. ...", "structural_lessons": "行動・意思決定パターンからの学び 1. ... 2. ... 3. ..."}',
        build_properties_fn=build_monthly_log_properties,
        upsert_fn=upsert_monthly_log,
        pre_upsert_hooks=[_monthly_review_pre_hook],
        post_upsert_hooks=[_monthly_review_post_hook],
        title_template="Monthly Review {period_name}",
    )


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="065 Monthly Review — month-end summary",
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Override date (YYYY-MM-DD)")
    parser.add_argument("--text", type=str, default=None,
                        help="Review text directly")

    audio_group = parser.add_argument_group("voice input")
    audio_group.add_argument(
        "--record", action="store_true", default=False,
        help="Launch browser-based recorder (single clip)",
    )
    audio_group.add_argument(
        "--record-wizard", action="store_true", default=False, dest="record_wizard",
        help="Launch guided section-by-section browser recorder",
    )
    audio_group.add_argument(
        "--record-seconds", type=int, default=120, dest="record_seconds",
        help="Maximum recording duration per section (default: 120)",
    )
    audio_group.add_argument(
        "--language", type=str, default="ja",
        help="Transcription language (default: ja)",
    )

    parser.add_argument("--no-llm", action="store_true", dest="no_llm",
                        help="Skip LLM summarization")
    parser.add_argument("--non-interactive", action="store_true", dest="non_interactive",
                        help="Skip interactive prompts")
    parser.add_argument("--port", type=int, default=8065,
                        help="Web UI server port (default: 8065)")
    parser.add_argument("--no-browser", action="store_true", dest="no_browser",
                        help="Don't auto-open browser")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")
    args = parser.parse_args()

    # CLI pipeline mode: when explicit input flags are provided
    if args.text or args.record or args.record_wizard or args.non_interactive:
        if args.record or args.record_wizard:
            try:
                from src.daily.audio import check_browser_dependencies
                check_browser_dependencies()
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)

        result = run_pipeline(
            date_override=args.date,
            text=args.text,
            record=args.record,
            record_wizard=args.record_wizard,
            record_seconds=args.record_seconds,
            language=args.language,
            no_llm=args.no_llm,
            non_interactive=args.non_interactive,
            verbose=args.verbose,
        )
        print(json.dumps(result, indent=2))
    else:
        # Default: launch Web UI
        from src.app.period_web_ui import run_period_server
        config = _build_web_ui_config()
        run_period_server(
            config,
            port=args.port,
            verbose=args.verbose,
            no_browser=args.no_browser,
        )


if __name__ == "__main__":
    main()
