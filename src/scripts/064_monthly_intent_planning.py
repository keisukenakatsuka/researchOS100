#!/usr/bin/env python
# src/scripts/064_monthly_intent_planning.py
"""Monthly Intent Planning — set this month's Big 3 via voice/text input.

Captures:
  - Big 3 for the month
  - Value links (which core values these connect to)
  - Breakdown to weeks (link existing Weekly Planning rows if possible)
  - Strategic rationale
  - Risks
  - Confidence level (1-10)

Mirrors the 061 UX: show existing Notion content, accept voice/text input,
write structured fields back to Notion MONTHLY_LOG with Log Type = Planning.

Usage::

    # Interactive (text prompt)
    python -m src.scripts.064_monthly_intent_planning

    # With voice input (wizard)
    python -m src.scripts.064_monthly_intent_planning --record-wizard

    # Specific date
    python -m src.scripts.064_monthly_intent_planning --date 2026-02-27

    # Debug logging
    python -m src.scripts.064_monthly_intent_planning -v
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

logger = logging.getLogger("064_monthly_intent_planning")

SCRIPT_NAME = "064_monthly_intent_planning"
JST = ZoneInfo("Asia/Tokyo")


# ── Display existing Notion content ──────────────────────────────

def _show_existing_content(page: Optional[Dict[str, Any]]) -> None:
    """Display existing MONTHLY_LOG Planning content from Notion."""
    if not page:
        print("\n  (No existing Monthly Planning found for this period.)\n")
        return

    from src.notion.period_upsert import extract_rich_text, extract_title_text

    props = page.get("properties", {})
    title = extract_title_text(props.get("Title", {}))
    big_3 = extract_rich_text(props.get("Big 3", {}))
    strategic_rationale = extract_rich_text(props.get("Strategic Rationale", {}))
    risks = extract_rich_text(props.get("Risks", {}))
    confidence = props.get("Confidence", {}).get("number")
    voice_transcript = extract_rich_text(props.get("Voice Transcript", {}))
    llm_summary = extract_rich_text(props.get("LLM Summary", {}))

    print(f"\n{'='*60}")
    print(f"  Existing Monthly Planning: {title}")
    print(f"{'='*60}")
    if big_3:
        print(f"\n  Big 3:\n    {big_3}")
    if strategic_rationale:
        print(f"\n  Strategic Rationale:\n    {strategic_rationale}")
    if risks:
        print(f"\n  Risks:\n    {risks}")
    if confidence is not None:
        print(f"\n  Confidence: {confidence}/10")
    if llm_summary:
        print(f"\n  LLM Summary:\n    {llm_summary[:300]}")
    if voice_transcript:
        print(f"\n  Voice Transcript: ({len(voice_transcript)} chars)")
    print(f"{'='*60}\n")


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
            {"key": "big_3", "title_ja": "Big 3 (今月の3大目標)", "title_en": "Big 3 goals for this month"},
            {"key": "strategic_rationale", "title_ja": "戦略的根拠", "title_en": "Strategic rationale"},
            {"key": "risks", "title_ja": "リスク", "title_en": "Risks and mitigations"},
            {"key": "weekly_breakdown", "title_ja": "週次ブレイクダウン", "title_en": "Breakdown to weeks"},
        ]
        out_dir = tempfile.mkdtemp(prefix="064_wizard_")
        section_results = launch_wizard_recorder(
            out_dir=out_dir,
            sections=section_defs,
            seconds=record_seconds,
            language=language,
            port=8064,
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
        out_path = tempfile.mktemp(prefix="064_audio_", suffix=".webm")
        launch_browser_recorder(
            output_path=out_path,
            seconds=record_seconds,
            language=language,
            port=8064,
        )
        # launch_browser_recorder transcribes internally;
        # retrieve via get_recorder_transcript().
        return get_recorder_transcript()


def _summarize_with_llm(transcript: str, date_iso: str, period_name: str) -> Dict[str, Any]:
    """Use LLM to extract structured monthly planning fields from transcript."""
    try:
        from src.llm.router import build_router_from_env, TASK_REASONING

        router = build_router_from_env()
        system = (
            "You are a monthly planning assistant. Extract structured planning "
            "fields from the user's voice transcript. Return valid JSON with:\n"
            '{"big_3": "1. ... 2. ... 3. ...", '
            '"strategic_rationale": "...", '
            '"risks": "...", '
            '"confidence": 7}\n'
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
                "big_3": str(parsed.get("big_3", "")),
                "strategic_rationale": str(parsed.get("strategic_rationale", "")),
                "risks": str(parsed.get("risks", "")),
                "confidence": parsed.get("confidence"),
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
    """Run the monthly intent planning pipeline."""
    load_env()
    setup_logging(logging.DEBUG if verbose else logging.INFO)

    now_jst = datetime.now(JST)
    date_iso = date_override or now_jst.strftime("%Y-%m-%d")
    wk = get_iso_week_context(tz=JST)

    logger.info("064 Monthly Intent Planning: date=%s", date_iso)

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
        log_label="064",
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
        log_label="064_cross_link",
    )
    logger.info(
        "Month↔Week cross-link: ok=%s weeks_found=%s weekly_logs_added=%s",
        cross_link.get("ok"),
        cross_link.get("weeks_found"),
        cross_link.get("weekly_logs_added"),
    )

    # ── 2. Query existing log ──
    existing = query_existing_log(
        db_env_name="NOTION_MONTHLY_LOG_ID",
        resolver_name="monthly_log",
        period_id=period_id,
        log_type="Planning",
        log_label="064_query",
    )
    _show_existing_content(existing)

    # ── 3. Collect input ──
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
            "Enter your monthly planning notes "
            "(Big 3, strategic rationale, risks, weekly breakdown)"
        )
        input_mode = "interactive"
    else:
        logger.info("Non-interactive mode with no input — skipping")
        input_mode = "none"

    if not raw_text and input_mode != "none":
        logger.warning("No input collected — aborting")
        return {"ok": False, "error": "No input collected"}

    logger.info("Input collected: %d chars (mode=%s)", len(raw_text), input_mode)

    # ── 4. LLM summarization ──
    llm_fields: Dict[str, Any] = {}
    if raw_text and not no_llm:
        llm_fields = _summarize_with_llm(raw_text, date_iso, period_name)
        logger.info("LLM fields: %s", list(llm_fields.keys()))

    # ── 5. Build properties ──
    big_3 = llm_fields.get("big_3", raw_text) if llm_fields else raw_text
    strategic_rationale = llm_fields.get("strategic_rationale", "")
    risks = llm_fields.get("risks", "")
    confidence = llm_fields.get("confidence")
    llm_summary = llm_fields.get("llm_summary", "")

    title = f"Monthly Planning {period_name}"

    props = build_monthly_log_properties(
        title=title,
        period_id=period_id,
        log_type="Planning",
        big_3=safe_truncate(big_3),
        strategic_rationale=safe_truncate(strategic_rationale),
        risks=safe_truncate(risks),
        confidence=int(confidence) if confidence is not None else None,
        voice_transcript=safe_truncate(raw_text),
        llm_summary=safe_truncate(llm_summary),
    )

    # ── 6. Upsert to Notion ──
    notion_result = upsert_monthly_log(
        period_id=period_id,
        log_type="Planning",
        properties=props,
        log_label="064_monthly_planning",
    )

    # ── 6b. Backfill Monthly Log → overlapping Weekly periods ──
    if notion_result.get("ok") and notion_result.get("page_id"):
        ml_backfill = backfill_monthly_log_to_weekly_periods(
            monthly_log_page_id=notion_result["page_id"],
            month_start=period["start_date"],
            month_end=period["end_date"],
            log_label="064_ml_backfill",
        )
        logger.info(
            "Monthly Log → Weekly periods backfill: ok=%s "
            "weeks_found=%s weeks_updated=%s",
            ml_backfill.get("ok"),
            ml_backfill.get("weeks_found"),
            ml_backfill.get("weeks_updated"),
        )

    # ── 7. Run metadata ──
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

    print(f"\nMonthly Planning saved for {period_name}")
    print(f"  Input: {input_mode} | {len(raw_text)} chars")
    if notion_result.get("ok"):
        print(f"  Notion: {notion_result['action']} -> {notion_result.get('page_url', '')}")
    else:
        print(f"  Notion: FAILED -> {notion_result.get('error', 'unknown')}")

    return result


# ── Web UI configuration ──────────────────────────────────────────

def _monthly_planning_pre_hook(period, notion_result):
    """Cross-link monthly period to overlapping weekly periods."""
    from src.notion.period_upsert import link_monthly_period_to_weeks
    link_monthly_period_to_weeks(
        monthly_period_id=period["page_id"],
        month_start=period["start_date"],
        month_end=period["end_date"],
        log_label="064_cross_link",
    )


def _monthly_planning_post_hook(period, notion_result):
    """Backfill monthly log to overlapping weekly periods."""
    from src.notion.period_upsert import backfill_monthly_log_to_weekly_periods
    if notion_result.get("ok") and notion_result.get("page_id"):
        backfill_monthly_log_to_weekly_periods(
            monthly_log_page_id=notion_result["page_id"],
            month_start=period["start_date"],
            month_end=period["end_date"],
            log_label="064_ml_backfill",
        )


def _build_web_ui_config():
    """Build PeriodWebUIConfig for 064 Monthly Planning."""
    from src.app.period_web_ui import PeriodWebUIConfig, SectionDef, FieldMapping
    from src.notion.period_schema import build_monthly_log_properties
    from src.notion.period_upsert import upsert_monthly_log

    return PeriodWebUIConfig(
        script_id="064",
        script_name=SCRIPT_NAME,
        title="Monthly Planning",
        port=8064,
        period_type="Monthly",
        log_type="Planning",
        db_env_name="NOTION_MONTHLY_LOG_ID",
        resolver_name="monthly_log",
        log_label="064_monthly_planning",
        sections=[
            SectionDef(key="big_3", title_ja="Big 3 (今月の3大目標)", title_en="Big 3 goals for this month"),
            SectionDef(key="strategic_rationale", title_ja="戦略的根拠", title_en="Strategic rationale"),
            SectionDef(key="risks", title_ja="リスク", title_en="Risks and mitigations"),
            SectionDef(key="weekly_breakdown", title_ja="週次ブレイクダウン", title_en="Breakdown to weeks"),
        ],
        field_mappings=[
            FieldMapping(section_key="big_3", notion_property="Big 3", llm_json_key="big_3", property_kwarg="big_3", is_primary=True),
            FieldMapping(section_key="strategic_rationale", notion_property="Strategic Rationale", llm_json_key="strategic_rationale", property_kwarg="strategic_rationale"),
            FieldMapping(section_key="risks", notion_property="Risks", llm_json_key="risks", property_kwarg="risks"),
        ],
        score_fields=[
            SectionDef(key="confidence", title_ja="Confidence (1-10)", title_en="Confidence level", input_type="number", min_value=1, max_value=10),
        ],
        llm_system_role="You are a monthly planning assistant.",
        llm_json_schema='{"big_3": "1. ... 2. ... 3. ...", "strategic_rationale": "...", "risks": "...", "confidence": 7}',
        build_properties_fn=build_monthly_log_properties,
        upsert_fn=upsert_monthly_log,
        pre_upsert_hooks=[_monthly_planning_pre_hook],
        post_upsert_hooks=[_monthly_planning_post_hook],
        title_template="Monthly Planning {period_name}",
    )


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="064 Monthly Intent Planning — set this month's Big 3",
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Override date (YYYY-MM-DD)")
    parser.add_argument("--text", type=str, default=None,
                        help="Planning text directly")

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
    parser.add_argument("--port", type=int, default=8064,
                        help="Web UI server port (default: 8064)")
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
