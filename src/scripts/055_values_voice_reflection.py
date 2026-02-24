#!/usr/bin/env python
# src/scripts/055_values_voice_reflection.py
"""Values Voice Reflection — interactive voice-based value alignment session.

Pipeline:
1. Load value domains from 054 output (or --values-json path)
2. For each selected domain:
   a. Display domain info and reflection question
   b. (Optional) Speak question via TTS  (--tts)
   c. Record user voice response via microphone
   d. Transcribe via OpenAI Whisper (through LLMRouter)
   e. Prompt importance_score (1–5) and alignment_score (1–5)
      — voice-first, keyboard fallback
3. Generate AI summary of the session (through LLMRouter)
4. Write JSON outputs to outputs/weekly/{week_id}/055_values_voice_reflection/
5. (Optional) Write entries to ROS_Alignment_Log in Notion  (--write)

Strict separation:
- 054 = batch generation / refinement (writes to ROS_Values_Codex)
- 055 = interactive voice reflection  (writes to ROS_Alignment_Log)

All OpenAI calls go through src/llm/router.py.

--dry-run mode:
- Does NOT record from microphone
- Does NOT call Whisper or TTS
- Does NOT write to Notion
- Uses deterministic dummy transcripts and scores

Usage::

    # Interactive reflection for a single domain
    python -m src.scripts.055_values_voice_reflection --run --domain family --lang ja

    # All domains with Notion write-back
    python -m src.scripts.055_values_voice_reflection --run --lang ja --write

    # Dry-run (no mic, no API calls, no Notion)
    python -m src.scripts.055_values_voice_reflection --run --dry-run

    # Load domains from a specific file
    python -m src.scripts.055_values_voice_reflection --run --values-json path/to/values.json

    # With TTS (speak questions aloud)
    python -m src.scripts.055_values_voice_reflection --run --tts --lang ja
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    RunMetadata,
    get_db_id,
    get_iso_week_context,
    get_output_dir,
    load_env,
    setup_logging,
)
from src.values.schema import (
    DOMAIN_IDS,
    AlignmentEntry,
    ValueDomain,
    ValueRecord,
    value_record_from_dict,
)
from src.values.voice import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    ReflectionResponse,
    VoiceConfig,
    summarize_reflection_session,
)

logger = logging.getLogger("055_values_voice_reflection")

SCRIPT_NAME = "055_values_voice_reflection"
JST = ZoneInfo("Asia/Tokyo")

# Dummy transcript for --dry-run mode.
_DRY_RUN_TRANSCRIPT = (
    "This is a dry-run dummy transcript. "
    "In a real session, voice input would be recorded and transcribed here."
)


# ================================================================
# Load 054 output
# ================================================================

def _find_latest_054_output() -> Optional[Path]:
    """Scan outputs/weekly/ for the latest 054_values_scale_setup/values.json."""
    base = Path("outputs/weekly")
    if not base.exists():
        return None
    week_dirs = sorted(base.iterdir(), reverse=True)
    for week_dir in week_dirs:
        candidate = week_dir / "054_values_scale_setup" / "values.json"
        if candidate.exists():
            return candidate
    return None


def _load_value_record(values_json_path: Optional[str]) -> ValueRecord:
    """Load ValueRecord from explicit path or latest 054 output.

    Parameters
    ----------
    values_json_path : str or None
        Explicit path to values.json. If None, scans for latest 054 output.

    Returns
    -------
    ValueRecord

    Raises
    ------
    FileNotFoundError
        If no values.json can be found.
    """
    if values_json_path:
        path = Path(values_json_path)
    else:
        path = _find_latest_054_output()

    if path is None or not path.exists():
        raise FileNotFoundError(
            "Cannot find 054 values.json. "
            "Run 054_values_scale_setup first, or use --values-json <path>."
        )
    logger.info("Loading value domains from: %s", path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return value_record_from_dict(data)


# ================================================================
# Main pipeline
# ================================================================

def run_pipeline(
    *,
    domain_filter: str = "all",
    values_json_path: Optional[str] = None,
    write_notion: bool = False,
    dry_run: bool = False,
    enable_tts: bool = False,
    review_type: str = "Daily",
    verbose: bool = False,
    language: Optional[str] = None,
) -> dict:
    """Execute the interactive voice reflection pipeline.

    Parameters
    ----------
    domain_filter : str
        Single domain_id or "all" (default).
    values_json_path : str or None
        Path to values.json. Falls back to latest 054 output.
    write_notion : bool
        Write entries to ROS_Alignment_Log.
    dry_run : bool
        No mic, no Whisper, no TTS, no Notion. Deterministic dummies.
    enable_tts : bool
        Speak questions aloud via TTS.
    review_type : str
        "Daily", "Weekly", or "Quarterly".
    verbose : bool
        Debug logging.
    language : str or None
        Voice language (ja/en). Default: ja.

    Returns
    -------
    dict
        Summary with counts and output paths.
    """
    load_env()
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    now_jst = datetime.now(tz=JST)
    wk = get_iso_week_context(tz=JST)
    date_iso = now_jst.date().isoformat()

    # ---- Resolve voice language ----
    if language is None:
        language = os.getenv("VOICE_LANGUAGE", "").strip() or DEFAULT_LANGUAGE
    voice_config = VoiceConfig(language=language)

    logger.info(
        "Starting %s week=%s (domain=%s, write=%s, dry_run=%s, tts=%s, lang=%s)",
        SCRIPT_NAME, wk.week_id, domain_filter,
        write_notion, dry_run, enable_tts, voice_config.language,
    )

    # ---- 1. Load value domains ----
    record = _load_value_record(values_json_path)
    logger.info("Loaded %d value domains", len(record.domains))

    # ---- 2. Filter domains ----
    if domain_filter == "all":
        domains = list(record.domains)
    else:
        domain = record.get_domain(domain_filter)
        if domain is None:
            raise ValueError(
                f"Unknown domain: {domain_filter!r}. "
                f"Available: {[d.domain_id for d in record.domains]}"
            )
        domains = [domain]

    # ---- 3. Build output directory ----
    out_dir = get_output_dir(SCRIPT_NAME, wk.week_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    domains_dir = out_dir / "domains"
    domains_dir.mkdir(exist_ok=True)

    # ---- 4. Build LLM router ----
    router = None
    if not dry_run:
        from src.llm.router import build_router_from_env
        router = build_router_from_env()

    # ---- 5. Interactive reflection loop ----
    entries: list[AlignmentEntry] = []
    responses: list[ReflectionResponse] = []

    print(f"\n{'='*60}")
    print(f"  Values Voice Reflection — {voice_config.display_name}")
    print(f"  Date: {date_iso}  |  Review: {review_type}")
    print(f"  Domains: {len(domains)}")
    if dry_run:
        print("  Mode: DRY RUN (no mic, no API calls, no Notion)")
    print(f"{'='*60}\n")

    for i, domain in enumerate(domains, 1):
        print(f"\n--- [{i}/{len(domains)}] {domain.domain_label} ---")
        print(f"  Definition: {domain.value_definition}")

        # Pick reflection question.
        question_text = ""
        if domain.reflection_questions:
            question_text = domain.reflection_questions[0]
            print(f"\n  Reflection question:")
            print(f"  {question_text}")
        else:
            question_text = (
                f"How are you currently living the value of "
                f"{domain.domain_label.lower()} in your life?"
            )
            print(f"\n  (No codex questions — using default)")
            print(f"  {question_text}")

        # Optional TTS.
        if enable_tts and not dry_run and router is not None:
            tts_path = audio_dir / f"{domain.domain_id}_question.mp3"
            try:
                router.speak_text(
                    question_text, tts_path, voice_config=voice_config,
                )
                _play_audio(tts_path)
            except Exception as e:
                logger.warning("TTS failed: %s", e)

        # Record voice response.
        transcript = ""
        audio_path = audio_dir / f"{domain.domain_id}.wav"

        if dry_run:
            transcript = _DRY_RUN_TRANSCRIPT
            logger.info("[dry-run] Using dummy transcript for %s", domain.domain_id)
            print(f"\n  [dry-run transcript] {transcript}")
        else:
            from src.values.audio import record_audio
            print(f"\n  Please share your reflection on this domain.")
            record_audio(audio_path)
            transcript = router.transcribe_audio(
                audio_path, voice_config=voice_config,
            )
            print(f"\n  [transcript] {transcript}")

        # Prompt scores.
        if dry_run:
            importance = 3
            alignment = 3
            logger.info("[dry-run] Using dummy scores (3, 3)")
            print(f"  [dry-run] Importance Score: {importance}")
            print(f"  [dry-run] Alignment Score: {alignment}")
        else:
            from src.values.audio import prompt_score
            importance = prompt_score(
                "Importance Score",
                router=router,
                voice_config=voice_config,
                audio_dir=audio_dir,
                domain_id=domain.domain_id,
            )
            alignment = prompt_score(
                "Alignment Score",
                router=router,
                voice_config=voice_config,
                audio_dir=audio_dir,
                domain_id=domain.domain_id,
            )

        # Auto-generate per-domain AI summary.
        per_domain_summary = ""
        if transcript and not dry_run and router is not None:
            per_domain_summary = _generate_domain_summary(
                transcript, domain.domain_id, voice_config, router,
            )
            if per_domain_summary:
                print(f"  [AI summary] {per_domain_summary[:120]}...")

        # Auto-generate next adjustment.
        next_adj = ""
        if not dry_run and router is not None:
            next_adj = _generate_domain_adjustment(
                domain.domain_id, domain.domain_label,
                importance, alignment, transcript,
                voice_config, router,
            )
            if next_adj:
                print(f"  [Next adjustment] {next_adj}")
        elif dry_run:
            per_domain_summary = "(dry-run) ダミー要約です。"
            next_adj = "(dry-run) 今週30分のブロックを確保して行動計画を立てる。"

        # Build entry.
        entry = AlignmentEntry(
            date_iso=date_iso,
            review_type=review_type,
            domain_id=domain.domain_id,
            importance_score=importance,
            alignment_score=alignment,
            reflection_text="",  # Raw transcript is stored separately.
            transcript=transcript,
            audio_url=str(audio_path) if audio_path.exists() else "",
            ai_summary=per_domain_summary,
            next_adjustment=next_adj,
        )
        entries.append(entry)

        response = ReflectionResponse(
            domain_id=domain.domain_id,
            question_text=question_text,
            transcript=transcript,
            audio_url=str(audio_path) if audio_path.exists() else "",
            language=voice_config.language,
        )
        responses.append(response)

        # Show gap info.
        gap_label = f"  Gap: {entry.gap_score}"
        if entry.significant_gap:
            gap_label += " (SIGNIFICANT)"
        print(f"\n  Importance: {importance} | Alignment: {alignment} | {gap_label}")

        # Write per-domain JSON.
        domain_result = {
            "domain_id": domain.domain_id,
            "domain_label": domain.domain_label,
            "question_text": question_text,
            "transcript": transcript,
            "importance_score": importance,
            "alignment_score": alignment,
            "gap_score": entry.gap_score,
            "significant_gap": entry.significant_gap,
            "ai_summary": per_domain_summary,
            "next_adjustment": next_adj,
        }
        domain_path = domains_dir / f"{domain.domain_id}.json"
        domain_path.write_text(
            json.dumps(domain_result, indent=2, ensure_ascii=False)
        )

    # ---- 6. AI summary ----
    ai_summary = ""
    if responses and not dry_run:
        print("\nGenerating AI summary...")
        ai_summary = summarize_reflection_session(
            tuple(responses),
            voice_config=voice_config,
            router=router,
        )
        print(f"\n{ai_summary}")
    elif dry_run:
        ai_summary = "[AI-Generated Summary]\n[dry-run] No summary generated."

    # ---- 7. Write reflections.json ----
    reflections_data = {
        "date": date_iso,
        "review_type": review_type,
        "language": voice_config.language,
        "language_display": voice_config.display_name,
        "domains_reflected": len(entries),
        "ai_summary": ai_summary,
        "entries": [
            {
                "domain_id": e.domain_id,
                "importance_score": e.importance_score,
                "alignment_score": e.alignment_score,
                "gap_score": e.gap_score,
                "significant_gap": e.significant_gap,
                "transcript": e.transcript,
                "audio_url": e.audio_url,
                "ai_summary": e.ai_summary,
                "next_adjustment": e.next_adjustment,
            }
            for e in entries
        ],
    }
    reflections_path = out_dir / "reflections.json"
    reflections_path.write_text(
        json.dumps(reflections_data, indent=2, ensure_ascii=False)
    )
    logger.info("Wrote reflections.json to %s", reflections_path)

    # ---- 8. Write summary markdown ----
    _write_summary_markdown(out_dir, entries, ai_summary, date_iso, review_type, voice_config)

    # ---- 9. Notion write-back ----
    notion_rows = 0
    if write_notion and not dry_run:
        notion_rows = _write_to_notion(entries, record)
    elif write_notion and dry_run:
        logger.info("[dry-run] Notion writes skipped")
    else:
        logger.info("Notion write-back disabled (use --write to enable)")

    # ---- 10. Run metadata ----
    run_meta = RunMetadata.build(
        notebook=SCRIPT_NAME,
        week_id=wk.week_id,
        date_from=wk.date_from_iso,
        date_to=wk.date_to_iso,
        counts={
            "domains_reflected": len(entries),
            "notion_rows_written": notion_rows,
            "review_type": review_type,
            "dry_run": dry_run,
            "tts_enabled": enable_tts,
            "voice_language": voice_config.language,
        },
    )
    run_meta.save(out_dir / "run_metadata.json")

    logger.info("Pipeline complete. Output: %s", out_dir)
    if router is not None:
        logger.info("Router usage: %s", router.usage_summary)

    return {
        "domains_reflected": len(entries),
        "review_type": review_type,
        "notion_rows_written": notion_rows,
        "dry_run": dry_run,
        "voice_language": voice_config.language,
        "voice_language_display": voice_config.display_name,
        "output_dir": str(out_dir),
    }


# ================================================================
# Audio playback (best-effort)
# ================================================================

def _play_audio(path: Path) -> None:
    """Best-effort audio playback. Fails silently."""
    try:
        import subprocess
        if sys.platform == "darwin":
            subprocess.run(["afplay", str(path)], check=True)
        else:
            logger.debug("Audio playback not supported on this platform")
    except Exception as e:
        logger.debug("Audio playback failed: %s", e)


# ================================================================
# Per-domain AI summary + next adjustment generators
# ================================================================

def _generate_domain_summary(
    transcript: str, domain_id: str, voice_config, router,
) -> str:
    """Generate a concise per-domain AI summary (1-3 sentences) from transcript."""
    if not transcript or not transcript.strip():
        return ""

    if voice_config.language == "ja":
        system = (
            "あなたは内省コーチです。以下の振り返り音声の書き起こしを、"
            "1〜3文で簡潔に要約してください。\n"
            "・本人が語った核心的な気づきや感情を捉える\n"
            "・内容や意味を変えない\n"
            "・ラベルや前置きは不要\n"
            '・JSON形式 {"summary": "要約テキスト"} で返す'
        )
    else:
        system = (
            "You are a reflection coach. Summarize the following voice reflection "
            "transcript in 1-3 concise sentences.\n"
            "- Capture the key insight or feeling expressed\n"
            "- Do NOT change the meaning\n"
            "- No labels or prefixes\n"
            '- Return as JSON: {"summary": "summary text here"}'
        )

    try:
        result = router.call_voice_processing(
            system=system,
            user=transcript,
            voice_config=voice_config,
        )
        summary = result.parsed.get("summary", "").strip()
        logger.info("AI summary generated for %s: %d chars", domain_id, len(summary))
        return summary
    except Exception as e:
        logger.warning("AI summary generation failed for %s: %s", domain_id, e)
        return ""


def _generate_domain_adjustment(
    domain_id: str, domain_label: str,
    importance: int, alignment: int, transcript: str,
    voice_config, router,
) -> str:
    """Generate a single concrete next-adjustment sentence."""
    gap = importance - alignment

    if voice_config.language == "ja":
        system = (
            "あなたは行動変容コーチです。\n"
            "以下の価値観ドメインについて、重要度と一致度のギャップに基づいて、"
            "今週できる具体的な行動提案を1文で生成してください。\n"
            "・抽象的なアドバイスではなく、具体的な行動を提案する\n"
            "・「〜する」の形で終わる\n"
            '・JSON形式 {"adjustment": "行動提案テキスト"} で返す'
        )
        user_prompt = (
            f"ドメイン: {domain_label}\n"
            f"重要度: {importance}/5\n"
            f"一致度: {alignment}/5\n"
            f"ギャップ: {gap}\n"
        )
    else:
        system = (
            "You are a behavioral change coach.\n"
            "Based on the gap between importance and alignment for this value domain, "
            "generate ONE concrete, actionable adjustment for this week.\n"
            "- Focus on a specific behavior, not abstract advice\n"
            "- Keep it to one sentence\n"
            '- Return as JSON: {"adjustment": "adjustment text here"}'
        )
        user_prompt = (
            f"Domain: {domain_label}\n"
            f"Importance: {importance}/5\n"
            f"Alignment: {alignment}/5\n"
            f"Gap: {gap}\n"
        )

    if transcript:
        user_prompt += f"\nReflection transcript:\n{transcript[:500]}"

    try:
        result = router.call_voice_processing(
            system=system,
            user=user_prompt,
            voice_config=voice_config,
        )
        adjustment = result.parsed.get("adjustment", "").strip()
        logger.info("Next adjustment generated for %s: %s", domain_id, adjustment[:80])
        return adjustment
    except Exception as e:
        logger.warning("Next adjustment generation failed for %s: %s", domain_id, e)
        return ""


# ================================================================
# Notion write-back
# ================================================================

def _write_to_notion(entries: list[AlignmentEntry], record: ValueRecord) -> int:
    """Write alignment entries to ROS_Alignment_Log. Returns row count."""
    from src.notion.alignment_repo import AlignmentLogRepo
    from src.notion.client import (
        NotionDataSourceResolver,
        build_notion_client_from_env,
    )
    from src.notion.truncation import TruncationTracker
    from src.notion.values_repo import ValuesCodexRepo

    client = build_notion_client_from_env()

    # Resolve Alignment Log.
    db_id = get_db_id("NOTION_ROS_Alignment_Log_ID")
    resolver = NotionDataSourceResolver(client=client)
    resolved = resolver.resolve_once(
        name="ROS_Alignment_Log",
        database_id=db_id,
    )
    data_source_id = resolved.data_source_id
    logger.info("Resolved data_source_id for ROS_Alignment_Log: %s", data_source_id)

    repo = AlignmentLogRepo(
        client=client,
        database_id=db_id,
        data_source_id=data_source_id,
    )
    repo.ensure_schema()

    # Resolve Values Codex for Domain relation.
    codex_repo = None
    try:
        codex_db_id = get_db_id("NOTION_ROS_Values_Codex_ID")
        codex_resolved = resolver.resolve_once(
            name="ROS_Values_Codex",
            database_id=codex_db_id,
        )
        codex_repo = ValuesCodexRepo(
            client=client,
            database_id=codex_db_id,
            data_source_id=codex_resolved.data_source_id,
        )
        logger.info("Values Codex repo initialized for Domain relation")
    except Exception as e:
        logger.warning("Could not initialize Values Codex repo: %s", e)

    tracker = TruncationTracker()
    review_quarter = record.review_quarter
    written = 0

    for entry in entries:
        # Resolve Domain relation.
        codex_page_id = None
        if codex_repo is not None:
            key = f"{review_quarter}:{entry.domain_id}"
            try:
                codex_page_id = codex_repo.fetch_domain_page_id(key)
                if codex_page_id:
                    logger.info("Domain relation resolved: %s -> %s", key, codex_page_id)
                else:
                    logger.warning("Domain relation: no codex page for key=%s", key)
            except Exception as e:
                logger.warning("Domain relation resolution failed: %s", e)

        repo.create_validated_entry(
            entry=entry,
            codex_page_id=codex_page_id,
            tracker=tracker,
        )
        logger.info(
            "Wrote alignment entry: %s (imp=%d, align=%d, gap=%d, summary=%s, adj=%s)",
            entry.domain_id,
            entry.importance_score,
            entry.alignment_score,
            entry.gap_score,
            "YES" if entry.ai_summary else "NO",
            "YES" if entry.next_adjustment else "NO",
        )
        written += 1

    if tracker.had_truncations:
        logger.warning(
            "Truncated fields during Notion write: %s",
            [e["field"] for e in tracker.report()],
        )

    return written


# ================================================================
# Summary markdown
# ================================================================

def _write_summary_markdown(
    out_dir: Path,
    entries: list[AlignmentEntry],
    ai_summary: str,
    date_iso: str,
    review_type: str,
    voice_config: VoiceConfig,
) -> None:
    """Write a human-readable summary of the reflection session."""
    lines = [
        f"# Values Voice Reflection — {date_iso}",
        "",
        f"Review Type: {review_type}",
        f"Language: {voice_config.display_name} ({voice_config.language})",
        f"Domains Reflected: {len(entries)}",
        "",
        "## Scores",
        "",
        "| Domain | Importance | Alignment | Gap | Significant? |",
        "|--------|-----------|-----------|-----|-------------|",
    ]
    for e in entries:
        sig = "YES" if e.significant_gap else ""
        lines.append(
            f"| {e.domain_id} | {e.importance_score} | "
            f"{e.alignment_score} | {e.gap_score} | {sig} |"
        )

    if ai_summary:
        lines.extend(["", "## AI Summary", "", ai_summary])

    lines.extend(["", "## Transcripts", ""])
    for e in entries:
        lines.append(f"### {e.domain_id}")
        lines.append(f"```")
        lines.append(e.transcript)
        lines.append(f"```")
        lines.append("")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info("Wrote summary.md to %s", summary_path)


# ================================================================
# CLI
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="055 Values Voice Reflection — interactive voice-based value alignment session",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Actually run the pipeline (default: shows help)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="all",
        help="Single domain_id to reflect on, or 'all' (default: all)",
    )
    parser.add_argument(
        "--values-json",
        type=str,
        default=None,
        help="Path to values.json (from 054). Falls back to latest 054 output.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        help="Enable Notion write-back to ROS_Alignment_Log",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="No mic, no Whisper, no TTS, no Notion. Uses dummy data.",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        default=False,
        help="Speak reflection questions aloud via TTS",
    )
    parser.add_argument(
        "--review-type",
        type=str,
        default="Daily",
        choices=["Daily", "Weekly", "Quarterly"],
        help="Review type (default: Daily)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        choices=list(SUPPORTED_LANGUAGES),
        help=(
            "Voice I/O language (default: ja). "
            "Also configurable via VOICE_LANGUAGE env var."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if not args.run:
        parser.print_help()
        print("\n  Pass --run to execute the pipeline.")
        sys.exit(0)

    if args.domain != "all" and args.domain not in DOMAIN_IDS:
        parser.error(
            f"Unknown domain: {args.domain!r}. "
            f"Available: {list(DOMAIN_IDS)}"
        )

    result = run_pipeline(
        domain_filter=args.domain,
        values_json_path=args.values_json,
        write_notion=args.write,
        dry_run=args.dry_run,
        enable_tts=args.tts,
        review_type=args.review_type,
        verbose=args.verbose,
        language=args.lang,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
