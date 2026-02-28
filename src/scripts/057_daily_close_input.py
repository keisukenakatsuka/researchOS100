#!/usr/bin/env python
# src/scripts/057_daily_close_input.py
"""Daily Close Input — capture raw evening close log.

Supports multiple input modes:

  1. Typed text   — ``--text "Did X, Y, Z today"``
  2. File input   — ``--from-file path/to/notes.txt``
  3. Audio file   — ``--audio-file path/to/recording.wav`` (transcribed via Whisper)
  4. Browser recording — ``--record`` (opens browser UI, single clip)
  5. Wizard recording  — ``--record-wizard`` (guided multi-section browser recording)
  6. Interactive   — default stdin prompt when no flags are given

Input precedence (first match wins):
  --text > --from-file > --audio-file > --record > --record-wizard > interactive

Pipeline:
1. Accept raw close text via one of the modes above
2. Optionally collect satisfaction (1–5) and energy level
3. Save raw text + metadata to data/daily/close_raw/YYYY-MM-DD/
4. For voice modes, also persist audio file(s) and transcript text file(s)

Usage::

    # Interactive (stdin prompt)
    python -m src.scripts.057_daily_close_input

    # From inline text
    python -m src.scripts.057_daily_close_input --text "Did X, Y, Z today"

    # From file
    python -m src.scripts.057_daily_close_input --from-file notes.txt --non-interactive

    # From existing audio file (transcribed via Whisper)
    python -m src.scripts.057_daily_close_input --audio-file recording.m4a --language ja

    # Browser-based single recording
    python -m src.scripts.057_daily_close_input --record --record-seconds 180

    # Wizard mode — guided section-by-section browser recording
    python -m src.scripts.057_daily_close_input --record-wizard --record-seconds 60

    # Wizard with custom sections
    python -m src.scripts.057_daily_close_input --record-wizard \\
        --wizard-sections done,friction,tomorrow
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
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
from src.daily.models import CloseRawInput
from src.daily.io import CLOSE_RAW_DIR, daily_output_dir, save_json

logger = logging.getLogger("057_daily_close_input")

SCRIPT_NAME = "057_daily_close_input"
JST = ZoneInfo("Asia/Tokyo")


# ── Interactive helpers ──────────────────────────────────────────

def _read_interactive() -> str:
    """Read multi-line text from stdin until EOF or empty line."""
    print("\nEnter your daily close log (press Ctrl-D or enter empty line to finish):\n")
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



# ── Audio handlers ───────────────────────────────────────────────

def _handle_audio_file(
    audio_file: str,
    out_dir: Path,
    language: str,
) -> tuple[str, str, str]:
    """Process an existing audio file: copy to output dir and transcribe.

    Returns (raw_text, audio_dest_path, transcript_path).
    """
    from src.daily.audio import transcribe_audio

    src_path = Path(audio_file)
    if not src_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    ext = src_path.suffix or ".wav"
    audio_dest = out_dir / f"close_raw_audio{ext}"
    shutil.copy2(str(src_path), str(audio_dest))
    logger.info("Copied audio file to %s", audio_dest)

    raw_text = transcribe_audio(str(src_path), language=language)
    logger.info("Transcribed audio (%s): %d chars", language, len(raw_text))

    transcript_path = out_dir / "close_raw_transcript.txt"
    transcript_path.write_text(raw_text, encoding="utf-8")
    logger.info("Saved transcript to %s", transcript_path)

    return raw_text, str(audio_dest), str(transcript_path)


def _handle_browser_record(
    out_dir: Path,
    record_seconds: int,
    language: str,
    port: int = 8057,
) -> tuple[str, str, str]:
    """Launch single-clip browser recorder, save audio + transcript.

    Returns (raw_text, audio_path, transcript_path).
    """
    from src.daily.audio import get_recorder_transcript, launch_browser_recorder

    audio_dest = out_dir / "close_raw_audio.webm"

    audio_path = launch_browser_recorder(
        output_path=str(audio_dest),
        seconds=record_seconds,
        language=language,
        port=port,
    )
    audio_path_str = str(audio_path)

    raw_text = get_recorder_transcript()
    if not raw_text:
        logger.warning("Browser recorder returned empty transcript")
        raw_text = ""

    transcript_path = out_dir / "close_raw_transcript.txt"
    transcript_path.write_text(raw_text, encoding="utf-8")
    logger.info("Saved transcript to %s", transcript_path)

    return raw_text, audio_path_str, str(transcript_path)


# ── Wizard handler ───────────────────────────────────────────────

def _parse_satisfaction_from_transcript(text: str) -> Optional[int]:
    """Try to extract a satisfaction score (1-5) from a transcript."""
    # Look for standalone digits 1-5.
    m = re.search(r'\b([1-5])\b', text)
    if m:
        return int(m.group(1))
    # Japanese number words.
    ja_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    for word, val in ja_map.items():
        if word in text:
            return val
    return None


def _resolve_wizard_sections(
    section_keys: Optional[str],
) -> list[dict[str, str]]:
    """Resolve wizard sections from comma-separated keys or use defaults."""
    from src.daily.audio import DEFAULT_WIZARD_SECTIONS

    if not section_keys:
        return list(DEFAULT_WIZARD_SECTIONS)

    all_by_key = {s["key"]: s for s in DEFAULT_WIZARD_SECTIONS}
    result = []
    for key in section_keys.split(","):
        key = key.strip()
        if key in all_by_key:
            result.append(all_by_key[key])
        else:
            # Allow custom sections with just a key.
            result.append({
                "key": key,
                "title_ja": key,
                "instruction": "",
            })
    return result


def _handle_wizard_record(
    out_dir: Path,
    record_seconds: int,
    language: str,
    wizard_sections: Optional[str] = None,
    non_interactive: bool = False,
    port: int = 8057,
    date_iso: str = "",
) -> dict[str, Any]:
    """Run the wizard recording flow.

    Returns dict with keys: raw_text, transcript_path, sections,
    satisfaction, value_domains, notion_result.
    """
    from src.daily.audio import (
        get_wizard_selected_values,
        get_wizard_notion_result,
        get_wizard_satisfaction,
        get_wizard_energy_level,
        launch_wizard_recorder,
    )

    sections = _resolve_wizard_sections(wizard_sections)

    section_results = launch_wizard_recorder(
        out_dir=str(out_dir),
        sections=sections,
        seconds=record_seconds,
        language=language,
        port=port,
        date_iso=date_iso,
    )

    # Build per-section metadata for JSON (strip raw transcript to avoid bloat).
    section_meta: List[Dict[str, Any]] = []
    for r in section_results:
        section_meta.append({
            "key": r["key"],
            "title_ja": r["title_ja"],
            "audio_path": r.get("audio_path"),
            "transcript_path": r.get("transcript_path"),
            "transcript_chars": r.get("transcript_chars", 0),
            "skipped": r.get("skipped", False),
        })

    # Assemble structured transcript from all sections.
    assembled_parts = []
    for r in section_results:
        title = r.get("title_ja", r.get("key", ""))
        transcript = r.get("transcript", "")
        if r.get("skipped"):
            assembled_parts.append(f"## {title}\n(skipped)\n")
        else:
            assembled_parts.append(f"## {title}\n{transcript}\n")
    assembled_text = "\n".join(assembled_parts).strip()

    # Save assembled transcript.
    transcript_path = out_dir / "close_raw_transcript.txt"
    transcript_path.write_text(assembled_text, encoding="utf-8")

    # Extract satisfaction from the "satisfaction" section if present.
    satisfaction: Optional[int] = None
    for r in section_results:
        if r.get("key") == "satisfaction" and not r.get("skipped"):
            satisfaction = _parse_satisfaction_from_transcript(r.get("transcript", ""))
            break

    # Satisfaction is extracted from the voice transcript only.
    # No CLI fallback — all input must come from the browser UI.

    # ── Value domains: prefer Notion checklist selection, fallback to keyword ──
    #
    # get_wizard_selected_values() returns the domain keys the user checked
    # in the browser UI (populated from Notion).  If the user didn't select
    # any (or Notion was unavailable), fall back to keyword extraction from
    # the voice transcript for the "values" section.
    value_domains: List[str] = get_wizard_selected_values()
    if not value_domains:
        for r in section_results:
            if r.get("key") == "values" and not r.get("skipped"):
                value_domains = _extract_value_domains(r.get("transcript", ""))
                break
    if value_domains:
        logger.info("Value domains: %s (source=%s)",
                     value_domains,
                     "notion_checklist" if get_wizard_selected_values() else "keyword")

    # Browser UI metadata (satisfaction / energy set by user in done-view).
    ui_satisfaction = get_wizard_satisfaction()
    if ui_satisfaction is not None:
        satisfaction = ui_satisfaction
    energy_level = get_wizard_energy_level()  # may be None if user skipped

    # Notion result (populated if user clicked "Submit to Notion" in browser).
    notion_result = get_wizard_notion_result()

    return {
        "raw_text": assembled_text,
        "transcript_path": str(transcript_path),
        "sections": section_meta,
        "satisfaction": satisfaction,
        "energy_level": energy_level,
        "value_domains": value_domains,
        "notion_result": notion_result,
    }


def _extract_value_domains(text: str) -> list[str]:
    """Simple keyword-based extraction of value domains from transcript."""
    # Known domain keywords (from 054_values_scale_setup).
    domain_keywords = {
        "健康": "Health",
        "家族": "Family",
        "仕事": "Work",
        "学び": "Learning",
        "創造": "Creativity",
        "つながり": "Connection",
        "冒険": "Adventure",
        "自由": "Freedom",
        "貢献": "Contribution",
        "成長": "Growth",
        "誠実": "Integrity",
        "感謝": "Gratitude",
    }
    found = []
    for ja, en in domain_keywords.items():
        if ja in text:
            found.append(en)
    return found


# ── Main pipeline ────────────────────────────────────────────────

def run_pipeline(
    *,
    date_override: Optional[str] = None,
    text: Optional[str] = None,
    from_file: Optional[str] = None,
    audio_file: Optional[str] = None,
    record: bool = False,
    record_wizard: bool = False,
    wizard_sections: Optional[str] = None,
    record_seconds: int = 120,
    language: str = "ja",
    satisfaction: Optional[int] = None,
    energy_level: Optional[str] = None,
    non_interactive: bool = False,
    verbose: bool = False,
) -> dict:
    """Execute the daily close input pipeline."""
    load_env()
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    now_jst = datetime.now(tz=JST)
    date_iso = date_override or now_jst.date().isoformat()
    wk = get_iso_week_context(tz=JST)

    logger.info("Starting %s date=%s non_interactive=%s", SCRIPT_NAME, date_iso, non_interactive)

    out_dir = daily_output_dir(CLOSE_RAW_DIR, date_iso)

    # ── 1. Get raw text (deterministic precedence) ──
    input_mode = "interactive"
    audio_path: Optional[str] = None
    transcript_path: Optional[str] = None
    sections_meta: Optional[List[Dict[str, Any]]] = None
    value_domains: Optional[List[str]] = None
    notion_result: Optional[Dict[str, Any]] = None

    if text:
        raw_text = text
        input_mode = "text"
    elif from_file:
        p = Path(from_file)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {from_file}")
        raw_text = p.read_text(encoding="utf-8").strip()
        input_mode = "file"
        logger.info("Read close log from file: %s (%d chars)", p, len(raw_text))
    elif audio_file:
        try:
            raw_text, audio_path, transcript_path = _handle_audio_file(
                audio_file, out_dir, language,
            )
            input_mode = "audio_file"
        except ImportError as e:
            _exit_missing_dep(str(e))
            return {}
    elif record:
        try:
            raw_text, audio_path, transcript_path = _handle_browser_record(
                out_dir, record_seconds, language,
            )
            input_mode = "browser_recorded_audio"
        except ImportError as e:
            _exit_missing_dep(str(e))
            return {}
    elif record_wizard:
        try:
            wizard_result = _handle_wizard_record(
                out_dir, record_seconds, language,
                wizard_sections=wizard_sections,
                non_interactive=non_interactive,
                date_iso=date_iso,
            )
            raw_text = wizard_result["raw_text"]
            transcript_path = wizard_result["transcript_path"]
            sections_meta = wizard_result["sections"]
            value_domains = wizard_result.get("value_domains") or None
            notion_result = wizard_result.get("notion_result")
            input_mode = "browser_recorded_audio_wizard"
            # Use satisfaction from wizard if extracted and not overridden by CLI.
            if satisfaction is None:
                satisfaction = wizard_result.get("satisfaction")
            # Use energy level from browser UI if not overridden by CLI.
            if energy_level is None:
                energy_level = wizard_result.get("energy_level")
        except ImportError as e:
            _exit_missing_dep(str(e))
            return {}
    elif non_interactive:
        logger.error(
            "Non-interactive mode requires --text, --from-file, --audio-file, --record, or --record-wizard."
        )
        print(
            "\nERROR: Non-interactive mode but no input source provided.\n"
            "Use one of:\n"
            "  --text 'your close log text'\n"
            "  --from-file path/to/file.txt\n"
            "  --audio-file path/to/recording.wav\n"
            "  --record  (browser-based single recording)\n"
            "  --record-wizard  (guided section-by-section recording)\n",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        raw_text = _read_interactive()

    if not raw_text:
        logger.error("Empty close log. Aborting.")
        sys.exit(1)

    # ── 2. Metadata ──
    # All metadata (satisfaction, energy) comes from CLI flags or
    # the browser UI.  No interactive terminal prompts.

    # ── 3. Build model ──
    is_audio = input_mode in ("audio_file", "browser_recorded_audio", "browser_recorded_audio_wizard")
    close_raw = CloseRawInput(
        date=date_iso,
        raw_text=raw_text,
        satisfaction=satisfaction,
        energy_level=energy_level,
        timestamp=now_jst.isoformat(timespec="seconds"),
        input_mode=input_mode,
        audio_path=audio_path,
        transcript_path=transcript_path,
        transcript_chars=len(raw_text) if is_audio else None,
        transcription_engine="whisper-1" if is_audio else None,
        language=language if is_audio else None,
        sections=sections_meta,
        value_domains=value_domains,
    )

    # ── 4. Save to data/daily/close_raw/YYYY-MM-DD/ ──
    out_path = save_json(out_dir / "close_raw.json", close_raw.to_dict())
    logger.info("Saved close_raw.json -> %s", out_path)

    # ── 5. Run metadata ──
    meta = RunMetadata.build(
        notebook=SCRIPT_NAME,
        week_id=wk.week_id,
        counts={
            "raw_text_chars": len(raw_text),
            "satisfaction": satisfaction,
            "energy_level": energy_level,
            "input_mode": input_mode,
        },
    )
    meta.save(out_dir / "run_metadata.json")

    result = {
        "date": date_iso,
        "output_dir": str(out_dir),
        "raw_text_chars": len(raw_text),
        "satisfaction": satisfaction,
        "energy_level": energy_level,
        "input_mode": input_mode,
        "audio_path": audio_path,
        "transcript_path": transcript_path,
    }
    if sections_meta:
        result["sections_count"] = len(sections_meta)
    if value_domains:
        result["value_domains"] = value_domains
    if notion_result:
        result["notion"] = notion_result

    print(f"\nClose log saved -> {out_dir}")
    print(f"  Mode: {input_mode} | Text: {len(raw_text)} chars | Satisfaction: {satisfaction} | Energy: {energy_level}")
    if audio_path:
        print(f"  Audio: {audio_path}")
    if transcript_path:
        print(f"  Transcript: {transcript_path}")
    if sections_meta:
        recorded = sum(1 for s in sections_meta if not s.get("skipped"))
        skipped = sum(1 for s in sections_meta if s.get("skipped"))
        print(f"  Wizard sections: {recorded} recorded, {skipped} skipped")
    if notion_result:
        if notion_result.get("ok"):
            print(f"  Notion: {notion_result['action']} -> {notion_result.get('page_url', notion_result.get('page_id', ''))}")
        else:
            print(f"  Notion: FAILED -> {notion_result.get('error', 'unknown error')}")

    return result


def _exit_missing_dep(msg: str) -> None:
    """Print a friendly error for missing dependencies and exit."""
    print(
        f"\nERROR: Audio/browser dependency missing.\n"
        f"{msg}\n\n"
        f"Text and file input modes still work without these dependencies.\n"
        f"To use voice input, install the required packages and try again.",
        file=sys.stderr,
    )
    sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="057 Daily Close Input — capture raw evening close log",
    )
    parser.add_argument("--date", type=str, default=None, help="Override date (YYYY-MM-DD)")
    parser.add_argument("--text", type=str, default=None, help="Close log text directly")
    parser.add_argument("--from-file", type=str, default=None, dest="from_file",
                        help="Read close log from file path")

    # ── Voice input flags ──
    audio_group = parser.add_argument_group("voice input")
    audio_group.add_argument(
        "--audio-file", type=str, default=None, dest="audio_file",
        help="Use an existing audio file (wav/mp3/m4a/webm/ogg) — transcribed via Whisper",
    )
    audio_group.add_argument(
        "--record", action="store_true", default=False,
        help="Launch browser-based recorder (single clip), then transcribe",
    )
    audio_group.add_argument(
        "--record-wizard", action="store_true", default=False, dest="record_wizard",
        help="Launch guided section-by-section browser recorder (wizard mode)",
    )
    audio_group.add_argument(
        "--wizard-sections", type=str, default=None, dest="wizard_sections",
        help="Comma-separated section keys for wizard (default: done,friction,tomorrow,mind,satisfaction,values)",
    )
    audio_group.add_argument(
        "--record-seconds", type=int, default=120, dest="record_seconds",
        help="Maximum recording duration per section in seconds (default: 120)",
    )
    audio_group.add_argument(
        "--language", type=str, default="ja",
        help="Transcription language — ISO 639-1 code (default: ja)",
    )

    parser.add_argument("--satisfaction", type=int, default=None, help="Satisfaction score (1-5)")
    parser.add_argument("--energy", type=str, default=None, choices=["Low", "Medium", "High"],
                        help="Energy level")
    parser.add_argument("--non-interactive", action="store_true", dest="non_interactive",
                        help="Skip all interactive prompts")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    # Gracefully check dependencies for audio modes before running.
    if args.audio_file or args.record or args.record_wizard:
        try:
            from src.daily.audio import AudioDependencyError
            if args.record or args.record_wizard:
                from src.daily.audio import check_browser_dependencies
                check_browser_dependencies()
        except (ImportError, Exception) as e:
            _exit_missing_dep(str(e))

    result = run_pipeline(
        date_override=args.date,
        text=args.text,
        from_file=args.from_file,
        audio_file=args.audio_file,
        record=args.record,
        record_wizard=args.record_wizard,
        wizard_sections=args.wizard_sections,
        record_seconds=args.record_seconds,
        language=args.language,
        satisfaction=args.satisfaction,
        energy_level=args.energy,
        non_interactive=args.non_interactive,
        verbose=args.verbose,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
