# src/daily/audio.py
"""Shared audio recording and transcription for the Daily System.

Provides:
- ``launch_browser_recorder()``: open browser UI to record a single audio clip
- ``launch_wizard_recorder()``: open browser UI for multi-section wizard recording
- ``transcribe_audio()``: transcribe an audio file via OpenAI Whisper (LLMRouter)
- ``check_browser_dependencies()``: verify required packages are installed

The browser recorder reuses the same FastAPI + MediaRecorder pattern as
``src/app/values_voice``, but with a minimal single-purpose UI focused on
capturing recordings and returning results.

Dependencies (for browser recording):
- uvicorn, fastapi, python-multipart  (pip install)

Dependencies (for transcription):
- OpenAI API key configured (via LLMRouter / .env)
"""

# NOTE: Do NOT use ``from __future__ import annotations`` here.
# FastAPI + Pydantic v2 need *runtime* type objects for endpoint
# parameter resolution (UploadFile, File, etc.).  The future-
# annotations import turns them into ForwardRef strings which
# triggers ``PydanticUserError: ... is not fully defined``.

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Dependency checks
# ----------------------------------------------------------------

_BROWSER_DEPS = ("uvicorn", "fastapi")
_BROWSER_INSTALL_HINT = "pip install uvicorn fastapi python-multipart"


class AudioDependencyError(RuntimeError):
    """Raised when required audio/browser dependencies are missing."""


def check_browser_dependencies() -> None:
    """Verify browser recording dependencies are installed.

    Raises
    ------
    AudioDependencyError
        With a user-friendly message listing missing packages.
    """
    missing = []
    for pkg in _BROWSER_DEPS:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise AudioDependencyError(
            f"Missing packages for browser recording: {', '.join(missing)}.\n"
            f"Install with:  {_BROWSER_INSTALL_HINT}"
        )


def check_transcription_dependencies() -> None:
    """Verify transcription dependencies (LLMRouter / OpenAI) are available.

    Raises
    ------
    AudioDependencyError
        If the LLM router cannot be initialized.
    """
    try:
        from src.config import load_env
        load_env()
        from src.llm.router import build_router_from_env
        build_router_from_env()
    except Exception as e:
        raise AudioDependencyError(
            f"Cannot initialize transcription engine (OpenAI Whisper via LLMRouter).\n"
            f"Ensure OPENAI_API_KEY is set in .env.\n"
            f"Error: {e}"
        )


# ----------------------------------------------------------------
# Transcription
# ----------------------------------------------------------------

def transcribe_audio(
    audio_path: Union[str, Path],
    *,
    language: str = "ja",
) -> str:
    """Transcribe an audio file via OpenAI Whisper through LLMRouter.

    Parameters
    ----------
    audio_path : str or Path
        Path to audio file (.wav, .mp3, .m4a, .webm, .ogg, etc.).
    language : str
        Language hint for Whisper (ISO 639-1). Default: "ja".

    Returns
    -------
    str
        Transcribed text.

    Raises
    ------
    AudioDependencyError
        If transcription engine is unavailable.
    FileNotFoundError
        If the audio file does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    from src.config import load_env
    load_env()
    from src.llm.router import build_router_from_env
    from src.values.voice import VoiceConfig

    router = build_router_from_env()
    voice_config = VoiceConfig(language=language)
    transcript = router.transcribe_audio(audio_path, voice_config=voice_config)
    logger.info(
        "Transcribed %s (%s): %d chars",
        audio_path.name, language, len(transcript),
    )
    return transcript


# ----------------------------------------------------------------
# Default wizard sections
# ----------------------------------------------------------------

DEFAULT_WIZARD_SECTIONS: List[Dict[str, str]] = [
    {
        "key": "done",
        "title_ja": "今日やったこと",
        "instruction": "今日取り組んだタスクや成果を話してください。",
    },
    {
        "key": "friction",
        "title_ja": "詰まりや違和感",
        "instruction": "詰まったこと、違和感があったこと、気になった摩擦を話してください。",
    },
    {
        "key": "tomorrow",
        "title_ja": "明日の予定",
        "instruction": "明日やること、予定していることを話してください。",
    },
    {
        "key": "mind",
        "title_ja": "気になっていること",
        "instruction": "頭の中にあること、気がかりなことを自由に話してください。",
    },
    {
        "key": "satisfaction",
        "title_ja": "満足度スコア",
        "instruction": "今日の満足度を1〜5で答えてください。理由も一言あればどうぞ。",
    },
    {
        "key": "values",
        "title_ja": "今日触れた価値領域",
        "instruction": "今日意識した、または触れた価値領域（バリュードメイン）を挙げてください。",
    },
]


# ----------------------------------------------------------------
# Browser-based recorder — single clip (existing API, unchanged)
# ----------------------------------------------------------------

# Shared state for the single-clip recorder server.
_recorder_state: dict = {}


def launch_browser_recorder(
    output_path: Union[str, Path],
    *,
    seconds: int = 120,
    language: str = "ja",
    port: int = 8057,
    no_browser: bool = False,
) -> str:
    """Launch browser-based audio recorder and return the saved audio file path.

    Opens a minimal web UI where the user can record audio via the browser's
    MediaRecorder API. After recording and optional transcription, the audio
    file is saved to ``output_path`` and the function returns.

    Parameters
    ----------
    output_path : str or Path
        Where to save the recorded audio file (e.g. .webm).
    seconds : int
        Maximum recording duration in seconds (default: 120).
    language : str
        Language hint for Whisper transcription (default: "ja").
    port : int
        Local server port (default: 8057).
    no_browser : bool
        If True, don't auto-open the browser (for testing).

    Returns
    -------
    str
        Path to the saved audio file.
    """
    check_browser_dependencies()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import uvicorn
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    _recorder_state.clear()
    _recorder_state.update({
        "output_path": str(output_path),
        "audio_saved": False,
        "transcript": "",
        "cancelled": False,
        "done": False,
        "language": language,
        "max_seconds": seconds,
    })

    app = FastAPI(title="Daily Close Audio Recorder")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _build_recorder_html(seconds=seconds, language=language)

    @app.get("/api/config")
    async def get_config():
        return {"max_seconds": seconds, "language": language}

    @app.post("/api/upload")
    async def upload_audio(audio: UploadFile = File(...)):
        content = await audio.read()
        if not content:
            return JSONResponse({"ok": False, "error": "Empty audio"}, status_code=400)
        ext = _ext_from_content_type(audio.content_type)
        final_path = output_path.with_suffix(ext)
        final_path.write_bytes(content)
        _recorder_state["output_path"] = str(final_path)
        _recorder_state["audio_saved"] = True
        logger.info("Audio saved: %s (%d bytes)", final_path, len(content))
        transcript = ""
        try:
            transcript = transcribe_audio(final_path, language=language)
            _recorder_state["transcript"] = transcript
        except Exception as e:
            logger.warning("Transcription failed: %s", e)
            _recorder_state["transcript"] = f"[transcription error: {e}]"
        return {"ok": True, "path": str(final_path), "transcript": transcript}

    @app.post("/api/done")
    async def mark_done():
        _recorder_state["done"] = True
        return {"ok": True}

    @app.post("/api/cancel")
    async def cancel():
        _recorder_state["cancelled"] = True
        _recorder_state["done"] = True
        return {"ok": True}

    server = _start_server(app, port=port)
    url = f"http://localhost:{port}"
    logger.info("Recorder server started at %s", url)
    if not no_browser:
        _auto_open_browser(url)

    try:
        while not _recorder_state.get("done"):
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
        _recorder_state["cancelled"] = True

    _stop_server(server)

    if _recorder_state.get("cancelled"):
        raise RuntimeError("Recording was cancelled by user.")
    if not _recorder_state.get("audio_saved"):
        raise RuntimeError("No audio was recorded.")

    final = _recorder_state["output_path"]
    logger.info("Browser recorder finished: %s", final)
    return final


def get_recorder_transcript() -> str:
    """Return the transcript from the last browser recording session."""
    return _recorder_state.get("transcript", "")


# ----------------------------------------------------------------
# Browser-based recorder — wizard (multi-section)
# ----------------------------------------------------------------

# Shared state for the wizard recorder server.
_wizard_state: dict = {}


def launch_wizard_recorder(
    out_dir: Union[str, Path],
    *,
    sections: Optional[List[Dict[str, str]]] = None,
    seconds: int = 120,
    language: str = "ja",
    port: int = 8057,
    no_browser: bool = False,
    date_iso: str = "",
) -> List[Dict[str, Any]]:
    """Launch browser-based multi-section wizard recorder.

    Opens a single browser tab that walks the user through each section,
    recording and transcribing one at a time. Audio files are saved into
    ``out_dir`` with deterministic names.

    Parameters
    ----------
    out_dir : str or Path
        Directory where per-section audio + transcript files are saved.
    sections : list of dicts, optional
        Each dict must have ``key``, ``title_ja``, ``instruction``.
        Defaults to ``DEFAULT_WIZARD_SECTIONS``.
    seconds : int
        Max recording duration per section (default: 120).
    language : str
        Language hint for Whisper (default: "ja").
    port : int
        Local server port (default: 8057).
    no_browser : bool
        If True, don't auto-open the browser.
    date_iso : str
        Target date (YYYY-MM-DD) for Notion upsert.

    Returns
    -------
    list of dict
        One entry per section with keys:
        ``key``, ``title_ja``, ``audio_path``, ``transcript_path``,
        ``transcript``, ``transcript_chars``, ``skipped``.
    """
    check_browser_dependencies()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sections is None:
        sections = DEFAULT_WIZARD_SECTIONS

    import uvicorn
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    # Build section metadata list (with indices for filenames).
    section_meta = []
    for idx, sec in enumerate(sections):
        section_meta.append({
            "key": sec["key"],
            "title_ja": sec["title_ja"],
            "instruction": sec.get("instruction", ""),
            "index": idx,
            "filename_prefix": f"close_raw_audio_{idx + 1:02d}_{sec['key']}",
        })

    # Pre-fetch Notion values (graceful fallback to empty).
    notion_values = fetch_values_from_notion()
    logger.info("Pre-fetched %d Notion value domains for wizard", len(notion_values))

    _wizard_state.clear()
    _wizard_state.update({
        "sections": section_meta,
        "results": [],  # list of dicts, one per completed section
        "cancelled": False,
        "done": False,
        "language": language,
        "max_seconds": seconds,
        "out_dir": str(out_dir),
        "notion_values": notion_values,
        "selected_values": [],  # user selections from checklist
        "date_iso": date_iso,
        "notion_result": None,  # populated after Notion submit
    })

    app = FastAPI(title="Daily Close Wizard Recorder")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _build_wizard_html(
            sections=section_meta,
            seconds=seconds,
            language=language,
            out_dir=str(out_dir),
            notion_values=notion_values,
        )

    @app.get("/api/wizard-config")
    async def wizard_config():
        return {
            "sections": section_meta,
            "max_seconds": seconds,
            "language": language,
        }

    @app.post("/api/wizard-upload")
    async def wizard_upload(
        audio: UploadFile = File(...),
        section_key: str = Form(...),
        section_index: int = Form(...),
    ):
        """Save audio for one section, transcribe, return transcript."""
        content = await audio.read()
        if not content:
            return JSONResponse({"ok": False, "error": "Empty audio"}, status_code=400)

        ext = _ext_from_content_type(audio.content_type)
        prefix = f"close_raw_audio_{section_index + 1:02d}_{section_key}"
        audio_path = out_dir / f"{prefix}{ext}"
        audio_path.write_bytes(content)
        logger.info("Wizard audio saved: %s (%d bytes)", audio_path, len(content))

        transcript = ""
        try:
            transcript = transcribe_audio(audio_path, language=language)
        except Exception as e:
            logger.warning("Wizard transcription failed for %s: %s", section_key, e)
            transcript = f"[transcription error: {e}]"

        # Save per-section transcript.
        transcript_path = out_dir / f"close_raw_transcript_{section_index + 1:02d}_{section_key}.txt"
        transcript_path.write_text(transcript, encoding="utf-8")

        result = {
            "key": section_key,
            "title_ja": section_meta[section_index]["title_ja"],
            "audio_path": str(audio_path),
            "transcript_path": str(transcript_path),
            "transcript": transcript,
            "transcript_chars": len(transcript),
            "skipped": False,
        }
        _wizard_state["results"].append(result)
        logger.info("Wizard section '%s' done: %d chars", section_key, len(transcript))

        return {"ok": True, "transcript": transcript, "section_key": section_key}

    @app.post("/api/wizard-skip")
    async def wizard_skip(
        section_key: str = Form(...),
        section_index: int = Form(...),
    ):
        """Mark a section as skipped."""
        result = {
            "key": section_key,
            "title_ja": section_meta[section_index]["title_ja"],
            "audio_path": None,
            "transcript_path": None,
            "transcript": "",
            "transcript_chars": 0,
            "skipped": True,
        }
        _wizard_state["results"].append(result)
        logger.info("Wizard section '%s' skipped", section_key)
        return {"ok": True, "section_key": section_key}

    @app.post("/api/wizard-done")
    async def wizard_done():
        _wizard_state["done"] = True
        return {"ok": True}

    @app.post("/api/wizard-cancel")
    async def wizard_cancel():
        _wizard_state["cancelled"] = True
        _wizard_state["done"] = True
        return {"ok": True}

    @app.get("/api/wizard-values")
    async def wizard_values():
        """Return Notion value domains for the checklist UI."""
        return {
            "ok": True,
            "values": _wizard_state.get("notion_values", []),
        }

    @app.post("/api/wizard-value-selection")
    async def wizard_value_selection(
        selected_json: str = Form(...),
    ):
        """Receive selected value domains from the checklist UI.

        ``selected_json`` is a JSON-encoded list of domain key strings.
        """
        selected = json.loads(selected_json)
        _wizard_state["selected_values"] = selected
        logger.info("User selected %d value domains: %s", len(selected), selected)
        return {"ok": True, "count": len(selected)}

    @app.post("/api/wizard-metadata")
    async def wizard_metadata(
        satisfaction: Optional[str] = Form(None),
        energy_level: Optional[str] = Form(None),
    ):
        """Receive satisfaction and energy level from the browser UI."""
        if satisfaction is not None and satisfaction.strip():
            try:
                val = int(satisfaction.strip())
                if 1 <= val <= 5:
                    _wizard_state["satisfaction"] = val
                    logger.info("Browser UI satisfaction: %d", val)
            except ValueError:
                pass
        if energy_level is not None and energy_level.strip():
            el = energy_level.strip()
            if el in ("Low", "Medium", "High"):
                _wizard_state["energy_level"] = el
                logger.info("Browser UI energy_level: %s", el)
        return {
            "ok": True,
            "satisfaction": _wizard_state.get("satisfaction"),
            "energy_level": _wizard_state.get("energy_level"),
        }

    @app.post("/api/notion-submit")
    async def notion_submit():
        """Submit close data to Notion Daily Logs (upsert by date).

        Reads from wizard state: results, selected_values, date_iso.
        Calls notion_upsert_daily_log() and returns result to the browser.
        """
        ws_date = _wizard_state.get("date_iso", "")
        if not ws_date:
            return JSONResponse(
                {"ok": False, "error": "No date set in wizard state"},
                status_code=400,
            )

        # Assemble raw text from section results.
        parts = []
        for r in _wizard_state.get("results", []):
            title = r.get("title_ja", r.get("key", ""))
            transcript = r.get("transcript", "")
            if r.get("skipped"):
                parts.append(f"## {title}\n(skipped)\n")
            else:
                parts.append(f"## {title}\n{transcript}\n")
        raw_text = "\n".join(parts).strip()

        # Extract satisfaction from results.
        satisfaction = None
        import re as _re
        for r in _wizard_state.get("results", []):
            if r.get("key") == "satisfaction" and not r.get("skipped"):
                txt = r.get("transcript", "")
                m = _re.search(r'\b([1-5])\b', txt)
                if m:
                    satisfaction = int(m.group(1))
                else:
                    ja_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
                    for word, val in ja_map.items():
                        if word in txt:
                            satisfaction = val
                            break
                break

        # Energy level from browser UI (stored by /api/wizard-metadata).
        energy_level = _wizard_state.get("energy_level") or None
        # Override satisfaction from browser UI if user set it explicitly.
        ui_satisfaction = _wizard_state.get("satisfaction")
        if ui_satisfaction is not None:
            satisfaction = ui_satisfaction

        value_domains = _wizard_state.get("selected_values") or None
        sections_count = len(_wizard_state.get("results", []))

        result = notion_upsert_daily_log(
            date_iso=ws_date,
            raw_text=raw_text,
            satisfaction=satisfaction,
            energy_level=energy_level,
            value_domains=value_domains,
            input_mode="browser_recorded_audio_wizard",
            sections_count=sections_count,
        )
        _wizard_state["notion_result"] = result
        return result

    server = _start_server(app, port=port)
    url = f"http://localhost:{port}"
    logger.info("Wizard recorder server started at %s", url)
    if not no_browser:
        _auto_open_browser(url)

    try:
        while not _wizard_state.get("done"):
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Wizard interrupted by user")
        _wizard_state["cancelled"] = True

    _stop_server(server)

    if _wizard_state.get("cancelled"):
        raise RuntimeError("Wizard recording was cancelled by user.")

    results = _wizard_state.get("results", [])
    logger.info("Wizard finished: %d sections recorded", len(results))
    return results


def get_wizard_selected_values() -> List[str]:
    """Return value domain keys selected by the user in the wizard checklist."""
    return _wizard_state.get("selected_values", [])


def get_wizard_notion_result() -> Optional[Dict[str, Any]]:
    """Return the Notion submission result from the wizard, if any."""
    return _wizard_state.get("notion_result")


def get_wizard_satisfaction() -> Optional[int]:
    """Return satisfaction score set in the browser UI, if any."""
    return _wizard_state.get("satisfaction")


def get_wizard_energy_level() -> Optional[str]:
    """Return energy level set in the browser UI, if any."""
    return _wizard_state.get("energy_level")


# ----------------------------------------------------------------
# Notion upsert for Daily Logs (used by wizard Submit button)
# ----------------------------------------------------------------

def notion_upsert_daily_log(
    *,
    date_iso: str,
    raw_text: str,
    satisfaction: Optional[int] = None,
    energy_level: Optional[str] = None,
    value_domains: Optional[List[str]] = None,
    input_mode: str = "",
    sections_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Upsert a Daily Log page to the Notion Daily Logs database.

    Uses LogDate as the idempotency key: if a page for the date exists,
    it is updated; otherwise a new page is created.

    Returns dict with ``ok``, ``page_id``, ``page_url``, ``action`` (created/updated).
    On failure, returns dict with ``ok=False`` and ``error``.
    """
    from src.notion.daily_schema import build_daily_log_properties
    from src.notion.daily_upsert import safe_truncate, upsert_daily_log

    title = f"Daily Log {date_iso}"
    props = build_daily_log_properties(
        title=title,
        date=date_iso,
        raw_close_log=safe_truncate(raw_text),
        satisfaction=satisfaction,
        value_domains=value_domains,
        energy_level=energy_level or "",
        stage="raw",
        publish_status="Draft",
    )

    return upsert_daily_log(
        date_iso=date_iso,
        properties=props,
        log_label="057_raw",
    )


# ----------------------------------------------------------------
# Server helpers
# ----------------------------------------------------------------

def _start_server(app: Any, *, port: int) -> Any:
    """Start uvicorn in a background thread and return (server, thread)."""
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return (server, thread)


def _stop_server(server_tuple: Any) -> None:
    """Shutdown a (server, thread) pair."""
    server, thread = server_tuple
    server.should_exit = True
    thread.join(timeout=5)


def _auto_open_browser(url: str) -> None:
    def _open():
        time.sleep(1.0)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def _ext_from_content_type(content_type: Optional[str]) -> str:
    """Map MIME content type to file extension."""
    if content_type:
        if "wav" in content_type:
            return ".wav"
        if "mp4" in content_type or "m4a" in content_type:
            return ".m4a"
        if "ogg" in content_type:
            return ".ogg"
    return ".webm"


# ----------------------------------------------------------------
# Notion ROS_Values_Codex integration
# ----------------------------------------------------------------

def fetch_values_from_notion() -> List[Dict[str, str]]:
    """Fetch value domains from the Notion ROS_Values_Codex DB.

    Returns a list of dicts with ``domain_key``, ``name`` (Japanese title),
    and ``definition`` for each value domain.

    Returns an empty list if Notion is unavailable or any error occurs
    (graceful fallback).
    """
    try:
        from src.config import load_env, get_db_id
        load_env()

        from src.notion.client import build_notion_client_from_env, NotionDataSourceResolver
        from src.notion.values_repo import ValuesCodexRepo

        client = build_notion_client_from_env()
        codex_db_id = get_db_id("NOTION_ROS_Values_Codex_ID")

        resolver = NotionDataSourceResolver(client=client)
        resolved = resolver.resolve_once(
            name="ROS_Values_Codex",
            database_id=codex_db_id,
        )

        repo = ValuesCodexRepo(
            client=client,
            database_id=codex_db_id,
            data_source_id=resolved.data_source_id,
        )

        # Fetch current quarter domains.
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(tz=ZoneInfo("Asia/Tokyo"))
        quarter = f"{now.year}-Q{(now.month - 1) // 3 + 1}"
        pages = repo.fetch_by_quarter(quarter)

        domains = []
        for page in pages:
            props = page.get("properties", {})
            # Extract domain_key from "Domain Key" property.
            dk_rt = props.get("Domain Key", {}).get("rich_text", [])
            domain_key = dk_rt[0]["plain_text"] if dk_rt else ""
            # Extract name from "Name" (title).
            name_rt = props.get("Name", {}).get("title", [])
            name = name_rt[0]["plain_text"] if name_rt else domain_key
            # Extract definition.
            def_rt = props.get("Value Definition", {}).get("rich_text", [])
            definition = def_rt[0]["plain_text"] if def_rt else ""

            if domain_key or name:
                domains.append({
                    "domain_key": domain_key,
                    "name": name,
                    "definition": definition,
                })

        logger.info("Fetched %d value domains from Notion (%s)", len(domains), quarter)
        return domains

    except Exception as e:
        logger.warning("Could not fetch values from Notion (fallback to keyword): %s", e)
        return []


# ----------------------------------------------------------------
# Single-clip recorder HTML
# ----------------------------------------------------------------

def _build_recorder_html(*, seconds: int, language: str) -> str:
    lang_display = {"ja": "Japanese", "en": "English"}.get(language, language)
    return f"""<!DOCTYPE html>
<html lang="{language}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daily Close — Voice Input</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --surface2: #242836;
    --border: #2e3345; --text: #e4e6ef; --text2: #9399b2;
    --accent: #7c3aed; --accent-light: #a78bfa;
    --green: #10b981; --red: #ef4444; --orange: #f59e0b;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; padding: 24px;
  }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 40px; max-width: 500px; width: 100%;
    text-align: center;
  }}
  h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 8px; }}
  .subtitle {{ color: var(--text2); font-size: 14px; margin-bottom: 28px; }}
  .timer {{
    font-size: 48px; font-weight: 700; font-variant-numeric: tabular-nums;
    margin-bottom: 24px; color: var(--text);
  }}
  .timer.recording {{ color: var(--red); }}
  .btn {{
    padding: 14px 32px; border: none; border-radius: 10px;
    font-size: 16px; font-weight: 600; cursor: pointer;
    transition: all 0.2s; margin: 6px;
  }}
  .btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .btn-record {{ background: var(--red); color: #fff; }}
  .btn-record.recording {{ animation: pulse 1.2s ease-in-out infinite; }}
  @keyframes pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }}
    50% {{ box-shadow: 0 0 0 12px rgba(239,68,68,0); }}
  }}
  .btn-done {{ background: var(--green); color: #fff; }}
  .btn-cancel {{ background: var(--surface2); color: var(--text2); }}
  .status {{ margin-top: 20px; font-size: 14px; color: var(--text2); min-height: 40px; }}
  .status.success {{ color: var(--green); }}
  .status.error {{ color: var(--red); }}
  .transcript-box {{
    background: var(--surface2); border-radius: 10px;
    padding: 16px; margin-top: 16px; font-size: 14px;
    line-height: 1.7; text-align: left; max-height: 200px;
    overflow-y: auto; display: none;
  }}
  .controls {{ margin-top: 20px; }}
</style>
</head>
<body>
<div class="card">
  <h1>Daily Close — Voice Input</h1>
  <div class="subtitle">Record your evening close log ({lang_display}) — max {seconds}s</div>
  <div class="timer" id="timer">0:{seconds:02d}</div>
  <div><button class="btn btn-record" id="rec-btn" onclick="toggleRecording()">Start Recording</button></div>
  <div class="status" id="status"></div>
  <div class="transcript-box" id="transcript-box"></div>
  <div class="controls" id="done-controls" style="display:none">
    <button class="btn btn-done" onclick="finish()">Use This Recording</button>
    <button class="btn btn-record" onclick="reRecord()">Re-record</button>
  </div>
  <div class="controls"><button class="btn btn-cancel" onclick="cancel()">Cancel</button></div>
</div>
<script>
const MAX_S={seconds};let mr=null,chunks=[],rec=false,ti=null,rem=MAX_S,has=false;
function ut(){{const m=Math.floor(rem/60),s=rem%60;document.getElementById('timer').textContent=m+':'+String(s).padStart(2,'0');}}
async function toggleRecording(){{if(!rec)await startRec();else stopRec();}}
async function startRec(){{try{{const st=await navigator.mediaDevices.getUserMedia({{audio:true}});chunks=[];const mt=MediaRecorder.isTypeSupported('audio/webm;codecs=opus')?'audio/webm;codecs=opus':'audio/webm';mr=new MediaRecorder(st,{{mimeType:mt}});mr.ondataavailable=e=>{{if(e.data.size>0)chunks.push(e.data);}};mr.onstop=()=>{{st.getTracks().forEach(t=>t.stop());upload();}};mr.start(250);rec=true;rem=MAX_S;ut();const b=document.getElementById('rec-btn');b.textContent='Stop Recording';b.classList.add('recording');document.getElementById('timer').classList.add('recording');document.getElementById('status').textContent='Recording...';document.getElementById('status').className='status';document.getElementById('done-controls').style.display='none';document.getElementById('transcript-box').style.display='none';ti=setInterval(()=>{{rem--;ut();if(rem<=0)stopRec();}},1000);}}catch(e){{document.getElementById('status').textContent='Mic error: '+e.message;document.getElementById('status').className='status error';}}}}
function stopRec(){{if(mr&&rec){{clearInterval(ti);mr.stop();rec=false;document.getElementById('rec-btn').textContent='Start Recording';document.getElementById('rec-btn').classList.remove('recording');document.getElementById('timer').classList.remove('recording');document.getElementById('status').textContent='Processing...';}}}}
async function upload(){{const b=new Blob(chunks,{{type:'audio/webm'}});const fd=new FormData();fd.append('audio',b,'recording.webm');try{{const r=await fetch('/api/upload',{{method:'POST',body:fd}});const d=await r.json();if(!r.ok||!d.ok)throw new Error(d.error||'Upload failed');has=true;document.getElementById('status').textContent='Recording saved and transcribed.';document.getElementById('status').className='status success';if(d.transcript){{const bx=document.getElementById('transcript-box');bx.textContent=d.transcript;bx.style.display='block';}}document.getElementById('done-controls').style.display='block';}}catch(e){{document.getElementById('status').textContent='Error: '+e.message;document.getElementById('status').className='status error';}}}}
function reRecord(){{has=false;document.getElementById('done-controls').style.display='none';document.getElementById('transcript-box').style.display='none';document.getElementById('status').textContent='';document.getElementById('status').className='status';rem=MAX_S;ut();}}
async function finish(){{if(!has)return;document.getElementById('status').textContent='Closing...';try{{await fetch('/api/done',{{method:'POST'}});}}catch(e){{}}document.getElementById('status').textContent='Done! You can close this tab.';document.getElementById('status').className='status success';}}
async function cancel(){{try{{await fetch('/api/cancel',{{method:'POST'}});}}catch(e){{}}document.getElementById('status').textContent='Cancelled. You can close this tab.';}}
ut();
</script>
</body>
</html>"""


# ----------------------------------------------------------------
# Wizard recorder HTML (multi-section, single page)
# ----------------------------------------------------------------

def _build_wizard_html(
    *,
    sections: List[Dict[str, Any]],
    seconds: int,
    language: str,
    out_dir: str = "",
    notion_values: Optional[List[Dict[str, str]]] = None,
) -> str:
    sections_json = json.dumps(sections, ensure_ascii=False)
    notion_values_json = json.dumps(notion_values or [], ensure_ascii=False)
    total = len(sections)
    return f"""<!DOCTYPE html>
<html lang="{language}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daily Close — Voice Wizard</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --surface2: #242836;
    --border: #2e3345; --text: #e4e6ef; --text2: #9399b2;
    --accent: #7c3aed; --accent-light: #a78bfa;
    --green: #10b981; --red: #ef4444; --orange: #f59e0b;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; padding: 24px;
  }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 40px; max-width: 560px; width: 100%;
    text-align: center;
  }}
  h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
  .progress {{ color: var(--text2); font-size: 13px; margin-bottom: 6px; }}
  .section-title {{ font-size: 28px; font-weight: 700; margin: 16px 0 4px; }}
  .section-instr {{ color: var(--text2); font-size: 14px; margin-bottom: 20px; line-height: 1.6; }}
  .timer {{
    font-size: 48px; font-weight: 700; font-variant-numeric: tabular-nums;
    margin-bottom: 20px; color: var(--text);
  }}
  .timer.recording {{ color: var(--red); }}
  .btn {{
    padding: 12px 28px; border: none; border-radius: 10px;
    font-size: 15px; font-weight: 600; cursor: pointer;
    transition: all 0.2s; margin: 5px;
  }}
  .btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .btn-record {{ background: var(--red); color: #fff; }}
  .btn-record.recording {{ animation: pulse 1.2s ease-in-out infinite; }}
  @keyframes pulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }}
    50% {{ box-shadow: 0 0 0 12px rgba(239,68,68,0); }}
  }}
  .btn-next {{ background: var(--green); color: #fff; }}
  .btn-skip {{ background: var(--surface2); color: var(--text2); }}
  .btn-cancel {{ background: var(--surface2); color: var(--text2); font-size: 13px; padding: 8px 20px; }}
  .btn-finish {{ background: var(--accent); color: #fff; font-size: 18px; padding: 16px 40px; }}
  .btn-finish:hover {{ background: #6d28d9; }}
  .status {{ margin-top: 16px; font-size: 14px; color: var(--text2); min-height: 36px; }}
  .status.success {{ color: var(--green); }}
  .status.error {{ color: var(--red); }}
  .transcript-box {{
    background: var(--surface2); border-radius: 10px;
    padding: 14px; margin-top: 12px; font-size: 13px;
    line-height: 1.7; text-align: left; max-height: 160px;
    overflow-y: auto; display: none;
  }}
  .controls {{ margin-top: 16px; }}
  .step-dots {{ display: flex; justify-content: center; gap: 8px; margin: 12px 0 20px; }}
  .dot {{
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--surface2); transition: background 0.3s;
  }}
  .dot.active {{ background: var(--accent); }}
  .dot.done {{ background: var(--green); }}
  .dot.skipped {{ background: var(--orange); }}
  .done-view {{ display: none; }}
  .confirm-view {{ display: none; }}
  .summary-list {{ text-align: left; margin: 16px 0; }}
  .summary-item {{ margin-bottom: 10px; padding: 10px; background: var(--surface2); border-radius: 8px; }}
  .summary-item .label {{ color: var(--text2); font-size: 12px; }}
  .summary-item .text {{ font-size: 13px; margin-top: 4px; line-height: 1.5; }}
  .summary-item .text.skip {{ color: var(--orange); font-style: italic; }}
  /* Values checklist styles */
  .values-checklist {{ text-align: left; margin: 16px 0; max-height: 300px; overflow-y: auto; }}
  .value-item {{
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 12px; margin-bottom: 6px;
    background: var(--surface2); border-radius: 8px;
    cursor: pointer; transition: background 0.2s;
  }}
  .value-item:hover {{ background: var(--border); }}
  .value-item.selected {{ background: rgba(124, 58, 237, 0.15); border: 1px solid var(--accent); }}
  .value-item input[type="checkbox"] {{
    margin-top: 3px; accent-color: var(--accent);
    width: 18px; height: 18px; flex-shrink: 0;
  }}
  .value-item .value-name {{ font-weight: 600; font-size: 14px; }}
  .value-item .value-def {{ color: var(--text2); font-size: 12px; margin-top: 2px; line-height: 1.4; }}
  .values-fallback-msg {{ color: var(--text2); font-size: 13px; margin: 12px 0; font-style: italic; }}
  /* Satisfaction / Energy selector buttons */
  .sat-btn, .energy-btn {{
    min-width: 48px; padding: 10px 16px;
    background: var(--surface2); color: var(--text);
    border: 2px solid transparent; border-radius: 8px;
    font-size: 16px; font-weight: 600; cursor: pointer;
    transition: all 0.2s;
  }}
  .sat-btn:hover, .energy-btn:hover {{ background: var(--border); }}
  .sat-btn.selected {{ background: rgba(124, 58, 237, 0.2); border-color: var(--accent); color: var(--accent); }}
  .energy-btn.selected {{ background: rgba(124, 58, 237, 0.2); border-color: var(--accent); color: var(--accent); }}
  /* Confirmation view styles */
  .confirm-icon {{ font-size: 64px; margin: 16px 0; }}
  .confirm-path {{
    background: var(--surface2); border-radius: 8px;
    padding: 12px 16px; font-size: 13px; font-family: monospace;
    color: var(--accent-light); word-break: break-all; margin: 12px 0;
    text-align: left;
  }}
  .auto-close-msg {{ color: var(--text2); font-size: 13px; margin-top: 12px; }}
</style>
</head>
<body>
<div class="card">
  <!-- Recording view -->
  <div id="rec-view">
    <h1>Daily Close \u2014 Voice Wizard</h1>
    <div class="progress" id="progress">Section 1 / {total}</div>
    <div class="step-dots" id="dots"></div>
    <div class="section-title" id="sec-title"></div>
    <div class="section-instr" id="sec-instr"></div>
    <!-- Standard recording UI -->
    <div id="rec-ui">
      <div class="timer" id="timer">0:00</div>
      <div>
        <button class="btn btn-record" id="rec-btn" onclick="toggleRec()">Start Recording</button>
      </div>
      <div class="status" id="status"></div>
      <div class="transcript-box" id="transcript-box"></div>
      <div class="controls" id="after-rec" style="display:none">
        <button class="btn btn-next" onclick="acceptAndNext()">Next Section</button>
        <button class="btn btn-record" onclick="reRecord()" style="font-size:13px;padding:10px 20px">Re-record</button>
      </div>
    </div>
    <!-- Values checklist UI (shown for "values" section when Notion data available) -->
    <div id="values-ui" style="display:none">
      <div class="values-checklist" id="values-checklist"></div>
      <div class="values-fallback-msg" id="values-fallback-msg" style="display:none">
        Notion\u304b\u3089\u306e\u4fa1\u5024\u9818\u57df\u306e\u53d6\u5f97\u306b\u5931\u6557\u3057\u307e\u3057\u305f\u3002\u97f3\u58f0\u3067\u4fa1\u5024\u9818\u57df\u3092\u8a71\u3057\u3066\u304f\u3060\u3055\u3044\u3002
      </div>
      <div class="controls" id="values-confirm" style="display:none">
        <button class="btn btn-next" onclick="submitValuesAndNext()">Next Section</button>
      </div>
    </div>
    <div class="controls" id="skip-controls">
      <button class="btn btn-skip" onclick="skipSection()">Skip this section</button>
    </div>
    <div class="controls" style="margin-top:24px">
      <button class="btn btn-cancel" onclick="cancelAll()">Cancel all</button>
    </div>
  </div>
  <!-- Done / Summary view -->
  <div id="done-view" class="done-view">
    <h1>\u5168\u30bb\u30af\u30b7\u30e7\u30f3\u5b8c\u4e86</h1>
    <p style="color:var(--text2);font-size:14px;margin:8px 0 16px;">\u5185\u5bb9\u3092\u78ba\u8a8d\u3057\u3001Satisfaction\u3068Energy\u3092\u9078\u3093\u3067\u300cSubmit to Notion\u300d\u3092\u62bc\u3057\u3066\u304f\u3060\u3055\u3044</p>
    <div class="summary-list" id="summary"></div>

    <!-- Satisfaction selector -->
    <div class="metadata-input" style="margin:20px 0 12px;text-align:left;">
      <label style="display:block;font-weight:600;font-size:14px;margin-bottom:8px;color:var(--text);">Satisfaction (1\u20135)</label>
      <div id="sat-buttons" style="display:flex;gap:8px;justify-content:center;">
        <button class="btn sat-btn" data-val="1" onclick="selectSatisfaction(1)">1</button>
        <button class="btn sat-btn" data-val="2" onclick="selectSatisfaction(2)">2</button>
        <button class="btn sat-btn" data-val="3" onclick="selectSatisfaction(3)">3</button>
        <button class="btn sat-btn" data-val="4" onclick="selectSatisfaction(4)">4</button>
        <button class="btn sat-btn" data-val="5" onclick="selectSatisfaction(5)">5</button>
      </div>
    </div>

    <!-- Energy Level selector -->
    <div class="metadata-input" style="margin:12px 0 20px;text-align:left;">
      <label style="display:block;font-weight:600;font-size:14px;margin-bottom:8px;color:var(--text);">Energy Level</label>
      <div id="energy-buttons" style="display:flex;gap:8px;justify-content:center;">
        <button class="btn energy-btn" data-val="Low" onclick="selectEnergy('Low')">Low</button>
        <button class="btn energy-btn" data-val="Medium" onclick="selectEnergy('Medium')">Medium</button>
        <button class="btn energy-btn" data-val="High" onclick="selectEnergy('High')">High</button>
      </div>
    </div>

    <button class="btn btn-finish" id="finish-btn" onclick="finishAll()">Submit to Notion</button>
    <div class="controls" style="margin-top:12px">
      <button class="btn btn-skip" id="save-only-btn" onclick="saveOnly()">\u4fdd\u5b58\u306e\u307f\uff08Notion\u306a\u3057\uff09</button>
      <button class="btn btn-cancel" onclick="cancelAll()">Cancel</button>
    </div>
    <div class="status" id="done-status"></div>
  </div>
  <!-- Confirmation view (after save) -->
  <div id="confirm-view" class="confirm-view">
    <div class="confirm-icon" id="confirm-icon">\u2705</div>
    <h1 id="confirm-title">\u4fdd\u5b58\u5b8c\u4e86\uff01</h1>
    <p style="color:var(--text2);font-size:14px;margin:8px 0 4px;" id="confirm-subtitle">\u4ee5\u4e0b\u306e\u30c7\u30a3\u30ec\u30af\u30c8\u30ea\u306b\u4fdd\u5b58\u3055\u308c\u307e\u3057\u305f\uff1a</p>
    <div class="confirm-path" id="confirm-path">{out_dir}</div>
    <!-- Notion result section -->
    <div id="notion-result" style="display:none;margin:12px 0;">
      <div id="notion-success" style="display:none;">
        <p style="color:var(--green);font-weight:600;font-size:15px;" id="notion-action-msg"></p>
        <a id="notion-url" href="#" target="_blank"
           style="display:inline-block;margin:8px 0;padding:10px 20px;background:var(--accent);color:#fff;border-radius:8px;text-decoration:none;font-weight:600;">
          Open in Notion
        </a>
        <p style="color:var(--text2);font-size:12px;margin-top:4px;">Page ID: <code id="notion-page-id" style="color:var(--accent-light);"></code></p>
      </div>
      <div id="notion-error" style="display:none;">
        <p style="color:var(--red);font-weight:600;">Notion submission failed</p>
        <div class="confirm-path" id="notion-error-msg" style="color:var(--red);"></div>
      </div>
    </div>
    <p class="auto-close-msg" id="auto-close-msg">\u3053\u306e\u30bf\u30d6\u306f <span id="countdown">5</span> \u79d2\u5f8c\u306b\u81ea\u52d5\u7684\u306b\u9589\u3058\u308b\u3053\u3068\u304c\u3067\u304d\u307e\u3059</p>
    <div class="controls" style="margin-top:16px">
      <button class="btn btn-skip" onclick="window.close()">Close now</button>
    </div>
  </div>
</div>
<script>
const SECTIONS = {sections_json};
const NOTION_VALUES = {notion_values_json};
const MAX_S = {seconds};
const TOTAL = SECTIONS.length;
const OUT_DIR = {json.dumps(out_dir)};
let cur = 0;
let mr = null, chunks = [], isRec = false, ti = null, rem = MAX_S;
let sectionHasRec = false;
let results = [];
let wizardFinished = false;
let selectedSatisfaction = null;
let selectedEnergy = null;

/* ── Satisfaction / Energy selectors ── */
function selectSatisfaction(val) {{
  selectedSatisfaction = val;
  document.querySelectorAll('.sat-btn').forEach(b => {{
    b.classList.toggle('selected', parseInt(b.dataset.val) === val);
  }});
}}

function selectEnergy(val) {{
  selectedEnergy = val;
  document.querySelectorAll('.energy-btn').forEach(b => {{
    b.classList.toggle('selected', b.dataset.val === val);
  }});
}}

/* ── beforeunload guard ── */
window.addEventListener('beforeunload', function(e) {{
  if (!wizardFinished) {{
    e.preventDefault();
    e.returnValue = '\u9014\u4e2d\u7d42\u4e86\u3059\u308b\u3068\u9332\u97f3\u304c\u5931\u308f\u308c\u307e\u3059\u3002\u672c\u5f53\u306b\u9589\u3058\u307e\u3059\u304b\uff1f';
    return e.returnValue;
  }}
}});

function initDots() {{
  const c = document.getElementById('dots');
  c.innerHTML = '';
  for (let i = 0; i < TOTAL; i++) {{
    const d = document.createElement('div');
    d.className = 'dot';
    d.id = 'dot-' + i;
    c.appendChild(d);
  }}
}}

function isValuesSection() {{
  return SECTIONS[cur] && SECTIONS[cur].key === 'values' && NOTION_VALUES.length > 0;
}}

function showSection() {{
  const sec = SECTIONS[cur];
  document.getElementById('progress').textContent = 'Section ' + (cur + 1) + ' / ' + TOTAL;
  document.getElementById('sec-title').textContent = sec.title_ja;
  document.getElementById('sec-instr').textContent = sec.instruction || '';
  document.getElementById('dots').querySelector('.active')?.classList.remove('active');
  document.getElementById('dot-' + cur).classList.add('active');
  rem = MAX_S;
  updateTimer();
  sectionHasRec = false;
  document.getElementById('status').textContent = '';
  document.getElementById('status').className = 'status';
  document.getElementById('transcript-box').style.display = 'none';
  document.getElementById('after-rec').style.display = 'none';
  document.getElementById('skip-controls').style.display = '';

  /* Show values checklist or recording UI */
  if (isValuesSection()) {{
    document.getElementById('rec-ui').style.display = 'none';
    document.getElementById('values-ui').style.display = '';
    buildValuesChecklist();
  }} else {{
    document.getElementById('rec-ui').style.display = '';
    document.getElementById('values-ui').style.display = 'none';
  }}
}}

/* ── Values checklist ── */
function buildValuesChecklist() {{
  const container = document.getElementById('values-checklist');
  container.innerHTML = '';
  if (NOTION_VALUES.length === 0) {{
    document.getElementById('values-fallback-msg').style.display = '';
    document.getElementById('rec-ui').style.display = '';
    document.getElementById('values-ui').style.display = 'none';
    return;
  }}
  document.getElementById('values-confirm').style.display = '';
  NOTION_VALUES.forEach((v, idx) => {{
    const item = document.createElement('label');
    item.className = 'value-item';
    item.innerHTML = '<input type="checkbox" data-key="' + (v.domain_key || v.name) + '" />'
      + '<div><div class="value-name">' + (v.name || v.domain_key) + '</div>'
      + (v.definition ? '<div class="value-def">' + v.definition + '</div>' : '')
      + '</div>';
    const cb = item.querySelector('input');
    cb.addEventListener('change', () => {{
      item.classList.toggle('selected', cb.checked);
      updateValuesButton();
    }});
    container.appendChild(item);
  }});
}}

function updateValuesButton() {{
  const checked = document.querySelectorAll('#values-checklist input:checked');
  const btn = document.querySelector('#values-confirm .btn-next');
  const count = checked.length;
  if (cur === TOTAL - 1) {{
    btn.textContent = count > 0 ? 'Finish (' + count + ' selected)' : 'Finish';
  }} else {{
    btn.textContent = count > 0 ? 'Next Section (' + count + ' selected)' : 'Next Section';
  }}
}}

async function submitValuesAndNext() {{
  const checked = document.querySelectorAll('#values-checklist input:checked');
  const selected = Array.from(checked).map(cb => cb.dataset.key);
  results[cur] = {{
    key: SECTIONS[cur].key,
    title: SECTIONS[cur].title_ja,
    transcript: selected.length > 0 ? selected.join(', ') : '',
    skipped: false,
    value_selection: selected,
  }};
  /* Send to server */
  try {{
    const fd = new FormData();
    fd.append('selected_json', JSON.stringify(selected));
    await fetch('/api/wizard-value-selection', {{ method: 'POST', body: fd }});
  }} catch(e) {{ console.warn('Value selection send error:', e); }}
  /* Also skip the audio upload for this section — mark as done server-side */
  try {{
    const fd = new FormData();
    fd.append('section_key', SECTIONS[cur].key);
    fd.append('section_index', String(cur));
    await fetch('/api/wizard-skip', {{ method: 'POST', body: fd }});
  }} catch(e) {{}}
  document.getElementById('dot-' + cur).classList.add('done');
  advanceSection();
}}

function updateTimer() {{
  const m = Math.floor(rem / 60), s = rem % 60;
  document.getElementById('timer').textContent = m + ':' + String(s).padStart(2, '0');
}}

async function toggleRec() {{
  if (!isRec) await startRec(); else stopRec();
}}

async function startRec() {{
  try {{
    const st = await navigator.mediaDevices.getUserMedia({{ audio: true }});
    chunks = [];
    const mt = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
    mr = new MediaRecorder(st, {{ mimeType: mt }});
    mr.ondataavailable = e => {{ if (e.data.size > 0) chunks.push(e.data); }};
    mr.onstop = () => {{ st.getTracks().forEach(t => t.stop()); uploadSection(); }};
    mr.start(250);
    isRec = true; rem = MAX_S; updateTimer();
    const b = document.getElementById('rec-btn');
    b.textContent = 'Stop Recording'; b.classList.add('recording');
    document.getElementById('timer').classList.add('recording');
    document.getElementById('status').textContent = 'Recording...';
    document.getElementById('status').className = 'status';
    document.getElementById('after-rec').style.display = 'none';
    document.getElementById('skip-controls').style.display = 'none';
    document.getElementById('transcript-box').style.display = 'none';
    ti = setInterval(() => {{ rem--; updateTimer(); if (rem <= 0) stopRec(); }}, 1000);
  }} catch (e) {{
    document.getElementById('status').textContent = 'Mic error: ' + e.message;
    document.getElementById('status').className = 'status error';
  }}
}}

function stopRec() {{
  if (mr && isRec) {{
    clearInterval(ti); mr.stop(); isRec = false;
    document.getElementById('rec-btn').textContent = 'Start Recording';
    document.getElementById('rec-btn').classList.remove('recording');
    document.getElementById('timer').classList.remove('recording');
    document.getElementById('status').textContent = 'Processing...';
  }}
}}

async function uploadSection() {{
  const sec = SECTIONS[cur];
  const blob = new Blob(chunks, {{ type: 'audio/webm' }});
  const fd = new FormData();
  fd.append('audio', blob, 'recording.webm');
  fd.append('section_key', sec.key);
  fd.append('section_index', String(cur));
  try {{
    const r = await fetch('/api/wizard-upload', {{ method: 'POST', body: fd }});
    const d = await r.json();
    if (!r.ok || !d.ok) throw new Error(d.error || 'Upload failed');
    sectionHasRec = true;
    results[cur] = {{ key: sec.key, title: sec.title_ja, transcript: d.transcript, skipped: false }};
    document.getElementById('status').textContent = 'Transcribed.';
    document.getElementById('status').className = 'status success';
    if (d.transcript) {{
      const bx = document.getElementById('transcript-box');
      bx.textContent = d.transcript; bx.style.display = 'block';
    }}
    document.getElementById('after-rec').style.display = '';
    document.getElementById('skip-controls').style.display = 'none';
    if (cur === TOTAL - 1) {{
      document.querySelector('#after-rec .btn-next').textContent = 'Finish';
    }}
  }} catch (e) {{
    document.getElementById('status').textContent = 'Error: ' + e.message;
    document.getElementById('status').className = 'status error';
    document.getElementById('skip-controls').style.display = '';
  }}
}}

function reRecord() {{
  sectionHasRec = false;
  document.getElementById('after-rec').style.display = 'none';
  document.getElementById('skip-controls').style.display = '';
  document.getElementById('transcript-box').style.display = 'none';
  document.getElementById('status').textContent = '';
  document.getElementById('status').className = 'status';
  rem = MAX_S; updateTimer();
}}

async function skipSection() {{
  const sec = SECTIONS[cur];
  const fd = new FormData();
  fd.append('section_key', sec.key);
  fd.append('section_index', String(cur));
  try {{ await fetch('/api/wizard-skip', {{ method: 'POST', body: fd }}); }} catch(e) {{}}
  results[cur] = {{ key: sec.key, title: sec.title_ja, transcript: '', skipped: true }};
  document.getElementById('dot-' + cur).classList.add('skipped');
  advanceSection();
}}

function acceptAndNext() {{
  document.getElementById('dot-' + cur).classList.add('done');
  advanceSection();
}}

function advanceSection() {{
  cur++;
  if (cur >= TOTAL) {{ showDoneView(); return; }}
  showSection();
}}

function showDoneView() {{
  document.getElementById('rec-view').style.display = 'none';
  document.getElementById('done-view').style.display = 'block';
  const sm = document.getElementById('summary');
  sm.innerHTML = '';
  for (let i = 0; i < TOTAL; i++) {{
    const r = results[i] || {{ title: SECTIONS[i].title_ja, transcript: '', skipped: true }};
    const div = document.createElement('div');
    div.className = 'summary-item';
    const lbl = document.createElement('div');
    lbl.className = 'label'; lbl.textContent = (i+1) + '. ' + r.title;
    div.appendChild(lbl);
    const txt = document.createElement('div');
    if (r.skipped) {{ txt.className = 'text skip'; txt.textContent = '(skipped)'; }}
    else if (r.value_selection && r.value_selection.length > 0) {{
      txt.className = 'text'; txt.textContent = '\u2705 ' + r.value_selection.join(', ');
    }}
    else {{ txt.className = 'text'; txt.textContent = r.transcript || '(empty)'; }}
    div.appendChild(txt);
    sm.appendChild(div);
  }}
}}

async function finishAll() {{
  /* Validate required fields */
  if (selectedSatisfaction === null) {{
    document.getElementById('done-status').textContent = 'Satisfaction\u3092\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044 (1\u20135)';
    document.getElementById('done-status').className = 'status error';
    return;
  }}
  if (selectedEnergy === null) {{
    document.getElementById('done-status').textContent = 'Energy Level\u3092\u9078\u629e\u3057\u3066\u304f\u3060\u3055\u3044';
    document.getElementById('done-status').className = 'status error';
    return;
  }}

  const btn = document.getElementById('finish-btn');
  btn.disabled = true;
  btn.textContent = 'Submitting to Notion...';
  document.getElementById('done-status').textContent = 'Notion\u306b\u9001\u4fe1\u4e2d...';
  document.getElementById('done-status').className = 'status';

  /* 0. Send satisfaction + energy to server */
  try {{
    const fd = new FormData();
    fd.append('satisfaction', String(selectedSatisfaction));
    fd.append('energy_level', selectedEnergy);
    await fetch('/api/wizard-metadata', {{ method: 'POST', body: fd }});
  }} catch(e) {{ /* non-fatal */ }}

  /* 1. Submit to Notion */
  let notionResult = null;
  try {{
    const r = await fetch('/api/notion-submit', {{ method: 'POST' }});
    notionResult = await r.json();
  }} catch(e) {{
    notionResult = {{ ok: false, error: e.message }};
  }}

  /* 2. Signal wizard done (saves locally) */
  try {{ await fetch('/api/wizard-done', {{ method: 'POST' }}); }} catch(e) {{}}
  wizardFinished = true;

  /* 3. Transition to confirmation view */
  document.getElementById('done-view').style.display = 'none';
  document.getElementById('confirm-view').style.display = 'block';

  /* 4. Show Notion result */
  document.getElementById('notion-result').style.display = 'block';
  if (notionResult && notionResult.ok) {{
    document.getElementById('notion-success').style.display = 'block';
    const action = notionResult.action === 'updated' ? 'Updated' : 'Created';
    document.getElementById('notion-action-msg').textContent =
      '\u2705 Notion Daily Log ' + action + ' (' + notionResult.date + ')';
    document.getElementById('notion-url').href = notionResult.page_url || '#';
    document.getElementById('notion-page-id').textContent = notionResult.page_id || '';
    document.getElementById('confirm-title').textContent = 'Notion\u9001\u4fe1\u5b8c\u4e86\uff01';
    document.getElementById('confirm-icon').textContent = '\u2705';
  }} else {{
    document.getElementById('notion-error').style.display = 'block';
    document.getElementById('notion-error-msg').textContent =
      (notionResult && notionResult.error) || 'Unknown error';
    document.getElementById('confirm-title').textContent = 'Notion\u9001\u4fe1\u5931\u6557';
    document.getElementById('confirm-icon').textContent = '\u274c';
  }}

  /* 5. Auto-close countdown */
  let remaining = 8;
  const cdEl = document.getElementById('countdown');
  cdEl.textContent = String(remaining);
  const cdTimer = setInterval(() => {{
    remaining--;
    cdEl.textContent = String(remaining);
    if (remaining <= 0) {{
      clearInterval(cdTimer);
      try {{ window.close(); }} catch(e) {{}}
    }}
  }}, 1000);
}}

async function saveOnly() {{
  const btn = document.getElementById('save-only-btn');
  btn.disabled = true;
  btn.textContent = '\u4fdd\u5b58\u4e2d...';
  try {{ await fetch('/api/wizard-done', {{ method: 'POST' }}); }} catch(e) {{}}
  wizardFinished = true;
  document.getElementById('done-view').style.display = 'none';
  document.getElementById('confirm-view').style.display = 'block';
  document.getElementById('confirm-subtitle').textContent =
    '\u30ed\u30fc\u30ab\u30eb\u306b\u4fdd\u5b58\u3055\u308c\u307e\u3057\u305f\uff08Notion\u306b\u306f\u9001\u4fe1\u3055\u308c\u3066\u3044\u307e\u305b\u3093\uff09';
  let remaining = 5;
  const cdEl = document.getElementById('countdown');
  const cdTimer = setInterval(() => {{
    remaining--;
    cdEl.textContent = String(remaining);
    if (remaining <= 0) {{
      clearInterval(cdTimer);
      try {{ window.close(); }} catch(e) {{}}
    }}
  }}, 1000);
}}

async function cancelAll() {{
  if (!confirm('\u9014\u4e2d\u7d42\u4e86\u3059\u308b\u3068\u9332\u97f3\u304c\u5931\u308f\u308c\u307e\u3059\u3002\u672c\u5f53\u306b\u30ad\u30e3\u30f3\u30bb\u30eb\u3057\u307e\u3059\u304b\uff1f')) return;
  wizardFinished = true;
  try {{ await fetch('/api/wizard-cancel', {{ method: 'POST' }}); }} catch(e) {{}}
  document.getElementById('status').textContent = 'Cancelled. You can close this tab.';
}}

initDots();
showSection();
</script>
</body>
</html>"""
