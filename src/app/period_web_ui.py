# src/app/period_web_ui.py
"""Shared FastAPI Web UI factory for period planning/review scripts (062-065).

Provides a declarative configuration approach: each script defines a
``PeriodWebUIConfig`` describing its sections, Notion field mappings,
LLM prompt, and upsert functions.  The factory then builds a complete
FastAPI SPA with voice-wizard input, Notion display, and save-back.

Architecture
------------
``build_period_web_app(config)`` → FastAPI app with routes:

    GET  /                   HTML SPA
    GET  /api/period-info    Resolve period metadata
    GET  /api/existing       Query existing Notion log content
    GET  /api/planning-context  (reviews only) Planning reference data
    POST /api/audio-upload   Upload webm + transcribe via Whisper
    POST /api/submit         Submit wizard sections → LLM → Notion upsert
    POST /api/submit-text    Direct text submit (--text equivalent)

``run_period_server(config, ...)`` launches uvicorn + auto-opens browser.

NOTE: Do NOT use ``from __future__ import annotations`` here.
FastAPI + Pydantic v2 need runtime type objects for endpoint parameters.
"""

import json
import logging
import os
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")


# ── Dataclasses ───────────────────────────────────────────────────


@dataclass
class SectionDef:
    """One wizard section for the Web UI."""

    key: str            # e.g. "big_3"
    title_ja: str       # e.g. "Big 3 (今週の3大目標)"
    title_en: str       # e.g. "Big 3 goals for this week"
    input_type: str = "textarea"   # "textarea" | "number" | "checkbox"
    min_value: Optional[int] = None
    max_value: Optional[int] = None


@dataclass
class FieldMapping:
    """Maps a wizard section → Notion property → LLM JSON key → builder kwarg."""

    section_key: str        # must match a SectionDef.key
    notion_property: str    # Notion property name, e.g. "Big 3"
    llm_json_key: str       # LLM response key, e.g. "big_3"
    property_kwarg: str     # kwarg name for build_*_properties(), e.g. "big_3"
    is_primary: bool = False  # if True, raw_text falls back to this field


@dataclass
class PeriodWebUIConfig:
    """Declarative configuration for a period planning/review Web UI."""

    # Identity
    script_id: str      # e.g. "062"
    script_name: str    # e.g. "062_weekly_intent_planning"
    title: str          # e.g. "Weekly Planning"
    port: int           # e.g. 8062

    # Period
    period_type: str    # "Weekly" or "Monthly"
    log_type: str       # "Planning" or "Review"

    # Notion
    db_env_name: str    # "NOTION_WEEKLY_LOG_ID" or "NOTION_MONTHLY_LOG_ID"
    resolver_name: str  # "weekly_log" or "monthly_log"
    log_label: str      # e.g. "062_weekly_planning"

    # UI sections (wizard textareas)
    sections: List[SectionDef] = field(default_factory=list)

    # Field mappings (section → Notion → LLM → builder kwarg)
    field_mappings: List[FieldMapping] = field(default_factory=list)

    # Score/numeric/checkbox fields outside wizard sections
    score_fields: List[SectionDef] = field(default_factory=list)

    # For review scripts: show Planning context as reference
    has_planning_context: bool = False
    planning_context_fields: List[str] = field(default_factory=list)

    # LLM prompt
    llm_system_role: str = ""
    llm_json_schema: str = ""

    # Notion write callables
    build_properties_fn: Optional[Callable] = None
    upsert_fn: Optional[Callable] = None

    # Hooks (each receives (period_dict, notion_result_dict))
    pre_upsert_hooks: List[Callable] = field(default_factory=list)
    post_upsert_hooks: List[Callable] = field(default_factory=list)

    # Title template for the Notion page
    title_template: str = "{log_type} {period_name}"


# ── Internal helpers ──────────────────────────────────────────────


def _resolve_period(config: PeriodWebUIConfig, date_iso: str) -> Dict[str, Any]:
    """Resolve the Notion period page."""
    from src.notion.period_upsert import resolve_period

    return resolve_period(
        date_iso=date_iso,
        period_type=config.period_type,
        log_label=config.script_id,
    )


def _query_existing(config: PeriodWebUIConfig, period_id: str) -> Optional[Dict[str, Any]]:
    """Query existing Notion log for the period."""
    from src.notion.period_upsert import query_existing_log

    return query_existing_log(
        db_env_name=config.db_env_name,
        resolver_name=config.resolver_name,
        period_id=period_id,
        log_type=config.log_type,
        log_label=f"{config.script_id}_query",
    )


def _extract_page_fields(
    config: PeriodWebUIConfig,
    page: Optional[Dict[str, Any]],
    field_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Extract Notion properties from a page into a flat dict."""
    if not page:
        return {}

    from src.notion.period_upsert import extract_rich_text, extract_title_text

    props = page.get("properties", {})
    result: Dict[str, Any] = {}

    # Always extract title
    result["title"] = extract_title_text(props.get("Title", {}))

    # Determine which fields to extract
    names_to_extract = field_names or []
    if not names_to_extract:
        # Auto-extract from field_mappings + score_fields
        for fm in config.field_mappings:
            names_to_extract.append(fm.notion_property)
        for sf in config.score_fields:
            names_to_extract.append(sf.title_en.split(" (")[0] if "(" in sf.title_en else sf.key.replace("_", " ").title())

    for name in names_to_extract:
        prop = props.get(name, {})
        prop_type = prop.get("type", "")

        if prop_type == "rich_text":
            result[name] = extract_rich_text(prop)
        elif prop_type == "number":
            result[name] = prop.get("number")
        elif prop_type == "checkbox":
            result[name] = prop.get("checkbox")
        elif prop_type == "select":
            sel = prop.get("select")
            result[name] = sel.get("name", "") if sel else ""
        else:
            result[name] = extract_rich_text(prop)

    # Also extract common fields
    voice_transcript = extract_rich_text(props.get("Voice Transcript", {}))
    if voice_transcript:
        result["Voice Transcript"] = f"({len(voice_transcript)} chars)"
    llm_summary = extract_rich_text(props.get("LLM Summary", {}))
    if llm_summary:
        result["LLM Summary"] = llm_summary[:300]

    return result


def _summarize_with_llm(
    config: PeriodWebUIConfig,
    transcript: str,
    date_iso: str,
    period_name: str,
) -> Dict[str, Any]:
    """Use LLM to extract structured fields from transcript."""
    try:
        from src.llm.router import build_router_from_env, TASK_REASONING

        router = build_router_from_env()
        system = (
            f"{config.llm_system_role}\n"
            f"Extract structured fields from the user's voice transcript. "
            f"Return valid JSON with:\n{config.llm_json_schema}\n"
            f"Keep the original language. Be concise but complete."
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
            return parsed
    except Exception as e:
        logger.warning("LLM summarization failed: %s", e)

    return {}


def _run_submit_pipeline(
    config: PeriodWebUIConfig,
    raw_text: str,
    section_texts: Dict[str, str],
    score_values: Dict[str, Any],
    date_iso: str,
    no_llm: bool = False,
) -> Dict[str, Any]:
    """Run the full submit pipeline: LLM → build props → upsert → hooks."""
    from src.notion.period_upsert import resolve_period, safe_truncate

    # 1. Resolve period
    period = _resolve_period(config, date_iso)
    if not period.get("ok"):
        return {"ok": False, "error": f"Period resolution failed: {period.get('error')}"}

    period_id = period["page_id"]
    period_name = period["name"]

    # 2. Run pre-upsert hooks (e.g. monthly cross-linking)
    for hook in config.pre_upsert_hooks:
        try:
            hook(period, {})
        except Exception as e:
            logger.warning("Pre-upsert hook failed: %s", e)

    # 3. LLM summarization
    llm_fields: Dict[str, Any] = {}
    if raw_text and not no_llm:
        llm_fields = _summarize_with_llm(config, raw_text, date_iso, period_name)
        logger.info("LLM fields: %s", list(llm_fields.keys()))

    # 4. Build property kwargs
    title = config.title_template.format(
        log_type=config.log_type,
        period_name=period_name,
    )
    # Start with common kwargs
    prop_kwargs: Dict[str, Any] = {
        "title": title,
        "period_id": period_id,
        "log_type": config.log_type,
        "voice_transcript": safe_truncate(raw_text),
    }
    if llm_fields:
        prop_kwargs["llm_summary"] = safe_truncate(
            json.dumps(llm_fields, ensure_ascii=False)
        )

    # Map each field from LLM output or raw section text
    for fm in config.field_mappings:
        if llm_fields and fm.llm_json_key in llm_fields:
            value = llm_fields[fm.llm_json_key]
        elif fm.is_primary:
            value = raw_text
        else:
            value = section_texts.get(fm.section_key, "")

        if isinstance(value, str):
            value = safe_truncate(value)
        if value:
            prop_kwargs[fm.property_kwarg] = value

    # Map score fields
    for sf in config.score_fields:
        val = score_values.get(sf.key)
        if val is not None:
            # Also check LLM for this field
            if llm_fields and sf.key in llm_fields:
                val = llm_fields[sf.key]
            if sf.input_type == "number" and val is not None:
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    val = None
            elif sf.input_type == "checkbox":
                val = bool(val)
            if val is not None:
                prop_kwargs[sf.key] = val

    # 5. Build Notion properties
    props = config.build_properties_fn(**prop_kwargs)

    # 6. Upsert to Notion
    notion_result = config.upsert_fn(
        period_id=period_id,
        log_type=config.log_type,
        properties=props,
        log_label=config.log_label,
    )

    # 7. Run post-upsert hooks
    for hook in config.post_upsert_hooks:
        try:
            hook(period, notion_result)
        except Exception as e:
            logger.warning("Post-upsert hook failed: %s", e)

    return {
        "ok": notion_result.get("ok", False),
        "period": period_name,
        "date": date_iso,
        "input_chars": len(raw_text),
        "llm_used": bool(llm_fields),
        "notion": notion_result,
    }


# ── FastAPI App Factory ───────────────────────────────────────────


def build_period_web_app(config: PeriodWebUIConfig):
    """Build a FastAPI app for a period planning/review Web UI."""
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title=f"{config.script_id} {config.title}")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _build_period_html(config)

    @app.get("/api/period-info")
    async def period_info(date: str = ""):
        date_iso = date or datetime.now(JST).strftime("%Y-%m-%d")
        period = _resolve_period(config, date_iso)
        if not period.get("ok"):
            return JSONResponse(
                {"ok": False, "error": period.get("error", "unknown")},
                status_code=500,
            )
        return {
            "ok": True,
            "date": date_iso,
            "period_name": period["name"],
            "period_type": config.period_type,
            "start_date": period.get("start_date", ""),
            "end_date": period.get("end_date", ""),
            "period_id": period["page_id"],
        }

    @app.get("/api/existing")
    async def get_existing(date: str = ""):
        date_iso = date or datetime.now(JST).strftime("%Y-%m-%d")
        period = _resolve_period(config, date_iso)
        if not period.get("ok"):
            return {"ok": False, "fields": {}}
        page = _query_existing(config, period["page_id"])
        fields = _extract_page_fields(config, page)
        return {
            "ok": True,
            "has_data": bool(page),
            "fields": fields,
            "period_name": period["name"],
        }

    if config.has_planning_context:
        @app.get("/api/planning-context")
        async def get_planning_context(date: str = ""):
            from src.notion.period_upsert import query_existing_log

            date_iso = date or datetime.now(JST).strftime("%Y-%m-%d")
            period = _resolve_period(config, date_iso)
            if not period.get("ok"):
                return {"ok": False, "fields": {}}

            planning_page = query_existing_log(
                db_env_name=config.db_env_name,
                resolver_name=config.resolver_name,
                period_id=period["page_id"],
                log_type="Planning",
                log_label=f"{config.script_id}_planning_ref",
            )
            fields = _extract_page_fields(
                config, planning_page,
                field_names=config.planning_context_fields,
            )
            return {
                "ok": True,
                "has_data": bool(planning_page),
                "fields": fields,
            }

    @app.post("/api/audio-upload")
    async def audio_upload(
        audio: UploadFile = File(...),
        section_key: str = Form(...),
    ):
        try:
            tmp_dir = tempfile.mkdtemp(prefix=f"{config.script_id}_audio_")
            audio_path = os.path.join(tmp_dir, f"{section_key}.webm")
            content = await audio.read()
            with open(audio_path, "wb") as f:
                f.write(content)

            from src.daily.audio import transcribe_audio
            transcript = transcribe_audio(audio_path, language="ja")

            return {"ok": True, "transcript": transcript, "section_key": section_key}
        except Exception as e:
            logger.error("Audio upload/transcribe failed: %s", e)
            return JSONResponse(
                {"ok": False, "error": str(e)}, status_code=500,
            )

    @app.post("/api/submit")
    async def submit(request_data: dict):
        sections = request_data.get("sections", {})
        scores = request_data.get("scores", {})
        date_iso = request_data.get("date", "") or datetime.now(JST).strftime("%Y-%m-%d")
        no_llm = request_data.get("no_llm", False)

        # Assemble raw_text from sections
        parts = []
        for sec_def in config.sections:
            text = sections.get(sec_def.key, "").strip()
            if text:
                parts.append(f"## {sec_def.title_ja}\n{text}")
        raw_text = "\n\n".join(parts)

        if not raw_text:
            return JSONResponse(
                {"ok": False, "error": "No input provided"},
                status_code=400,
            )

        result = _run_submit_pipeline(
            config, raw_text, sections, scores, date_iso, no_llm,
        )
        return result

    @app.post("/api/submit-text")
    async def submit_text(request_data: dict):
        text = request_data.get("text", "").strip()
        date_iso = request_data.get("date", "") or datetime.now(JST).strftime("%Y-%m-%d")
        no_llm = request_data.get("no_llm", False)

        if not text:
            return JSONResponse(
                {"ok": False, "error": "No text provided"},
                status_code=400,
            )

        result = _run_submit_pipeline(
            config, text, {}, {}, date_iso, no_llm,
        )
        return result

    return app


# ── HTML Builder ──────────────────────────────────────────────────


def _build_period_html(config: PeriodWebUIConfig) -> str:
    """Build the inline HTML SPA for a period planning/review UI."""

    # Generate sections JSON for JS
    sections_json = json.dumps([
        {"key": s.key, "title_ja": s.title_ja, "title_en": s.title_en, "input_type": s.input_type}
        for s in config.sections
    ], ensure_ascii=False)

    score_fields_json = json.dumps([
        {
            "key": s.key, "title_ja": s.title_ja, "title_en": s.title_en,
            "input_type": s.input_type,
            "min_value": s.min_value, "max_value": s.max_value,
        }
        for s in config.score_fields
    ], ensure_ascii=False)

    # Field names for existing content display
    existing_field_names = json.dumps(
        [fm.notion_property for fm in config.field_mappings],
        ensure_ascii=False,
    )

    planning_context_fields_json = json.dumps(
        config.planning_context_fields,
        ensure_ascii=False,
    )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{config.script_id} {config.title}</title>
<style>
:root {{
  --bg: #0f1117; --surface: #1a1d27; --surface2: #242836;
  --border: #2e3345; --text: #e4e6ef; --text2: #9399b2;
  --accent: #7c3aed; --accent-light: #a78bfa;
  --green: #10b981; --red: #ef4444; --orange: #f59e0b;
  --blue: #3b82f6; --cyan: #06b6d4;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg); color: var(--text);
  min-height: 100vh; display: flex;
}}

/* Sidebar */
.sidebar {{
  width: 300px; min-width: 300px;
  background: var(--surface); border-right: 1px solid var(--border);
  display: flex; flex-direction: column; height: 100vh;
  position: sticky; top: 0; overflow-y: auto;
}}
.sidebar-header {{
  padding: 20px; border-bottom: 1px solid var(--border);
}}
.sidebar-header h2 {{
  font-size: 16px; color: var(--accent-light); margin-bottom: 4px;
}}
.sidebar-header .period-info {{
  font-size: 13px; color: var(--text2);
}}
.sidebar-section {{
  padding: 16px 20px; border-bottom: 1px solid var(--border);
}}
.sidebar-section h3 {{
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
  color: var(--text2); margin-bottom: 10px;
}}
.sidebar-field {{
  margin-bottom: 10px;
}}
.sidebar-field .field-label {{
  font-size: 11px; color: var(--accent-light); margin-bottom: 2px;
  text-transform: uppercase; letter-spacing: 0.3px;
}}
.sidebar-field .field-value {{
  font-size: 13px; color: var(--text); line-height: 1.5;
  white-space: pre-wrap; word-break: break-word;
  max-height: 120px; overflow-y: auto;
}}
.empty-note {{
  font-size: 13px; color: var(--text2); font-style: italic;
}}

/* Main */
.main {{
  flex: 1; padding: 32px; overflow-y: auto; height: 100vh;
  max-width: 800px;
}}
.main h1 {{
  font-size: 24px; margin-bottom: 8px;
}}
.main .subtitle {{
  font-size: 14px; color: var(--text2); margin-bottom: 24px;
}}

/* Planning context banner (reviews only) */
.planning-context {{
  background: var(--surface); border: 1px solid var(--border);
  border-left: 3px solid var(--blue);
  border-radius: 8px; padding: 16px; margin-bottom: 20px;
}}
.planning-context h3 {{
  font-size: 13px; color: var(--blue); margin-bottom: 10px;
  text-transform: uppercase; letter-spacing: 0.5px;
}}
.planning-context .ctx-field {{
  margin-bottom: 8px;
}}
.planning-context .ctx-label {{
  font-size: 11px; color: var(--text2); text-transform: uppercase;
}}
.planning-context .ctx-value {{
  font-size: 13px; color: var(--text); white-space: pre-wrap;
  line-height: 1.5;
}}

/* Section cards */
.section-card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px; margin-bottom: 16px;
}}
.section-card h3 {{
  font-size: 14px; color: var(--accent-light); margin-bottom: 12px;
  display: flex; align-items: center; gap: 8px;
}}
.section-card textarea {{
  width: 100%; min-height: 100px; padding: 12px;
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 8px; color: var(--text); font-size: 14px;
  font-family: inherit; resize: vertical; line-height: 1.6;
}}
.section-card textarea:focus {{
  outline: none; border-color: var(--accent);
}}

/* Recording controls */
.record-bar {{
  display: flex; align-items: center; gap: 10px; margin-top: 10px;
}}
.btn {{
  padding: 8px 18px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface2); color: var(--text); cursor: pointer;
  font-size: 13px; font-weight: 500; transition: all 0.15s;
}}
.btn:hover {{ background: var(--border); }}
.btn-record {{ border-color: var(--red); color: var(--red); }}
.btn-record.recording {{
  background: var(--red); color: #fff; animation: pulse 1.5s infinite;
}}
@keyframes pulse {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.7; }}
}}
.btn-primary {{
  background: var(--accent); border-color: var(--accent); color: #fff;
}}
.btn-primary:hover {{ background: #6d28d9; }}
.btn-primary:disabled {{
  opacity: 0.5; cursor: not-allowed;
}}
.rec-status {{
  font-size: 12px; color: var(--text2);
}}

/* Score inputs */
.score-section {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px; margin-bottom: 16px;
}}
.score-section h3 {{
  font-size: 14px; color: var(--accent-light); margin-bottom: 12px;
}}
.score-row {{
  display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
}}
.score-row label {{
  font-size: 13px; color: var(--text); min-width: 150px;
}}
.score-row input[type="number"] {{
  width: 80px; padding: 8px; background: var(--surface2);
  border: 1px solid var(--border); border-radius: 6px;
  color: var(--text); font-size: 14px; text-align: center;
}}
.score-row input[type="number"]:focus {{
  outline: none; border-color: var(--accent);
}}
.checkbox-row {{
  display: flex; align-items: center; gap: 8px;
}}
.checkbox-row input[type="checkbox"] {{
  width: 18px; height: 18px; accent-color: var(--accent);
}}

/* Submit */
.submit-section {{
  margin-top: 24px; display: flex; align-items: center; gap: 16px;
}}
.no-llm-toggle {{
  display: flex; align-items: center; gap: 6px;
  font-size: 13px; color: var(--text2);
}}
.no-llm-toggle input {{ accent-color: var(--accent); }}

/* Result */
.result-banner {{
  margin-top: 16px; padding: 16px; border-radius: 8px;
  font-size: 14px; display: none;
}}
.result-banner.success {{
  background: rgba(16,185,129,0.1); border: 1px solid var(--green);
  color: var(--green); display: block;
}}
.result-banner.error {{
  background: rgba(239,68,68,0.1); border: 1px solid var(--red);
  color: var(--red); display: block;
}}

/* Loading overlay */
.loading {{
  display: none; align-items: center; gap: 8px;
  font-size: 13px; color: var(--text2);
}}
.loading.active {{ display: flex; }}
.spinner {{
  width: 16px; height: 16px; border: 2px solid var(--border);
  border-top-color: var(--accent); border-radius: 50%;
  animation: spin 0.8s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>

<!-- Sidebar: period info + existing content -->
<div class="sidebar">
  <div class="sidebar-header">
    <h2>{config.script_id} {config.title}</h2>
    <div class="period-info" id="period-info">Loading...</div>
  </div>

  <div id="planning-context-sidebar"></div>

  <div class="sidebar-section">
    <h3>Existing Content</h3>
    <div id="existing-content">
      <div class="empty-note">Loading...</div>
    </div>
  </div>
</div>

<!-- Main: wizard input -->
<div class="main">
  <h1>{config.title}</h1>
  <div class="subtitle" id="main-subtitle">{config.period_type} {config.log_type}</div>

  <div id="planning-context-banner"></div>

  <div id="wizard-sections"></div>

  <div id="score-section"></div>

  <div class="submit-section">
    <button class="btn btn-primary" id="submit-btn" onclick="submitAll()">
      Save to Notion
    </button>
    <div class="no-llm-toggle">
      <input type="checkbox" id="no-llm-cb">
      <label for="no-llm-cb">Skip LLM</label>
    </div>
    <div class="loading" id="submit-loading">
      <div class="spinner"></div>
      <span>Processing...</span>
    </div>
  </div>

  <div class="result-banner" id="result-banner"></div>
</div>

<script>
const CONFIG = {{
  scriptId: "{config.script_id}",
  title: "{config.title}",
  periodType: "{config.period_type}",
  logType: "{config.log_type}",
  hasPlanningContext: {'true' if config.has_planning_context else 'false'},
  sections: {sections_json},
  scoreFields: {score_fields_json},
  existingFieldNames: {existing_field_names},
  planningContextFields: {planning_context_fields_json},
}};

let mediaRecorders = {{}};
let audioChunks = {{}};

// ── Init ──
document.addEventListener('DOMContentLoaded', async () => {{
  renderWizardSections();
  renderScoreFields();
  await loadPeriodInfo();
  await loadExisting();
  if (CONFIG.hasPlanningContext) await loadPlanningContext();
}});

// ── Render wizard sections ──
function renderWizardSections() {{
  const container = document.getElementById('wizard-sections');
  container.innerHTML = CONFIG.sections.map(sec => `
    <div class="section-card" id="section-${{sec.key}}">
      <h3>${{sec.title_ja}}</h3>
      <textarea id="text-${{sec.key}}" placeholder="${{sec.title_en}}..."
                rows="4"></textarea>
      <div class="record-bar">
        <button class="btn btn-record" id="rec-btn-${{sec.key}}"
                onclick="toggleRecord('${{sec.key}}')">
          Record
        </button>
        <span class="rec-status" id="rec-status-${{sec.key}}"></span>
      </div>
    </div>
  `).join('');
}}

// ── Render score fields ──
function renderScoreFields() {{
  const container = document.getElementById('score-section');
  if (CONFIG.scoreFields.length === 0) return;
  let html = '<div class="score-section"><h3>Scores</h3>';
  for (const sf of CONFIG.scoreFields) {{
    if (sf.input_type === 'number') {{
      html += `
        <div class="score-row">
          <label>${{sf.title_ja}}</label>
          <input type="number" id="score-${{sf.key}}"
                 min="${{sf.min_value || 1}}" max="${{sf.max_value || 10}}"
                 placeholder="${{sf.min_value || 1}}-${{sf.max_value || 10}}">
        </div>`;
    }} else if (sf.input_type === 'checkbox') {{
      html += `
        <div class="score-row checkbox-row">
          <input type="checkbox" id="score-${{sf.key}}">
          <label>${{sf.title_ja}}</label>
        </div>`;
    }}
  }}
  html += '</div>';
  container.innerHTML = html;
}}

// ── Load period info ──
async function loadPeriodInfo() {{
  try {{
    const resp = await fetch('/api/period-info');
    const data = await resp.json();
    if (data.ok) {{
      document.getElementById('period-info').innerHTML =
        `${{data.period_name}}<br><small>${{data.start_date}} ~ ${{data.end_date}}</small>`;
      document.getElementById('main-subtitle').textContent =
        `${{CONFIG.periodType}} ${{CONFIG.logType}} — ${{data.period_name}}`;
    }}
  }} catch (e) {{
    document.getElementById('period-info').textContent = 'Failed to load';
  }}
}}

// ── Load existing Notion content ──
async function loadExisting() {{
  try {{
    const resp = await fetch('/api/existing');
    const data = await resp.json();
    const container = document.getElementById('existing-content');

    if (!data.ok || !data.has_data) {{
      container.innerHTML = '<div class="empty-note">No existing data for this period.</div>';
      return;
    }}

    let html = '';
    for (const [key, value] of Object.entries(data.fields)) {{
      if (!value || value === '') continue;
      html += `
        <div class="sidebar-field">
          <div class="field-label">${{key}}</div>
          <div class="field-value">${{escapeHtml(String(value))}}</div>
        </div>`;
    }}
    container.innerHTML = html || '<div class="empty-note">No fields populated yet.</div>';
  }} catch (e) {{
    document.getElementById('existing-content').innerHTML =
      '<div class="empty-note">Failed to load existing content.</div>';
  }}
}}

// ── Load planning context (reviews only) ──
async function loadPlanningContext() {{
  try {{
    const resp = await fetch('/api/planning-context');
    const data = await resp.json();

    if (!data.ok || !data.has_data) return;

    let html = '<div class="planning-context"><h3>Reference — Planning</h3>';
    for (const [key, value] of Object.entries(data.fields)) {{
      if (!value || value === '') continue;
      html += `
        <div class="ctx-field">
          <div class="ctx-label">${{key}}</div>
          <div class="ctx-value">${{escapeHtml(String(value))}}</div>
        </div>`;
    }}
    html += '</div>';
    document.getElementById('planning-context-banner').innerHTML = html;
  }} catch (e) {{
    // Silently skip
  }}
}}

// ── Voice recording ──
async function toggleRecord(sectionKey) {{
  const btn = document.getElementById(`rec-btn-${{sectionKey}}`);
  const status = document.getElementById(`rec-status-${{sectionKey}}`);

  if (mediaRecorders[sectionKey] && mediaRecorders[sectionKey].state === 'recording') {{
    // Stop
    mediaRecorders[sectionKey].stop();
    btn.textContent = 'Record';
    btn.classList.remove('recording');
    status.textContent = 'Processing...';
    return;
  }}

  // Start
  try {{
    const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
    const recorder = new MediaRecorder(stream, {{ mimeType: 'audio/webm' }});
    audioChunks[sectionKey] = [];

    recorder.ondataavailable = (e) => {{
      if (e.data.size > 0) audioChunks[sectionKey].push(e.data);
    }};

    recorder.onstop = async () => {{
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(audioChunks[sectionKey], {{ type: 'audio/webm' }});
      status.textContent = 'Transcribing...';

      const formData = new FormData();
      formData.append('audio', blob, `${{sectionKey}}.webm`);
      formData.append('section_key', sectionKey);

      try {{
        const resp = await fetch('/api/audio-upload', {{ method: 'POST', body: formData }});
        const data = await resp.json();
        if (data.ok) {{
          const textarea = document.getElementById(`text-${{sectionKey}}`);
          textarea.value = (textarea.value ? textarea.value + '\\n' : '') + data.transcript;
          status.textContent = 'Transcribed!';
        }} else {{
          status.textContent = 'Transcription failed: ' + (data.error || 'unknown');
        }}
      }} catch (err) {{
        status.textContent = 'Upload failed: ' + err.message;
      }}
    }};

    recorder.start();
    mediaRecorders[sectionKey] = recorder;
    btn.textContent = 'Stop';
    btn.classList.add('recording');
    status.textContent = 'Recording...';
  }} catch (err) {{
    status.textContent = 'Mic access denied: ' + err.message;
  }}
}}

// ── Submit ──
async function submitAll() {{
  const btn = document.getElementById('submit-btn');
  const loading = document.getElementById('submit-loading');
  const resultBanner = document.getElementById('result-banner');

  btn.disabled = true;
  loading.classList.add('active');
  resultBanner.className = 'result-banner';
  resultBanner.style.display = 'none';

  // Collect section texts
  const sections = {{}};
  for (const sec of CONFIG.sections) {{
    sections[sec.key] = document.getElementById(`text-${{sec.key}}`).value;
  }}

  // Collect score values
  const scores = {{}};
  for (const sf of CONFIG.scoreFields) {{
    const el = document.getElementById(`score-${{sf.key}}`);
    if (sf.input_type === 'checkbox') {{
      scores[sf.key] = el.checked;
    }} else {{
      scores[sf.key] = el.value ? Number(el.value) : null;
    }}
  }}

  const noLlm = document.getElementById('no-llm-cb').checked;

  try {{
    const resp = await fetch('/api/submit', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ sections, scores, no_llm: noLlm }}),
    }});
    const data = await resp.json();

    if (data.ok) {{
      resultBanner.className = 'result-banner success';
      const pageUrl = data.notion?.page_url || '';
      resultBanner.innerHTML =
        `Saved to Notion! Period: ${{data.period}}` +
        (pageUrl ? ` — <a href="${{pageUrl}}" target="_blank" style="color:var(--green)">Open in Notion</a>` : '') +
        `<br><small>Input: ${{data.input_chars}} chars | LLM: ${{data.llm_used ? 'Yes' : 'No'}}</small>`;
      // Refresh sidebar
      await loadExisting();
    }} else {{
      resultBanner.className = 'result-banner error';
      resultBanner.textContent = 'Save failed: ' + (data.error || JSON.stringify(data));
    }}
  }} catch (err) {{
    resultBanner.className = 'result-banner error';
    resultBanner.textContent = 'Request failed: ' + err.message;
  }} finally {{
    btn.disabled = false;
    loading.classList.remove('active');
  }}
}}

// ── Utilities ──
function escapeHtml(str) {{
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}}
</script>
</body>
</html>'''


# ── Server Launcher ───────────────────────────────────────────────


def run_period_server(
    config: PeriodWebUIConfig,
    *,
    port: Optional[int] = None,
    verbose: bool = False,
    no_browser: bool = False,
) -> None:
    """Launch the period Web UI server."""
    from src.config import load_env, setup_logging

    load_env()
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    actual_port = port or config.port
    logger.info("Starting %s Web UI on port %d", config.script_name, actual_port)

    import uvicorn

    app = build_period_web_app(config)

    url = f"http://localhost:{actual_port}"
    if not no_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    logger.info("%s available at %s", config.title, url)
    uvicorn.run(app, host="127.0.0.1", port=actual_port, log_level="warning")
