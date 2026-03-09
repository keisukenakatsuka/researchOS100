#!/usr/bin/env python
# src/scripts/059_next_day_preparation.py
"""Next Day Preparation — reduce next-day friction by preparing meeting briefs.

Pipeline:
1. Load 058 output (CloseStructured) from local JSON
2. LLM extraction: cluster research targets by inferred meeting context
3. Web UI: present targets grouped by meeting cluster for user editing
4. Research: search Notion Events DB + external (Google CSE, NewsAPI)
5. LLM synthesis: generate meeting briefs from research results
6. Write Meeting Briefs to Notion + local JSON
7. Update Daily Log with prep layer (stage=prepared)

Usage::

    # Interactive (opens Web UI for editing targets)
    python -m src.scripts.059_next_day_preparation

    # Specific date
    python -m src.scripts.059_next_day_preparation --date 2026-02-20

    # Skip Web UI (auto-accept extracted targets)
    python -m src.scripts.059_next_day_preparation --no-ui

    # Skip external research (Notion Events only)
    python -m src.scripts.059_next_day_preparation --no-external
"""

# NOTE: Do NOT use ``from __future__ import annotations`` here.
# FastAPI + Pydantic v2 need *runtime* type objects for endpoint
# parameter resolution (UploadFile, File, etc.).

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
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
    RunMetadata,
    get_iso_week_context,
    get_db_id,
    get_optional_db_id,
)
from src.daily.models import (
    CloseStructured,
    FollowUp,
    MeetingBrief,
    NextDayPrep,
)
from src.daily.io import (
    CLOSE_STRUCTURED_DIR,
    NEXT_DAY_PREP_DIR,
    daily_output_dir,
    save_json,
    load_json,
)

from src.deep_research.session import run_single_pipeline
from src.deep_research import generate_run_id
from src.llm.claude_client import ClaudeClient

logger = logging.getLogger("059_next_day_preparation")

SCRIPT_NAME = "059_next_day_preparation"
JST = ZoneInfo("Asia/Tokyo")

# Cache directories
_LLM_CACHE_DIR = Path("data/cache/next_day_prep")
_SEARCH_CACHE_DIR = Path("data/cache/next_day_prep_search")

# Web UI port
_UI_PORT = 8059


# ═══════════════════════════════════════════════════════════════════
# 1. INPUT: Load 058 structured output
# ═══════════════════════════════════════════════════════════════════

def _load_close_structured(date_iso: str) -> CloseStructured:
    """Load 058 output for a given date."""
    s_dir = CLOSE_STRUCTURED_DIR / date_iso
    s_path = s_dir / "close_structured.json"
    if not s_path.exists():
        raise FileNotFoundError(
            f"No 058 output for {date_iso}. Run 058_daily_close_structuring first.\n"
            f"Expected: {s_path}"
        )
    data = load_json(s_path)
    return CloseStructured.from_dict(data)


# ═══════════════════════════════════════════════════════════════════
# 2. LLM EXTRACTION: Cluster research targets by meeting context
# ═══════════════════════════════════════════════════════════════════

_EXTRACTION_SYSTEM_PROMPT = """\
You are a meeting preparation assistant. Your job is to analyze a daily close log
and extract research targets needed for tomorrow's meetings/work.

Rules:
- Group targets by inferred meeting or work context ("meeting cluster").
- Each cluster should have a descriptive meeting_title.
- Extract organizations, people, and topics that need research.
- If the source text doesn't clearly separate meetings, infer 1-N clusters based on context.
- Include URLs if they are mentioned in the source text; otherwise set url to null.
- Set inferred_date if you can determine when the meeting is; otherwise null.
- Set purpose_hint to a brief description of the meeting's purpose if inferrable.
- **出力は日本語で記述してください（meeting_title, purpose_hint は日本語）。**
  人名・組織名は原語のままで構いません。
- Output valid JSON only. No markdown fences.

Output schema:
{
  "meetings": [
    {
      "meeting_title": "string（日本語）",
      "inferred_date": "YYYY-MM-DD" or null,
      "purpose_hint": "string（日本語）" or null,
      "organizations": [{"name": "string", "url": "string or null"}],
      "people": [{"name": "string", "url": "string or null", "org": "string or null"}],
      "topics": [{"name": "string（日本語可）", "url": "string or null"}]
    }
  ]
}
"""


def _input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _extract_meeting_targets(
    structured: CloseStructured,
    date_iso: str,
    *,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """Call LLM to extract meeting-clustered research targets."""
    # Build input text from structured fields
    parts = []
    if structured.friction_blockers:
        parts.append("## Friction / Blockers\n" + "\n".join(f"- {fb}" for fb in structured.friction_blockers))
    if structured.open_questions:
        parts.append("## Open Questions\n" + "\n".join(f"- {oq}" for oq in structured.open_questions))
    if structured.provisional_top3:
        parts.append("## Provisional Top 3 (tomorrow)\n" + "\n".join(f"- {t}" for t in structured.provisional_top3))
    # Also include structured items for richer context
    if structured.items:
        item_lines = []
        for item in structured.items:
            people_str = f" [people: {', '.join(item.people)}]" if item.people else ""
            item_lines.append(f"- [{item.category}] {item.text}{people_str}")
        parts.append("## Daily Log Items\n" + "\n".join(item_lines))
    if structured.contact_candidates:
        parts.append("## Contact Candidates\n" + "\n".join(f"- {cc}" for cc in structured.contact_candidates))
    if structured.research_candidates:
        parts.append("## Research Candidates\n" + "\n".join(f"- {rc}" for rc in structured.research_candidates))

    input_text = "\n\n".join(parts)
    if not input_text.strip():
        logger.warning("Empty input for LLM extraction — returning no meetings")
        return {"meetings": []}

    # Check cache
    cache_key = _input_hash(input_text)
    cache_path = _LLM_CACHE_DIR / f"{structured.date}_{cache_key}.json"
    if cache_path.exists():
        logger.info("LLM extraction cache hit: %s", cache_path.name)
        return load_json(cache_path)

    # LLM call
    from src.llm.router import build_router_from_env, TASK_REASONING
    router = build_router_from_env(cache_dir=_LLM_CACHE_DIR)

    user_prompt = (
        f"Date of close log: {structured.date}\n"
        f"Meeting date: {date_iso}\n\n"
        f"{input_text}"
    )

    logger.info("Calling LLM for meeting target extraction (model=%s)", model)
    result = router.call(
        task_type=TASK_REASONING,
        system=_EXTRACTION_SYSTEM_PROMPT,
        user=user_prompt,
        model_override=model,
        temperature_override=0.3,
        use_cache=False,
    )

    data = result.parsed
    # Sanity: result.parsed must be a dict; guard against future attr changes
    assert isinstance(data, dict), f"LLM result.parsed expected dict, got {type(data).__name__}"
    if "meetings" not in data:
        data = {"meetings": []}

    # Cache
    _LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(cache_path, data)
    in_tok = getattr(result, "input_tokens", None)
    out_tok = getattr(result, "output_tokens", None)
    if in_tok is not None and out_tok is not None:
        logger.info(
            "Extracted %d meeting clusters (tokens: %d prompt, %d completion)",
            len(data["meetings"]), in_tok, out_tok,
        )
    else:
        logger.info("Extracted %d meeting clusters (tokens: n/a)", len(data["meetings"]))
    return data


# ═══════════════════════════════════════════════════════════════════
# 3. WEB UI: Edit meeting targets (FastAPI-based, like 057)
# ═══════════════════════════════════════════════════════════════════

_ui_state: dict = {}


def _launch_target_editor_ui(
    meetings: List[Dict[str, Any]],
    *,
    port: int = _UI_PORT,
    no_browser: bool = False,
) -> List[Dict[str, Any]]:
    """Launch Web UI for editing meeting-clustered targets.

    Returns the user-confirmed meetings list (possibly edited).
    """
    from src.daily.audio import check_browser_dependencies
    check_browser_dependencies()

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    _ui_state.clear()
    _ui_state.update({
        "meetings": meetings,
        "confirmed": False,
        "cancelled": False,
        "done": False,
    })

    app = FastAPI(title="059 Next Day Preparation — Target Editor")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _build_target_editor_html(meetings)

    @app.get("/api/meetings")
    async def get_meetings():
        return {"meetings": _ui_state["meetings"]}

    @app.post("/api/update-meetings")
    async def update_meetings(payload: dict):
        _ui_state["meetings"] = payload.get("meetings", [])
        return {"ok": True}

    @app.post("/api/confirm")
    async def confirm(payload: dict):
        confirmed = payload.get("meetings", _ui_state["meetings"])
        _ui_state["meetings"] = confirmed
        _ui_state["confirmed"] = True
        _ui_state["done"] = True
        logger.info(
            "[UI] Run Research clicked — received %d clusters: %s",
            len(confirmed),
            [m.get("meeting_title", "(untitled)") for m in confirmed],
        )
        return {"ok": True}

    @app.post("/api/cancel")
    async def cancel():
        _ui_state["cancelled"] = True
        _ui_state["done"] = True
        return {"ok": True}

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    url = f"http://localhost:{port}"
    logger.info("Target editor UI started at %s", url)
    if not no_browser:
        def _open():
            time.sleep(1.0)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        while not _ui_state.get("done"):
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("UI interrupted by user")
        _ui_state["cancelled"] = True

    server.should_exit = True
    thread.join(timeout=5)

    if _ui_state.get("cancelled"):
        raise RuntimeError("Target editing was cancelled by user.")

    return _ui_state["meetings"]


def _build_target_editor_html(meetings: List[Dict[str, Any]]) -> str:
    """Build the HTML for the target editor UI.

    Uses string.Template for the JavaScript portion to avoid f-string
    escaping issues with JS braces and quotes.
    """
    from string import Template

    meetings_json = json.dumps(meetings, ensure_ascii=False)

    # CSS is plain text — use raw string (no f-string needed)
    css = """
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #242836;
    --border: #2e3345; --text: #e4e6ef; --text2: #9399b2;
    --accent: #7c3aed; --accent-light: #a78bfa;
    --green: #10b981; --red: #ef4444; --orange: #f59e0b;
    --blue: #3b82f6;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; padding: 24px;
  }
  .container { max-width: 800px; margin: 0 auto; }
  h1 { font-size: 24px; font-weight: 700; margin-bottom: 8px; text-align: center; }
  .subtitle { color: var(--text2); font-size: 14px; margin-bottom: 24px; text-align: center; }
  .meeting-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 24px; margin-bottom: 20px;
  }
  .meeting-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
  .meeting-header input {
    flex: 1; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; color: var(--text);
    font-size: 16px; font-weight: 600;
  }
  .meeting-header input:focus { outline: none; border-color: var(--accent); }
  .btn-delete-meeting {
    background: none; border: 1px solid var(--red); color: var(--red);
    border-radius: 6px; padding: 6px 12px; cursor: pointer; font-size: 13px;
  }
  .btn-delete-meeting:hover { background: rgba(239,68,68,0.1); }
  .section-label {
    font-size: 13px; font-weight: 600; color: var(--text2);
    margin: 12px 0 6px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .target-list { display: flex; flex-direction: column; gap: 6px; }
  .target-item {
    display: flex; align-items: center; gap: 8px;
    background: var(--surface2); border-radius: 8px; padding: 8px 12px;
  }
  .target-item input {
    flex: 1; background: transparent; border: none;
    color: var(--text); font-size: 14px;
  }
  .target-item input:focus { outline: none; }
  .target-item input.url-input { color: var(--accent-light); font-size: 12px; flex: 0.6; }
  .target-item input.org-input { color: var(--orange); font-size: 12px; flex: 0.4; }
  .btn-delete {
    background: none; border: none; color: var(--red);
    cursor: pointer; font-size: 16px; padding: 2px 6px; opacity: 0.6;
  }
  .btn-delete:hover { opacity: 1; }
  .btn-add {
    background: none; border: 1px dashed var(--border);
    border-radius: 8px; padding: 6px 12px; color: var(--text2);
    cursor: pointer; font-size: 13px; margin-top: 6px; width: 100%;
  }
  .btn-add:hover { border-color: var(--accent); color: var(--accent); }
  .actions {
    display: flex; justify-content: center; gap: 16px;
    margin-top: 32px; padding: 24px 0;
  }
  .btn {
    padding: 14px 36px; border: none; border-radius: 10px;
    font-size: 16px; font-weight: 600; cursor: pointer;
    transition: all 0.2s;
  }
  .btn-run { background: var(--green); color: #fff; font-size: 18px; padding: 16px 48px; }
  .btn-run:hover { filter: brightness(1.1); }
  .btn-run:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-cancel { background: var(--surface2); color: var(--text2); }
  .btn-add-meeting {
    background: none; border: 2px dashed var(--border);
    border-radius: 12px; padding: 20px; color: var(--text2);
    cursor: pointer; font-size: 15px; width: 100%; margin-bottom: 20px;
  }
  .btn-add-meeting:hover { border-color: var(--accent); color: var(--accent); }
  .status { text-align: center; margin-top: 16px; font-size: 14px; color: var(--text2); min-height: 20px; }
  .status.error { color: var(--red); }
  .status.success { color: var(--green); }
  .empty-state {
    text-align: center; padding: 40px; color: var(--text2);
    background: var(--surface); border-radius: 12px; border: 1px solid var(--border);
    margin-bottom: 20px;
  }
  .empty-state p { margin-bottom: 12px; }
  .purpose-hint {
    color: var(--text2); font-size: 13px; margin-bottom: 12px;
    font-style: italic;
  }
"""

    # JavaScript — use string.Template ($-substitution) to avoid f-string
    # escaping issues with JS braces and single quotes.
    js_template = Template("""
let meetings = $MEETINGS_JSON;

function render() {
  const container = document.getElementById('meetings-container');
  container.innerHTML = '';

  if (meetings.length === 0) {
    container.innerHTML = '<div class="empty-state"><p>No meeting clusters extracted.</p>'
      + '<p>Click "+ Add Meeting Cluster" to add one manually.</p></div>';
    return;
  }

  meetings.forEach(function(m, mi) {
    const card = document.createElement('div');
    card.className = 'meeting-card';

    // --- header ---
    const header = document.createElement('div');
    header.className = 'meeting-header';
    const titleInput = document.createElement('input');
    titleInput.type = 'text';
    titleInput.value = m.meeting_title || '';
    titleInput.placeholder = 'Meeting title';
    titleInput.addEventListener('change', function() { meetings[mi].meeting_title = this.value; });
    header.appendChild(titleInput);
    const delBtn = document.createElement('button');
    delBtn.className = 'btn-delete-meeting';
    delBtn.textContent = 'Remove';
    delBtn.addEventListener('click', function() { deleteMeeting(mi); });
    header.appendChild(delBtn);
    card.appendChild(header);

    // --- purpose hint ---
    if (m.purpose_hint) {
      const hint = document.createElement('div');
      hint.className = 'purpose-hint';
      hint.textContent = m.purpose_hint;
      card.appendChild(hint);
    }

    // --- sections: organizations, people, topics ---
    buildSection(card, mi, 'organizations', 'Organizations', '+ Add Organization');
    buildPeopleSection(card, mi);
    buildSection(card, mi, 'topics', 'Topics', '+ Add Topic');

    container.appendChild(card);
  });
}

function buildSection(card, mi, type, label, addLabel) {
  var m = meetings[mi];
  var lbl = document.createElement('div');
  lbl.className = 'section-label';
  lbl.textContent = label;
  card.appendChild(lbl);

  var list = document.createElement('div');
  list.className = 'target-list';
  (m[type] || []).forEach(function(item, idx) {
    list.appendChild(buildTargetRow(mi, type, idx, item));
  });
  card.appendChild(list);

  var addBtn = document.createElement('button');
  addBtn.className = 'btn-add';
  addBtn.textContent = addLabel;
  addBtn.addEventListener('click', function() { addTarget(mi, type); });
  card.appendChild(addBtn);
}

function buildPeopleSection(card, mi) {
  var m = meetings[mi];
  var lbl = document.createElement('div');
  lbl.className = 'section-label';
  lbl.textContent = 'People';
  card.appendChild(lbl);

  var list = document.createElement('div');
  list.className = 'target-list';
  (m.people || []).forEach(function(p, idx) {
    list.appendChild(buildPersonRow(mi, idx, p));
  });
  card.appendChild(list);

  var addBtn = document.createElement('button');
  addBtn.className = 'btn-add';
  addBtn.textContent = '+ Add Person';
  addBtn.addEventListener('click', function() { addTarget(mi, 'people'); });
  card.appendChild(addBtn);
}

function buildTargetRow(mi, type, idx, item) {
  var row = document.createElement('div');
  row.className = 'target-item';

  var nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = item.name || '';
  nameInput.placeholder = 'Name';
  nameInput.addEventListener('change', function() { updateTarget(mi, type, idx, 'name', this.value); });
  row.appendChild(nameInput);

  var urlInput = document.createElement('input');
  urlInput.type = 'text';
  urlInput.className = 'url-input';
  urlInput.value = item.url || '';
  urlInput.placeholder = 'URL (optional)';
  urlInput.addEventListener('change', function() { updateTarget(mi, type, idx, 'url', this.value); });
  row.appendChild(urlInput);

  var delBtn = document.createElement('button');
  delBtn.className = 'btn-delete';
  delBtn.innerHTML = '&times;';
  delBtn.addEventListener('click', function() { removeTarget(mi, type, idx); });
  row.appendChild(delBtn);

  return row;
}

function buildPersonRow(mi, idx, person) {
  var row = document.createElement('div');
  row.className = 'target-item';

  var nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.value = person.name || '';
  nameInput.placeholder = 'Name';
  nameInput.addEventListener('change', function() { updateTarget(mi, 'people', idx, 'name', this.value); });
  row.appendChild(nameInput);

  var orgInput = document.createElement('input');
  orgInput.type = 'text';
  orgInput.className = 'org-input';
  orgInput.value = person.org || '';
  orgInput.placeholder = 'Org';
  orgInput.addEventListener('change', function() { updateTarget(mi, 'people', idx, 'org', this.value); });
  row.appendChild(orgInput);

  var urlInput = document.createElement('input');
  urlInput.type = 'text';
  urlInput.className = 'url-input';
  urlInput.value = person.url || '';
  urlInput.placeholder = 'URL (optional)';
  urlInput.addEventListener('change', function() { updateTarget(mi, 'people', idx, 'url', this.value); });
  row.appendChild(urlInput);

  var delBtn = document.createElement('button');
  delBtn.className = 'btn-delete';
  delBtn.innerHTML = '&times;';
  delBtn.addEventListener('click', function() { removeTarget(mi, 'people', idx); });
  row.appendChild(delBtn);

  return row;
}

function updateTitle(mi, val) { meetings[mi].meeting_title = val; }

function updateTarget(mi, type, idx, field, val) {
  if (meetings[mi][type] && meetings[mi][type][idx]) {
    meetings[mi][type][idx][field] = val || null;
  }
}

function removeTarget(mi, type, idx) {
  if (meetings[mi][type]) {
    meetings[mi][type].splice(idx, 1);
    render();
  }
}

function addTarget(mi, type) {
  if (!meetings[mi][type]) meetings[mi][type] = [];
  if (type === 'people') meetings[mi][type].push({name: '', url: null, org: null});
  else meetings[mi][type].push({name: '', url: null});
  render();
}

function addMeeting() {
  meetings.push({
    meeting_title: 'New Meeting',
    inferred_date: null,
    purpose_hint: null,
    organizations: [],
    people: [],
    topics: []
  });
  render();
}

function deleteMeeting(mi) {
  if (confirm('Remove this meeting cluster?')) {
    meetings.splice(mi, 1);
    render();
  }
}

async function runResearch() {
  var cleaned = meetings.map(function(m) {
    return Object.assign({}, m, {
      organizations: (m.organizations || []).filter(function(o) { return o.name && o.name.trim(); }),
      people: (m.people || []).filter(function(p) { return p.name && p.name.trim(); }),
      topics: (m.topics || []).filter(function(t) { return t.name && t.name.trim(); })
    });
  }).filter(function(m) {
    return m.meeting_title && m.meeting_title.trim() &&
      ((m.organizations || []).length > 0 || (m.people || []).length > 0 || (m.topics || []).length > 0);
  });

  if (cleaned.length === 0) {
    document.getElementById('status').textContent = 'No valid targets to research. Add at least one target.';
    document.getElementById('status').className = 'status error';
    return;
  }

  var btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Researching...';
  document.getElementById('status').textContent = 'Sending targets to research pipeline...';
  document.getElementById('status').className = 'status';

  try {
    var r = await fetch('/api/confirm', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({meetings: cleaned})
    });
    var d = await r.json();
    if (d.ok) {
      document.getElementById('status').textContent = 'Research started! This tab will close shortly...';
      document.getElementById('status').className = 'status success';
      setTimeout(function() { try { window.close(); } catch(e) {} }, 3000);
    } else {
      throw new Error(d.error || 'Unknown error');
    }
  } catch(e) {
    document.getElementById('status').textContent = 'Error: ' + e.message;
    document.getElementById('status').className = 'status error';
    btn.disabled = false;
    btn.textContent = 'Run Research';
  }
}

async function cancelAll() {
  try { await fetch('/api/cancel', {method: 'POST'}); } catch(e) {}
  document.getElementById('status').textContent = 'Cancelled. You can close this tab.';
  document.getElementById('status').className = 'status';
}

render();
""")

    js_code = js_template.substitute(MEETINGS_JSON=meetings_json)

    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>059 Next Day Prep — Target Editor</title>\n'
        '<style>' + css + '</style>\n'
        '</head>\n'
        '<body>\n'
        '<div class="container">\n'
        '  <h1>Next Day Preparation</h1>\n'
        '  <div class="subtitle">Review and edit research targets for tomorrow\'s meetings, then click Run.</div>\n'
        '  <div id="meetings-container"></div>\n'
        '  <button class="btn-add-meeting" onclick="addMeeting()">+ Add Meeting Cluster</button>\n'
        '  <div class="actions">\n'
        '    <button class="btn btn-cancel" onclick="cancelAll()">Cancel</button>\n'
        '    <button class="btn btn-run" id="run-btn" onclick="runResearch()">Run Research</button>\n'
        '  </div>\n'
        '  <div class="status" id="status"></div>\n'
        '</div>\n'
        '<script>' + js_code + '</script>\n'
        '</body>\n'
        '</html>'
    )


# ═══════════════════════════════════════════════════════════════════
# 4. RESEARCH LOGIC
# ═══════════════════════════════════════════════════════════════════

def _search_notion_events(
    query: str,
    *,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """Search Notion Events DB for pages matching the query.

    Returns list of dicts with 'page_id', 'title', 'date', 'summary'.
    Returns empty list if Events DB is not configured or query fails.
    """
    try:
        events_db_id = get_optional_db_id("NOTION_EVENTS_DB_ID")
        if not events_db_id:
            logger.debug("NOTION_EVENTS_DB_ID not set, skipping Events search")
            return []

        from src.notion.client import (
            build_notion_client_from_env,
            NotionDataSourceResolver,
        )
        client = build_notion_client_from_env()
        resolver = NotionDataSourceResolver(client=client)
        resolved = resolver.resolve_once(name="events", database_id=events_db_id)

        # Search by title contains
        pages = client.query_data_source(
            data_source_id=resolved.data_source_id,
            filter={
                "property": "Name",
                "title": {"contains": query},
            },
            page_size=max_results,
            fetch_all=False,
        )

        results = []
        for page in pages[:max_results]:
            props = page.get("properties", {})
            # Extract title
            name_prop = props.get("Name", {})
            title_parts = name_prop.get("title", [])
            title = title_parts[0]["plain_text"] if title_parts else ""
            # Extract date
            date_prop = props.get("Date", {}).get("date", {})
            date_str = date_prop.get("start", "") if date_prop else ""
            # Extract summary/description
            desc_prop = props.get("Description", {}).get("rich_text", [])
            summary = desc_prop[0]["plain_text"][:200] if desc_prop else ""

            results.append({
                "page_id": page.get("id", ""),
                "title": title,
                "date": date_str,
                "summary": summary,
            })

        logger.info("Notion Events search '%s': %d results", query[:40], len(results))
        return results

    except Exception as e:
        logger.warning("Notion Events search failed (non-fatal): %s", e)
        return []


def _search_google(query: str, *, max_results: int = 3) -> List[Dict[str, Any]]:
    """Search Google CSE. Returns empty list on failure."""
    try:
        from src.search.google_cse import build_google_cse_from_env
        cse = build_google_cse_from_env()
        return cse.search(query, num=max_results)
    except Exception as e:
        logger.warning("Google CSE search failed (non-fatal): %s", e)
        return []


def _search_news(query: str, *, max_results: int = 3) -> List[Dict[str, Any]]:
    """Search NewsAPI. Returns empty list on failure."""
    try:
        from src.search.newsapi import build_newsapi_from_env
        news = build_newsapi_from_env()
        return news.search(query, page_size=max_results)
    except Exception as e:
        logger.warning("NewsAPI search failed (non-fatal): %s", e)
        return []


def _research_target(
    target_name: str,
    target_type: str,  # "organization" | "person" | "topic"
    *,
    meeting_context: str = "",
) -> Dict[str, Any]:
    """Research a single target using ALL sources unconditionally.

    All three sources are always executed:
      A) Notion Events DB (internal knowledge)
      B) Google CSE (external knowledge — strategic/overview)
      C) NewsAPI (external knowledge — recent developments)

    Returns dict with 'name', 'type', 'events', 'search_results',
    'news_results', 'source_urls'.
    """
    # Check search cache
    cache_key = _input_hash(f"{target_type}:{target_name}")
    cache_path = _SEARCH_CACHE_DIR / f"{cache_key}.json"
    if cache_path.exists():
        logger.debug("Search cache hit: %s (%s)", target_name, target_type)
        return load_json(cache_path)

    result = {
        "name": target_name,
        "type": target_type,
        "events": [],
        "search_results": [],
        "news_results": [],
        "source_urls": [],
    }

    # A) Search Notion Events — internal knowledge
    events = _search_notion_events(target_name)
    result["events"] = events

    # B) Google CSE — strategic/overview enrichment (ALWAYS runs)
    if target_type == "organization":
        google_query = f"{target_name} 戦略 事業 最新動向"
        google_query_en = f"{target_name} strategy investment news"
    elif target_type == "person":
        google_query = f"{target_name} 経歴 所属 活動"
        google_query_en = f"{target_name} career role profile"
    else:
        google_query = f"{target_name} 最新 動向"
        google_query_en = f"{target_name} latest developments"

    # Run both JP and EN queries for broader coverage
    google_results = _search_google(google_query, max_results=3)
    if len(google_results) < 2:
        google_results.extend(_search_google(google_query_en, max_results=2))
    result["search_results"] = google_results
    result["source_urls"].extend(
        r.get("link", "") for r in google_results if r.get("link")
    )

    # C) NewsAPI — recent news (ALWAYS runs)
    news_results = _search_news(target_name, max_results=5)
    result["news_results"] = news_results
    result["source_urls"].extend(
        r.get("url", "") for r in news_results if r.get("url")
    )

    # Filter empty source URLs
    result["source_urls"] = [u for u in result["source_urls"] if u]

    # Save to cache
    _SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(cache_path, result)

    logger.info(
        "Researched %s '%s': %d events, %d search, %d news",
        target_type, target_name,
        len(result["events"]),
        len(result["search_results"]),
        len(result["news_results"]),
    )
    return result


def _research_meeting_cluster(
    meeting: Dict[str, Any],
) -> Dict[str, Any]:
    """Research all targets in a meeting cluster.

    ALL three sources (Events, Google CSE, NewsAPI) are always executed
    for every target — there is no "skip external" path per target.

    Returns the meeting dict enriched with research results.
    """
    enriched = dict(meeting)
    enriched["research"] = {}

    meeting_context = meeting.get("purpose_hint", "") or meeting.get("meeting_title", "")
    all_events = []  # Track for Related Event relation
    all_source_urls = []

    # Research organizations
    for org in meeting.get("organizations", []):
        if org.get("name"):
            r = _research_target(org["name"], "organization", meeting_context=meeting_context)
            enriched["research"][f"org:{org['name']}"] = r
            all_events.extend(r.get("events", []))
            all_source_urls.extend(r.get("source_urls", []))

    # Research people
    for person in meeting.get("people", []):
        if person.get("name"):
            r = _research_target(person["name"], "person", meeting_context=meeting_context)
            enriched["research"][f"person:{person['name']}"] = r
            all_events.extend(r.get("events", []))
            all_source_urls.extend(r.get("source_urls", []))

    # Research topics
    for topic in meeting.get("topics", []):
        if topic.get("name"):
            r = _research_target(topic["name"], "topic", meeting_context=meeting_context)
            enriched["research"][f"topic:{topic['name']}"] = r
            all_events.extend(r.get("events", []))
            all_source_urls.extend(r.get("source_urls", []))

    # Collect unique Related Event page IDs
    enriched["related_event_ids"] = list(set(
        e["page_id"] for e in all_events if e.get("page_id")
    ))
    enriched["all_source_urls"] = list(set(all_source_urls))

    return enriched


# ═══════════════════════════════════════════════════════════════════
# 5. LLM SYNTHESIS: Generate meeting brief from research
# ═══════════════════════════════════════════════════════════════════

_BRIEF_SYNTHESIS_SYSTEM = """\
あなたは戦略的ミーティング準備アシスタントです。
リサーチ結果（内部イベント情報・Web検索・直近ニュース）を統合し、
意思決定に直結するレベルのミーティングブリーフを生成してください。

**出力はすべて日本語。** 人名・組織名は原語のまま可。

## Contextの書き方（最重要）
Contextは「事実の列挙」ではなく「戦略的文脈の統合」を目指してください。
以下の3層を統合して1つのナラティブにまとめること：
  1. 直近ニュース・動向（NewsAPI結果）→ タイムリーな外部シグナル
  2. 組織・人物の戦略的ポジション（Google CSE結果）→ 構造的な背景
  3. 過去のやり取り・内部知識（Notion Events結果）→ 関係性の文脈

悪い例: 「X社はGPを任命した。Y氏が参加する。」
良い例: 「X社は直近でGP体制を強化しており、特にヘルスケア領域への投資拡大を
示唆している。本ラウンドテーブルはその戦略的文脈と整合的であり、日本市場での
ポジショニング議論は当社の○○戦略に直接影響する可能性がある。」

## Key Questionsの書き方
表面的な質問（「目的は何か？」）ではなく、
リサーチ結果から導かれる具体的・戦略的な質問を生成すること。

出力JSON:
{
  "purpose": "string — 会議の目的と、なぜ今この会議が重要なのか",
  "context": "string — 3層統合の戦略的文脈（上記ガイドラインに従う）",
  "key_questions": "string — リサーチに基づく具体的な質問（箇条書き）",
  "desired_outcomes": "string — この会議で得たい具体的成果",
  "prep_checklist": "string — 事前に必要な準備（箇条書き）",
  "people_summary": "string — 参加者の背景と会議での役割"
}

箇条書きには (- ) を使用。各フィールドは簡潔だが実行可能な内容にすること。
"""


def _synthesize_meeting_brief(
    enriched_meeting: Dict[str, Any],
    date_iso: str,
    daily_log_id: str = "",
) -> MeetingBrief:
    """Use LLM to synthesize research into a meeting brief.

    Parameters
    ----------
    date_iso : str
        The close/log date (same as 058 Daily Log). This becomes the
        Meeting Brief's ``Date`` property in Notion so that 058, 059,
        and 060 records share a consistent date key.
    """
    meeting_title = enriched_meeting.get("meeting_title", "Untitled Meeting")
    research = enriched_meeting.get("research", {})

    # Build research summary for LLM input — clearly separate 3 source layers
    # so the LLM can integrate them into a strategic narrative.
    events_parts = []
    search_parts = []
    news_parts = []
    for key, data in research.items():
        if data.get("events"):
            events_parts.append(f"[{key}]")
            for ev in data["events"]:
                events_parts.append(
                    f"  - {ev.get('title', '')} ({ev.get('date', '')}): "
                    f"{ev.get('summary', '')}"
                )
        if data.get("search_results"):
            search_parts.append(f"[{key}]")
            for sr in data["search_results"]:
                search_parts.append(
                    f"  - {sr.get('title', '')}: {sr.get('snippet', '')}"
                )
        if data.get("news_results"):
            news_parts.append(f"[{key}]")
            for nr in data["news_results"]:
                search_src = nr.get("source", {})
                source_name = search_src.get("name", "") if isinstance(search_src, dict) else str(search_src)
                news_parts.append(
                    f"  - [{source_name}] {nr.get('title', '')}: "
                    f"{nr.get('description', '')}"
                )

    sections = []
    if news_parts:
        sections.append(
            "## 層1: 直近ニュース（NewsAPI — タイムリーな外部シグナル）\n"
            + "\n".join(news_parts)
        )
    if search_parts:
        sections.append(
            "## 層2: Web検索（Google CSE — 戦略的ポジション・概要）\n"
            + "\n".join(search_parts)
        )
    if events_parts:
        sections.append(
            "## 層3: 内部ナレッジ（Notion Events — 過去のやり取り）\n"
            + "\n".join(events_parts)
        )

    research_text = "\n\n".join(sections) if sections else "(リサーチ結果なし)"

    # Check if we have enough research to warrant an LLM call
    total_results = sum(
        len(d.get("events", [])) + len(d.get("search_results", [])) + len(d.get("news_results", []))
        for d in research.values()
    )

    if total_results == 0:
        # No research results — generate a minimal brief without LLM
        logger.info("No research results for '%s' — generating minimal brief", meeting_title)
        orgs = [o["name"] for o in enriched_meeting.get("organizations", []) if o.get("name")]
        people = [p["name"] for p in enriched_meeting.get("people", []) if p.get("name")]
        topics = [t["name"] for t in enriched_meeting.get("topics", []) if t.get("name")]

        return MeetingBrief(
            title=meeting_title,
            date=date_iso,
            people=people,
            purpose=enriched_meeting.get("purpose_hint", "") or f"会議テーマ: {', '.join(topics or orgs or ['(未指定)'])}",
            context=f"組織: {', '.join(orgs)}\n参加者: {', '.join(people)}\nトピック: {', '.join(topics)}",
            key_questions="- 主要な目的は何か？\n- どのような意思決定が必要か？",
            desired_outcomes="- 明確な次のステップとアクションアイテム",
            prep_checklist="- 関連資料を確認する\n- 質問を準備する",
            links_materials="",
            status="Draft",
            created_by="Auto",
            related_event_ids=enriched_meeting.get("related_event_ids", []),
            daily_log_id=daily_log_id,
        )

    # LLM synthesis
    cache_key = _input_hash(f"{meeting_title}:{research_text[:500]}")
    cache_path = _LLM_CACHE_DIR / f"brief_{cache_key}.json"
    if cache_path.exists():
        logger.info("Brief synthesis cache hit for '%s'", meeting_title)
        synth = load_json(cache_path)
    else:
        from src.llm.router import build_router_from_env, TASK_REASONING
        router = build_router_from_env(cache_dir=_LLM_CACHE_DIR)

        # Build a richer user prompt with meeting structure
        orgs_str = ", ".join(
            o["name"] for o in enriched_meeting.get("organizations", []) if o.get("name")
        ) or "(なし)"
        people_str = ", ".join(
            p["name"] for p in enriched_meeting.get("people", []) if p.get("name")
        ) or "(なし)"
        topics_str = ", ".join(
            t["name"] for t in enriched_meeting.get("topics", []) if t.get("name")
        ) or "(なし)"

        user_prompt = (
            f"会議名: {meeting_title}\n"
            f"日付: {date_iso}\n"
            f"目的ヒント: {enriched_meeting.get('purpose_hint', '未指定')}\n"
            f"関連組織: {orgs_str}\n"
            f"参加者: {people_str}\n"
            f"トピック: {topics_str}\n\n"
            f"以下の3層リサーチ結果を統合して、戦略的なブリーフを生成してください。\n"
            f"Contextでは事実の列挙ではなく、会議の意思決定や議論にどう影響するかまで\n"
            f"踏み込んでください。\n\n"
            f"{research_text}"
        )

        logger.info("Calling LLM for brief synthesis: '%s'", meeting_title)
        result = router.call(
            task_type=TASK_REASONING,
            system=_BRIEF_SYNTHESIS_SYSTEM,
            user=user_prompt,
            model_override="gpt-4o",
            temperature_override=0.3,
            use_cache=False,
        )
        synth = result.parsed
        save_json(cache_path, synth)

    # Build source URLs string
    source_urls = enriched_meeting.get("all_source_urls", [])
    links_str = "\n".join(f"- {url}" for url in source_urls[:15]) if source_urls else ""

    people_names = [p["name"] for p in enriched_meeting.get("people", []) if p.get("name")]

    return MeetingBrief(
        title=meeting_title,
        date=date_iso,
        people=people_names,
        purpose=synth.get("purpose", ""),
        context=synth.get("context", ""),
        key_questions=synth.get("key_questions", ""),
        desired_outcomes=synth.get("desired_outcomes", ""),
        prep_checklist=synth.get("prep_checklist", ""),
        links_materials=links_str,
        status="Draft",
        created_by="Auto",
        related_event_ids=enriched_meeting.get("related_event_ids", []),
        daily_log_id=daily_log_id,
    )


# ═══════════════════════════════════════════════════════════════════
# 5b. DEEP RESEARCH PATH: Generate meeting brief via 073 pipeline
# ═══════════════════════════════════════════════════════════════════


def _build_research_question(meeting: Dict[str, Any]) -> str:
    """Build a research question from meeting metadata."""
    title = meeting.get("meeting_title", "")
    orgs = [o["name"] for o in meeting.get("organizations", []) if o.get("name")]
    people = [p["name"] for p in meeting.get("people", []) if p.get("name")]
    topics = [t["name"] for t in meeting.get("topics", []) if t.get("name")]

    parts = []
    if orgs:
        parts.append("、".join(orgs))
    if people:
        parts.append("、".join(people))
    if topics:
        parts.append("、".join(topics))

    target = "、".join(parts) if parts else title
    return (
        f"{target}について調べてください。"
        f"ミーティング「{title}」の準備として、"
        f"事業概要・最新動向・戦略的ポジション・直近のニュースを調査してください。"
    )


def _init_deep_research_clients() -> Dict[str, Any]:
    """Initialize clients needed by run_single_pipeline().

    Returns dict with llm_client, search_client, news_client, notion_client,
    enable_writeback.
    """
    llm_client = ClaudeClient()

    search_client = None
    try:
        from src.search.google_cse import build_google_cse_from_env
        search_client = build_google_cse_from_env()
    except Exception as e:
        logger.warning("Google CSE client init failed (non-fatal): %s", e)

    news_client = None
    try:
        from src.search.newsapi import build_newsapi_from_env
        news_client = build_newsapi_from_env()
    except Exception as e:
        logger.warning("NewsAPI client init failed (non-fatal): %s", e)

    notion_client = None
    try:
        from src.notion.client import build_notion_client_from_env
        notion_client = build_notion_client_from_env()
    except Exception:
        pass

    enable_writeback = os.getenv("ENABLE_NOTION_WRITEBACK", "").lower() == "true"

    return {
        "llm_client": llm_client,
        "search_client": search_client,
        "news_client": news_client,
        "notion_client": notion_client,
        "enable_writeback": enable_writeback,
    }


def _map_deep_research_to_brief(
    deep_result: Dict[str, Any],
    events: List[Dict[str, Any]],
    meeting: Dict[str, Any],
    date_iso: str,
    daily_log_id: str = "",
) -> MeetingBrief:
    """Map Deep Research results + Events to a MeetingBrief via LLM."""
    meeting_title = meeting.get("meeting_title", "Untitled Meeting")
    people_names = [p["name"] for p in meeting.get("people", []) if p.get("name")]

    # Build research input for the synthesis prompt
    sections = []

    # Deep Research claims
    claims = deep_result.get("claims", [])
    if claims:
        claim_lines = [f"  - {c.get('statement', '')}" for c in claims if c.get("statement")]
        if claim_lines:
            sections.append("## 調査で判明した重要な主張\n" + "\n".join(claim_lines))

    # Deep Research top evidence
    top_evidence = deep_result.get("top_evidence", [])
    if top_evidence:
        ev_lines = [f"  - {e}" for e in top_evidence if e]
        if ev_lines:
            sections.append("## 根拠となるエビデンス\n" + "\n".join(ev_lines))

    # Deep Research memo summary
    memo_summary = deep_result.get("memo_summary", "")
    if memo_summary:
        sections.append(f"## 調査概要\n{memo_summary}")

    # Events from Notion
    if events:
        ev_parts = [
            f"  - {ev.get('title', '')} ({ev.get('date', '')}): {ev.get('summary', '')}"
            for ev in events
        ]
        if ev_parts:
            sections.append("## 内部ナレッジ（Notion Events）\n" + "\n".join(ev_parts))

    research_text = "\n\n".join(sections) if sections else "(リサーチ結果なし)"

    # Reuse the same synthesis prompt as 3-layer research
    from src.llm.router import build_router_from_env, TASK_REASONING
    router = build_router_from_env(cache_dir=_LLM_CACHE_DIR)

    orgs_str = ", ".join(
        o["name"] for o in meeting.get("organizations", []) if o.get("name")
    ) or "(なし)"
    people_str = ", ".join(people_names) or "(なし)"
    topics_str = ", ".join(
        t["name"] for t in meeting.get("topics", []) if t.get("name")
    ) or "(なし)"

    user_prompt = (
        f"会議名: {meeting_title}\n"
        f"日付: {date_iso}\n"
        f"目的ヒント: {meeting.get('purpose_hint', '未指定')}\n"
        f"関連組織: {orgs_str}\n"
        f"参加者: {people_str}\n"
        f"トピック: {topics_str}\n\n"
        f"以下のDeep Research調査結果とイベント情報を統合して、"
        f"戦略的なブリーフを生成してください。\n"
        f"Contextでは事実の列挙ではなく、会議の意思決定や議論にどう影響するかまで\n"
        f"踏み込んでください。\n\n"
        f"{research_text}"
    )

    logger.info("Calling LLM for Deep Research brief mapping: '%s'", meeting_title)
    result = router.call(
        task_type=TASK_REASONING,
        system=_BRIEF_SYNTHESIS_SYSTEM,
        user=user_prompt,
        model_override="gpt-4o",
        temperature_override=0.3,
        use_cache=False,
    )
    synth = result.parsed

    # Build source domains string
    source_domains = deep_result.get("source_domains", [])
    links_str = "\n".join(f"- {d}" for d in source_domains[:15]) if source_domains else ""

    return MeetingBrief(
        title=meeting_title,
        date=date_iso,
        people=people_names,
        purpose=synth.get("purpose", ""),
        context=synth.get("context", ""),
        key_questions=synth.get("key_questions", ""),
        desired_outcomes=synth.get("desired_outcomes", ""),
        prep_checklist=synth.get("prep_checklist", ""),
        links_materials=links_str,
        status="Draft",
        created_by="059_deep_research",
        related_event_ids=[e.get("page_id", "") for e in events if e.get("page_id")],
        daily_log_id=daily_log_id,
    )


def _generate_brief_deep_research(
    meeting: Dict[str, Any],
    date_iso: str,
    daily_log_id: str = "",
) -> MeetingBrief:
    """Generate a meeting brief via Deep Research pipeline (067-072).

    Raises on failure so caller can fall back to 3-layer research.
    """
    question = _build_research_question(meeting)
    run_id = generate_run_id()
    clients = _init_deep_research_clients()

    logger.info(
        "Running Deep Research for '%s' (run_id=%s)",
        meeting.get("meeting_title", ""), run_id,
    )

    result = run_single_pipeline(
        question,
        run_id,
        llm_client=clients["llm_client"],
        search_client=clients["search_client"],
        news_client=clients["news_client"],
        notion_client=clients["notion_client"],
        enable_writeback=clients["enable_writeback"],
    )

    if result.get("status") == "failed":
        raise RuntimeError(f"Deep Research pipeline failed: {result.get('error', 'unknown')}")

    # Supplement with Notion Events (Deep Research doesn't search Events DB)
    all_events = []
    for org in meeting.get("organizations", []):
        if org.get("name"):
            all_events.extend(_search_notion_events(org["name"]))
    for person in meeting.get("people", []):
        if person.get("name"):
            all_events.extend(_search_notion_events(person["name"]))

    brief = _map_deep_research_to_brief(
        result, all_events, meeting, date_iso, daily_log_id,
    )
    logger.info(
        "Deep Research brief generated: '%s' (sources=%d, evidence=%d, claims=%d)",
        brief.title,
        result.get("sources_count", 0),
        result.get("evidence_count", 0),
        result.get("claims_count", 0),
    )
    return brief


# ═══════════════════════════════════════════════════════════════════
# 6. NOTION OUTPUT: Write Meeting Briefs + update Daily Log
# ═══════════════════════════════════════════════════════════════════

import re as _re


def _notion_write_with_retry(
    client,
    db_id: str,
    props: Dict[str, Any],
    existing_id: Optional[str],
    label: str,
    *,
    _max_retries: int = 3,
) -> tuple:
    """Create or update a Notion page, retrying on validation errors.

    On a 400 validation_error that names a specific property (e.g.
    ``"Links / Materials is expected to be url"``), the offending property
    is stripped from *props* and the request is retried.

    Returns ``(page_id, action)`` where action is "created" or "updated".
    Raises on non-recoverable errors.
    """
    for attempt in range(1, _max_retries + 1):
        try:
            if existing_id:
                client.update_page(page_id=existing_id, properties=props)
                return existing_id, "updated"
            else:
                page = client.create_page(parent_db_id=db_id, properties=props)
                return page.get("id", ""), "created"
        except Exception as exc:
            msg = str(exc)
            # Detect "PropertyName is expected to be <type>" pattern
            m = _re.search(
                r"(?:status=400|validation_error).*?[\"']?([A-Za-z /]+?)[\"']?\s+is expected to be\s+(\w+)",
                msg,
            )
            if m and attempt < _max_retries:
                bad_prop = m.group(1).strip()
                expected_type = m.group(2)
                if bad_prop in props:
                    logger.warning(
                        "Notion validation: '%s' expected %s — stripping and retrying "
                        "(attempt %d/%d, brief='%s')",
                        bad_prop, expected_type, attempt, _max_retries, label,
                    )
                    del props[bad_prop]
                    continue
            # Non-recoverable — re-raise
            raise


def _write_meeting_brief_to_notion(
    brief: MeetingBrief,
) -> Dict[str, Any]:
    """Create or update a Meeting Brief page in Notion.

    Returns dict with 'ok', 'page_id', 'action'.
    """
    try:
        meeting_briefs_db_id = get_optional_db_id("NOTION_Meeting_Briefs_ID")
        logger.info(
            "NOTION_Meeting_Briefs_ID from env: '%s'",
            meeting_briefs_db_id or "(empty)",
        )
        if not meeting_briefs_db_id:
            logger.info("NOTION_Meeting_Briefs_ID not set, skipping Notion write")
            return {"ok": False, "error": "NOTION_Meeting_Briefs_ID not configured"}

        logger.info(
            "Writing Meeting Brief to Notion: title='%s' date=%s",
            brief.title, brief.date,
        )

        from src.notion.daily_schema import (
            build_meeting_brief_properties,
            resolve_daily_log_relation_key,
        )
        from src.notion.daily_upsert import safe_truncate
        from src.notion.client import (
            build_notion_client_from_env,
            NotionDataSourceResolver,
        )

        client = build_notion_client_from_env()
        resolver = NotionDataSourceResolver(client=client)
        resolved = resolver.resolve_once(name="meeting_briefs", database_id=meeting_briefs_db_id)

        props = build_meeting_brief_properties(
            title=brief.title,
            date=brief.date,
            daily_log_id=brief.daily_log_id,
            people=brief.people if brief.people else None,
            purpose=safe_truncate(brief.purpose),
            context=safe_truncate(brief.context),
            key_questions=safe_truncate(brief.key_questions),
            desired_outcomes=safe_truncate(brief.desired_outcomes),
            prep_checklist=safe_truncate(brief.prep_checklist),
            links_materials=safe_truncate(brief.links_materials),
            status=brief.status,
            created_by=brief.created_by,
            related_event_ids=brief.related_event_ids if brief.related_event_ids else None,
        )

        # Defensive: resolve "Daily Logs" vs "Daily Log" against actual DB
        # schema.  The Notion API sometimes returns empty `properties` from
        # GET /databases/{id} — in that case, fall back to schema inference
        # by sampling one page via the already-resolved data_source_id.
        db_prop_names: Optional[set] = None  # None = "schema unknown"
        try:
            from src.notion.client import infer_schema_types_from_sample_page

            db_meta = client.get_database(database_id=meeting_briefs_db_id)
            raw_props = db_meta.get("properties", {})

            db_title_parts = db_meta.get("title", [])
            db_title_str = (
                db_title_parts[0].get("plain_text", "")
                if db_title_parts else "(no title)"
            )
            logger.info(
                "Meeting Briefs DB: id=%s title='%s'",
                db_meta.get("id", "?"), db_title_str,
            )

            if raw_props:
                # Normal path — API returned the schema
                db_prop_names = set(raw_props.keys())
                logger.info(
                    "Meeting Briefs DB property names (%d): %s",
                    len(db_prop_names), sorted(db_prop_names),
                )
                for pname, pval in sorted(raw_props.items()):
                    logger.debug("  DB prop: '%s' -> type=%s", pname, pval.get("type", "?"))
            else:
                # Fallback: properties empty — infer from a sample page
                logger.warning(
                    "GET /databases returned 0 properties — "
                    "inferring schema from sample page via data_source_id=%s",
                    resolved.data_source_id[:8],
                )
                try:
                    sample_pages = client.query_data_source(
                        data_source_id=resolved.data_source_id,
                        page_size=1,
                        fetch_all=False,
                    )
                    if sample_pages:
                        inferred = infer_schema_types_from_sample_page(sample_pages[0])
                        db_prop_names = set(inferred.keys())
                        logger.info(
                            "Inferred %d properties from sample page: %s",
                            len(db_prop_names), sorted(db_prop_names),
                        )
                        for pname, ptype in sorted(inferred.items()):
                            logger.debug("  inferred prop: '%s' -> type=%s", pname, ptype)
                    else:
                        logger.warning("No sample pages found — skipping schema validation")
                except Exception as sample_exc:
                    logger.warning("Sample page inference failed (non-fatal): %s", sample_exc)

            # Resolve "Daily Logs" vs "Daily Log"
            if db_prop_names is not None:
                props = resolve_daily_log_relation_key(props, db_prop_names)

            logger.info(
                "Props we want to write (%d): %s",
                len(props), sorted(props.keys()),
            )

            # Only strip unknown properties when we have positive
            # schema evidence (non-empty property list).
            if db_prop_names:
                for key in list(props.keys()):
                    if key not in db_prop_names and key != "Title":
                        logger.warning(
                            "Skipping prop '%s' — not found in Meeting Briefs DB "
                            "(DB has: %s)",
                            key, sorted(db_prop_names),
                        )
                        del props[key]
            else:
                logger.warning(
                    "Schema unknown — sending all %d props without validation",
                    len(props),
                )
        except Exception as e:
            logger.debug("Could not probe Meeting Briefs DB schema (non-fatal): %s", e)

        # Idempotent: check if page exists by Date + Title
        existing_id = None
        try:
            pages = client.query_data_source(
                data_source_id=resolved.data_source_id,
                filter={
                    "and": [
                        {"property": "Date", "date": {"equals": brief.date}},
                        {"property": "Title", "title": {"equals": brief.title}},
                    ]
                },
                page_size=1,
                fetch_all=False,
            )
            if pages:
                existing_id = pages[0].get("id")
        except Exception as e:
            logger.debug("Meeting brief lookup failed (non-fatal): %s", e)

        # Sanity check: "Daily Log" (singular) must never be in Meeting Briefs props
        if "Daily Log" in props:
            logger.error(
                "BUG: 'Daily Log' (singular) found in Meeting Briefs props — "
                "renaming to 'Daily Logs'. Keys before fix: %s",
                sorted(props.keys()),
            )
            props["Daily Logs"] = props.pop("Daily Log")

        logger.info(
            "Meeting Brief Notion props for '%s': %s",
            brief.title, sorted(props.keys()),
        )

        # Write to Notion with retry: on validation_error, strip the
        # offending property and retry once.
        page_id, action = _notion_write_with_retry(
            client, meeting_briefs_db_id, props, existing_id, brief.title,
        )

        return {"ok": True, "page_id": page_id, "action": action}

    except Exception as e:
        logger.error("Meeting brief Notion write failed: %s", e, exc_info=True)
        return {"ok": False, "error": str(e)}


def _get_daily_log_page_id(date_iso: str) -> str:
    """Look up the Daily Log page ID for a given date.

    Returns page_id or empty string.
    """
    try:
        daily_logs_db_id = get_optional_db_id("NOTION_Daily_Logs_ID")
        if not daily_logs_db_id:
            return ""

        from src.notion.client import (
            build_notion_client_from_env,
            NotionDataSourceResolver,
        )
        client = build_notion_client_from_env()
        resolver = NotionDataSourceResolver(client=client)
        resolved = resolver.resolve_once(name="daily_logs", database_id=daily_logs_db_id)

        pages = client.query_data_source(
            data_source_id=resolved.data_source_id,
            filter={
                "property": "LogDate",
                "date": {"equals": date_iso},
            },
            page_size=1,
            fetch_all=False,
        )
        if pages:
            return pages[0].get("id", "")
    except Exception as e:
        logger.warning("Daily Log lookup failed (non-fatal): %s", e)
    return ""


# ═══════════════════════════════════════════════════════════════════
# 7. FOLLOW-UPS & MONITORING (kept from original 059)
# ═══════════════════════════════════════════════════════════════════

def _generate_follow_ups(
    structured: CloseStructured,
    date_iso: str,
) -> List[FollowUp]:
    """Generate follow-up actions from blockers, contacts, and open questions.

    Parameters
    ----------
    date_iso : str
        The close/log date. Follow-ups are dated to the same LogDate
        as the Daily Log (no "next day" offset).
    """
    follow_ups = []

    due_normal = (datetime.fromisoformat(date_iso) + timedelta(days=3)).date().isoformat()
    due_urgent = (datetime.fromisoformat(date_iso) + timedelta(days=1)).date().isoformat()

    for item in structured.items:
        if item.category == "Blocker":
            follow_ups.append(FollowUp(
                title=f"解決: {item.text[:60]}",
                date=date_iso,
                due_date=due_urgent,
                source="Research",
                status="Open",
                priority="P0",
                next_action=f"調査してブロッカーを解消: {item.text[:100]}",
                notes=f"{structured.date} のブロッカー",
            ))

    for person in structured.contact_candidates:
        follow_ups.append(FollowUp(
            title=f"{person} にフォローアップ",
            date=date_iso,
            due_date=due_normal,
            source="Meeting",
            status="Open",
            priority="P1",
            next_action=f"{person} に議論事項について連絡する",
            notes=f"{structured.date} の連絡候補",
        ))

    for q in structured.open_questions:
        follow_ups.append(FollowUp(
            title=f"調査: {q[:60]}",
            date=date_iso,
            due_date=due_normal,
            source="Research",
            status="Open",
            priority="P1",
            next_action=f"調査して判断: {q[:100]}",
            notes=f"{structured.date} の未解決質問",
        ))

    return follow_ups


def _generate_monitoring_suggestions(structured: CloseStructured) -> List[str]:
    """Generate monitoring suggestions from research candidates and progress items."""
    suggestions = []
    for item in structured.items:
        if item.category == "Progress":
            suggestions.append(f"進捗確認: {item.text[:80]}")
    for rc in structured.research_candidates:
        suggestions.append(f"リサーチ注視: {rc}")
    return suggestions


def _build_tomorrow_plan(
    structured: CloseStructured,
    briefs: List[MeetingBrief],
    follow_ups: List[FollowUp],
) -> str:
    """Build a raw tomorrow plan from all preparation outputs (日本語)."""
    lines = [f"明日の計画（{structured.date} の振り返りから生成）:", ""]

    blockers = [f for f in follow_ups if f.priority == "P0"]
    if blockers:
        lines.append("## 最優先 — ブロッカー解消")
        for b in blockers:
            lines.append(f"  - {b.title}")
        lines.append("")

    if briefs:
        lines.append("## ミーティング")
        for brief in briefs:
            people_str = ", ".join(brief.people) if brief.people else ""
            suffix = f"（参加者: {people_str}）" if people_str else ""
            lines.append(f"  - {brief.title}{suffix}")
        lines.append("")

    normal_fups = [f for f in follow_ups if f.priority != "P0"]
    if normal_fups:
        lines.append("## フォローアップ")
        for f in normal_fups:
            lines.append(f"  - [{f.priority}] {f.title}")
        lines.append("")

    if structured.provisional_top3:
        lines.append("## 本日からの継続事項")
        for item in structured.provisional_top3:
            lines.append(f"  - {item}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(
    *,
    date_override: Optional[str] = None,
    verbose: bool = False,
    no_ui: bool = False,
    no_external: bool = False,  # deprecated — kept for CLI compat; ignored
    no_browser: bool = False,
    model: str = "gpt-4o",
) -> dict:
    """Execute the next-day preparation pipeline."""
    load_env()
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    if no_external:
        logger.warning("--no-external is deprecated and ignored. All 3 sources always run.")

    now_jst = datetime.now(tz=JST)
    date_iso = date_override or now_jst.date().isoformat()
    wk = get_iso_week_context(tz=JST)

    logger.info("Starting %s date=%s", SCRIPT_NAME, date_iso)

    # ── 1. Load 058 output ──
    structured = _load_close_structured(date_iso)
    logger.info("Loaded close_structured: %d items", len(structured.items))

    # ── 2. LLM extraction of meeting-clustered targets ──
    extraction = _extract_meeting_targets(structured, date_iso, model=model)
    meetings = extraction.get("meetings", [])
    logger.info("Extracted %d meeting clusters", len(meetings))

    # ── 3. Web UI for editing targets ──
    if not no_ui and meetings:
        try:
            meetings = _launch_target_editor_ui(
                meetings,
                port=_UI_PORT,
                no_browser=no_browser,
            )
            logger.info("User confirmed %d meeting clusters", len(meetings))
        except RuntimeError as e:
            logger.warning("UI cancelled: %s", e)
            meetings = []

    # ── 4. Research each meeting cluster ──
    # All three sources (Events, CSE, NewsAPI) are always executed.
    # --no-external only suppresses Google CSE and NewsAPI at the
    # top-level search functions (graceful no-op if keys are missing).
    # ── 5. Look up Daily Log page ID for linking ──
    daily_log_id = _get_daily_log_page_id(date_iso)
    if daily_log_id:
        logger.info("Found Daily Log page: %s", daily_log_id[:8])

    # ── 6. Synthesize meeting briefs (Deep Research first, fallback to 3-layer) ──
    # Meeting Brief date = date_iso (close date), NOT tomorrow, so that
    # 058 Daily Log, 059 Meeting Briefs, and 060 Morning Commit all
    # share a consistent date key in Notion.
    briefs = []
    for meeting in meetings:
        try:
            brief = _generate_brief_deep_research(meeting, date_iso, daily_log_id=daily_log_id)
        except Exception:
            logger.warning(
                "Deep Research failed for '%s', falling back to 3-layer research",
                meeting.get("meeting_title", ""),
                exc_info=True,
            )
            enriched = _research_meeting_cluster(meeting)
            brief = _synthesize_meeting_brief(enriched, date_iso, daily_log_id=daily_log_id)
        briefs.append(brief)
    logger.info("Generated %d meeting briefs (date=%s)", len(briefs), date_iso)

    # ── 7. Generate follow-ups ──
    follow_ups = _generate_follow_ups(structured, date_iso)
    logger.info("Generated %d follow-ups", len(follow_ups))

    # ── 8. Monitoring suggestions ──
    monitoring = _generate_monitoring_suggestions(structured)
    logger.info("Generated %d monitoring suggestions", len(monitoring))

    # ── 9. Build tomorrow plan ──
    tomorrow_plan = _build_tomorrow_plan(structured, briefs, follow_ups)

    # ── 10. Build NextDayPrep model ──
    prep = NextDayPrep(
        date=date_iso,
        meeting_briefs=briefs,
        follow_ups=follow_ups,
        monitoring_suggestions=monitoring,
        tomorrow_plan_raw=tomorrow_plan,
    )

    # ── 11. Save local output ──
    out_dir = daily_output_dir(NEXT_DAY_PREP_DIR, date_iso)
    out_path = save_json(out_dir / "next_day_prep.json", prep.to_dict())
    logger.info("Saved next_day_prep.json -> %s", out_path)

    # Save enriched research data for debugging
    if enriched_meetings:
        save_json(out_dir / "research_data.json", {
            "meetings": enriched_meetings,
            "date": date_iso,
        })

    # ── 12. Write Meeting Briefs to Notion ──
    brief_page_ids = []
    for brief in briefs:
        notion_result = _write_meeting_brief_to_notion(brief)
        if notion_result.get("ok"):
            brief_page_ids.append(notion_result["page_id"])
            logger.info(
                "Meeting brief '%s': %s -> %s",
                brief.title, notion_result["action"], notion_result["page_id"][:8],
            )
        else:
            logger.warning(
                "Meeting brief '%s' Notion write failed: %s",
                brief.title, notion_result.get("error", "unknown"),
            )

    # ── 13. Notion upsert — prepared layer (Daily Log) ──
    from src.notion.daily_schema import build_daily_log_properties
    from src.notion.daily_upsert import safe_truncate, upsert_daily_log

    notion_props = build_daily_log_properties(
        title=f"Daily Log {date_iso}",
        date=date_iso,
        prep_notes=safe_truncate(tomorrow_plan),
        tomorrow_plan_raw=safe_truncate(tomorrow_plan),
        meeting_brief_ids=brief_page_ids if brief_page_ids else None,
        stage="prepared",
    )
    notion_result = upsert_daily_log(
        date_iso=date_iso,
        properties=notion_props,
        log_label="059_prepared",
    )

    # ── 14. Run metadata ──
    meta = RunMetadata.build(
        notebook=SCRIPT_NAME,
        week_id=wk.week_id,
        counts={
            "meeting_clusters": len(meetings),
            "meeting_briefs": len(briefs),
            "follow_ups": len(follow_ups),
            "monitoring_suggestions": len(monitoring),
            "brief_page_ids": len(brief_page_ids),
            "model": model,
            "no_ui": no_ui,
            "no_external": no_external,
        },
    )
    meta.save(out_dir / "run_metadata.json")

    # ── 15. Summary output ──
    result = {
        "date": date_iso,
        "output_dir": str(out_dir),
        "meeting_clusters": len(meetings),
        "meeting_briefs": len(briefs),
        "follow_ups": len(follow_ups),
        "monitoring_suggestions": len(monitoring),
        "brief_page_ids": brief_page_ids,
    }
    if notion_result:
        result["notion_daily_log"] = notion_result

    print(f"\nNext-day prep saved -> {out_dir}")
    print(f"  Meeting briefs: {len(briefs)} | Follow-ups: {len(follow_ups)} | Monitoring: {len(monitoring)}")
    if brief_page_ids:
        print(f"  Meeting brief Notion pages: {len(brief_page_ids)}")
    if notion_result.get("ok"):
        print(f"  Daily Log: {notion_result['action']} -> {notion_result.get('page_url', '')}")
    else:
        print(f"  Daily Log: FAILED -> {notion_result.get('error', 'unknown')}")

    return result


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="059 Next Day Preparation — meeting briefs with research",
    )
    parser.add_argument("--date", type=str, default=None, help="Override date (YYYY-MM-DD)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--no-ui", action="store_true", help="Skip Web UI (auto-accept targets)")
    parser.add_argument("--no-external", action="store_true", help="Skip external research (Notion Events only)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser for UI")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model (default: gpt-4o)")
    args = parser.parse_args()

    result = run_pipeline(
        date_override=args.date,
        verbose=args.verbose,
        no_ui=args.no_ui,
        no_external=args.no_external,
        no_browser=args.no_browser,
        model=args.model,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
