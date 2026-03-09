#!/usr/bin/env python
# src/scripts/066_plan_and_review_lp.py
"""Plan & Review LP — Web-based landing page for 061-065.

Launches a browser-based dashboard with cards linking to each sub-script's
Web UI.  Each card opens the target script's server in a new tab.

Usage::

    # Launch Web UI landing page (opens browser)
    python -m src.scripts.066_plan_and_review_lp

    # Without auto-open browser
    python -m src.scripts.066_plan_and_review_lp --no-browser

    # CLI fallback: run a sub-script directly (preserves legacy behavior)
    python -m src.scripts.066_plan_and_review_lp --choice 062 --text "my plan"

    # Debug logging
    python -m src.scripts.066_plan_and_review_lp -v
"""

# NOTE: Do NOT use ``from __future__ import annotations`` here.
# FastAPI + Pydantic v2 need runtime type objects for endpoint parameters.

import argparse
import importlib
import json
import logging
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env, setup_logging

logger = logging.getLogger("066_plan_and_review_lp")

SCRIPT_NAME = "066_plan_and_review_lp"
_UI_PORT = 8066

MENU_OPTIONS = {
    "061": {
        "label": "Daily Console",
        "label_ja": "日次オペレーション",
        "description": "Morning View — pipeline management & Notion publish",
        "module": "src.scripts.061_notion_publish_morning_view",
        "port": 8061,
        "is_server": True,
    },
    "062": {
        "label": "Weekly Planning",
        "label_ja": "週次計画",
        "description": "Set this week's Big 3, success criteria, execution plan",
        "module": "src.scripts.062_weekly_intent_planning",
        "port": 8062,
        "is_server": True,
    },
    "063": {
        "label": "Weekly Review",
        "label_ja": "週次振り返り",
        "description": "Wins 3, improvements, value alignment, adjustments",
        "module": "src.scripts.063_weekly_review",
        "port": 8063,
        "is_server": True,
    },
    "064": {
        "label": "Monthly Planning",
        "label_ja": "月次計画",
        "description": "Monthly Big 3, strategic rationale, risks, weekly breakdown",
        "module": "src.scripts.064_monthly_intent_planning",
        "port": 8064,
        "is_server": True,
    },
    "065": {
        "label": "Monthly Review",
        "label_ja": "月次振り返り",
        "description": "Successes, improvements, value adjustments, structural lessons",
        "module": "src.scripts.065_monthly_review",
        "port": 8065,
        "is_server": True,
    },
}


def _is_port_in_use(port: int) -> bool:
    """Check if a local port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ── FastAPI App ───────────────────────────────────────────────────


def _build_app():
    """Create the FastAPI landing page application."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="066 Plan & Review LP")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _build_html()

    @app.get("/api/status")
    async def get_status():
        """Check which sub-servers are running."""
        status = {}
        for key, opt in MENU_OPTIONS.items():
            status[key] = {
                "label": opt["label"],
                "port": opt["port"],
                "running": _is_port_in_use(opt["port"]),
            }
        return {"ok": True, "status": status}

    @app.post("/api/launch/{script_id}")
    async def launch_script(script_id: str):
        """Launch a sub-script server as a background subprocess."""
        if script_id not in MENU_OPTIONS:
            return JSONResponse({"ok": False, "error": "Unknown script"}, status_code=400)

        opt = MENU_OPTIONS[script_id]
        port = opt["port"]

        if _is_port_in_use(port):
            return {"ok": True, "status": "already_running", "port": port}

        cmd = [sys.executable, "-m", opt["module"], "--no-browser"]
        logger.info("Launching %s: %s", script_id, " ".join(cmd))

        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        return {"ok": True, "status": "launched", "port": port}

    return app


# ── HTML Builder ──────────────────────────────────────────────────


def _build_html() -> str:
    """Build the landing page HTML."""
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
    now = datetime.now(JST)
    date_str = now.strftime("%Y-%m-%d")
    week_num = now.isocalendar()[1]
    month_name = now.strftime("%B %Y")

    # Build card HTML
    cards_html = ""
    for key, opt in MENU_OPTIONS.items():
        cards_html += f'''
        <div class="card" id="card-{key}">
          <div class="card-header">
            <span class="card-badge">{key}</span>
            <span class="card-status" id="status-{key}"></span>
          </div>
          <div class="card-title">{opt["label"]}</div>
          <div class="card-title-ja">{opt["label_ja"]}</div>
          <div class="card-desc">{opt["description"]}</div>
          <div class="card-actions">
            <button class="btn btn-primary" onclick="openScript('{key}', {opt['port']})">
              Open
            </button>
            <button class="btn btn-secondary" id="launch-btn-{key}"
                    onclick="launchScript('{key}')">
              Launch Server
            </button>
          </div>
        </div>'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Plan & Review</title>
<style>
:root {{
  --bg: #ffffff; --surface: #f8f9fa; --surface2: #e9ecef;
  --border: #dee2e6; --text: #212529; --text2: #6c757d;
  --accent: #7c3aed; --accent-light: #6d28d9;
  --green: #10b981; --red: #ef4444; --orange: #f59e0b;
  --blue: #3b82f6; --cyan: #06b6d4;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg); color: var(--text);
  min-height: 100vh; display: flex; flex-direction: column;
  align-items: center; padding: 40px 20px;
}}

.header {{
  text-align: center; margin-bottom: 40px;
}}
.header h1 {{
  font-size: 28px; color: var(--accent-light); margin-bottom: 4px;
}}
.header .subtitle {{
  font-size: 14px; color: var(--text2);
}}

.cards-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px; max-width: 960px; width: 100%;
}}

.card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 24px;
  transition: border-color 0.2s, transform 0.1s;
}}
.card:hover {{
  border-color: var(--accent); transform: translateY(-2px);
}}
.card-header {{
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 12px;
}}
.card-badge {{
  font-size: 12px; font-weight: 700; color: var(--accent-light);
  background: rgba(124,58,237,0.15); padding: 4px 10px;
  border-radius: 6px; letter-spacing: 0.5px;
}}
.card-status {{
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--border);
}}
.card-status.running {{ background: var(--green); }}
.card-title {{
  font-size: 18px; font-weight: 600; margin-bottom: 2px;
}}
.card-title-ja {{
  font-size: 13px; color: var(--text2); margin-bottom: 8px;
}}
.card-desc {{
  font-size: 13px; color: var(--text2); line-height: 1.5;
  margin-bottom: 16px;
}}
.card-actions {{
  display: flex; gap: 8px;
}}

.btn {{
  padding: 8px 18px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface2); color: var(--text); cursor: pointer;
  font-size: 13px; font-weight: 500; transition: all 0.15s;
}}
.btn:hover {{ background: var(--border); }}
.btn-primary {{
  background: var(--accent); border-color: var(--accent); color: #fff;
}}
.btn-primary:hover {{ background: #6d28d9; }}
.btn-secondary {{
  font-size: 12px; padding: 6px 12px;
}}

.footer {{
  margin-top: 40px; text-align: center;
  font-size: 13px; color: var(--text2);
}}
.footer .date-info {{
  display: flex; gap: 16px; justify-content: center;
}}
.footer .date-info span {{
  display: flex; align-items: center; gap: 4px;
}}
</style>
</head>
<body>

<div class="header">
  <h1>Plan & Review</h1>
  <div class="subtitle">researchOS - Plan &amp; Review Layer</div>
</div>

<div class="cards-grid">
  {cards_html}
</div>

<div class="footer">
  <div class="date-info">
    <span>{date_str} JST</span>
    <span>Week {week_num:02d}</span>
    <span>{month_name}</span>
  </div>
</div>

<script>
// Check server status on load and periodically
async function checkStatus() {{
  try {{
    const resp = await fetch('/api/status');
    const data = await resp.json();
    if (!data.ok) return;

    for (const [id, info] of Object.entries(data.status)) {{
      const dot = document.getElementById(`status-${{id}}`);
      const launchBtn = document.getElementById(`launch-btn-${{id}}`);
      if (dot) {{
        dot.className = info.running ? 'card-status running' : 'card-status';
      }}
      if (launchBtn) {{
        if (info.running) {{
          launchBtn.textContent = 'Running';
          launchBtn.disabled = true;
          launchBtn.style.opacity = '0.5';
        }} else {{
          launchBtn.textContent = 'Launch Server';
          launchBtn.disabled = false;
          launchBtn.style.opacity = '1';
        }}
      }}
    }}
  }} catch (e) {{
    // Silently skip
  }}
}}

function openScript(id, port) {{
  window.open(`http://localhost:${{port}}`, `_blank_${{id}}`);
}}

async function launchScript(id) {{
  const btn = document.getElementById(`launch-btn-${{id}}`);
  btn.textContent = 'Launching...';
  btn.disabled = true;

  try {{
    const resp = await fetch(`/api/launch/${{id}}`, {{ method: 'POST' }});
    const data = await resp.json();
    if (data.ok) {{
      // Wait for server to start, then check status
      setTimeout(async () => {{
        await checkStatus();
        // Auto-open if newly launched
        if (data.status === 'launched') {{
          setTimeout(() => openScript(id, data.port), 1500);
        }}
      }}, 2000);
    }}
  }} catch (e) {{
    btn.textContent = 'Launch Failed';
    setTimeout(() => {{
      btn.textContent = 'Launch Server';
      btn.disabled = false;
    }}, 2000);
  }}
}}

// Initial check + periodic refresh
checkStatus();
setInterval(checkStatus, 5000);
</script>
</body>
</html>'''


# ── CLI fallback ──────────────────────────────────────────────────


def _run_pipeline_script(
    module_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Import and run a pipeline script's run_pipeline function."""
    mod = importlib.import_module(module_name)
    run_fn = getattr(mod, "run_pipeline")

    kwargs: Dict[str, Any] = {
        "verbose": args.verbose,
    }
    if args.date:
        kwargs["date_override"] = args.date
    if args.text:
        kwargs["text"] = args.text
    if args.record:
        kwargs["record"] = True
    if args.record_wizard:
        kwargs["record_wizard"] = True
    if args.record_seconds != 120:
        kwargs["record_seconds"] = args.record_seconds
    if args.language != "ja":
        kwargs["language"] = args.language
    if args.no_llm:
        kwargs["no_llm"] = True
    if args.non_interactive:
        kwargs["non_interactive"] = True

    return run_fn(**kwargs)


# ── Server launcher ──────────────────────────────────────────────


def run_server(
    *,
    port: int = _UI_PORT,
    verbose: bool = False,
    no_browser: bool = False,
) -> None:
    """Launch the landing page Web UI server."""
    load_env()
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    logger.info("Starting %s on port %d", SCRIPT_NAME, port)

    import uvicorn

    app = _build_app()

    url = f"http://localhost:{port}"
    if not no_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    logger.info("Plan & Review LP available at %s", url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ── CLI ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="066 Plan & Review LP — Web-based landing page for 061-065",
    )
    parser.add_argument("--choice", type=str, default=None,
                        choices=list(MENU_OPTIONS.keys()),
                        help="Skip Web UI and run this script directly (CLI mode)")

    # Common flags forwarded to sub-scripts (CLI mode only)
    parser.add_argument("--date", type=str, default=None,
                        help="Override date (YYYY-MM-DD)")
    parser.add_argument("--text", type=str, default=None,
                        help="Input text directly")

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
    parser.add_argument("--port", type=int, default=_UI_PORT,
                        help=f"Web UI server port (default: {_UI_PORT})")
    parser.add_argument("--no-browser", action="store_true", dest="no_browser",
                        help="Don't auto-open browser")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")
    args = parser.parse_args()

    # CLI fallback mode: --choice with sub-script flags
    if args.choice:
        load_env()
        setup_logging(logging.DEBUG if args.verbose else logging.INFO)

        opt = MENU_OPTIONS[args.choice]
        print(f"\n  Running: [{args.choice}] {opt['label']}")
        print(f"  {'---' * 14}\n")

        if args.choice == "061":
            # 061 is always a server subprocess
            cmd = [sys.executable, "-m", opt["module"]]
            if args.verbose:
                cmd.append("-v")
            try:
                subprocess.run(cmd, check=True)
            except KeyboardInterrupt:
                print("\n  Server stopped.")
            except subprocess.CalledProcessError as e:
                print(f"\n  Script exited with code {e.returncode}")
        else:
            result = _run_pipeline_script(opt["module"], args)
            if result:
                print(json.dumps(result, indent=2))
    else:
        # Default: launch Web UI landing page
        run_server(
            port=args.port,
            verbose=args.verbose,
            no_browser=args.no_browser,
        )


if __name__ == "__main__":
    main()
