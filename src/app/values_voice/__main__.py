#!/usr/bin/env python
# src/app/values_voice/__main__.py
"""CLI launcher for the Values Voice Reflection web UI.

Starts the FastAPI server and opens the browser.

Usage::

    python -m src.app.values_voice --run --lang ja
    python -m src.app.values_voice --run --lang en --write
    python -m src.app.values_voice --run --dry-run
    python -m src.app.values_voice --run --port 8055
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.values.voice import SUPPORTED_LANGUAGES


DEFAULT_PORT = 8055


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Values Voice Reflection — browser UI (FastAPI + Whisper + TTS)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Start the web server (default: shows help)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})",
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
        help="No Whisper, no TTS, no Notion. Returns dummy transcripts.",
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
        default="ja",
        choices=list(SUPPORTED_LANGUAGES),
        help="Voice I/O language (default: ja)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="Don't auto-open the browser",
    )
    args = parser.parse_args()

    if not args.run:
        parser.print_help()
        print(f"\n  Pass --run to start the server on http://localhost:{DEFAULT_PORT}")
        sys.exit(0)

    # Import here to defer FastAPI/uvicorn dependency check.
    try:
        import uvicorn
    except ImportError:
        print(
            "ERROR: uvicorn is required but not installed.\n"
            "Install with:  pip install uvicorn fastapi python-multipart\n"
        )
        sys.exit(1)

    try:
        import fastapi  # noqa: F401
    except ImportError:
        print(
            "ERROR: fastapi is required but not installed.\n"
            "Install with:  pip install uvicorn fastapi python-multipart\n"
        )
        sys.exit(1)

    from src.app.values_voice.server import create_app

    app = create_app(
        values_json_path=args.values_json,
        language=args.lang,
        write_notion=args.write,
        dry_run=args.dry_run,
        review_type=args.review_type,
    )

    url = f"http://localhost:{args.port}"
    print(f"\n  Values Voice Reflection")
    print(f"  Language : {args.lang}")
    print(f"  Mode     : {'DRY RUN' if args.dry_run else 'Live'}")
    print(f"  Notion   : {'Write enabled' if args.write else 'Disabled'}")
    print(f"  URL      : {url}")
    print()

    # Auto-open browser after a short delay.
    if not args.no_browser:
        def _open_browser():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
