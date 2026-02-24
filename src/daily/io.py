# src/daily/io.py
"""I/O helpers for the Daily System pipeline (057–061).

Provides deterministic output paths under ``data/daily/`` and
JSON load/save helpers used across all five scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


# ── Output directory roots ────────────────────────────────────────

_DATA_ROOT = Path("data/daily")

CLOSE_RAW_DIR = _DATA_ROOT / "close_raw"
CLOSE_STRUCTURED_DIR = _DATA_ROOT / "close_structured"
NEXT_DAY_PREP_DIR = _DATA_ROOT / "next_day_prep"
MORNING_COMMIT_DIR = _DATA_ROOT / "morning_commit"
NOTION_PUBLISH_DIR = _DATA_ROOT / "notion_publish"


def daily_output_dir(root: Path, date_iso: str) -> Path:
    """Return ``root/YYYY-MM-DD/`` and create it if missing."""
    d = root / date_iso
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── JSON helpers ──────────────────────────────────────────────────

def save_json(path: Path, data: Dict[str, Any]) -> Path:
    """Write *data* as pretty-printed JSON. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_json(path: Path) -> Dict[str, Any]:
    """Read JSON from *path*."""
    return json.loads(path.read_text(encoding="utf-8"))
