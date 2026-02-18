# src/telemetry/llm_trace.py
from __future__ import annotations
import json, os, re, time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

SENSITIVE_KEYS = [
    "NOTION_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
]

def _now_iso() -> str:
    # localtime with offset
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def _clip(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= max_chars else s[:max_chars] + "\n...(truncated)"

def _redact_text(s: str) -> str:
    if not s:
        return s
    # simple key=value redaction
    for k in SENSITIVE_KEYS:
        v = os.getenv(k)
        if v:
            s = s.replace(v, f"[REDACTED:{k}]")
    # also redact "Bearer xxx"
    s = re.sub(r"Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*", "Bearer [REDACTED]", s)
    return s

def _safe(obj: Any, max_chars: int = 12000) -> Any:
    # Convert dataclass / objects to dict when possible
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, (dict, list, int, float, bool)) or obj is None:
        # Recursively redact strings inside dict/list
        return _safe_walk(obj, max_chars=max_chars)
    # fallback to string
    return _clip(_redact_text(repr(obj)), max_chars)

def _safe_walk(x: Any, max_chars: int) -> Any:
    if isinstance(x, str):
        return _clip(_redact_text(x), max_chars)
    if isinstance(x, list):
        return [_safe_walk(v, max_chars) for v in x]
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            ks = str(k)
            if ks in SENSITIVE_KEYS:
                out[ks] = f"[REDACTED:{ks}]"
            else:
                out[ks] = _safe_walk(v, max_chars)
        return out
    return x

def _run_dir(project_root: Path, run_id: str) -> Path:
    return project_root / "outputs" / "runs" / run_id

def append_llm_trace(
    *,
    project_root: Path,
    run_id: str,
    event: Dict[str, Any],
    filename_prefix: str = "llm_trace",
) -> Path:
    run_dir = _run_dir(project_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    # stable per-run filename (or time-based if you prefer)
    # Here: 1 file per run_id
    path = run_dir / f"{filename_prefix}.jsonl"

    line = json.dumps(_safe(event), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path
