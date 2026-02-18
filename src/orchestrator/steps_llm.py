# src/orchestrator/steps_llm.py
from __future__ import annotations
from typing import Tuple
from typing import Optional
"""
LLM steps for the Notebook Builder loop.

Implements two step handlers meant to be called from src/orchestrator/one_loop.py:
  - LLM_PLAN:
      Read Task (+ optional last error + notebook excerpt), ask Claude for a structured plan (JSON),
      write plan into state.json, and enqueue downstream steps:
        * SCAFFOLD_HEADERS (structure override)
        * LLM_IMPLEMENT for PATCH_CELL steps
        * VERIFY_NOTEBOOK for VERIFY steps

  - LLM_IMPLEMENT:
      Take one planned PATCH_CELL step, ask Claude for a patch (JSON),
      enqueue APPLY_PATCH (and optionally VERIFY_NOTEBOOK).

Design principles
-----------------
- state.json is the single source of truth for execution.
- Notion is an audit log (best-effort appends where possible).
- Claude outputs MUST be JSON (schema-enforced).
- Keep patches small and verifiable: PATCH -> PREFIX verify -> (eventually FULL verify).

IMPORTANT (Claude schema compatibility)
--------------------------------------
Some Claude structured-output implementations do NOT support JSON Schema constructs
that compile down to `oneOf` (e.g., using `type: ["integer","null"]`).
This file avoids such constructs by making nullable fields OPTIONAL (omitted) instead of null-typed.

Policy decisions applied in this version
----------------------------------------
- REPOS_IS_MODULE fix path: REMOVED entirely.
- build_repos: forbidden in notebook patches (LLM_IMPLEMENT output) to keep Notebook side "BaseRepo-only" policy.
- scaffold_headers helpers: imported with robust fallbacks (never hard-fail on ImportError).
- _maybe_append_plan_to_notion: hardened with hasattr checks (safe when repos is partial).
"""

import json
import os
import time

from dataclasses import dataclass
from pathlib import Path
from src.telemetry.llm_trace import append_llm_trace, _now_iso  # _now_isoはexportしてもよい
from typing import Any, Dict, List, Optional, Union
import re
import nbformat

from src.llm.claude_client import ClaudeClient, ClaudeStructuredOutputError
from src.state.state_store import (
    StateStore,
    enqueue,
    new_task_item,
    update_task_item,
    mark_done,
    mark_failed,
    bump_metric,
)

# --- Discover project root (directory containing "src/") ---
cwd = Path.cwd()
project_root = None

cursor = cwd
for _ in range(8):  # safety limit to avoid infinite loop
    if (cursor / "src").is_dir():
        project_root = cursor
        break
    if cursor.parent == cursor:
        break
    cursor = cursor.parent
    
# -----------------------------
# Structure seed schema (LLM produces ONLY structure)
# -----------------------------
STRUCTURE_SEED_JSON_SCHEMA: Dict[str, Any] = {
    "name": "structure_seed_schema_v1",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "structure": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "cell_index": {"type": "integer"},
                        "title": {"type": "string"},
                        "overview": {"type": "string"},
                        "io": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["cell_index", "title"],
                },
            },
            "seed_summary": {"type": "string"},
       },
        "required": ["structure"],
    },
}

# -----------------------------
# Debug: Structure early visibility
# -----------------------------
def _debug_print_structure(
    *,
    prefix: str,
    task_page_id: str,
    proposal_page_id: str,
    notebook_path: str,
    structure: List[Dict[str, Any]],
    plan_summary: str = "",
    max_chars: int = 6000,
) -> None:
    """
    Print planned STRUCTURE as early as possible (LLM_PLAN),
    so the operator sees it BEFORE SCAFFOLD_HEADERS runs.
    Also stores a trimmed copy into state via caller (_state_patch) when needed.
    """
    try:
        payload = {
            "task_page_id": str(task_page_id),
            "proposal_page_id": str(proposal_page_id),
            "notebook_path": str(notebook_path),
            "plan_summary": str(plan_summary or ""),
            "structure": structure,
        }
        s = json.dumps(payload, ensure_ascii=False, indent=2)
        if len(s) > int(max_chars):
            s = s[: int(max_chars)] + "\n...(truncated)"
        print(f"\n{prefix} Planned STRUCTURE (early):")
        print(s)
    except Exception as e:
        print(f"\n{prefix} Planned STRUCTURE print failed: {type(e).__name__}: {e}")

 
# -----------------------------
# StateStore compatibility
# -----------------------------

def _normalize_structure_bootstrap(structure, *, min_cells: int = 3) -> list[dict]:
    """
    - cell_index を int 化
    - 重複/欠損を除去
    - 1..N を最低 min_cells まで埋める
    """
    out = []
    seen = set()

    if isinstance(structure, list):
        for it in structure:
            if not isinstance(it, dict):
                continue
            raw_idx = it.get("cell_index")
            idx = None
            if isinstance(raw_idx, int):
                idx = raw_idx
            elif isinstance(raw_idx, str):
                m = re.search(r"\d+", raw_idx)
                if m:
                    idx = int(m.group(0))
            
            if idx is None or idx <= 0:
                continue
            if idx <= 0 or idx in seen:
                continue
            title = str(it.get("title") or "").strip()
            if not title:
                continue
            seen.add(idx)
            out.append(
                {
                    "cell_index": idx,
                    "title": title,
                    "overview": str(it.get("overview") or ""),
                    "io": str(it.get("io") or ""),
                    "notes": str(it.get("notes") or ""),
                }
            )

    out.sort(key=lambda d: d["cell_index"])

    # ---- 必須セル（01/02/03..）を最低限埋める ----
    def _ensure(idx: int, title: str):
        nonlocal out, seen
        if idx in seen:
            return
        seen.add(idx)
        out.append({"cell_index": idx, "title": title, "overview": "", "io": "", "notes": ""})

    _ensure(1, "Setup & config")
    _ensure(2, "Schema introspection (Notion repos)")
    _ensure(3, "Discovery Candidate Extraction")

    # min_cells まで placeholder を埋める（後で 1つずつ置換する想定）
    i = 4
    while len(out) < min_cells:
        _ensure(i, f"Placeholder / awaiting task ({i})")
        i += 1

    out.sort(key=lambda d: d["cell_index"])
    return out
    
def _state_read(store: StateStore) -> Dict[str, Any]:
    """
    Support multiple StateStore variants:
      - store.load() -> dict
      - store.get_state() -> dict
      - functions read_state(store) -> dict
    """
    if hasattr(store, "load"):
        return store.load()  # type: ignore[no-any-return]
    if hasattr(store, "get_state"):
        return store.get_state()  # type: ignore[no-any-return]

    try:
        from src.state.state_store import read_state  # type: ignore
    except Exception as e:
        raise AttributeError(
            "StateStore has no load()/get_state(), and read_state() import failed."
        ) from e
    return read_state(store)  # type: ignore[no-any-return]


def _state_write(store: StateStore, st: Dict[str, Any]) -> None:
    """
    Support multiple StateStore variants.

    - If store.update(fn) style, pass a lambda.
    - Else if store.save(dict) style, call save.
    - Else fallback to function write_state(store, dict).

    NOTE:
    This method may fully overwrite state in some StateStore variants.
    Prefer _state_patch() for small updates.
    """
    # 1) update(fn)
    if hasattr(store, "update"):
        try:
            store.update(lambda _old: st)  # type: ignore[misc]
            return
        except TypeError:
            pass

    # 2) save(dict)
    if hasattr(store, "save"):
        store.save(st)  # type: ignore[misc]
        return

    # 3) function fallback
    try:
        from src.state.state_store import write_state  # type: ignore
    except Exception as e:
        raise AttributeError(
            "StateStore has no update(fn)/save(dict), and write_state() import failed."
        ) from e

    write_state(store, st)  # type: ignore[misc]


def _state_patch(store: StateStore, patch: Dict[str, Any]) -> None:
    """
    Merge patch into state (shallow merge at top-level).
    Prevents overwriting queue/memory_refs updated by other steps.
    """

    def _fn(old: Dict[str, Any]) -> Dict[str, Any]:
        st = dict(old or {})
        st.update(patch)
        return st

    if hasattr(store, "update"):
        store.update(_fn)  # type: ignore[misc]
        return

    st = _state_read(store)
    st = _fn(st)
    _state_write(store, st)


def _cancel_todo_for_same_proposal(
    *,
    store: StateStore,
    task_page_id: str,
    proposal_page_id: str,
    keep_task_item_id: str,
) -> int:
    """
    Cancel duplicated TODO items for the same task/proposal before enqueueing a new plan.
    Safer than deleting; keeps auditability.

    Cancels: SCAFFOLD_HEADERS / LLM_IMPLEMENT / APPLY_PATCH / VERIFY_NOTEBOOK
    """
    cancelled = {"n": 0}

    def _fn(st: Dict[str, Any]) -> Dict[str, Any]:
        q = list(st.get("queue") or [])
        newq = []
        for it in q:
            if not isinstance(it, dict):
                newq.append(it)
                continue
            if it.get("task_item_id") == keep_task_item_id:
                newq.append(it)
                continue
            if (it.get("status") or "TODO") != "TODO":
                newq.append(it)
                continue

            tgt = it.get("target") or {}
            if not isinstance(tgt, dict):
                newq.append(it)
                continue

            same = (str(tgt.get("task_page_id") or "") == str(task_page_id)) and (
                str(tgt.get("proposal_page_id") or "") == str(proposal_page_id)
            )
            if not same:
                newq.append(it)
                continue

            t = (it.get("type") or "").upper()
            if t in ("SCAFFOLD_HEADERS", "LLM_IMPLEMENT", "APPLY_PATCH", "VERIFY_NOTEBOOK"):
                it2 = dict(it)
                it2["status"] = "CANCELLED"
                it2["last_error"] = "Superseded by new LLM_PLAN"
                it2["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                newq.append(it2)
                cancelled["n"] += 1
            else:
                newq.append(it)

        st["queue"] = newq
        return st

    if hasattr(store, "update"):
        store.update(_fn)  # type: ignore[misc]
    else:
        st = _state_read(store)
        st = _fn(st)
        _state_write(store, st)

    return int(cancelled["n"])


# -----------------------------
# JSON Schemas for Claude
# -----------------------------
# NOTE: Avoid constructs that compile to oneOf (e.g., type arrays like ["integer","null"]).
# Make fields OPTIONAL instead (omit key when absent).

PLAN_JSON_SCHEMA: Dict[str, Any] = {
    "name": "plan_schema_v1",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "plan_summary": {"type": "string"},
            # per-cell structure for Cell01+
            "structure": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "cell_index": {"type": "integer"},
                        "title": {"type": "string"},
                        "overview": {"type": "string"},
                        "io": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["cell_index", "title"],
                },
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": ["PATCH_CELL", "VERIFY"]},
                        "mode": {"type": "string", "enum": ["REPLACE", "APPEND", "INSERT"]},
                        "cell_index": {"type": "integer"},
                        "cell_type": {"type": "string", "enum": ["code", "markdown"]},
                        "intent": {"type": "string"},
                        "acceptance": {"type": "array", "items": {"type": "string"}},
                        "run_mode": {"type": "string", "enum": ["PREFIX", "FULL"]},
                        "up_to_cell_index": {"type": "integer"},
                    },
                    "required": ["kind"],
                },
            },
            "notion_update": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "append_to_task": {"type": "string"},
                    "append_to_proposal": {"type": "string"},
                },
            },
        },
        "required": ["plan_summary", "structure", "steps"],
    },
}

IMPLEMENT_JSON_SCHEMA: Dict[str, Any] = {
    "name": "implement_schema_v1",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {"type": "string", "enum": ["REPLACE", "APPEND", "INSERT"]},
            "cell_index": {"type": "integer"},  # OPTIONAL (omit for APPEND)
            "cell_type": {"type": "string", "enum": ["code", "markdown"]},
            "new_source": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["mode", "cell_type", "new_source"],
    },
}

# -----------------------------
# Fixed context pack (5 points)
# -----------------------------

REPO_CONTRACT_SNIPPET = """\
Repo API contract (MUST FOLLOW)
- Use repo_query(repo, filter=..., sorts=..., page_size=...) only.
- DO NOT call repo.query_pages directly.
- DO NOT use Notion REST (no notion_get/notion_post/requests/"/v1/").
- Property names you use MUST exist in schema_snapshot for the target DB.
"""
def _extract_runtime_context_pack(target: Dict[str, Any]) -> Dict[str, Any]:
    """
    one_loop.py may inject a context_pack into LLM_PLAN / LLM_IMPLEMENT targets.
    We keep this best-effort and optional to avoid breaking older queues.
    Expected keys (best-effort):
      - available_symbols: list[str]
      - cell_facts_tail: list[dict]
      - upstream_sources: list[{cell_index:int, source:str}]
      - cell_facts_path: str (artifacts path pointer)
      - schema_snapshot / last_error (already handled elsewhere)
    """
    cp = target.get("context_pack")
    return cp if isinstance(cp, dict) else {}

def _build_upstream_context_text(
    *,
    runtime_cp: Dict[str, Any],
    max_chars: int = 9000,
) -> str:
    """
    Human-readable block (not part of 5-point context pack) to reduce symbol drift across cells.
    """
    if not isinstance(runtime_cp, dict) or not runtime_cp:
        return ""

    payload: Dict[str, Any] = {}
    # Keep these small + stable
    if isinstance(runtime_cp.get("available_symbols"), list):
        payload["available_symbols"] = runtime_cp.get("available_symbols")
    if isinstance(runtime_cp.get("cell_facts_tail"), list):
        payload["cell_facts_tail"] = runtime_cp.get("cell_facts_tail")
    if isinstance(runtime_cp.get("upstream_sources"), list):
        payload["upstream_sources"] = runtime_cp.get("upstream_sources")
    if runtime_cp.get("cell_facts_path"):
        payload["cell_facts_path"] = str(runtime_cp.get("cell_facts_path"))

    if not payload:
        return ""

    s = _safe_json_dump(payload, max_chars=max_chars)
    return (
        "UPSTREAM CONTEXT (runtime facts; MUST FOLLOW):\n"
        "- available_symbols: names known to exist upstream (avoid NameError)\n"
        "- upstream_sources: prior cell code sources (prefer reusing variables)\n"
        "- cell_facts_tail: recent per-cell facts\n"
        "- cell_facts_path: pointer to artifacts snapshot (if present)\n"
        "----- BEGIN UPSTREAM CONTEXT JSON -----\n"
        f"{s}\n"
        "----- END UPSTREAM CONTEXT JSON -----\n\n"
    )

def _extract_structure_json_min(cell00_src: str, *, max_chars: int = 2500) -> str:
    """
    Extract the STRUCTURE_JSON block (commented JSON) from Cell00.
    Returns a MINIMAL string (may be empty if not found).
    """
    if not cell00_src:
        return ""

    begin = "=== STRUCTURE_JSON:BEGIN ==="
    end = "=== STRUCTURE_JSON:END ==="

    a = cell00_src.find(begin)
    b = cell00_src.find(end, a + 1) if a != -1 else -1
    block = ""
    if a != -1 and b != -1 and b > a:
        block = cell00_src[a + len(begin) : b]
    else:
        # If markers are missing, try best-effort: take only lines that look like "# {", "# [", "#  }", etc.
        # This still might be empty, which is OK (we fallback).
        block = cell00_src

    import re
    lines = []
    for ln in (block or "").splitlines():
        m = re.match(r"^\s*#\s?(.*)$", ln)
        if not m:
            continue
        s = (m.group(1) or "").rstrip()
        if not s:
            continue
        # Keep likely JSON-ish lines (conservative)
        if s.startswith("[") or s.startswith("{") or s.startswith("]") or s.startswith("}") or s.startswith('"') or s.startswith(",") or re.match(r'^\s*"\w', s):
            lines.append(s)

    txt = "\n".join(lines).strip()
    if len(txt) > max_chars:
        txt = txt[:max_chars] + "\n...(truncated)"
    return txt

def _ensure_structure_payload(
    *,
    cell00_src: str,
    hint: Dict[str, Any],
    plan_structure: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Ensure we ALWAYS provide something non-empty for "Notebook structure".
    If STRUCTURE_JSON is empty/unparseable, we fallback to:
      1) plan_structure (if present)
      2) stable minimal skeleton (Cell00/01/02+)
    Output is a dict so the caller can dump it as JSON.
    """
    txt = _extract_structure_json_min(cell00_src)

    # 1) Try parsing the extracted JSON text if it looks like a JSON array
    if txt and (txt.lstrip().startswith("[") and txt.rstrip().endswith("]")):
        try:
            arr = json.loads(txt)
            if isinstance(arr, list) and arr:
                # keep minimal fields only
                out = []
                for it in arr:
                    if not isinstance(it, dict):
                        continue
                    if it.get("cell_index") is None:
                        continue
                    try:
                        idx = int(it.get("cell_index"))
                    except Exception:
                        continue
                    out.append(
                        {
                            "cell_index": idx,
                            "title": str(it.get("title") or ""),
                            "overview": str(it.get("overview") or ""),
                            "io": str(it.get("io") or ""),
                            "notes": str(it.get("notes") or ""),
                        }
                    )
                out = [x for x in out if x.get("cell_index") is not None]
                if out:
                    out.sort(key=lambda d: int(d["cell_index"]))
                    return {"kind": "STRUCTURE_JSON", "items": out}
        except Exception:
            pass

    # 2) Fallback to plan_structure (already normalized usually)
    if isinstance(plan_structure, list) and plan_structure:
        out2 = []
        for it in plan_structure:
            if not isinstance(it, dict):
                continue
            if it.get("cell_index") is None or not it.get("title"):
                continue
            try:
                idx = int(it.get("cell_index"))
            except Exception:
                continue
            out2.append(
                {
                    "cell_index": idx,
                    "title": str(it.get("title") or ""),
                    "overview": str(it.get("overview") or ""),
                    "io": str(it.get("io") or ""),
                    "notes": str(it.get("notes") or ""),
                }
            )
        if out2:
            out2.sort(key=lambda d: int(d["cell_index"]))
            return {"kind": "PLAN_STRUCTURE_FALLBACK", "items": out2}

    # 3) Stable minimal skeleton (guaranteed non-empty)
    start_cell = 2
    try:
        start_cell = int((hint or {}).get("start_cell_index") or 2)
    except Exception:
        start_cell = 2

    skeleton = [
        {"cell_index": 0, "title": "Locked overview", "overview": "", "io": "", "notes": "Cell00 locked. May not contain STRUCTURE_JSON yet."},
        {"cell_index": 1, "title": "Setup & config", "overview": "", "io": "", "notes": "Cell01 setup is the source of truth for repos/claude."},
        {"cell_index": 2, "title": "Schema introspection (fixed)", "overview": "", "io": "Out: DB_PROP_TYPES, SCHEMA_REPORT", "notes": "Cell02 is fixed and must NOT be modified."},
        {"cell_index": max(3, start_cell), "title": "Work cells start here", "overview": "", "io": "", "notes": "Planner should design Cell03+."},
    ]
    return {"kind": "SKELETON_FALLBACK", "items": skeleton}

def _build_context_pack_5(
    *,
    task_objective_constraints: str,
    structure_payload: Dict[str, Any],
    schema_snapshot: Dict[str, Any],
    last_error_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the ONLY context that LLM receives (5 points fixed).
    """
    return {
        "task": task_objective_constraints,
        "notebook_structure": structure_payload,
        "repo_contract": REPO_CONTRACT_SNIPPET,
        "schema_snapshot": schema_snapshot,
        "last_error": last_error_summary,
    }

def _min_last_error_summary(last_error: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep last_error super small and stable.
    """
    if not isinstance(last_error, dict):
        return {}
    out: Dict[str, Any] = {}
    # IMPORTANT: do NOT emit nulls for optional integer fields (Claude schema compatibility)
    fci = last_error.get("failing_cell_index")
    if fci is None:
        fci = last_error.get("cell_index")
    try:
        if fci is not None:
            out["failing_cell_index"] = int(fci)
    except Exception:
        pass
    out["exception"] = str(last_error.get("exception_message") or "")
    out["error_summary"] = str(last_error.get("error_summary") or "")
    out["cause_hint"] = str(last_error.get("next_action") or last_error.get("category") or "")

    # keep traceback to 3-10 lines max
    tb = str(last_error.get("traceback") or "")
    if tb:
        lines = [ln for ln in tb.splitlines() if ln.strip()]
        out["traceback"] = "\n".join(lines[-8:])  # last 8 lines
    else:
        out["traceback"] = ""
    return out

# -----------------------------
# Utilities: notebook context
# -----------------------------


def _read_notebook_cells(notebook_path: Union[str, Path]) -> List[Dict[str, Any]]:
    nb = nbformat.read(str(Path(notebook_path).resolve()), as_version=4)
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(nb.cells):
        out.append(
            {
                "index": i,
                "cell_type": c.get("cell_type"),
                "source": (c.get("source") or ""),
            }
        )
    return out

def _read_notebook_cell_sources(notebook_path: Union[str, Path], indices: List[int], *, max_chars: int = 4000) -> Dict[int, str]:
    """
    Read selected cells by index (0-based) and return {index: source}.
    """
    nb = nbformat.read(str(Path(notebook_path).resolve()), as_version=4)
    out: Dict[int, str] = {}
    for i in indices:
        if 0 <= int(i) < len(nb.cells):
            src = str(nb.cells[int(i)].get("source") or "")
        else:
            src = ""
        if len(src) > max_chars:
            src = src[:max_chars] + "\n# ...(truncated)"
        out[int(i)] = src
    return out


def _extract_cell_meta_from_cell00(cell00_src: str, *, cell_index: int) -> Dict[str, str]:
    """
    Best-effort: parse Cell00 and extract the structure entry for a specific cell_index.
    Returns: {title, overview, io, notes} (missing keys -> "").
    Works even if Cell00 stores STRUCTURE_JSON as commented JSON lines.
    """
    import re

    # Try to find the STRUCTURE_JSON block markers if available
    begin = "=== STRUCTURE_JSON:BEGIN ==="
    end = "=== STRUCTURE_JSON:END ==="

    block = ""
    a = cell00_src.find(begin)
    b = cell00_src.find(end, a + 1) if a != -1 else -1
    if a != -1 and b != -1 and b > a:
        block = cell00_src[a + len(begin) : b]
    else:
        # fallback: try to just use entire cell00 (still best-effort)
        block = cell00_src

    # Extract commented JSON lines: leading "# " or "#"
    lines = []
    for ln in block.splitlines():
        m = re.match(r"^\s*#\s?(.*)$", ln)
        if m:
            lines.append(m.group(1))
    txt = "\n".join(lines).strip()

    # If no commented JSON, bail
    if not txt:
        return {"title": "", "overview": "", "io": "", "notes": ""}

    # Try to parse JSON array
    try:
        arr = json.loads(txt)
    except Exception:
        return {"title": "", "overview": "", "io": "", "notes": ""}

    if not isinstance(arr, list):
        return {"title": "", "overview": "", "io": "", "notes": ""}

    for it in arr:
        if not isinstance(it, dict):
            continue
        try:
            if int(it.get("cell_index")) != int(cell_index):
                continue
        except Exception:
            continue
        return {
            "title": str(it.get("title") or ""),
            "overview": str(it.get("overview") or ""),
            "io": str(it.get("io") or ""),
            "notes": str(it.get("notes") or ""),
        }

    return {"title": "", "overview": "", "io": "", "notes": ""}

def _excerpt_cells(
    cells: List[Dict[str, Any]],
    *,
    center: Optional[int],
    radius: int = 1,
    max_chars_per_cell: int = 1200,
) -> List[Dict[str, Any]]:
    if not cells:
        return []

    if center is None:
        center = 0

    lo = max(0, int(center) - int(radius))
    hi = min(len(cells), int(center) + int(radius) + 1)

    ex: List[Dict[str, Any]] = []
    for c in cells[lo:hi]:
        src = c.get("source", "") or ""
        if len(src) > max_chars_per_cell:
            src = src[:max_chars_per_cell] + "\n# ...(truncated)"
        ex.append({**c, "source": src})
    return ex


def _safe_json_dump(obj: Any, max_chars: int = 5000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = repr(obj)
    if len(s) > max_chars:
        return s[:max_chars] + "\n...(truncated)"
    return s


# -----------------------------
# scaffold_headers helpers (robust imports + fallbacks)
# -----------------------------

# We try to import helper functions; if missing, we use minimal heuristics.
try:
    from src.nb.scaffold_headers import looks_like_autoheader as _looks_like_autoheader  # type: ignore
except Exception:

    def _looks_like_autoheader(src: str) -> bool:
        s = src or ""
        # Heuristic: detect typical scaffold markers or cell header framing
        if "=== AUTOHEADER" in s:
            return True
        if "# ============================================================" in s and "# Cell " in s:
            return True
        return False


try:
    from src.nb.scaffold_headers import strip_existing_autoheader as _strip_existing_autoheader  # type: ignore
except Exception:

    def _strip_existing_autoheader(src: str) -> str:
        """
        Fallback stripper:
        - If we see an explicit AUTOHEADER block, drop up to its END marker.
        - Else return src as-is (idempotent and safe).
        """
        s = src or ""
        # Common patterns seen in your notebooks:
        #   === AUTOHEADER:BEGIN === ... === AUTOHEADER:END ===
        #   === AUTOHEADER:END ===
        for end_marker in ("=== AUTOHEADER:END ===", "=== AUTOHEADER:END===", "=== AUTOHEADER END ==="):
            pos = s.find(end_marker)
            if pos != -1:
                # keep content after the marker line
                nl = s.find("\n", pos)
                return s[nl + 1 :] if nl != -1 else ""
        return s


# Try to use explicit markers if present (optional)
try:
    from src.nb.scaffold_headers import CELLHEADER_BEGIN as _CELLHEADER_BEGIN  # type: ignore
    from src.nb.scaffold_headers import CELLHEADER_END as _CELLHEADER_END  # type: ignore
except Exception:
    _CELLHEADER_BEGIN, _CELLHEADER_END = None, None


# -----------------------------
# Notion append/update (best-effort, safe for partial repos)
# -----------------------------


def _maybe_append_plan_to_notion(
    *,
    repos: Any,
    task_page_id: str,
    proposal_page_id: str,
    append_to_task: Optional[str],
    append_to_proposal: Optional[str],
) -> None:
    """
    Best-effort audit logging. Do NOT fail the loop if Notion logging fails.

    Hardened:
    - Guard access via hasattr(repos, ...) and hasattr(repos.<repo>, ...)
    """
    try:
        if append_to_proposal:
            if hasattr(repos, "proposals") and repos.proposals is not None:
                if hasattr(repos.proposals, "append_log"):
                    repos.proposals.append_log(proposal_page_id=proposal_page_id, text=append_to_proposal)
                elif hasattr(repos.proposals, "set_next_action"):
                    repos.proposals.set_next_action(proposal_page_id=proposal_page_id, text=append_to_proposal)

        if append_to_task:
            if hasattr(repos, "tasks") and repos.tasks is not None:
                if hasattr(repos.tasks, "append_log"):
                    repos.tasks.append_log(task_page_id=task_page_id, text=append_to_task)
    except Exception:
        return


# -----------------------------
# Step handlers
# -----------------------------

IMPLEMENT_PRIORITY = 60
VERIFY_PRIORITY = 80
APPLY_PRIORITY = 90
POST_PATCH_VERIFY_PRIORITY = 80


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    kind: str  # PATCH_CELL | VERIFY
    # PATCH
    mode: Optional[str] = None
    cell_index: Optional[int] = None
    cell_type: Optional[str] = None
    intent: Optional[str] = None
    acceptance: Optional[List[str]] = None
    # VERIFY
    run_mode: Optional[str] = None
    up_to_cell_index: Optional[int] = None


def llm_plan_step(
    *,
    store: StateStore,
    repos: Any,
    claude: ClaudeClient,
    task_item_id: str,
    target: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Planner (updated for bootstrap-first workflow):
      - FIRST RUN: scaffold ONLY Cell00 + Cell01 (setup), verify PREFIX up to 1,
        then enqueue another LLM_PLAN to plan Cell02+.
      - AFTER BOOTSTRAP: plan/implement loop starts from Cell02.
      - Cancels duplicated TODO items for same task/proposal to avoid queue bloat
      - Never writes null for optional ints (omit keys instead)
      - Uses _state_patch to avoid overwriting queue/memory_refs
    """
    import os
    import time
    _trace_root = project_root or Path.cwd()

    task_page_id = target.get("task_page_id")
    proposal_page_id = target.get("proposal_page_id")
    notebook_path = target.get("notebook_path")
    if not task_page_id or not proposal_page_id or not notebook_path:
        mark_failed(
            store,
            task_item_id=task_item_id,
            reason="LLM_PLAN missing task_page_id/proposal_page_id/notebook_path",
        )
        return {"ok": False, "message": "LLM_PLAN missing required fields"}
    _trace_group = f"proposal={proposal_page_id}|task_item={task_item_id}"
    # Prevent queue bloat: cancel existing TODO steps for the same proposal before enqueueing a fresh plan
    try:
        cancelled_n = _cancel_todo_for_same_proposal(
            store=store,
            task_page_id=str(task_page_id),
            proposal_page_id=str(proposal_page_id),
            keep_task_item_id=str(task_item_id),
        )
        if cancelled_n:
            bump_metric(store, "queue_cancelled_by_llm_plan", cancelled_n)
    except Exception:
        pass

    llm_cfg = dict(target.get("llm") or {})
    model = llm_cfg.get("model") or os.getenv("CLAUDE_MODEL") or "claude-3-5-sonnet-latest"
    max_tokens = int(llm_cfg.get("max_tokens") or 2500)
    temperature = float(llm_cfg.get("temperature") or 0.0)

    goal = (target.get("goal") or "").strip()
    hint = target.get("hint") or {}
    run_evidence = target.get("run_evidence") or {}
    last_error = (run_evidence.get("last_error") or {}) if isinstance(run_evidence, dict) else {}
    last_error_min = _min_last_error_summary(last_error if isinstance(last_error, dict) else {})

    # ---------------------------------------------------------
    # ✅ NEW: runtime upstream facts injected by one_loop (optional)
    # ---------------------------------------------------------
    runtime_cp = _extract_runtime_context_pack(target)
    upstream_ctx_text = _build_upstream_context_text(runtime_cp=runtime_cp, max_chars=8000)
 
    # ---------------------------------------------------------
    # Task fields: ALWAYS try to hydrate from Notion task page
    # (target.task_fields is often missing in queue items)
    # ---------------------------------------------------------
    def _prop_to_text(p: dict) -> str:
        if not isinstance(p, dict):
            return str(p or "")
        t = (p.get("type") or "").lower()
        if t in ("title", "rich_text"):
            arr = p.get(t) or []
            return "".join([x.get("plain_text","") for x in arr if isinstance(x, dict)]).strip()
        if t == "select":
            s = p.get("select") or {}
            return str(s.get("name") or "").strip()
        if t == "multi_select":
            arr = p.get("multi_select") or []
            return ", ".join([x.get("name","") for x in arr if isinstance(x, dict)]).strip()
        if t == "checkbox":
            return str(bool(p.get("checkbox")))
        if t == "number":
            return "" if p.get("number") is None else str(p.get("number"))
        if t == "url":
            return str(p.get("url") or "").strip()
        if t == "date":
            d = p.get("date") or {}
            return str(d.get("start") or "").strip()
        if t == "relation":
            arr = p.get("relation") or []
            return ", ".join([x.get("id","") for x in arr if isinstance(x, dict)]).strip()
        return str(p)

    # NOTE: task_fields is SINGLE source of truth in this function (avoid double-init / accidental overwrite)
    task_fields: Dict[str, Any] = target.get("task_fields") if isinstance(target.get("task_fields"), dict) else {}
    hydrated_from_task = False
    try:
        if (not task_fields) and hasattr(repos, "tasks") and hasattr(repos.tasks, "retrieve_page"):
            page = repos.tasks.retrieve_page(page_id=str(task_page_id))
            props = page.get("properties") or {}
            s = repos.tasks.schema
            def _t(name: str) -> str:
                return _prop_to_text(props.get(name) or {})
            task_fields = {
                "Title": _t(s.TASK_TITLE),
                "Objective": _t(s.TASK_OBJECTIVE),
                "Constraints": _t(s.TASK_CONSTRAINTS),
                "Acceptance Criteria": _t(s.TASK_AC),
                "Scope": _t(s.TASK_SCOPE),
                "Entry Point": _t(s.TASK_ENTRY),
                "Run Policy": _t(s.TASK_RUN_POLICY),
            }
            hydrated_from_task = True
    except Exception as e:
        # keep going; planner will fall back, but we also record debug for inspection
        try:
            _state_patch(store, {"plan_debug": {"task_hydrate_error": str(e)}})
        except Exception:
            pass
    # -----------------------------
    # Task fields / brief (robust)
    # -----------------------------
    # NOTE:
    # In many runs, one_loop enqueues LLM_PLAN with only:
    #   task_page_id/proposal_page_id/notebook_path/hint
    # so we MUST be able to fetch task context from repos.tasks.query_tasks().

    def _rt_plain(v: Any) -> str:
        """Best-effort normalize Notion property value to plain text."""
        if v is None:
            return ""
        # Already plain
        if isinstance(v, str):
            return v.strip()
        # Common shapes: {"type":"rich_text","rich_text":[{"plain_text":...}]}
        if isinstance(v, dict):
            t = (v.get("type") or "").lower()
            if t in ("rich_text", "title"):
                arr = v.get(t) or []
                if isinstance(arr, list):
                    return "".join([str(x.get("plain_text") or "") for x in arr if isinstance(x, dict)]).strip()
            if "plain_text" in v:
                return str(v.get("plain_text") or "").strip()
        # Fallback
        return str(v).strip()

    def _props_to_fields(props: Dict[str, Any]) -> Dict[str, str]:
        """Convert Notion page.properties -> {name: plain_text}."""
        out: Dict[str, str] = {}
        if not isinstance(props, dict):
            return out
        for name, pv in props.items():
            if not name:
                continue
            out[str(name)] = _rt_plain(pv)
        return out

    # 1) Normalize explicit target.task_fields (if present) WITHOUT overwriting hydrated task_fields
    if isinstance(task_fields, dict) and task_fields:
        _tmp: Dict[str, str] = {}
        for k, v in (task_fields or {}).items():
            _tmp[str(k)] = _rt_plain(v)
        task_fields = _tmp
    else:
        task_fields = {}

    # 2) If missing, fetch from Notion via repos.tasks.query_tasks(page_id=task_page_id)
    #    (repos.tasks has only: create_task/get_database_meta/query_tasks)
    task_page = None
    if (not task_fields) and hasattr(repos, "tasks") and hasattr(repos.tasks, "query_tasks"):
        try:
            # Try the most likely signature: query_tasks(filter=..., page_size=...)
            # Filter style depends on your BaseRepo wrapper; we try a couple of safe variants.
            rows = None
            try:
                rows = repos.tasks.query_tasks(
                    filter={"property": "id", "equals": str(task_page_id)},
                    page_size=1,
                )
            except Exception:
                # Fallback: some repos expect "page_id" directly
                rows = repos.tasks.query_tasks(page_id=str(task_page_id), page_size=1)

            # rows may be list[page] or dict with "results"
            if isinstance(rows, dict) and isinstance(rows.get("results"), list):
                task_page = rows["results"][0] if rows["results"] else None
            elif isinstance(rows, list):
                task_page = rows[0] if rows else None
        except Exception:
            task_page = None

    if (not task_fields) and isinstance(task_page, dict):
        props = task_page.get("properties") if isinstance(task_page.get("properties"), dict) else {}
        task_fields = _props_to_fields(props)

    # 3) task_brief: prefer explicit target.task_brief, else derive from fields
    task_brief = ""
    if "task_brief" in target and target.get("task_brief") is not None:
        task_brief = str(target.get("task_brief") or "").strip()
    else:
        # Heuristic: if there is a "Task Brief" like property, use it
        for k in ("Task Brief", "task_brief", "Brief", "概要", "説明", "Description"):
            if task_fields.get(k):
                task_brief = str(task_fields.get(k) or "").strip()
                break

    # Objective used for Cell00 overview (best-effort)
    objective_raw = str(task_fields.get("Objective") or task_fields.get("objective") or "").strip()
    # -----------------------------
    # Task context (Objective/Constraints ONLY)
    # -----------------------------
    obj = str(task_fields.get("Objective") or task_fields.get("objective") or "").strip()
    con = str(task_fields.get("Constraints") or task_fields.get("constraints") or "").strip()
    ac  = str(task_fields.get("Acceptance Criteria") or "").strip()
    scope = str(task_fields.get("Scope") or "").strip()
    entry = str(task_fields.get("Entry Point") or "").strip()
    policy = str(task_fields.get("Run Policy") or "").strip()

    # If still empty, fall back to "best available" text fields (prevents placeholder plans)
    if (not obj) and task_brief:
        obj = task_brief.strip()

    task_obj_con = ""
    if obj:
        task_obj_con += "Objective:\n" + obj.strip() + "\n"
    if con:
        task_obj_con += "\nConstraints:\n" + con.strip() + "\n"
    if ac:
        task_obj_con += "\nAcceptance Criteria:\n" + ac.strip() + "\n"
    if scope:
        task_obj_con += "\nScope:\n" + scope.strip() + "\n"
    if entry:
        task_obj_con += "\nEntry Point:\n" + entry.strip() + "\n"
    if policy:
        task_obj_con += "\nRun Policy:\n" + policy.strip() + "\n"
    task_obj_con = task_obj_con.strip()

    # record debug so notebook can inspect why placeholder happened
    try:
        _state_patch(
            store,
            {"plan_debug": {
                "hydrated_from_task": hydrated_from_task,
                "task_fields_keys": sorted(list(task_fields.keys())) if isinstance(task_fields, dict) else None,
                "task_obj_con_len": len(task_obj_con or ""),
                "task_obj_con_head": (task_obj_con[:400] + "...(clip)") if len(task_obj_con or "") > 400 else task_obj_con,
            }},
        )
    except Exception:
        pass
    # Best-effort: include task_brief so planner doesn't fall back to placeholders
    if task_brief.strip():
        if task_obj_con:
            task_obj_con = task_obj_con + "\n\nTask Brief:\n" + task_brief.strip()
        else:
            task_obj_con = "Task Brief:\n" + task_brief.strip()

    # ---------------------------------------------------------
    # ✅ NEW: STRUCTURE seed-only mode (Tasks -> LLM -> structure)
    # ---------------------------------------------------------
    seed_only = False
    try:
        seed_only = bool(target.get("structure_seed_only")) or (isinstance(hint, dict) and str(hint.get("phase") or "").upper() in ("STRUCTURE_SEED", "STRUCTURE_ONLY"))
    except Exception:
        seed_only = False

    if seed_only:
        system_seed = (
            "You are the STRUCTURE SEEDER.\n"
            "Return ONLY valid JSON that matches the provided schema.\n"
            "No markdown fences, no commentary, no extra keys.\n"
            "IMPORTANT: Only define notebook cell structure.\n"
            "- Cell00 is locked.\n"
            "- Cell01 is setup.\n"
            "- Cell02 is fixed (schema truth).\n"
            "- You MUST propose Cell03+ structure.\n"
        )

        user_seed = (
            "TASK (from Notion Tasks):\n"
            f"{task_obj_con or '(Objective/Constraints not provided)'}\n\n"
            "INSTRUCTIONS:\n"
            "- Produce structure for cells 1..N (at least up to cell 6).\n"
            "- Include cell_index >= 1.\n"
            "- Keep titles concrete.\n"
            "- DO NOT include steps. Structure only.\n"
        )

        try:
            res_seed = claude.call_json(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_seed,
                messages=[{"role": "user", "content": user_seed}],
                json_schema=STRUCTURE_SEED_JSON_SCHEMA,
            )
        except ClaudeStructuredOutputError as e:
        # Trace: error
            # Trace: error (STRUCTURE_SEED)
            try:
                append_llm_trace(
                    project_root=_trace_root,
                    run_id=str(proposal_page_id),
                    event={
                        "ts": _now_iso(),
                        "group": _trace_group,
                        "step_type": "LLM_STRUCTURE_SEED_ERROR",
                        "task_item_id": str(task_item_id),
                        "task_page_id": str(task_page_id),
                        "proposal_page_id": str(proposal_page_id),
                        "notebook_path": str(notebook_path),
                        "model": str(model),
                        "error": {"type": type(e).__name__, "message": str(e)},
                    },
                )
            except Exception:
                pass
            mark_failed(store, task_item_id=task_item_id, reason=f"STRUCTURE_SEED JSON error: {e}")
            return {"ok": False, "message": f"STRUCTURE_SEED JSON error: {e}"}

        # Trace: response (STRUCTURE_SEED)
        try:
            append_llm_trace(
                project_root=_trace_root,
                run_id=str(proposal_page_id),
                event={
                    "ts": _now_iso(),
                    "group": _trace_group,
                    "step_type": "LLM_STRUCTURE_SEED_RESULT",
                    "task_item_id": str(task_item_id),
                    "task_page_id": str(task_page_id),
                    "proposal_page_id": str(proposal_page_id),
                    "notebook_path": str(notebook_path),
                    "model": str(model),
                    "response": {
                        "parsed_json": (res_seed.parsed_json or {}),
                        "usage": getattr(res_seed.usage, "__dict__", res_seed.usage),
                    },
                },
            )
        except Exception:
            pass
 


        seed_obj = res_seed.parsed_json or {}
        seed_structure_raw = seed_obj.get("structure") or []
        seed_structure = _normalize_structure_bootstrap(seed_structure_raw)
        if not seed_structure:
            mark_failed(store, task_item_id=task_item_id, reason="STRUCTURE_SEED returned empty structure")
            return {"ok": False, "message": "STRUCTURE_SEED returned empty structure"}

        # Save to state for scaffold/debug visibility
        _state_patch(store, {
            "plan_seed_structure": seed_structure,
            "plan_seed_summary": str(seed_obj.get("seed_summary") or ""),
        })

        update_task_item(
            store,
            task_item_id=task_item_id,
            status="DONE",
            patch={"result": {"seed_structure": seed_structure, "usage": getattr(res_seed.usage, "__dict__", res_seed.usage)}},
        )
        bump_metric(store, "llm_structure_seed_count", 1)
        # ✅ enqueue scaffold using the seeded structure
        enqueue(
            store,
            new_task_item(
                type="SCAFFOLD_HEADERS",
                intent="Scaffold headers using seeded structure from Tasks (seed-only)",
                assignee="SYSTEM",
                priority=99,
                target={
                    "task_page_id": str(task_page_id),
                    "proposal_page_id": str(proposal_page_id),
                    "notebook_path": str(notebook_path),
                    "task_fields": dict(task_fields or {}),
                    "policy": dict(target.get("policy") or {}),
                    "structure": seed_structure,
                    "cleanup_queue": True,
                    "preserve_existing": True,
                    # もし Cell00 locked + STRUCTURE_JSON empty を埋めたいなら scaffold 側でこのフラグを見る
                    "update_cell00_structure_json": True,
                },
            ),
        )

        mark_done(store, task_item_id=task_item_id)
        return {"ok": True, "message": f"Seeded structure ({len(seed_structure)} cells) from Tasks", "structure": seed_structure}
    # Read notebook to detect bootstrap state (Cell00 lock existence)
    cells = _read_notebook_cells(notebook_path)

    # ---------------------------------------------------------
    # ✅ NEW: Always capture Cell00 + Cell01 sources (best-effort)
    #   - Used to constrain LLM_PLAN so it follows Cell01 Repo contract
    #   - Safe even when notebook has <2 cells
    # ---------------------------------------------------------
    def _read_cell00_01_sources() -> tuple[str, str]:
        try:
            picked = _read_notebook_cell_sources(notebook_path, [0, 1], max_chars=6000)
            c0 = str(picked.get(0) or "")
            c1 = str(picked.get(1) or "")
            return c0, c1
        except Exception:
            return "", ""

    # NOTE: define these unconditionally so later code never NameErrors
    cell00_src_for_prompt, cell01_src_for_prompt = _read_cell00_01_sources()
    def _cell_source_text(x) -> str:
        if x is None:
            return ""
        if isinstance(x, list):
            return "".join([str(s) for s in x])
        return str(x)

    cell0_src = ""
    try:
        if isinstance(cells, list) and cells:
            cell0_src = _cell_source_text(cells[0].get("source"))
    except Exception:
        cell0_src = ""

    # We treat "not yet scaffolded" as bootstrap phase
    try:
        from src.nb.scaffold_headers import CELL0_LOCK_BEGIN
        is_bootstrap = (CELL0_LOCK_BEGIN not in (cell0_src or ""))
    except Exception:
        # conservative: if we can't import marker, don't assume scaffold exists
        is_bootstrap = True

    # Replan-on-failure flag (if verify failed etc.)
    has_error = False
    if isinstance(last_error, dict) and last_error:
        # one_loop からは has_error が来ないことが多いので、存在判定を広げる
        has_error = bool(last_error.get("has_error")) or (last_error.get("ok") is False)
        if not has_error:
            # error_summary / category / next_action / traceback / failing_cell_index 等があればエラー扱い
            if any(
                last_error.get(k)
                for k in (
                    "error_summary",
                    "category",
                    "next_action",
                    "traceback",
                    "failing_cell_index",
                    "cell_index",
                )
            ):
                has_error = True


    # -----------------------------
    # Bootstrap plan: ONLY 00 + 01
    # -----------------------------
    if is_bootstrap and (not has_error):
        structure_list = [
            {
                "cell_index": 1,
                "title": "Setup & config",
                "overview": "Project-root enforcement, env load, RESOLVED_DB resolution, BaseRepo wiring, LLM client init.",
                "io": "In: env.txt. Out: repos + helpers + claude client.",
                "notes": "Cell00 is comment-only; setup code lives in Cell01.",
            }
        ]

        # Persist plan for visibility
        plan_payload = {
            "plan_id": f"plan_bootstrap_{int(time.time())}",
            "summary": "Bootstrap: scaffold only Cell00 + Cell01, then verify PREFIX up to Cell01. Plan Cell02+ afterward.",
            "structure": structure_list,
            "steps": [
                {
                    "step_id": "bootstrap_verify_01",
                    "kind": "VERIFY",
                    "run_mode": "PREFIX",
                    "up_to_cell_index": 1,
                    "status": "TODO",
                    "attempts": 0,
                }
            ],
            "task_page_id": task_page_id,
            "proposal_page_id": proposal_page_id,
            "notebook_path": notebook_path,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _state_patch(store, {"plan": plan_payload})

        # Enqueue scaffold (ONLY structure for Cell01)
        scaffold_task_fields = dict(task_fields or {})
        if objective_raw:
            scaffold_task_fields["objective"] = objective_raw
            scaffold_task_fields.setdefault("Objective", objective_raw)


        enqueue(
            store,
            new_task_item(
                type="SCAFFOLD_HEADERS",
                intent="Bootstrap scaffold: create Cell00 + Cell01 only",
                assignee="SYSTEM",
                priority=99,
                target={
                    "task_page_id": str(task_page_id),
                    "proposal_page_id": str(proposal_page_id),
                    "notebook_path": str(notebook_path),
                    "task_fields": scaffold_task_fields,
                    "policy": dict(target.get("policy") or {}),
                    "structure": structure_list,          # <- ONLY Cell01
                    "update_cell00_structure_json": False,  # <- IMPORTANT: don't overwrite STRUCTURE_JSON during bootstrap
                    "cleanup_queue": True,
                    "preserve_existing": True,
                },
            ),
        )

        # Verify up to Cell01
        # ---------------------------------------------
        # DO NOT enqueue VERIFY if scaffold previously failed
        # ---------------------------------------------
        
        st = store.load() if hasattr(store, "load") else {}
        sc = (st.get("scaffold") or {}) if isinstance(st.get("scaffold"), dict) else {}
        le = (st.get("last_error") or {}) if isinstance(st.get("last_error"), dict) else {}
        
        scaffold_failed = False
        
        if str(sc.get("status", "")).upper() in ("FAILED", "ERROR"):
            scaffold_failed = True
        
        if "Scaffold failed" in str(le.get("error_summary", "")):
            scaffold_failed = True
        
        
        if not scaffold_failed:
            enqueue(
                store,
                new_task_item(
                    type="VERIFY_NOTEBOOK",
                    intent="Bootstrap verify: PREFIX up to Cell01",
                    assignee="VERIFIER",
                    priority=98,
                    target={
                        "task_page_id": task_page_id,
                        "proposal_page_id": proposal_page_id,
                        "notebook_path": notebook_path,
                        "run_mode": "PREFIX",
                        "up_to_cell_index": 1,
                        "timeout_sec": int(target.get("timeout_sec") or 300),
                        "quality_gates": target.get("quality_gates")
                        or {
                            "ruff": {"enabled": False, "args": ["check", "."], "timeout_sec": 300, "cwd": "."},
                            "pytest": {"enabled": False, "args": ["-q"], "timeout_sec": 900, "cwd": "."},
                        },
                    },
                ),
            )
        else:
            bump_metric(store, "bootstrap_verify_suppressed_scaffold_failed", 1)

        # -------------------------------------------------
        # Enqueue follow-up planner for post-bootstrap cells
        # -------------------------------------------------
        enqueue(
            store,
            new_task_item(
                type="LLM_PLAN",
                intent="Post-bootstrap plan: plan Cell03+ from Task fields",
                assignee="PLANNER",
                priority=97,
                target={
                    "task_page_id": str(task_page_id),
                    "proposal_page_id": str(proposal_page_id),
                    "notebook_path": str(notebook_path),
                    "task_fields": dict(task_fields or {}),
                    "llm": dict(target.get("llm") or {}),
                    "hint": {
                        "phase": "POST_BOOTSTRAP_ONE_BY_ONE",
                        "start_cell_index": 3,
                        "mode": "ONE_BY_ONE",
                        "target_cell_index": 3,
                    },
                    "policy": dict(target.get("policy") or {}),
                },
            ),
        )        
        # IMPORTANT: Stop here. Next loop will run after scaffold/verify updates state,
        # then plan Cell02+ with real STRUCTURE_JSON present in Cell00.
        update_task_item(
            store,
            task_item_id=task_item_id,
            status="DONE",
            patch={"result": {"plan_summary": plan_payload["summary"], "structure": structure_list}},
        )
        bump_metric(store, "llm_plan_count", 1)
        mark_done(store, task_item_id=task_item_id)
        return {
            "ok": True,
            "message": "Bootstrap enqueued: SCAFFOLD_HEADERS (Cell00+01) and VERIFY_NOTEBOOK (PREFIX up to 1).",
            "plan": plan_payload,
        }

    # ---------------------------------------------------------
    # From here: normal planning (POST_BOOTSTRAP / or on failure)
    # ---------------------------------------------------------

    # Notebook excerpt focus (prefer failing cell; but default to Cell02+ after bootstrap)
    center_idx: Optional[int] = None
    try:
        if isinstance(last_error, dict):
            if last_error.get("cell_index") is not None:
                center_idx = int(last_error.get("cell_index"))
            elif last_error.get("failing_cell_index") is not None:
                center_idx = int(last_error.get("failing_cell_index"))
        if center_idx is None and isinstance(hint, dict) and hint.get("target_cell_index") is not None:
            center_idx = int(hint.get("target_cell_index"))
    except Exception:
        center_idx = None

    # Default focus for post-bootstrap: start at 2
    if center_idx is None:
        try:
            start_cell = int((hint or {}).get("start_cell_index") or 2)
        except Exception:
            start_cell = 2
        center_idx = start_cell

    # -----------------------------
    # STRUCTURE payload (ALWAYS non-empty)
    # -----------------------------
    # Prefer the explicitly read Cell00 (may be empty for brand new notebooks)
    cell00_src = str(cell00_src_for_prompt or "")
    if not cell00_src:
        try:
            if isinstance(cells, list) and len(cells) > 0:
                cell00_src = str(cells[0].get("source") or "")
        except Exception:
            cell00_src = ""

    # Plan structure might exist in state already (for fallback)
    st_now = _state_read(store)
    plan_structure_fallback = None
    try:
        plan_structure_fallback = (st_now.get("plan") or {}).get("structure")
    except Exception:
        plan_structure_fallback = None

    structure_payload = _ensure_structure_payload(
        cell00_src=cell00_src,
        hint=(hint if isinstance(hint, dict) else {}),
        plan_structure=plan_structure_fallback if isinstance(plan_structure_fallback, list) else None,
    )



    def _fallback_structure_post_bootstrap() -> List[Dict[str, Any]]:
        return [
            {
                "cell_index": 2,
                "title": "Load sources",
                "overview": "Query Notion data sources for papers/events within scope.",
                "io": "In: repos + time window. Out: lists/dataframes.",
                "notes": "Use repos only; no direct Notion API calls; last 7 days only.",
            },
            {
                "cell_index": 3,
                "title": "Extract candidates",
                "overview": "Extract candidate entities from concrete signals with evidence pointers.",
                "io": "In: papers/events rows. Out: candidate list with evidence.",
                "notes": "No keyword frequency alone; keep evidence URLs/snippets.",
            },
            {
                "cell_index": 4,
                "title": "Rank & filter",
                "overview": "Score candidates against RQ/criteria and pick actionable ones.",
                "io": "In: candidates + RQs. Out: shortlist.",
                "notes": "Skip weak evidence; keep rationale.",
            },
            {
                "cell_index": 5,
                "title": "Write proposals",
                "overview": "Append proposal rows to Weekly Target Update DB.",
                "io": "In: shortlist. Out: created proposal pages/rows.",
                "notes": "Append-only; dedupe by Week+Name+Field; truncate rich_text fields.",
            },
            {
                "cell_index": 6,
                "title": "Summary & diagnostics",
                "overview": "Emit counts, duplicates skipped, and review links.",
                "io": "In: run stats. Out: printed summary.",
                "notes": "No edits to monitoring targets DB.",
            },
        ]

    # NOTE: task_brief already merged into task_obj_con above

    # -----------------------------
    # Schema snapshot (prefer state)
    # -----------------------------
    schema_snapshot: Dict[str, Any] = {}
    try:
        st_now2 = _state_read(store)
        cache = st_now2.get("schema_cache_latest") if isinstance(st_now2, dict) else None
        if isinstance(cache, dict) and isinstance(cache.get("schema_snapshot"), dict):
            schema_snapshot = dict(cache.get("schema_snapshot") or {})
    except Exception:
        schema_snapshot = {}

    # Fallback: accept injected one_loop payload if present
    if not schema_snapshot:
        try:
            injected = (runtime_cp or {}).get("schema_snapshot")
            if isinstance(injected, dict):
                schema_snapshot = dict(injected)
        except Exception:
            schema_snapshot = {}
    
    
    # Planner prompt (ask for structure + steps)
    system = (
        "You are the PLANNER agent.\n"
        "Return ONLY valid JSON that matches the provided schema.\n"
        "No markdown fences, no commentary, no extra keys.\n"
        "IMPORTANT: Omit optional integer fields instead of using null.\n"
        "\n"
        "Hard rules:\n"
        "- Cell00 is locked (comment-only).\n"
        "- Cell01 is setup and should not be modified unless absolutely necessary.\n"
        "- Cell02 is FIXED schema truth cell and must NOT be modified.\n"
        "- Plan work cells from Cell03 onward.\n"
        "\n"
        "CRITICAL:\n"
        "- You MUST follow the Repo contract defined in Cell01.\n"
        "- Do NOT invent alternate repo wiring.\n"
        "- Assume downstream cells will reuse variables/functions defined in Cell01.\n"
         "\n"
         "Repo wiring reality check (do NOT assume a `repos` dict):\n"
         "- Cell01 exposes individual BaseRepo variables like:\n"
         "  papers_repo, events_repo, rq_repo, weekly_target_update_repo\n"
         "- Prefer those variables over any imagined registry/dict.\n"
        "Use ONLY the 5-point context pack provided by the system.\n"
        "Do not assume any property name unless it exists in schema_snapshot.\n"
        "\n"
        "CONSISTENCY RULES (critical for Cell03+):\n"
        "- Reuse names that already exist upstream (see UPSTREAM CONTEXT if provided).\n"
        "- If UPSTREAM CONTEXT provides available_symbols, prefer those names.\n"  
    )


    context_pack = _build_context_pack_5(
        task_objective_constraints=(task_obj_con or "(Objective/Constraints not provided)"),
        structure_payload=structure_payload,
        schema_snapshot=schema_snapshot,
        last_error_summary=last_error_min,
    )

    # ---------------------------------------------------------
    # ✅ NEW: Provide Cell01 (Repo contract + canonical helpers) to LLM_PLAN
    #   - Prevents "safe fallback" / inventing repos/build_repos/etc.
    #   - Still keeps the 5-point CONTEXT_PACK intact (we add "GLOBAL CONTEXT" separately)
    # ---------------------------------------------------------
    global_ctx = ""
    if str(cell01_src_for_prompt or "").strip():
        global_ctx = (
            "GLOBAL CONTEXT (MUST READ): Cell01 (setup/wiring + Repo contract)\n"
            "----- Cell01 (setup code: env, repos, helpers; treat as source of truth) -----\n"
            f"{cell01_src_for_prompt}\n\n"
        )

    user = (
        "CONTEXT_PACK (5 points, fixed):\n"
        f"{_safe_json_dump(context_pack, max_chars=9000)}\n\n"
        f"{upstream_ctx_text}"
        f"{global_ctx}"
        "INSTRUCTIONS:\n"
        "- Produce STRUCTURE for Cell03+ (cell_index >= 3).\n"
        "- Then propose steps (PATCH_CELL / VERIFY) to implement those cells.\n"
        "- Do NOT schedule PATCH_CELL for Cell00/01/02.\n"
        "- Follow Cell01 Repo contract strictly (repo_query only; BaseRepo only).\n"
        "- Use only schema_snapshot property names.\n"
        "- Ensure variable/function names stay consistent across cells (Cell03 -> Cell04+).\n"
        "- Prefer upstream-defined names from available_symbols if provided.\n"
        "- Keep patches small.\n"
    )

    # -----------------------------
    # Trace: request (LLM_PLAN)
    # -----------------------------
    try:
        append_llm_trace(
            project_root=_trace_root,
            run_id=str(proposal_page_id),  # using proposal as "run bucket"
            event={
                "ts": _now_iso(),
                "group": _trace_group,
                "step_type": "LLM_PLAN",
                "task_item_id": str(task_item_id),
                "task_page_id": str(task_page_id),
                "proposal_page_id": str(proposal_page_id),
                "notebook_path": str(notebook_path),
                "model": str(model),
                "request": {
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "json_schema_name": str(PLAN_JSON_SCHEMA.get("name") or ""),
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                },
                "meta": {
                    "cell01_included": bool(str(cell01_src_for_prompt or "").strip()),
                    "cell00_len": len(str(cell00_src_for_prompt or "")),
                    "cell01_len": len(str(cell01_src_for_prompt or "")),
                    "upstream_ctx_included": bool(upstream_ctx_text.strip()),
                },
            },
        )
    except Exception:
        pass

    # Call Claude
    try:
        res = claude.call_json(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            json_schema=PLAN_JSON_SCHEMA,
        )
    except ClaudeStructuredOutputError as e:
        mark_failed(store, task_item_id=task_item_id, reason=f"LLM_PLAN JSON error: {e}")
        return {"ok": False, "message": f"LLM_PLAN JSON error: {e}"}

    plan_obj = res.parsed_json or {}
    plan_summary = str(plan_obj.get("plan_summary") or "")
    raw_steps = list(plan_obj.get("steps") or [])
    notion_update = dict(plan_obj.get("notion_update") or {})

    raw_structure = plan_obj.get("structure")
    structure_list = _normalize_structure_bootstrap(raw_structure)
    if not structure_list:
        # IMPORTANT: if structure is empty, treat as hard failure for this new flow
        mark_failed(store, task_item_id=task_item_id, reason="LLM_PLAN returned empty structure")
        update_task_item(
            store,
            task_item_id=task_item_id,
            status="FAILED",
            patch={"result": {"plan_summary": plan_summary, "structure_preview": [], "error": "empty_structure"}},
        )

        return {"ok": False, "message": "LLM_PLAN returned empty structure", "plan": {"summary": plan_summary}}

    # ✅ NEW: ensure the loop log shows Structure even if update_task_item is not printed by caller
    try:
        print(f"[STRUCTURE][LLM_PLAN] structure_preview_len={len(structure_list)}")
    except Exception:
        pass

    # ✅ NEW: show structure immediately in logs (before SCAFFOLD_HEADERS runs)
    _debug_print_structure(
        prefix="[STRUCTURE][LLM_PLAN]",
        task_page_id=str(task_page_id),
        proposal_page_id=str(proposal_page_id),
        notebook_path=str(notebook_path),
        structure=structure_list,
        plan_summary=plan_summary,
    )
    # ✅ NEW: also persist a trimmed copy for notebook-side inspection
    try:
        _state_patch(store, {"plan_debug": {
            "printed_structure_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "structure_len": len(structure_list),
        }})
    except Exception:
        pass

    # Save structure immediately so Builder can inspect before scaffold runs
    _state_patch(store, {"plan": {
        "plan_id": f"plan_{int(time.time())}",
        "summary": plan_summary,
        "structure": structure_list,
        "task_page_id": task_page_id,
        "proposal_page_id": proposal_page_id,
        "notebook_path": notebook_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }})

    update_task_item(
        store,
        task_item_id=task_item_id,
        status="DOING",
        patch={"result": {"plan_summary": plan_summary, "structure_preview": structure_list}},
    )

    # If has_error: focus on failing cell only (but never below 1; also avoid rewriting 1 unless necessary)
    if has_error:
        failing = 2
        try:
            if isinstance(last_error, dict):
                if last_error.get("failing_cell_index") is not None:
                    failing = int(last_error["failing_cell_index"])
                elif last_error.get("cell_index") is not None:
                    failing = int(last_error["cell_index"])
        except Exception:
            failing = 2
        if failing < 1:
            failing = 1

        normalized_steps: List[Dict[str, Any]] = [
            {
                "step_id": "fix_failing_cell",
                "kind": "PATCH_CELL",
                "mode": "REPLACE",
                "cell_type": "code",
                "cell_index": failing,
                "intent": "Fix the failing cell based on the latest traceback (minimal change).",
                "acceptance": ["PREFIX verify passes up to the failing cell."],
                "status": "TODO",
                "attempts": 0,
            },
            {
                "step_id": "verify_after_fix",
                "kind": "VERIFY",
                "run_mode": "PREFIX",
                "up_to_cell_index": failing,
                "status": "TODO",
                "attempts": 0,
            },
        ]
        if not plan_summary:
            plan_summary = f"Replan: fix failing cell {failing} then verify PREFIX."
    else:
        # Normalize steps deterministically
        normalized_steps = []
        sid = 0
        for s in raw_steps:
            if not isinstance(s, dict) or "kind" not in s:
                continue
            sid += 1
            step_id = str(s.get("step_id") or f"s{sid}")
            kind = str(s.get("kind") or "").upper()

            if kind == "PATCH_CELL":
                mode2 = str(s.get("mode") or "REPLACE").upper()
                if mode2 == "INSERT":
                    mode2 = "REPLACE"  # never insert into scaffolded notebooks
                cell_type2 = str(s.get("cell_type") or "code").lower()

                cell_index_val = s.get("cell_index", None)
                cell_index2: Optional[int] = None
                if cell_index_val is not None:
                    try:
                        cell_index2 = int(cell_index_val)
                    except Exception:
                        cell_index2 = None

                # IMPORTANT: in normal phase, do not schedule edits to Cell01 by default
                if cell_index2 is not None and cell_index2 == 1:
                    continue
                # HARD: do not edit fixed cells
                if cell_index2 is not None and cell_index2 in (0, 1, 2):
                    continue

                step_d: Dict[str, Any] = {
                    "step_id": step_id,
                    "kind": "PATCH_CELL",
                    "mode": mode2,
                    "cell_type": cell_type2,
                    "intent": str(s.get("intent") or ""),
                    "acceptance": list(s.get("acceptance") or []),
                    "status": "TODO",
                    "attempts": 0,
                }
                if cell_index2 is not None:
                    step_d["cell_index"] = cell_index2
                normalized_steps.append(step_d)

            elif kind == "VERIFY":
                run_mode = str(s.get("run_mode") or "PREFIX").upper()
                up_to_val = s.get("up_to_cell_index", None)
                up_to: Optional[int] = None
                if up_to_val is not None:
                    try:
                        up_to = int(up_to_val)
                    except Exception:
                        up_to = None

                step_d2: Dict[str, Any] = {
                    "step_id": step_id,
                    "kind": "VERIFY",
                    "run_mode": run_mode,
                    "status": "TODO",
                    "attempts": 0,
                }
                if run_mode == "PREFIX":
                    if up_to is None:
                        up_to = int(center_idx) if center_idx is not None else 2
                    step_d2["up_to_cell_index"] = int(up_to)
                normalized_steps.append(step_d2)

    if not normalized_steps:
        normalized_steps = [
            {
                "step_id": "s1",
                "kind": "PATCH_CELL",
                "mode": "REPLACE",
                "cell_type": "code",
                "cell_index": 3,
                "intent": "Implement Cell03 (first work cell) using repos and scope from Cell00.",
                "acceptance": ["PREFIX verify passes up to Cell03."],
                "status": "TODO",
                "attempts": 0,
            },
            {
                "step_id": "v1",
                "kind": "VERIFY",
                "run_mode": "PREFIX",
                "up_to_cell_index": 3,
                "status": "TODO",
                "attempts": 0,
            },
        ]
        if not plan_summary:
            plan_summary = "Fallback plan: implement Cell03 then verify PREFIX."

    # Persist plan (patch update; avoid overwriting queue)
    plan_payload = {
        "plan_id": f"plan_{int(time.time())}",
        "summary": plan_summary,
        "structure": structure_list,
        "steps": normalized_steps,
        "task_page_id": task_page_id,
        "proposal_page_id": proposal_page_id,
        "notebook_path": notebook_path,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _state_patch(store, {"plan": plan_payload})

    # ✅ NEW: store the final planned structure + steps count for quick inspection in state
    try:
        _state_patch(store, {"plan_debug": {
            "final_steps_len": len(normalized_steps),
            "final_structure_len": len(structure_list),
            "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }})
    except Exception:
        pass

        # ✅ NEW: show structure immediately in logs (before SCAFFOLD_HEADERS runs)
        _debug_print_structure(
            prefix="[STRUCTURE][LLM_PLAN][BOOTSTRAP]",
            task_page_id=str(task_page_id),
            proposal_page_id=str(proposal_page_id),
            notebook_path=str(notebook_path),
            structure=structure_list,
            plan_summary=str(plan_payload.get("summary") or ""),
        )
    # Enqueue SCAFFOLD_HEADERS for Cell02+ headers (Cell00/01 already exist/locked)
    scaffold_task_fields = dict(task_fields or {})
    if objective_raw:
        scaffold_task_fields["objective"] = objective_raw
        scaffold_task_fields.setdefault("Objective", objective_raw)
    # IMPORTANT: never ask scaffold to touch Cell00/01/02 in post-bootstrap planning.
    structure_for_scaffold = [it for it in structure_list if int(it.get("cell_index") or 0) >= 3]

    enqueue(
        store,
        new_task_item(
            type="SCAFFOLD_HEADERS",
            intent="Scaffold per-cell headers for planned Cell02+ structure",
            assignee="SYSTEM",
            priority=99,
            target={
                "task_page_id": str(task_page_id),
                "proposal_page_id": str(proposal_page_id),
                "notebook_path": str(notebook_path),
                "task_fields": scaffold_task_fields,
                "policy": dict(target.get("policy") or {}),
                "structure": structure_for_scaffold,
                "cleanup_queue": True,
                "preserve_existing": True,
            },
        ),
    )

    # Enqueue downstream work (implement/verify)
    for s in normalized_steps:
        if s.get("kind") == "PATCH_CELL":
            enqueue(
                store,
                new_task_item(
                    type="LLM_IMPLEMENT",
                    intent=f"Implement {s.get('step_id')}: {s.get('intent','')}",
                    assignee="IMPLEMENTER",
                    priority=IMPLEMENT_PRIORITY,
                    target={
                        "task_page_id": task_page_id,
                        "proposal_page_id": proposal_page_id,
                        "notebook_path": notebook_path,
                        "plan_step": s,
                        "llm": llm_cfg,
                        "hint": hint,
                        "run_evidence": run_evidence,
                        "timeout_sec": int(target.get("timeout_sec") or 300),
                        "quality_gates": target.get("quality_gates"),
                        "auto_verify_after_patch": target.get("auto_verify_after_patch", True),
                    },
                ),
            )
        else:
            run_mode = str(s.get("run_mode") or "PREFIX").upper()
            verify_target: Dict[str, Any] = {
                "task_page_id": task_page_id,
                "proposal_page_id": proposal_page_id,
                "notebook_path": notebook_path,
                "run_mode": run_mode,
                "timeout_sec": int(target.get("timeout_sec") or 300),
                "quality_gates": target.get("quality_gates")
                or {
                    "ruff": {"enabled": False, "args": ["check", "."], "timeout_sec": 300, "cwd": "."},
                    "pytest": {"enabled": False, "args": ["-q"], "timeout_sec": 900, "cwd": "."},
                },
            }
            if run_mode == "PREFIX":
                u = s.get("up_to_cell_index")
                if u is None:
                    u = center_idx if center_idx is not None else 2
                verify_target["up_to_cell_index"] = int(u)

            enqueue(
                store,
                new_task_item(
                    type="VERIFY_NOTEBOOK",
                    intent=f"Verify {run_mode} after {s.get('step_id')}",
                    assignee="VERIFIER",
                    priority=VERIFY_PRIORITY,
                    target=verify_target,
                ),
            )

    _maybe_append_plan_to_notion(
        repos=repos,
        task_page_id=str(task_page_id),
        proposal_page_id=str(proposal_page_id),
        append_to_task=notion_update.get("append_to_task"),
        append_to_proposal=notion_update.get("append_to_proposal"),
    )

    update_task_item(
        store,
        task_item_id=task_item_id,
        status="DONE",
        patch={
            "result": {
                "plan_summary": plan_summary,
                "structure": structure_list,
                "steps": normalized_steps,
                "usage": getattr(res.usage, "__dict__", res.usage),
            }
        },
    )
    bump_metric(store, "llm_plan_count", 1)
    mark_done(store, task_item_id=task_item_id)

    return {
        "ok": True,
        "message": f"Planned {len(normalized_steps)} steps (structure={len(structure_list)} cells)",
        "plan": plan_payload,
    }



def llm_implement_step(
    *,
    store: StateStore,
    repos: Any,
    claude: ClaudeClient,
    task_item_id: str,
    target: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Implementer:
      - For scaffolded notebooks, never INSERT new cells; always patch existing cell_index via REPLACE
      - Preserve per-cell autoheader block (best-effort, robust to marker differences)
      - Enqueues APPLY_PATCH with mode/cell_index/cell_type/new_source
      - Optionally enqueues VERIFY_NOTEBOOK PREFIX immediately after patch

    Policy in this version:
      - Notebook patches MUST NOT include build_repos() (BaseRepo-only policy).
    """
    import re

    task_page_id = target.get("task_page_id")
    proposal_page_id = target.get("proposal_page_id")
    notebook_path = target.get("notebook_path")

    _trace_root = project_root or Path.cwd()
    _trace_group = f"proposal={proposal_page_id}|task_item={task_item_id}"

    # trace root (project_root may be None)
    _trace_root = project_root or Path.cwd()
    # "run_id" is optional; we can group by proposal+task_item+ts
    _trace_group = f"proposal={proposal_page_id}|task_item={task_item_id}"

    plan_step = target.get("plan_step") or {}
    has_plan_step = isinstance(plan_step, dict) and bool(plan_step)

    if not task_page_id or not proposal_page_id or not notebook_path:
        mark_failed(store, task_item_id=task_item_id, reason="LLM_IMPLEMENT missing required fields")
        return {"ok": False, "message": "LLM_IMPLEMENT missing required fields"}

    if not has_plan_step:
        mark_failed(store, task_item_id=task_item_id, reason="LLM_IMPLEMENT missing plan_step")
        return {"ok": False, "message": "LLM_IMPLEMENT missing plan_step"}

    if str(plan_step.get("kind") or "").upper() != "PATCH_CELL":
        mark_failed(store, task_item_id=task_item_id, reason="LLM_IMPLEMENT requires plan_step.kind=PATCH_CELL")
        return {"ok": False, "message": "LLM_IMPLEMENT requires plan_step.kind=PATCH_CELL"}

    llm_cfg = dict(target.get("llm") or {})
    model = llm_cfg.get("model") or os.getenv("CLAUDE_MODEL") or "claude-3-5-sonnet-latest"
    max_tokens = int(llm_cfg.get("max_tokens") or 2500)
    temperature = float(llm_cfg.get("temperature") or 0.0)

    # Plan intent
    mode = str(plan_step.get("mode") or "REPLACE").upper()
    cell_type = str(plan_step.get("cell_type") or "code").lower()
    plan_cell_index = plan_step.get("cell_index", None)

    cells = _read_notebook_cells(notebook_path)
    # ---------------------------------------------------------
    # ✅ NEW: Always capture Cell00 + Cell01 sources (best-effort)
    #   - Used to constrain LLM_PLAN so it follows Cell01 Repo contract
    #   - Safe even when notebook has <2 cells
    # ---------------------------------------------------------
    def _read_cell00_01_sources() -> Tuple[str, str]:
        try:
            picked = _read_notebook_cell_sources(notebook_path, [0, 1], max_chars=6000)
            c0 = str(picked.get(0) or "")
            c1 = str(picked.get(1) or "")
            return c0, c1
        except Exception:
            return "", ""

    cell00_src_for_prompt, cell01_src_for_prompt = _read_cell00_01_sources()

    # If Cell00 isn't readable yet (e.g., empty nb), fall back to existing logic later
    # NOTE: For bootstrap phase, LLM_PLAN returns early and does not call Claude,
    #       so this is primarily for POST_BOOTSTRAP normal planning.
     
    # Determine focus cell (the cell we are patching)
    if mode == "APPEND":
        focus_idx = max(0, len(cells) - 1)
    else:
        if plan_cell_index is None:
            hint = target.get("hint") if isinstance(target.get("hint"), dict) else {}
            tci = hint.get("target_cell_index")
            focus_idx = int(tci) if tci is not None else 0
        else:
            focus_idx = int(plan_cell_index)
    
    # --- ALWAYS include Cell00/Cell01 + the target cell + neighbors ---
    # This is the key improvement: LLM sees global policy + setup wiring.
    idxs = sorted(set([0, 1, focus_idx, max(0, focus_idx - 1), min(len(cells) - 1, focus_idx + 1)]))
    picked = _read_notebook_cell_sources(notebook_path, idxs, max_chars=4000)
    
    # Keep an excerpt-like object (same shape as before) for compatibility
    excerpt = []
    for i in idxs:
        excerpt.append({"index": i, "cell_type": (cells[i]["cell_type"] if i < len(cells) else "code"), "source": picked.get(i, "")})
    
    cell00_src = picked.get(0, "")
    cell01_src = picked.get(1, "")
    cellX_src = picked.get(focus_idx, "")
    cellX_meta = _extract_cell_meta_from_cell00(cell00_src, cell_index=focus_idx)

    # ---------------------------------------------------------
    # ✅ NEW: runtime upstream facts injected by one_loop (optional)
    # ---------------------------------------------------------
    runtime_cp = _extract_runtime_context_pack(target)
    upstream_ctx_text = _build_upstream_context_text(runtime_cp=runtime_cp, max_chars=9000)
 
    # Pull last_error injected by one_loop.py (if any)
    run_evidence = target.get("run_evidence") if isinstance(target.get("run_evidence"), dict) else {}
    last_error = run_evidence.get("last_error") if isinstance(run_evidence, dict) else {}
    if not isinstance(last_error, dict):
        last_error = {}

    # ✅ FIX MODE hint from one_loop.py (optional)
    hint = target.get("hint") if isinstance(target.get("hint"), dict) else {}
    fix_mode = bool(hint.get("fix_mode")) if isinstance(hint, dict) else False
    freeze_structure = bool(hint.get("freeze_structure")) if isinstance(hint, dict) else False
    tb_is_compressed = bool(last_error.get("traceback_is_compressed"))
 
    failing_cell_index = last_error.get("failing_cell_index")
    executed_up_to = last_error.get("executed_up_to")
    tb = last_error.get("traceback")
    tb_str = "" if tb is None else str(tb)
    if len(tb_str) > 12000:
        tb_str = tb_str[:12000] + "\n...(truncated)"

    # -----------------------------
    # Schema snapshot (prefer state)
    # -----------------------------
    schema_snapshot: Dict[str, Any] = {}
    try:
        st_now = _state_read(store)
        cache = st_now.get("schema_cache_latest") if isinstance(st_now, dict) else None
        if isinstance(cache, dict) and isinstance(cache.get("schema_snapshot"), dict):
            schema_snapshot = dict(cache.get("schema_snapshot") or {})
    except Exception:
        schema_snapshot = {}

    if not schema_snapshot:
        try:
            injected = (runtime_cp or {}).get("schema_snapshot")
            if isinstance(injected, dict):
                schema_snapshot = dict(injected)
        except Exception:
            schema_snapshot = {}

    # Prompts
    system = (
        "You are the IMPLEMENTER agent.\n"
        "\n"
        "Return ONLY valid JSON matching the provided schema.\n"
        "No markdown fences, no commentary, no extra keys.\n"
        "Generate executable notebook cell source.\n"
        "IMPORTANT: Omit optional integer fields instead of using null.\n"
        "IMPORTANT: If the plan targets an existing cell, do NOT use INSERT.\n"
        "\n"
        "NOTEBOOK POLICY:\n"
        "- You MUST NOT use build_repos(). Notebook side is BaseRepo-only.\n"
        "- Do NOT import or construct NotionRepos.\n"
        "\n"
        "NOTION ACCESS RULES (strict):\n"
        "- Notebook Cell00 MAY import from `src.notion.client` for bootstrapping only.\n"
        "- HARD BAN: `import src.notion.repos as repos` is forbidden.\n"
        "- HARD BAN: `from src.notion import repos` is forbidden.\n"
        "CRITICAL:\n"
         "- You MUST rely on Cell00 policy/structure and Cell01 wiring.\n"
         "- Do not invent new setup; reuse what Cell01 provides.\n"
         "\n"
         "Repo wiring reality check:\n"
         "- DO NOT assume a `repos` dict exists in the notebook.\n"
         "- Use these variables from Cell01 (BaseRepo instances):\n"
         "  papers_repo, events_repo, rq_repo, weekly_target_update_repo\n"
         "- Use `repo_query(repo, ...)` for reads.\n"
        "- Do not change other cells; output only the target cell patch.\n"
        "\n"
        "CONSISTENCY RULES (critical):\n"
        "- Reuse variable names that already exist upstream.\n"
        "- If UPSTREAM CONTEXT provides available_symbols, ONLY reference names from it (unless defining a new name intentionally).\n"
    )

    # ✅ Stronger constraints in FIX mode (Option A)
    if fix_mode:
        system += (
            "\n"
            "FIX MODE (after runtime failure):\n"
            "- Your job is to make the failing cell run, with MINIMAL changes.\n"
            "- DO NOT refactor broadly. DO NOT rename upstream variables.\n"
            "- Keep the cell's I/O contract stable (inputs/outputs names must remain consistent).\n"
            "- Prefer adding small guards (type checks, safe accessors) over changing data flow.\n"
        )
        if freeze_structure:
            system += "- STRUCTURE IS FROZEN: do NOT propose new cells, do NOT change cell roles.\n"
        if tb_is_compressed:
            system += "- Traceback is compressed to the actionable part; focus on the exception type/message and fix that precisely.\n"
 
    context_pack = _build_context_pack_5(
        task_objective_constraints="(Planner context omitted in implement step)",
        structure_payload={"kind": "IMPLEMENT_VIEW", "items": []},
        schema_snapshot=schema_snapshot,
        last_error_summary=_min_last_error_summary(last_error),
    )

    user = (
        "CONTEXT_PACK (5 points, fixed):\n"
        f"{_safe_json_dump(context_pack, max_chars=7000)}\n\n"
        f"{upstream_ctx_text}"
        "PLAN STEP (what to build/fix):\n"
        f"{_safe_json_dump(plan_step)}\n\n"
        "GLOBAL CONTEXT (MUST READ): Cell00 (policy/structure) and Cell01 (setup/wiring)\n"
        "----- Cell00 (locked overview/policy/structure) -----\n"
        f"{cell00_src}\n\n"
        "----- Cell01 (setup code: env, repos, helpers) -----\n"
        f"{cell01_src}\n\n"
        f"TARGET CELL: Cell{focus_idx:02d}\n"
        "----- Current source of target Cell (before your change) -----\n"
        f"{cellX_src}\n\n"
        "TARGET CELL ROLE (from Cell00 structure; may be empty if not found):\n"
        f"{_safe_json_dump(cellX_meta)}\n\n"
        "ERROR CONTEXT (if any):\n"
        f"{_safe_json_dump({'failing_cell_index': failing_cell_index, 'executed_up_to': executed_up_to})}\n\n"
        "TRACEBACK (truncated):\n"
        f"{tb_str}\n\n"
        "NEARBY NOTEBOOK EXCERPT (target +/-1):\n"
        f"{_safe_json_dump(excerpt)}\n\n"
        "INSTRUCTIONS:\n"
        "- You must implement/fix ONLY the target cell (the plan_step.cell_index).\n"
        "- First, infer the root cause from traceback + current cell source.\n"
        "- Then produce corrected executable code for CellX that matches the role in Cell00 and uses the wiring from Cell01.\n"
        "- IMPORTANT: Keep names consistent with upstream_sources / available_symbols if provided.\n"
        "- In FIX MODE: do the smallest possible patch that makes the cell execute successfully.\n"
        "- Keep changes minimal and localized.\n"
        "- Output JSON must include: mode, cell_type, new_source.\n"
        "- If plan_step.mode is REPLACE, you must patch that cell_index.\n"
        "- Do NOT output null for integers; omit the key instead.\n"
        "- Notebook policy: DO NOT use build_repos().\n"
        "- IMPORTANT: Do NOT reference a `repos` dict. Use papers_repo/events_repo/rq_repo/weekly_target_update_repo.\n"

    )

    # -----------------------------
    # Trace: request (LLM_IMPLEMENT)
    # -----------------------------
    try:
        append_llm_trace(
            project_root=_trace_root,
            run_id=str(proposal_page_id),
            event={
                "ts": _now_iso(),
                "group": _trace_group,
                "step_type": "LLM_IMPLEMENT",
                "task_item_id": str(task_item_id),
                "task_page_id": str(task_page_id),
                "proposal_page_id": str(proposal_page_id),
                "notebook_path": str(notebook_path),
                "model": str(model),
                "request": {
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "json_schema_name": str(IMPLEMENT_JSON_SCHEMA.get("name") or ""),
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                },
                "meta": {
                    "focus_cell_index": int(focus_idx),
                    "plan_step": plan_step,
                    "upstream_ctx_included": bool(upstream_ctx_text.strip()),
                },
            },
        )
    except Exception:
        pass

    try:
        res = claude.call_json(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            json_schema=IMPLEMENT_JSON_SCHEMA,
        )
    except ClaudeStructuredOutputError as e:

        # Trace: error
        try:
            append_llm_trace(
                project_root=_trace_root,
                run_id=str(proposal_page_id),
                event={
                    "ts": _now_iso(),
                    "group": _trace_group,
                    "step_type": "LLM_IMPLEMENT_ERROR",
                    "task_item_id": str(task_item_id),
                    "task_page_id": str(task_page_id),
                    "proposal_page_id": str(proposal_page_id),
                    "notebook_path": str(notebook_path),
                    "model": str(model),
                    "error": {"type": type(e).__name__, "message": str(e)},
                },
            )
        except Exception:
            pass
        mark_failed(store, task_item_id=task_item_id, reason=f"LLM_IMPLEMENT JSON error: {e}")
        return {"ok": False, "message": f"LLM_IMPLEMENT JSON error: {e}"}

 
    # Trace: response (LLM_IMPLEMENT_RESULT)
    try:
        append_llm_trace(
            project_root=_trace_root,
            run_id=str(proposal_page_id),
            event={
                "ts": _now_iso(),
                "group": _trace_group,
                "step_type": "LLM_IMPLEMENT_RESULT",
                "task_item_id": str(task_item_id),
                "task_page_id": str(task_page_id),
                "proposal_page_id": str(proposal_page_id),
                "notebook_path": str(notebook_path),
                "model": str(model),
                "response": {
                    "parsed_json": (res.parsed_json or {}),
                    "usage": getattr(res.usage, "__dict__", res.usage),
                },
            },
        )
    except Exception:
        pass
        

    out = res.parsed_json or {}
    out_mode = str(out.get("mode") or mode).upper()
    if out_mode == "INSERT":
        out_mode = "REPLACE"  # hard forbid

    # cell_index required for non-APPEND
    if out_mode != "APPEND":
        if plan_cell_index is None:
            mark_failed(store, task_item_id=task_item_id, reason="LLM_IMPLEMENT requires cell_index for non-APPEND")
            return {"ok": False, "message": "LLM_IMPLEMENT requires cell_index for non-APPEND steps"}
        plan_cell_index = int(plan_cell_index)
    # HARD: Cell02 is fixed schema truth cell (must not be modified)
    try:
        if plan_cell_index is not None and int(plan_cell_index) == 2:
            mark_failed(store, task_item_id=task_item_id, reason="Cell02 is fixed and must not be modified.")
            return {"ok": False, "message": "Refused: Cell02 is fixed and must not be modified."}
    except Exception:
        pass
    
    out_cell_type = str(out.get("cell_type") or cell_type).lower()
    new_code = str(out.get("new_source") or "").strip()
    notes = str(out.get("notes") or "")

    # -------------------------
    # Hard policy guard: forbid build_repos() in notebook patches
    # -------------------------
    if "build_repos(" in new_code:
        mark_failed(
            store,
            task_item_id=task_item_id,
            reason="LLM_IMPLEMENT produced build_repos() which is forbidden in notebook patches (BaseRepo-only policy).",
        )
        return {"ok": False, "message": "LLM output violates notebook policy: build_repos() is forbidden."}
     # -------------------------
     # Hard correctness guard: do NOT assume `repos` dict exists (Cell01 uses individual repo vars)
     # -------------------------
    banned_repos_dict_patterns = [
        "repos[",
        "repos.get(",
        "if 'repos' in dir()",
        "if \"repos\" in dir()",
        "if 'repos' not in dir()",
        "if \"repos\" not in dir()",
        "REQUIRED_REPO_KEYS",
    ]
    hits = [p for p in banned_repos_dict_patterns if p in new_code]
    if hits:
        mark_failed(
            store,
            task_item_id=task_item_id,
            reason=f"LLM_IMPLEMENT referenced a `repos` dict pattern which is not part of Cell01 wiring: {hits}",
        )
        return {
            "ok": False,
            "message": f"LLM output violates notebook wiring: do not reference `repos` dict. Use papers_repo/events_repo/rq_repo/weekly_target_update_repo. hits={hits}",
        }
 
    
    # Additional guard: forbid common wrong imports around repos module
    banned_snippets = [
        "from src.notion import repos",
        "import src.notion.repos as repos",
        "from src.notion.repos import repos",
    ]
    bad_hits = [b for b in banned_snippets if b in new_code]
    if bad_hits:
        mark_failed(
            store,
            task_item_id=task_item_id,
            reason=f"LLM_IMPLEMENT produced banned repos import(s): {bad_hits}",
        )
        return {"ok": False, "message": f"LLM output violates notebook policy (banned repos imports): {bad_hits}"}

    if not new_code:
        mark_failed(store, task_item_id=task_item_id, reason="LLM_IMPLEMENT returned empty new_source")
        return {"ok": False, "message": "LLM_IMPLEMENT returned empty new_source"}

    # Determine patch cell_index
    if out_mode != "APPEND":
        cell_index = int(plan_cell_index)  # type: ignore[arg-type]
    else:
        cell_index = None

    def _extract_cell_header_prefix(src: str) -> str:
        s = src or ""

        # Prefer explicit markers if available
        if _CELLHEADER_BEGIN and _CELLHEADER_END and (_CELLHEADER_BEGIN in s) and (_CELLHEADER_END in s):
            a = s.find(_CELLHEADER_BEGIN)
            b = s.find(_CELLHEADER_END, a)
            if b != -1:
                b2 = b + len(_CELLHEADER_END)
                return s[a:b2].rstrip() + "\n\n"

        # Fallback: autoheader detector + stripper (idempotent)
        if s and _looks_like_autoheader(s):
            rest = _strip_existing_autoheader(s)
            header = s[: max(0, len(s) - len(rest))]
            return header.rstrip() + "\n\n" if header.strip() else ""

        return ""

    # Build new_source while preserving header prefix if present
    if cell_index is not None:
        try:
            cur_cells = _read_notebook_cells(notebook_path)
            cur_src = str(cur_cells[cell_index].get("source") or "") if cell_index < len(cur_cells) else ""
        except Exception:
            cur_src = ""
        header_prefix = _extract_cell_header_prefix(cur_src)
        if header_prefix:
            new_source = header_prefix + new_code.strip() + "\n"
        else:
            new_source = new_code.strip() + "\n"
    else:
        new_source = new_code.strip() + "\n"

    patch_target: Dict[str, Any] = {
        "task_page_id": task_page_id,
        "proposal_page_id": proposal_page_id,
        "notebook_path": notebook_path,
        "new_source": new_source,
        "mode": out_mode,
        "cell_type": out_cell_type,
    }
    if cell_index is not None:
        patch_target["cell_index"] = int(cell_index)

    enqueue(
        store,
        new_task_item(
            type="APPLY_PATCH",
            intent=f"Apply patch for {plan_step.get('step_id')}",
            assignee="IMPLEMENTER",
            priority=APPLY_PRIORITY,
            target=patch_target,
        ),
    )

    # Auto verify after patch
    if bool(target.get("auto_verify_after_patch", True)):
        base = int(cell_index) if cell_index is not None else int(len(cells))
        if executed_up_to is not None:
            try:
                up_to_i = max(int(executed_up_to), base)
            except Exception:
                up_to_i = base
        else:
            up_to_i = base

        verify_target: Dict[str, Any] = {
            "task_page_id": task_page_id,
            "proposal_page_id": proposal_page_id,
            "notebook_path": notebook_path,
            "run_mode": "PREFIX",
            "up_to_cell_index": int(up_to_i),
            "timeout_sec": int(target.get("timeout_sec") or 300),
            "quality_gates": target.get("quality_gates")
            or {
                "ruff": {"enabled": False, "args": ["check", "."], "timeout_sec": 300, "cwd": "."},
                "pytest": {"enabled": False, "args": ["-q"], "timeout_sec": 900, "cwd": "."},
            },
        }

        enqueue(
            store,
            new_task_item(
                type="VERIFY_NOTEBOOK",
                intent=f"Verify PREFIX after {plan_step.get('step_id')}",
                assignee="VERIFIER",
                priority=POST_PATCH_VERIFY_PRIORITY,
                target=verify_target,
            ),
        )

    update_task_item(
        store,
        task_item_id=task_item_id,
        status="DONE",
        patch={
            "result": {
                "patch_target": patch_target,
                "usage": getattr(res.usage, "__dict__", res.usage),
                "notes": notes,
            }
        },
    )
    bump_metric(store, "llm_implement_count", 1)
    mark_done(store, task_item_id=task_item_id)

    return {"ok": True, "message": f"Enqueued APPLY_PATCH ({out_mode})", "patch": patch_target}


def handle_llm_step(
    *,
    step_type: str,
    store: StateStore,
    repos: Any,
    claude: ClaudeClient,
    task_item_id: str,
    target: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience dispatcher.
    """
    t = str(step_type or "").upper()
    if t == "LLM_PLAN":
        return llm_plan_step(store=store, repos=repos, claude=claude, task_item_id=task_item_id, target=target)
    if t == "LLM_IMPLEMENT":
        return llm_implement_step(store=store, repos=repos, claude=claude, task_item_id=task_item_id, target=target)
    raise ValueError(f"Unknown LLM step_type: {step_type}")
