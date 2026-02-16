# src/orchestrator/one_loop.py
"""
One-loop orchestrator (single-process) that ties together:
- state/state_store.py   (queue + machine state)
- notion/repos.py        (TASKS/PROPOSALS/RUNS/DECISIONS)
- artifacts/layout.py    (run folders + manifests)
- nb/patcher.py          (apply patch)
- nb/scaffold_headers.py (deterministic Cell00 + per-cell headers)
- exec/nb_runner.py      (execute notebook prefix/full)
- exec/quality_gate.py   (pytest/ruff gates)
- verify/error_parser.py (categorize + next_action)

Intent
------
This is a *single* deterministic loop function you can call repeatedly from a notebook
(or CLI) to advance the system by one step.

Important
---------
- This orchestrator does NOT resolve data_source_id. You must resolve once in setup
  cell and build repos using cached RESOLVED_DB mapping (your strict rule).
- LLM planning is handled elsewhere (e.g., steps_llm.py). This orchestrator
  handles deterministic steps and can enqueue LLM_PLAN on failures.

Queue item contract (minimal)
-----------------------------
We use state_store.new_task_item(..., target=...) with these fields in target:

For planning:
  type: "PLAN_CHANGESET"
  target: {
    "task_page_id": "...",            # Notion TASK page id
    "proposal_title": "...",          # Title for ChangeSet
    "notebook_path": "notebooks/xxx.ipynb",
    "cell_index": 12,
    "intent": "...",
    "acceptance": "...",
    "risk": "LOW|MEDIUM|HIGH"
  }

For scaffolding:
  type: "SCAFFOLD_HEADERS"
  target: {
    "task_page_id": "...",
    "proposal_page_id": "...",
    "notebook_path": "...",
    "task_fields": {...},             # dict from TASKS_DB
    "policy": {...},                  # policy dict
    "structure": [                    # optional override; list[dict]
      {"cell_index": 1, "title": "...", "overview": "...", "io": "...", "notes": "..."},
      ...
    ],
    "cleanup_queue": true,            # optional
    "preserve_existing": true         # optional
  }

For implementation:
  type: "APPLY_PATCH"
  target: {
    "task_page_id": "...",
    "proposal_page_id": "...",
    "notebook_path": "...",
    "cell_index": 12,
    "new_source": "....",             # patched cell source (string)
  }

For verification:
  type: "VERIFY_NOTEBOOK"
  target: {
    "task_page_id": "...",
    "proposal_page_id": "...",
    "notebook_path": "...",
    "run_mode": "PREFIX|FULL",
    "up_to_cell_index": 12,           # required if PREFIX
    "timeout_sec": 300,
    "quality_gates": {
        "pytest": {"enabled": true, "args": ["-q"], "timeout_sec": 900, "cwd": "."},
        "ruff":   {"enabled": true, "args": ["check", "."], "timeout_sec": 300, "cwd": "."}
    }
  }

This loop will:
- create proposal if needed (PLAN_CHANGESET)
- scaffold Cell00 + Cell headers (SCAFFOLD_HEADERS)
- apply patch + write artifacts (APPLY_PATCH)
- execute notebook + quality gates + write RUN to RUNS_DB (VERIFY_NOTEBOOK)
- update proposal status (VERIFIED/FAILED) and link run
- update state queue item status
"""

from __future__ import annotations
import re
import hashlib
import nbformat
import json
import time
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace  # ✅ ADD
from typing import Any, Dict, List, Literal, Optional
from types import ModuleType
from src.orchestrator.steps_llm import handle_llm_step
from src.llm.claude_client import ClaudeClient
from src.artifacts.layout import ensure_run_dir, next_run_id
from src.exec.nb_runner import execute_notebook_full, execute_notebook_prefix, write_run_reports
from src.exec.quality_gate import run_pytest, run_ruff
from src.nb.patcher import patch_cell_source_with_artifacts
from src.nb.scaffold_headers import (
    build_cell00,
    build_cell01_setup,  # ✅ ADD
    build_cell02_schema_introspection_repo_first,
    build_cell_header,
    build_scaffold_digest,
    looks_like_autoheader,
    strip_existing_autoheader,
    extract_structure_from_cell00,
    normalize_cell_source_for_header_insertion,
    CELL0_LOCK_BEGIN,
    CELL0_LOCK_END,
    STRUCTURE_JSON_BEGIN,
    STRUCTURE_JSON_END,
)


from src.notion.repos import NotionRepos, run_payload_from_notebook_result
from src.state.state_store import (
    StateStore,
    bump_metric,
    enqueue,
    mark_done,
    mark_failed,
    new_task_item,
    pop_next_todo,
    update_task_item,
    set_last_error,  # ✅ ADD
)

from src.verify.error_parser import (
    suggest_next_action_from_notebook_result,
    suggest_next_action_from_quality_gate,
)


def _get_plan_seed_structure(st: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pull STRUCTURE seed from state. Expected shapes:
      - st["plan_seed_structure"] == [ {cell_index,title,...}, ... ]
      - st["plan_seed_structure"] == {"structure":[...]} (rare, but allow)
    """
    if not isinstance(st, dict):
        return []
    seed = st.get("plan_seed_structure")
    if isinstance(seed, list):
        return seed
    if isinstance(seed, dict) and isinstance(seed.get("structure"), list):
        return seed["structure"]
    return []

def _has_nonempty_structure(obj: Any) -> bool:
    if isinstance(obj, list):
        return any(isinstance(x, dict) and x.get("cell_index") and x.get("title") for x in obj)
    if isinstance(obj, dict):
        s = obj.get("structure")
        return isinstance(s, list) and any(isinstance(x, dict) and x.get("cell_index") and x.get("title") for x in s)
    return False

def _enqueue_structure_seed_then_scaffold(
    *,
    store: StateStore,
    target: Dict[str, Any],
    task_item_id: str,
    reason: str,
) -> None:
    """
    If STRUCTURE seed is missing, enqueue:
      1) LLM_PLAN (structure_seed_only=True) with high priority
      2) SCAFFOLD_HEADERS again (so scaffold happens after seed)
    And cancel current scaffold item to avoid infinite loop.
    """
    # cancel current item (audit-safe)
    update_task_item(store, task_item_id=task_item_id, status="CANCELLED", last_error=reason)

    tgt = dict(target or {})
    task_page_id = str(tgt.get("task_page_id") or "")
    proposal_page_id = str(tgt.get("proposal_page_id") or "")
    notebook_path = str(tgt.get("notebook_path") or "")
    if not (task_page_id and proposal_page_id and notebook_path):
        return

    # 1) seed-only LLM_PLAN
    enqueue(
        store,
        new_task_item(
            type="LLM_PLAN",
            intent="Seed STRUCTURE from Notion Task (structure-only, pre-scaffold).",
            assignee="PLANNER",
            priority=1000,
            target={
                "task_page_id": task_page_id,
                "proposal_page_id": proposal_page_id,
                "notebook_path": notebook_path,
                "hint": {"phase": "STRUCTURE_SEED"},
                "structure_seed_only": True,
            },
        ),
    )

    # 2) re-enqueue scaffold (will read state.plan_seed_structure next time)
    enqueue(
        store,
        new_task_item(
            type="SCAFFOLD_HEADERS",
            intent="Scaffold notebook headers (after STRUCTURE seed).",
            assignee="SYSTEM",
            priority=999,
            target=tgt,
        ),
    )

# -------------------------
# Helpers: debug structure dump
# -------------------------
def _debug_dump_structure_from_notebook(
    *,
    notebook_path: str,
    artifacts_dir: str | Path | None = None,
    prefix: str = "[STRUCTURE]",
) -> Optional[List[Dict[str, Any]]]:
    """
    Debug helper:
      - reads Cell00
      - extracts STRUCTURE_JSON
      - prints it to stdout
      - optionally saves to <artifacts_dir>/debug/structure.json
    """
    try:
        nb_path = Path(str(notebook_path)).expanduser().resolve()
        if not nb_path.exists():
            print(f"{prefix} notebook not found: {nb_path}")
            return None

        nb = nbformat.read(str(nb_path), as_version=4)
        if not nb.cells:
            print(f"{prefix} notebook has no cells: {nb_path}")
            return None

        cell0_src = str(nb.cells[0].get("source") or "")
        structure = extract_structure_from_cell00(cell0_src) or []

        print(f"\n{prefix} extracted from Cell00:")
        print(json.dumps(structure, ensure_ascii=False, indent=2))

        if artifacts_dir:
            out_dir = Path(str(artifacts_dir)) / "debug"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "structure.json").write_text(
                json.dumps(structure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return structure
    except Exception as e:
        print(f"{prefix} dump failed: {type(e).__name__}: {e}")
        return None

def _step_apply_patch(
    *,
    store,
    repos,
    task_item_id: str,
    target: dict,
    backup_dir: str | Path = "state/backups",
    run_prefix: Optional[str] = None,
) -> Any:
    """
    Deterministic step: apply a notebook patch.
    Enhancements:
      - persist last_patch info into state to detect repeated no-op loops
    """
    notebook_path = target.get("notebook_path")
    new_source = (target.get("new_source") or "")
    mode = (target.get("mode") or "REPLACE").upper()
    cell_type = (target.get("cell_type") or "code").lower()
    cell_index = target.get("cell_index", None)

    
    # ✅ REPLACE/INSERT は cell_index 必須（ここで確実に落とす）
    if mode in ("REPLACE", "INSERT"):
        if cell_index is None:
            raise ValueError(f"APPLY_PATCH requires cell_index for mode={mode}")
        try:
            cell_index = int(cell_index)
        except Exception:
            raise ValueError(f"APPLY_PATCH invalid cell_index={cell_index!r}")
    
    # ✅ APPEND の場合は cell_index 無しでもOK（あるなら int 化だけ）
    if mode == "APPEND" and cell_index is not None:
        try:
            cell_index = int(cell_index)
        except Exception:
            cell_index = None


    # ---- Guardrails ----
    if mode == "INSERT":
        mode = "REPLACE"
    if not notebook_path:
        raise ValueError("APPLY_PATCH target missing notebook_path")
    if not new_source.strip():
        raise ValueError("APPLY_PATCH target missing new_source")

    # artifacts
    artifacts = target.get("artifacts")
    if artifacts is None:
        base = target.get("artifacts_dir")
        if not base:
            base = f"artifacts/{run_prefix}" if run_prefix else "artifacts/run"
        artifacts = SimpleNamespace(base_dir=Path(base))

    # Ensure notebook length (safety)
    if mode == "REPLACE" and cell_index is not None:
        import nbformat
        nb_path = Path(str(notebook_path)).expanduser().resolve()
        nb = nbformat.read(str(nb_path), as_version=4)
        while len(nb.cells) <= int(cell_index):
            nb.cells.append(nbformat.v4.new_code_cell(""))
        nbformat.write(nb, str(nb_path))

    res = patch_cell_source_with_artifacts(
        notebook_path=notebook_path,
        new_source=new_source,
        artifacts=artifacts,
        backup_dir=backup_dir,
        backup_tag=(run_prefix or "apply_patch"),
        mode=mode,
        cell_index=int(cell_index) if cell_index is not None else None,
        cell_type=cell_type,
    )
    if not res.ok:
        raise RuntimeError(res.error or "patch failed")

    # queue item DONE + result
    update_task_item(
        store,
        task_item_id=task_item_id,
        status="DONE",
        patch={"result": {"before_hash": res.before_hash, "after_hash": res.after_hash, "diff_path": res.diff_path}},
    )
    bump_metric(store, "patch_count", 1)
    mark_done(store, task_item_id=task_item_id)
    # ✅ DEBUG: print Structure immediately after scaffolding (so it appears early in logs)
    _debug_dump_structure_from_notebook(
        notebook_path=str(nb_path),
        artifacts_dir=str(art.base_dir),
        prefix="[STRUCTURE][AFTER_SCAFFOLD]",
    )

    
    # ✅ NEW: persist last_patch (for loop detection & audit)
    def _fn(st: Dict[str, Any]) -> Dict[str, Any]:
        mem = st.setdefault("memory_refs", {})
        mem["last_patch"] = {
            "task_item_id": str(task_item_id),
            "proposal_page_id": str(target.get("proposal_page_id") or ""),
            "notebook_path": str(notebook_path),
            "cell_index": int(cell_index) if cell_index is not None else None,
            "mode": str(mode),
            "cell_type": str(cell_type),
            "before_hash": res.before_hash,
            "after_hash": res.after_hash,
            "diff_path": res.diff_path,
            "updated_at": _now_iso_jst(),
        }
        st["memory_refs"] = mem
        return st

    _store_update_state(store, _fn)

    return LoopResult(
        did_work=True,
        task_item_id=task_item_id,
        step_type="APPLY_PATCH",
        ok=True,
        message=f"Patched notebook (mode={mode} cell_index={cell_index}) before={res.before_hash} after={res.after_hash}",
        proposal_page_id=str(target.get("proposal_page_id") or "") or None,
    )


    
StepType = Literal[
    "PLAN_CHANGESET",
    "SCAFFOLD_HEADERS",
    "APPLY_PATCH",
    "VERIFY_NOTEBOOK",
    "LLM_PLAN",
    "LLM_IMPLEMENT",
]



# -------------------------
# Helpers: misc
# -------------------------

def _now_iso_jst() -> str:
    # Robust JST ISO8601 (with +09:00)
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
    except Exception:
        # fallback (no tzinfo) — still stable format
        return datetime.now().isoformat(timespec="seconds")



def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def _extract_actionable_traceback(tb: str, *, max_lines: int = 25) -> str:
    """
    ノイズを落として LLM が理解しやすい最小トレースを作る。
    - 最後の 'TypeError: ...' / 'ValueError: ...' 近辺を優先
    - 'Cell In[...] line ...' があればその周辺を優先
    """
    if not tb:
        return ""

    lines = tb.splitlines()

    # 1) 末尾から "TypeError:" などを探してそこから上に少し含める
    err_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if re.search(r"\b(TypeError|ValueError|AttributeError|KeyError|RuntimeError)\b:", lines[i]):
            err_idx = i
            break

    if err_idx is not None:
        start = max(0, err_idx - (max_lines - 1))
        return "\n".join(lines[start : err_idx + 1])[-12000:]  # 念のため上限

    # 2) "Cell In[" っぽいところがあればそこから
    cell_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "Cell In[" in lines[i]:
            cell_idx = i
            break
    if cell_idx is not None:
        start = max(0, cell_idx - (max_lines - 1))
        end = min(len(lines), cell_idx + 6)
        return "\n".join(lines[start:end])[-12000:]

    # 3) fallback: 最後の max_lines
    return "\n".join(lines[-max_lines:])[-12000:]

# -------------------------
# Helpers: last_error normalization
# -------------------------

def _get_attr_any(obj: Any, names: List[str], default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default

def _normalize_nb_fail_fields(nb_res: Any) -> Dict[str, Any]:
    """
    Normalize NotebookRunResult-ish objects to a stable dict.
    This is the single source of truth for state["last_error"] population.
    """
    ok = bool(getattr(nb_res, "ok", False))

    # indices
    failing_cell_index = None
    for n in ("failing_cell_index", "failed_cell_index", "cell_index"):
        v = getattr(nb_res, n, None)
        try:
            if v is not None:
                failing_cell_index = int(v)
                break
        except Exception:
            pass

    executed_up_to = None
    for n in ("executed_up_to_cell", "executed_up_to", "executed_up_to_cell_index"):
        v = getattr(nb_res, n, None)
        try:
            if v is not None:
                executed_up_to = int(v)
                break
        except Exception:
            pass

    # traceback-ish
    tb = _get_attr_any(nb_res, ["error_traceback", "traceback", "error_trace", "tb"], default="") or ""
    tb_str = str(tb)
    if len(tb_str) > 20000:
        tb_str = tb_str[:20000] + "\n...(truncated)"
    tb_min = _extract_actionable_traceback(tb_str) or tb_str

    err_sum = str(getattr(nb_res, "error_summary", "") or "")
    err_short = str(getattr(nb_res, "error_trace_short", "") or "")

    return {
        "ok": ok,
        "failing_cell_index": failing_cell_index,
        "executed_up_to": executed_up_to,
        "error_summary": err_sum,
        "error_trace_short": err_short,
        "traceback": tb_min,
    }

def _write_last_error_from_verify(
    *,
    store: StateStore,
    has_error: bool,
    run_mode: str,
    run_page_id: str,
    artifacts_path: str,
    nb_fields: Dict[str, Any],
    rep: Optional[Any] = None,
    up_to_for_sig: Optional[int] = None,
) -> None:
    """
    Single canonical writer for state["last_error"].
    Uses set_last_error() for the core keys + attaches extras in one state.update.
    """
    category = str(getattr(rep, "category", "") or "UNKNOWN") if rep is not None else "UNKNOWN"
    next_action = str(getattr(rep, "next_action", "") or "") if rep is not None else ""
    extracted = getattr(rep, "extracted", {}) if rep is not None else {}
    hints = getattr(rep, "hints", []) if rep is not None else []

    # core fields via existing helper (already resilient with **extra)
    set_last_error(
        store,
        has_error=bool(has_error),
        failing_cell_index=nb_fields.get("failing_cell_index"),
        error_summary=nb_fields.get("error_summary", "") or "",
        error_trace_path="",  # you can wire an actual path if you have it
        executed_up_to=(up_to_for_sig if str(run_mode).upper() == "PREFIX" else None),
        run_mode=str(run_mode),
        run_page_id=str(run_page_id or ""),
        artifacts_path=str(artifacts_path or ""),
    )

    # attach extras (traceback/category/next_action/extracted/hints/etc.)
    def _fn(st: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(st, dict):
            return st
        le = st.get("last_error")
        if not isinstance(le, dict):
            le = {}
        le["traceback"] = nb_fields.get("traceback", "") or le.get("traceback", "")
        le["category"] = category
        le["next_action"] = next_action
        if isinstance(extracted, dict):
            le["extracted"] = extracted
        if isinstance(hints, list):
            le["hints"] = hints
        # keep executed_up_to even on success if you want (optional)
        if not has_error and str(run_mode).upper() == "PREFIX" and up_to_for_sig is not None:
            le["executed_up_to"] = int(up_to_for_sig) + 1
        le["updated_at"] = _now_iso_jst()
        st["last_error"] = le
        return st

    _store_update_state(store, _fn)

def _maybe_enqueue_post_bootstrap_plan_one_by_one(
    *,
    store: StateStore,
    task_page_id: str,
    proposal_page_id: str,
    notebook_path: str,
) -> None:
    """
    If state has post_bootstrap_plan[proposal].pending == True,
    enqueue a single LLM_PLAN that targets exactly one next cell (start_cell_index),
    then mark pending False (consumed) or advance start_cell_index as needed.
    """
    def _consume_and_get(st: Dict[str, Any]) -> Dict[str, Any]:
        pb = st.get("post_bootstrap_plan") if isinstance(st.get("post_bootstrap_plan"), dict) else {}
        rec = pb.get(str(proposal_page_id)) if isinstance(pb, dict) else None
        if not isinstance(rec, dict) or not rec.get("pending"):
            st["_pb_out"] = None
            return st

        start = rec.get("start_cell_index", 3)
        try:
            start_i = int(start)
        except Exception:
            start_i = 3

        st["_pb_out"] = {
            "start_cell_index": start_i,
            "mode": rec.get("mode") or "ONE_BY_ONE",
        }

        # consume (set pending False)
        rec2 = dict(rec)
        rec2["pending"] = False
        rec2["updated_at"] = _now_iso_jst()
        pb2 = dict(pb)
        pb2[str(proposal_page_id)] = rec2
        st["post_bootstrap_plan"] = pb2
        return st

    st2 = _store_update_state(store, _consume_and_get)
    out = None
    try:
        out = (st2 or {}).get("_pb_out")
    except Exception:
        out = None

    if not isinstance(out, dict):
        return

    cell_i = int(out.get("start_cell_index") or 3)

    enqueue(
        store,
        new_task_item(
            type="LLM_PLAN",
            intent=f"Post-bootstrap plan: plan and implement exactly Cell{cell_i:02d} (one-by-one).",
            assignee="PLANNER",
            priority=90,
            target={
                "task_page_id": str(task_page_id),
                "proposal_page_id": str(proposal_page_id),
                "notebook_path": str(notebook_path),
                "hint": {
                    "phase": "POST_BOOTSTRAP_ONE_BY_ONE",
                    "target_cell_index": cell_i,
                    "mode": "ONE_BY_ONE",
                },
            },
        ),
    )


def _advance_executed_up_to_on_prefix_pass(
    *,
    store: StateStore,
    proposal_page_id: str,
    notebook_path: str,
    passed_up_to: int,
) -> None:
    """
    On successful PREFIX verify (up_to=N), advance executed_up_to -> N+1.
    We store it in state["last_error"]["executed_up_to"] even when has_error=False,
    because downstream LLM_IMPLEMENT uses executed_up_to as the next verification boundary.

    Also keeps a per-proposal progress mirror under memory_refs.prefix_progress for robustness.
    """

    next_up_to = int(passed_up_to) + 1

    def _fn(st: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(st, dict):
            return st

        # ---- last_error: keep progress even if not error ----
        le = st.get("last_error")
        if not isinstance(le, dict):
            le = {}

        le["has_error"] = False
        le["error_summary"] = ""          # clear
        le["traceback"] = ""              # clear
        le["failing_cell_index"] = None   # clear
        le["run_mode"] = "PREFIX"
        le["executed_up_to"] = next_up_to
        le["notebook_path"] = str(notebook_path)
        le["proposal_page_id"] = str(proposal_page_id)
        le["updated_at"] = _now_iso_jst()
        st["last_error"] = le

        # ---- mirror progress per proposal (optional but useful) ----
        mem = st.setdefault("memory_refs", {})
        prog = mem.setdefault("prefix_progress", {})
        rec = prog.setdefault(str(proposal_page_id), {})
        rec["executed_up_to"] = next_up_to
        rec["notebook_path"] = str(notebook_path)
        rec["updated_at"] = _now_iso_jst()
        prog[str(proposal_page_id)] = rec
        mem["prefix_progress"] = prog
        st["memory_refs"] = mem

        return st

    _store_update_state(store, _fn)

def _assert_repos_ready(repos: Any) -> None:
    """
    Fail-fast: repos is NotionRepos instance (NOT src.notion.repos module).
    This prevents infinite VERIFY_NOTEBOOK failures like:
      AttributeError: module 'src.notion.repos' has no attribute 'runs'
    """
    if repos is None:
        raise TypeError("repos is None. You must pass a NotionRepos instance.")

    # Typical mistake: `from src.notion import repos` (module) passed in
    import types
    if isinstance(repos, types.ModuleType):
        raise TypeError(
            "repos is a module (src.notion.repos), not a NotionRepos instance. "
            "Pass build_repos(...) result instead."
        )

    # Minimal contract we rely on
    for attr in ("tasks", "proposals", "runs"):
        if not hasattr(repos, attr):
            raise TypeError(f"repos missing '{attr}'. Expected NotionRepos; got={type(repos)}")

    if not hasattr(repos.runs, "create_run") or not callable(repos.runs.create_run):
        raise TypeError("repos.runs.create_run missing/not callable. Check repos wiring.")

def _make_failure_signature(
    *,
    proposal_page_id: str,
    run_mode: str,
    up_to_cell_index: Optional[int],
    category: str,
    next_action: str,
    error_summary: str,
    cell_index: Optional[int],
) -> str:
    raw = "|".join(
        [
            str(proposal_page_id),
            str(run_mode),
            str(up_to_cell_index) if up_to_cell_index is not None else "None",
            str(category or ""),
            str(next_action or ""),
            str(error_summary or ""),
            str(cell_index) if cell_index is not None else "None",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _store_update_state(store: StateStore, fn):
    """
    StateStore.update expects a callable: fn(st)->st
    """
    if hasattr(store, "update"):
        return store.update(fn)
    raise AttributeError("StateStore has no update(fn)")

def _state_get(store: StateStore) -> Dict[str, Any]:
    if hasattr(store, "load"):
        st = store.load()
        return st if isinstance(st, dict) else {}
    return {}


def _save_schema_cache_from_artifacts(
    *,
    store: StateStore,
    proposal_page_id: str,
    notebook_path: str,
    artifacts_dir: str,
) -> Optional[str]:
    """
    Best-effort: read schema snapshot produced by Cell02 and store into state.
    Expected file: <artifacts_dir>/schema/schema_snapshot.json
    Fallback: <artifacts_dir>/schema_snapshot.json
    """
    try:
        base = Path(str(artifacts_dir))
        p = base / "schema" / "schema_snapshot.json"
        if not p.exists():
            p2 = base / "schema_snapshot.json"
            if p2.exists():
                p = p2
            else:
                return None

        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None

        def _fn(st: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(st, dict):
                st = {}
            cache = st.setdefault("schema_cache", {})
            key = str(proposal_page_id) if proposal_page_id else str(notebook_path)
            cache[key] = {
                "schema_snapshot": data,
                "artifacts_path": str(base),
                "notebook_path": str(notebook_path),
                "updated_at": _now_iso_jst(),
            }
            st["schema_cache"] = cache
            st["schema_cache_latest"] = cache[key]
            return st

        _store_update_state(store, _fn)
        return str(p)
    except Exception:
        return None


def _inject_context_pack_into_target(store: StateStore, target: Dict[str, Any]) -> None:
    """
    Minimal context pack injector:
      - last_error (summary + traceback + indices)
      - latest schema snapshot (from state)
    """
    try:
        st = _state_get(store)
        if not isinstance(st, dict):
            return

        le = st.get("last_error") if isinstance(st.get("last_error"), dict) else {}
        latest = st.get("schema_cache_latest") if isinstance(st.get("schema_cache_latest"), dict) else {}

        target.setdefault("context_pack", {})
        cp = target.get("context_pack")
        if not isinstance(cp, dict):
            return

        if isinstance(le, dict) and le:
            cp.setdefault("last_error", {})
            if isinstance(cp["last_error"], dict):
                cp["last_error"].setdefault("failing_cell_index", le.get("failing_cell_index"))
                cp["last_error"].setdefault("error_summary", le.get("error_summary"))
                cp["last_error"].setdefault("traceback", le.get("traceback"))
                cp["last_error"].setdefault("executed_up_to", le.get("executed_up_to"))
                cp["last_error"].setdefault("run_mode", le.get("run_mode"))
                cp["last_error"].setdefault("run_page_id", le.get("run_page_id"))
                cp["last_error"].setdefault("artifacts_path", le.get("artifacts_path"))

        if isinstance(latest, dict) and latest.get("schema_snapshot"):
            cp.setdefault("schema_snapshot", latest.get("schema_snapshot"))
    except Exception:
        return

def _debounce_should_replan(
    *,
    store: StateStore,
    proposal_page_id: str,
    failure_sig: str,
    debounce_sec: int,
    max_replans_per_sig: int = 2,
) -> bool:
    """
    Returns True if we SHOULD enqueue replan.
    Uses state["memory_refs"]["replan_debounce"][proposal_id] records.
    """
    out = {"should": True}

    def _fn(st: Dict[str, Any]) -> Dict[str, Any]:
        mem = st.setdefault("memory_refs", {})
        db = mem.setdefault("replan_debounce", {})
        rec = db.setdefault(str(proposal_page_id), {})
        last_sig = rec.get("last_sig")
        last_ts = rec.get("last_ts")
        sig_counts = rec.setdefault("sig_counts", {})

        now = time.time()

        # same signature within debounce window -> block
        if last_sig == failure_sig and isinstance(last_ts, (int, float)) and (now - float(last_ts)) < debounce_sec:
            out["should"] = False
            return st

        # cap per signature to avoid infinite loops
        cnt = int(sig_counts.get(failure_sig, 0))
        if cnt >= int(max_replans_per_sig):
            out["should"] = False
            return st

        rec["last_sig"] = failure_sig
        rec["last_ts"] = now
        sig_counts[failure_sig] = cnt + 1
        rec["sig_counts"] = sig_counts
        db[str(proposal_page_id)] = rec
        mem["replan_debounce"] = db
        st["memory_refs"] = mem
        out["should"] = True
        return st

    _store_update_state(store, _fn)
    return bool(out["should"])


def _cleanup_queue_for_task_proposal(
    *,
    store: StateStore,
    task_page_id: str,
    proposal_page_id: str,
    keep_task_item_id: Optional[str] = None,
    cancel_types: Optional[List[str]] = None,
) -> int:
    """
    Cancel (mark status=CANCELLED) all TODO queue items for the same task/proposal
    that are likely stale after a replan.

    We DO NOT delete items (safer audit-wise). We mark as CANCELLED.
    """
    if cancel_types is None:
        cancel_types = ["VERIFY_NOTEBOOK", "APPLY_PATCH", "LLM_IMPLEMENT", "LLM_PLAN", "SCAFFOLD_HEADERS"]

    cancelled = {"n": 0}

    def _fn(st: Dict[str, Any]) -> Dict[str, Any]:
        q = list(st.get("queue") or [])
        newq = []
        for it in q:
            try:
                if keep_task_item_id and it.get("task_item_id") == keep_task_item_id:
                    newq.append(it)
                    continue

                if (it.get("status") or "TODO") != "TODO":
                    newq.append(it)
                    continue

                if (it.get("type") or "") not in cancel_types:
                    newq.append(it)
                    continue

                tgt = it.get("target") or {}
                if str(tgt.get("task_page_id") or "") != str(task_page_id):
                    newq.append(it)
                    continue
                if str(tgt.get("proposal_page_id") or "") != str(proposal_page_id):
                    newq.append(it)
                    continue

                it2 = dict(it)
                it2["status"] = "CANCELLED"
                it2["last_error"] = "Superseded by replan"
                it2["updated_at"] = _now_iso_jst()
                newq.append(it2)
                cancelled["n"] += 1
            except Exception:
                newq.append(it)

        st["queue"] = newq
        return st

    _store_update_state(store, _fn)
    return int(cancelled["n"])


def _enabled(qg_cfg: Dict[str, Any], name: str) -> bool:
    cfg = qg_cfg.get(name)
    if cfg is None:
        return False
    if isinstance(cfg, dict):
        return bool(cfg.get("enabled", False))
    return False


def _fail(store: StateStore, task_item_id: str, msg: str) -> "LoopResult":
    mark_failed(store, task_item_id=task_item_id, reason=msg)
    return LoopResult(did_work=True, task_item_id=task_item_id, step_type=None, ok=False, message=msg)


def _policy_normalize_for_scaffold(policy: Dict[str, Any], structure_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize arbitrary policy shapes to scaffold_headers.build_cell00 expectations.
    - Accepts your current policy shape with policy["writes"] dict, and produces
      top-level keys writes_allowed / writes_forbidden.
    - Forces structure to list[dict] (structure_list).
    """
    p = dict(policy or {})
    writes = p.get("writes") if isinstance(p.get("writes"), dict) else {}

    # back-compat: allow either shape
    p.setdefault("writes_allowed", p.get("writes_allowed") or (writes.get("allowed") if isinstance(writes, dict) else []) or [])
    p.setdefault(
        "writes_forbidden",
        p.get("writes_forbidden") or (writes.get("forbidden") if isinstance(writes, dict) else []) or [],
    )

    # keep structure strictly a list (cell plan)
    p["structure"] = structure_list

    # optional niceties (don’t override if already present)
    if "required_env" not in p and isinstance(p.get("data_sources"), dict):
        # common env list can be filled by caller; leave empty by default
        p["required_env"] = p.get("required_env") or []
    return p


# -------------------------
# Result container
# -------------------------

@dataclass
class LoopResult:
    did_work: bool
    task_item_id: Optional[str]
    step_type: Optional[str]
    ok: bool
    message: str
    run_page_id: Optional[str] = None
    proposal_page_id: Optional[str] = None


# -------------------------
# Main loop
# -------------------------

def run_one_step(
    *,
    store: StateStore,
    repos: NotionRepos,
    # optional context
    default_task_id: str = "TASK",
    default_run_prefix: str = "run",
    backup_dir: str | Path = "state/backups",
) -> LoopResult:
    """
    Pop the next TODO item from the state queue, mark it DOING, execute it, and
    update Notion + state accordingly.

    Returns LoopResult. If no TODO exists: did_work=False.
    """
    item = pop_next_todo(store)
    if item is None:
        return LoopResult(did_work=False, task_item_id=None, step_type=None, ok=True, message="No TODO items.")

    task_item_id = item.get("task_item_id")
    step_type: StepType = item.get("type")  # type: ignore[assignment]
    target: Dict[str, Any] = dict(item.get("target") or {})

    # ============================================================
    # ✅ STRUCTURE injection / pre-seed gate for SCAFFOLD_HEADERS
   # - If target.structure is empty but state has plan_seed_structure -> inject it
    # - If BOTH are empty -> enqueue seed-only plan before scaffold
    # ============================================================
    if str(step_type).upper() == "SCAFFOLD_HEADERS":
        st_now = _state_get(store)
        seed = _get_plan_seed_structure(st_now)
        tgt_struct = target.get("structure")
        has_target_struct = _has_nonempty_structure(tgt_struct)
        has_seed = _has_nonempty_structure(seed)

        # 1) Inject seed into target if missing
        if (not has_target_struct) and has_seed:
            # important: don't mutate original queue item dict accidentally
            target = dict(target)
            target["structure"] = copy.deepcopy(seed)

        # 2) If still empty: schedule seed first, then reschedule scaffold
        if (not _has_nonempty_structure(target.get("structure"))) and (not has_seed):
            _enqueue_structure_seed_then_scaffold(
                store=store,
                target=target,
                task_item_id=str(task_item_id),
                reason="SCAFFOLD_HEADERS requires STRUCTURE seed; enqueued structure_seed_only LLM_PLAN first.",
            )
            return LoopResult(
                did_work=True,
                task_item_id=str(task_item_id),
                step_type="SCAFFOLD_HEADERS",
                ok=True,
                message="Seeded STRUCTURE job enqueued before scaffold (scaffold re-queued).",
            )
 
    # ✅ Inject stable context pack for LLM steps (schema + last_error)
    if str(step_type).upper() in ("LLM_PLAN", "LLM_IMPLEMENT"):
        _inject_context_pack_into_target(store, target)
    # -------------------------
    # LLM step dispatcher
    # -------------------------
    if str(step_type).upper() in ("LLM_PLAN", "LLM_IMPLEMENT"):
        try:
            # ClaudeClient の生成方法は環境により異なるので、
            # 可能な限り「from_env / build」系があればそれを使い、なければ素直に生成する
            claude = None
            if hasattr(ClaudeClient, "from_env") and callable(getattr(ClaudeClient, "from_env")):
                claude = ClaudeClient.from_env()
            else:
                claude = ClaudeClient()

            out = handle_llm_step(
                step_type=str(step_type),
                store=store,
                repos=repos,
                claude=claude,
                task_item_id=str(task_item_id),
                target=target,
            )
            ok = bool(out.get("ok", False))
            msg = str(out.get("message") or "")
            return LoopResult(did_work=True, task_item_id=task_item_id, step_type=str(step_type), ok=ok, message=msg)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            update_task_item(store, task_item_id=task_item_id, status="FAILED", last_error=err)
            mark_failed(store, task_item_id=task_item_id, reason=err)
            return LoopResult(did_work=True, task_item_id=task_item_id, step_type=str(step_type), ok=False, message=f"Exception in {step_type}: {err}")

    # =========================
    # ✅ NEW: Always inject last_error into LLM_IMPLEMENT target
    # =========================
    if str(step_type).upper() == "LLM_IMPLEMENT":
        try:
            st_now = store.load() if hasattr(store, "load") else {}
            le = (st_now.get("last_error") or {}) if isinstance(st_now, dict) else {}
            if isinstance(le, dict) and le.get("has_error"):
                tb0 = str(le.get("traceback", "") or "")
                tb_min = _extract_actionable_traceback(tb0)
    
                target.setdefault("run_evidence", {})
                if isinstance(target["run_evidence"], dict):
                    target["run_evidence"].setdefault("last_error", {})
                    if isinstance(target["run_evidence"]["last_error"], dict):
                        target["run_evidence"]["last_error"]["error_summary"] = le.get("error_summary", "")
                        target["run_evidence"]["last_error"]["failing_cell_index"] = le.get("failing_cell_index", None)
                        target["run_evidence"]["last_error"]["traceback"] = tb_min
                        target["run_evidence"]["last_error"]["executed_up_to"] = le.get("executed_up_to", None)
                        target["run_evidence"]["last_error"]["run_mode"] = le.get("run_mode", "")
                        target["run_evidence"]["last_error"]["run_page_id"] = le.get("run_page_id", "")
                        target["run_evidence"]["last_error"]["artifacts_path"] = le.get("artifacts_path", "")
    
                target.setdefault("failing_cell_index", le.get("failing_cell_index", None))
                target.setdefault("executed_up_to", le.get("executed_up_to", None))
                target.setdefault("traceback", tb_min)
        except Exception:
            pass



    # =========================
    # ✅ NEW: repos preflight
    # =========================
    REPOS_REQUIRED_STEPS = {"PLAN_CHANGESET", "VERIFY_NOTEBOOK"}
    
    def _set_once_flag(store: StateStore, key: str) -> bool:
        """Return True only the first time for this key (persisted in state)."""
        out = {"fresh": False}
    
        def _fn(st: Dict[str, Any]) -> Dict[str, Any]:
            mem = st.setdefault("memory_refs", {})
            flags = mem.setdefault("once_flags", {})
            if flags.get(key):
                out["fresh"] = False
                return st
            flags[key] = True
            mem["once_flags"] = flags
            st["memory_refs"] = mem
            out["fresh"] = True
            return st
    
        _store_update_state(store, _fn)
        return bool(out["fresh"])
    
    
    if isinstance(repos, ModuleType) and str(step_type) in REPOS_REQUIRED_STEPS:
        msg = (
            "[NON-RECOVERABLE] TypeError: repos is a module (src.notion.repos), "
            "not a NotionRepos instance. Pass build_repos(...) result instead."
        )
    
        # 1) mark this queue item failed
        update_task_item(store, task_item_id=task_item_id, status="FAILED", last_error=msg)
        mark_failed(store, task_item_id=task_item_id, reason=msg)
    
        # 2) enqueue LLM_IMPLEMENT ONCE to patch Cell 01 (or the setup cell) to use build_repos()
        nb_path = str(target.get("notebook_path") or "")
        proposal_page_id = str(target.get("proposal_page_id") or "")
        task_page_id = str(target.get("task_page_id") or "")
    
        # key includes notebook + proposal so we don't spam the queue
        once_key = f"enqueue_fix_repos_module:{nb_path}:{proposal_page_id}:{task_page_id}"
    
        if nb_path and _set_once_flag(store, once_key):
            enqueue(
                store,
                new_task_item(
                    type="LLM_IMPLEMENT",
                    intent="Fix notebook setup: ensure repos is a NotionRepos instance by using build_repos() (do not import src.notion.repos module).",
                    assignee="IMPLEMENTER",
                    priority=99,
                    target={
                        "task_page_id": task_page_id,
                        "proposal_page_id": proposal_page_id,
                        "notebook_path": nb_path,
                        "cell_index": 1,  # <- your convention: setup cell
                        "fix_kind": "REPOS_IS_MODULE",
                        "error_summary": msg,
                        "desired_change": "Replace any `import src.notion.factory` usage with `from src.notion.repos import build_repos; repos = build_repos()` (or your actual builder).",
                    },
                ),
            )
            bump_metric(store, "llm_implement_enqueued", 1)
    
            return LoopResult(
                did_work=True,
                task_item_id=task_item_id,
                step_type=str(step_type),
                ok=False,
                message=msg + " (Enqueued LLM_IMPLEMENT once.)",
            )
    
        # If already enqueued once, just stop (no infinite loop)
        return LoopResult(
            did_work=True,
            task_item_id=task_item_id,
            step_type=str(step_type),
            ok=False,
            message=msg + " (LLM_IMPLEMENT already enqueued; suppressed.)",
        )



    try:
        # ✅ Fail-fast for Notion wiring issues (prevents infinite VERIFY loops)
        if step_type in ("PLAN_CHANGESET", "VERIFY_NOTEBOOK"):
            _assert_repos_ready(repos)

        if step_type == "PLAN_CHANGESET":
            return _step_plan_changeset(store=store, repos=repos, task_item_id=task_item_id, target=target)

        if step_type == "SCAFFOLD_HEADERS":
            return _step_scaffold_headers(
                store=store,
                repos=repos,
                task_item_id=task_item_id,
                target=target,
                backup_dir=backup_dir,
                run_prefix=default_run_prefix,
                task_id_fallback=default_task_id,
            )

        if step_type == "APPLY_PATCH":
            return _step_apply_patch(
                store=store,
                repos=repos,
                task_item_id=task_item_id,
                target=target,
                backup_dir=backup_dir,
                run_prefix=default_run_prefix,
            )

        if step_type == "VERIFY_NOTEBOOK":
            return _step_verify_notebook(
                store=store,
                repos=repos,
                task_item_id=task_item_id,
                target=target,
                run_prefix=default_run_prefix,
                task_id_fallback=default_task_id,
            )

        mark_failed(store, task_item_id=task_item_id, reason=f"Unknown step type: {step_type}")
        return LoopResult(did_work=True, task_item_id=task_item_id, step_type=str(step_type), ok=False, message=f"Unknown step type: {step_type}")


    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        update_task_item(store, task_item_id=task_item_id, status="FAILED", last_error=err)
    
        # ✅ NEW: known config error → enqueue LLM_PLAN
        if "repos is a module (src.notion.repos)" in err:
            planner_target = {
                "task_page_id": str(target.get("task_page_id") or ""),
                "proposal_page_id": str(target.get("proposal_page_id") or ""),
                "notebook_path": str(target.get("notebook_path") or ""),
                "run_evidence": {"last_error": {"category": "ORCHESTRATOR_CONFIG", "next_action": "FIX_REPOS_WIRING", "error_summary": err}},
                "hint": {"fix_kind": "REPOS_IS_MODULE", "target_cell_index": 1},
            }
            enqueue(
                store,
                new_task_item(
                    type="LLM_PLAN",
                    intent="Fix repos wiring: convert repos module to NotionRepos instance via build_repos().",
                    assignee="PLANNER",
                    priority=99,
                    target=planner_target,
                ),
            )
            bump_metric(store, "llm_replan_enqueued", 1)
    
        return LoopResult(
            did_work=True,
            task_item_id=task_item_id,
            step_type=str(step_type),
            ok=False,
            message=f"Exception: {err}",
        )




# -------------------------
# Step implementations
# -------------------------

def _step_plan_changeset(
    *,
    store: StateStore,
    repos: NotionRepos,
    task_item_id: str,
    target: Dict[str, Any],
) -> LoopResult:
    """
    Planner step: create a ChangeSet page in PROPOSALS_DB and link to TASKS_DB.
    """
    task_page_id = target.get("task_page_id")
    if not task_page_id:
        return _fail(store, task_item_id, "PLAN_CHANGESET missing target.task_page_id")

    title = target.get("proposal_title") or "ChangeSet"
    notebook_path = target.get("notebook_path") or ""
    cell_index = target.get("cell_index")
    if cell_index is None:
        return _fail(store, task_item_id, "PLAN_CHANGESET missing target.cell_index")

    proposal = repos.proposals.create_changeset(
        title=title,
        task_page_id=task_page_id,
        status="DRAFT",
        notebook_path=notebook_path or None,
        cell_index=int(cell_index),
        intent=target.get("intent"),
        acceptance=target.get("acceptance"),
        risk=target.get("risk"),
        owner_text=target.get("owner_text"),
    )
    proposal_page_id = proposal["id"]

    repos.tasks.link_proposal(task_page_id=task_page_id, proposal_page_id=proposal_page_id, set_latest=True)

    update_task_item(
        store,
        task_item_id=task_item_id,
        status="DONE",
        patch={"result": {"proposal_page_id": proposal_page_id}},
    )
    bump_metric(store, "patch_count", 0)

    enqueue(
        store,
        new_task_item(
            type="SCAFFOLD_HEADERS",
            intent="Scaffold notebook headers (Cell00-02 + per-cell headers).",
            assignee="SYSTEM",
            priority=95,
            target={
                "task_page_id": str(task_page_id),
                "proposal_page_id": str(proposal_page_id),
                "notebook_path": str(notebook_path),
                "task_fields": target.get("task_fields") or {},   # あるならここで渡す
                "policy": target.get("policy") or {},             # あるならここで渡す
                "structure": target.get("structure"),             # optional
                "cleanup_queue": True,
                "preserve_existing": True,
            },
        ),
    )

    return LoopResult(
        did_work=True,
        task_item_id=task_item_id,
        step_type="PLAN_CHANGESET",
        ok=True,
        message=f"Created proposal {proposal_page_id}",
        proposal_page_id=proposal_page_id,
    )


def _step_scaffold_headers(
    *,
    store: StateStore,
    repos: NotionRepos,  # kept for parity / future extension
    task_item_id: str,
    target: Dict[str, Any],
    backup_dir: str | Path,
    run_prefix: str,
    task_id_fallback: str,
) -> LoopResult:
    """
    Deterministic scaffold step:
      - Generate Cell 00 (notebook overview header)
      - Generate per-cell headers for Cell 01.. based on provided structure
      - Debounce using a digest in state.json
      - Avoid duplicate headers (signature check)
      - Optionally prune duplicate queued SCAFFOLD_HEADERS/LLM_PLAN items for same task/proposal

    Required target:
      - task_page_id
      - proposal_page_id
      - notebook_path
      - task_fields: dict (from TASKS_DB)
      - policy: dict
    Optional:
      - structure: list[dict] overrides policy-derived structure
      - cleanup_queue: bool (default True)
      - preserve_existing: bool (default True)
    """

    task_page_id = target.get("task_page_id")
    proposal_page_id = target.get("proposal_page_id")
    notebook_path = target.get("notebook_path")
    task_fields = target.get("task_fields") or {}
    policy_in = target.get("policy") or {}
    # ----------------------------------------------------------
    # Structure source of truth:
    #   - If Cell00 is locked, read STRUCTURE_JSON from Cell00.
    #   - Else (first run), use target["structure"] (preferred) or policy["structure"].
    # ----------------------------------------------------------
    structure_in = target.get("structure")
    structure_fallback = (
        structure_in if isinstance(structure_in, list)
        else (policy_in.get("structure") if isinstance(policy_in.get("structure"), list) else [])
    )

    if not task_page_id or not proposal_page_id or not notebook_path:
        return _fail(store, task_item_id, "SCAFFOLD_HEADERS missing task_page_id/proposal_page_id/notebook_path")
    if not isinstance(task_fields, dict) or not isinstance(policy_in, dict):
        return _fail(store, task_item_id, "SCAFFOLD_HEADERS requires target.task_fields and target.policy dict")

    cleanup_queue = bool(target.get("cleanup_queue", True))
    preserve_existing = bool(target.get("preserve_existing", True))

    # ---- load notebook early (needed to resolve structure) ----
    nb_path = Path(str(notebook_path)).expanduser().resolve()
    if not nb_path.exists():
        return _fail(store, task_item_id, f"SCAFFOLD_HEADERS notebook not found: {nb_path}")
    
    nb = nbformat.read(str(nb_path), as_version=4)
    cell0_src = (nb.cells[0].get("source") or "") if nb.cells else ""
    cell0_locked = (CELL0_LOCK_BEGIN in cell0_src) and (CELL0_LOCK_END in cell0_src)
    # ---- create run artifacts dir (single run for this scaffold) ----
    run_id = next_run_id(prefix=run_prefix)
    art = ensure_run_dir(run_id=run_id, task_id=str(task_page_id) or task_id_fallback, notebook_path=str(nb_path))

    
    # ---- structure source of truth ----
    if cell0_locked:
        extracted = extract_structure_from_cell00(cell0_src)
        if extracted:
            structure = extracted
        else:
            # ✅ RECOVERY: Cell00 locked but embedded structure is empty.
            # If caller provides a non-empty structure, patch ONLY the STRUCTURE_JSON block once.
            structure = structure_fallback
            if isinstance(structure, list) and structure:
                import json

                def _normalize(x):
                    out = []
                    for it in x:
                        if not isinstance(it, dict):
                            continue
                        if it.get("cell_index") is None or not it.get("title"):
                            continue
                        out.append(
                            {
                                "cell_index": int(it.get("cell_index")),
                                "title": str(it.get("title") or ""),
                                "overview": str(it.get("overview") or ""),
                                "io": str(it.get("io") or ""),
                                "notes": str(it.get("notes") or ""),
                            }
                        )
                    out.sort(key=lambda d: d["cell_index"])
                    return out

                payload = _normalize(structure)
                new_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)

                def _replace_structure_block(src: str) -> str:
                    a = src.find(STRUCTURE_JSON_BEGIN)
                    b = src.find(STRUCTURE_JSON_END, a + 1)
                    if a == -1 or b == -1 or b <= a:
                        return src  # give up safely
                    head = src[: a + len(STRUCTURE_JSON_BEGIN)]
                    tail = src[b:]
                    mid_lines = []
                    for ln in new_json.splitlines():
                        mid_lines.append("# " + ln)
                    mid = "\n" + "\n".join(mid_lines) + "\n"
                    return head + mid + tail

                cell0_new = _replace_structure_block(cell0_src)
                if cell0_new != cell0_src:
                    pr0_fix = patch_cell_source_with_artifacts(
                        notebook_path=str(nb_path),
                        new_source=cell0_new.rstrip() + "\n",
                        artifacts=art,
                        backup_dir=backup_dir,
                        backup_tag=f"proposal_{proposal_page_id}",
                        diff_name_hint=f"scaffold_cell00_structure_fix_{proposal_page_id}",
                        mode="REPLACE",
                        cell_index=0,
                        cell_type="code",
                    )
                    if not pr0_fix.ok:
                        return _fail(store, task_item_id, f"Failed to patch Cell00 structure JSON: {pr0_fix.error}")
    else:
        structure = structure_fallback

    # Parse + validate structure items
    items: List[Dict[str, Any]] = []
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

            title = str(it.get("title") or "").strip()
            if not title:
                continue
    
            items.append(
                {
                    "cell_index": idx,
                    "title": title,
                    "overview": str(it.get("overview") or ""),
                    "io": str(it.get("io") or ""),
                    "notes": str(it.get("notes") or ""),
                }
            )
    
    # dedupe by cell_index (最後勝ち)
    dedup = {}
    for it in items:
        dedup[int(it["cell_index"])] = it
    items = list(dedup.values())
    items.sort(key=lambda x: x["cell_index"])
    max_idx = max([2] + [it["cell_index"] for it in items])  # 02 までは必ず作る


    # Normalize policy to scaffold_headers expectations (writes_allowed/forbidden, structure=list)
    policy = _policy_normalize_for_scaffold(policy_in, items)

    # ---- debounce digest (stable) ----
    digest = build_scaffold_digest(
        task_page_id=str(task_page_id),
        proposal_page_id=str(proposal_page_id),
        notebook_path=str(nb_path),
        structure=items,
        policy=policy,
    )


    st0 = _state_get(store)
    prev = (st0.get("scaffold") or {}) if isinstance(st0.get("scaffold"), dict) else {}
    if prev.get("digest") == digest and prev.get("status") == "DONE":
        mark_done(store, task_item_id=task_item_id)
        return LoopResult(
            did_work=True,
            task_item_id=task_item_id,
            step_type="SCAFFOLD_HEADERS",
            ok=True,
            message=f"Scaffold unchanged (debounced digest={digest})",
            proposal_page_id=str(proposal_page_id),
        )

    # ---- ensure notebook has enough cells (CRITICAL: must WRITE before patcher REPLACE) ----
    while len(nb.cells) < max_idx + 1:
        nb.cells.append(nbformat.v4.new_code_cell(""))

    # WRITE here so patcher reads the expanded notebook (patcher re-reads from disk)
    nbformat.write(nb, str(nb_path))

    # ---- Cell00: create ONLY if not locked ----
    if not cell0_locked:
        # 初回のみCell00を確定（以降は不変）
        # structure は fallback を使い、Cell00に埋め込む
        policy_for_cell00 = _policy_normalize_for_scaffold(policy_in, items)
    
        cell00_new = build_cell00(
            task_fields=task_fields,
            policy=policy_for_cell00,
            structure=items,
        )
    
        # 既存のCell0コードは原則保持しない（事故源なので）
        # preserve_existing を尊重したいなら下の merged0 を少し変えればOK
        merged0 = cell00_new + "\n"
    
        pr0 = patch_cell_source_with_artifacts(
            notebook_path=str(nb_path),
            new_source=merged0,
            artifacts=art,
            backup_dir=backup_dir,
            backup_tag=f"proposal_{proposal_page_id}",
            diff_name_hint=f"scaffold_cell00_{proposal_page_id}",
            mode="REPLACE",
            cell_index=0,
            cell_type="code",
        )
        if not pr0.ok:
            update_task_item(store, task_item_id=task_item_id, status="FAILED", last_error=pr0.error)
            return LoopResult(
                did_work=True,
                task_item_id=task_item_id,
                step_type="SCAFFOLD_HEADERS",
                ok=False,
                message=f"Scaffold failed at Cell 00: {pr0.error}",
                proposal_page_id=str(proposal_page_id),
            )
    
        patched = 1
    else:
        # locked: we never touch Cell00 again
        patched = 0

    # ============================================================
    # ✅ FORCE Cell01: always create/replace the real setup cell
    #   - Do NOT depend on `items` (structure may omit cell_index=1)
    #   - Guarantees Cell01 exists even when structure starts from 2
    # ============================================================
    
    # Ensure notebook has at least 2 cells (cell0 + cell1)
    nb = nbformat.read(str(nb_path), as_version=4)
    while len(nb.cells) <= 1:
        nb.cells.append(nbformat.v4.new_code_cell(""))
    nbformat.write(nb, str(nb_path))
    
    req_env = policy.get("required_env") if isinstance(policy, dict) else []
    if not isinstance(req_env, (list, tuple)):
        req_env = []
    
    cell01_src = build_cell01_setup(required_env=list(req_env))
    
    pr1 = patch_cell_source_with_artifacts(
        notebook_path=str(nb_path),
        new_source=cell01_src.rstrip() + "\n",
        artifacts=art,
        backup_dir=backup_dir,
        backup_tag=f"proposal_{proposal_page_id}",
        diff_name_hint=f"scaffold_cell001_setup_{proposal_page_id}",
        mode="REPLACE",
        cell_index=1,
        cell_type="code",
    )
    if not pr1.ok:
        update_task_item(store, task_item_id=task_item_id, status="FAILED", last_error=pr1.error)
        return LoopResult(
            did_work=True,
            task_item_id=task_item_id,
            step_type="SCAFFOLD_HEADERS",
            ok=False,
            message=f"Scaffold failed at Cell 01 (setup): {pr1.error}",
            proposal_page_id=str(proposal_page_id),
        )
    
    patched += 1
    # ============================================================
    # ✅ FORCE Cell02: always create/replace schema introspection cell
    # ============================================================
    
    nb = nbformat.read(str(nb_path), as_version=4)
    while len(nb.cells) <= 2:
        nb.cells.append(nbformat.v4.new_code_cell(""))
    nbformat.write(nb, str(nb_path))
    
    cell02_src = build_cell02_schema_introspection_repo_first()
    
    pr2 = patch_cell_source_with_artifacts(
        notebook_path=str(nb_path),
        new_source=cell02_src.rstrip() + "\n",
        artifacts=art,
        backup_dir=backup_dir,
        backup_tag=f"proposal_{proposal_page_id}",
        diff_name_hint=f"scaffold_cell002_schema_{proposal_page_id}",
        mode="REPLACE",
        cell_index=2,
        cell_type="code",
    )
    
    if not pr2.ok:
        update_task_item(store, task_item_id=task_item_id, status="FAILED", last_error=pr2.error)
        return LoopResult(
            did_work=True,
            task_item_id=task_item_id,
            step_type="SCAFFOLD_HEADERS",
            ok=False,
            message=f"Scaffold failed at Cell 02 (schema): {pr2.error}",
            proposal_page_id=str(proposal_page_id),
        )
    
    patched += 1

    def _set_post_bootstrap_pending_once(st: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(st, dict):
            st = {}
        pb = st.setdefault("post_bootstrap_plan", {})
        key = str(proposal_page_id)
    
        rec = pb.get(key)
        if isinstance(rec, dict) and rec.get("initialized"):
            return st  # 既に設定済みなら触らない
    
        pb[key] = {
            "pending": True,
            "start_cell_index": 3,
            "mode": "ONE_BY_ONE",
            "initialized": True,
            "updated_at": _now_iso_jst(),
        }
        st["post_bootstrap_plan"] = pb
        return st
    
    _store_update_state(store, _set_post_bootstrap_pending_once)

    
    # ---- patch each cell header ----
    for it in items:
        i = int(it["cell_index"])    
        # ✅ Cell01/02 are fixed cells (forced above). Do not touch here.
        if i in (1, 2):
            continue
        # ---- default: header-only scaffolding for other cells ----
        header = build_cell_header(
            cell_index=i,
            title=it["title"],
            overview=it["overview"],
            io=it["io"],
            notes=it["notes"],
        )
    
        nb = nbformat.read(str(nb_path), as_version=4)
        cur = (nb.cells[i].get("source") or "") if i < len(nb.cells) else ""
        body = normalize_cell_source_for_header_insertion(cur)
        rest = body.strip()
        new_src = header + ("\n" + rest + "\n" if (preserve_existing and rest) else "\n")
    
        pr = patch_cell_source_with_artifacts(
            notebook_path=str(nb_path),
            new_source=new_src,
            artifacts=art,
            backup_dir=backup_dir,
            backup_tag=f"proposal_{proposal_page_id}",
            diff_name_hint=f"scaffold_cell{i:03d}_{proposal_page_id}",
            mode="REPLACE",
            cell_index=i,
            cell_type="code",
        )
        if not pr.ok:
            update_task_item(store, task_item_id=task_item_id, status="FAILED", last_error=pr.error)
            return LoopResult(
                did_work=True,
                task_item_id=task_item_id,
                step_type="SCAFFOLD_HEADERS",
                ok=False,
                message=f"Scaffold failed at Cell {i}: {pr.error}",
                proposal_page_id=str(proposal_page_id),
            )
    
        patched += 1


    # ---- update state: scaffold digest + metadata ----
    def _apply_scaffold_state(st: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(st, dict):
            st = {}
        st["scaffold"] = {
            "status": "DONE",
            "digest": digest,
            "run_id": run_id,
            "artifacts_path": str(art.base_dir),
            "patched_cells": patched,
            "updated_at": _now_iso_jst(),
            "task_page_id": str(task_page_id),
            "proposal_page_id": str(proposal_page_id),
            "notebook_path": str(nb_path),
        }

        if cleanup_queue:
            q = st.get("queue") or []
            if isinstance(q, list):
                new_q = []
                for item in q:
                    if not isinstance(item, dict):
                        new_q.append(item)
                        continue
        
                    if item.get("task_item_id") == task_item_id:
                        new_q.append(item)
                        continue
        
                    t = (item.get("type") or "").upper()
                    tgt = item.get("target") or {}
                    if not isinstance(tgt, dict):
                        new_q.append(item)
                        continue
        
                    same = (str(tgt.get("task_page_id")) == str(task_page_id)) and (str(tgt.get("proposal_page_id")) == str(proposal_page_id))
                    is_todo = (item.get("status") or "TODO") == "TODO"
        
                    # ✅ Scaffold is allowed to cancel only duplicate scaffolds.
                    # Do NOT cancel LLM_PLAN here (we want LLM_PLAN to run after scaffolding).
                    if same and is_todo and t in ("SCAFFOLD_HEADERS",):
                        it2 = dict(item)
                        it2["status"] = "CANCELLED"
                        it2["last_error"] = "Superseded by scaffold (cleanup_queue)"
                        it2["updated_at"] = _now_iso_jst()
                        new_q.append(it2)
                        continue

        
                    new_q.append(item)
        
                st["queue"] = new_q
        return st

    _store_update_state(store, _apply_scaffold_state)

    # ---- mark DONE in queue item + metrics ----
    update_task_item(
        store,
        task_item_id=task_item_id,
        status="DONE",
        patch={
            "result": {
                "digest": digest,
                "patched_cells": patched,
                "run_id": run_id,
                "artifacts_path": str(art.base_dir),
                "notebook_path": str(nb_path),
            }
        },
    )
    bump_metric(store, "scaffold_headers_count", 1)
    mark_done(store, task_item_id=task_item_id)
    # ✅ DEBUG: even when debounced, print current structure so operator sees it first
    _debug_dump_structure_from_notebook(
        notebook_path=str(nb_path),
        artifacts_dir=str(prev.get("artifacts_path") or ""),
        prefix="[STRUCTURE][DEBOUNCED]",
    )
    # Bootstrap verify (Cell00-02まで通す)
    enqueue(
        store,
        new_task_item(
            type="VERIFY_NOTEBOOK",
            intent="Verify bootstrap cells (PREFIX up to Cell02).",
            assignee="SYSTEM",
            priority=90,
            target={
                "task_page_id": str(task_page_id),
                "proposal_page_id": str(proposal_page_id),
                "notebook_path": str(nb_path),
                "run_mode": "PREFIX",
                "up_to_cell_index": 2,
                "timeout_sec": 300,
                "quality_gates": target.get("quality_gates") or {},
                "auto_replan_on_fail": True,
            },
        ),
    )

    return LoopResult(
        did_work=True,
        task_item_id=task_item_id,
        step_type="SCAFFOLD_HEADERS",
        ok=True,
        message=f"Scaffolded headers (cells patched={patched}, digest={digest})",
        proposal_page_id=str(proposal_page_id),
    )



def _step_verify_notebook(
    *,
    store: StateStore,
    repos: NotionRepos,
    task_item_id: str,
    target: Dict[str, Any],
    run_prefix: str,
    task_id_fallback: str,
) -> LoopResult:
    """
    Verifier step (enhanced):
      - Execute notebook (prefix or full)
      - Create RUN page in RUNS_DB
      - On failure:
          - mark proposal failed
          - update state queue item failed
          - debounce + cleanup stale queue
          - enqueue LLM_PLAN with run_evidence (next_action/error_summary/cell_index)
      - On success:
          - optional quality gates
          - mark verified
    """
    task_page_id = target.get("task_page_id")
    proposal_page_id = target.get("proposal_page_id")
    notebook_path = target.get("notebook_path")
    nb_path = Path(str(notebook_path)).expanduser().resolve()
    if not nb_path.exists():
        return _fail(store, task_item_id, f"VERIFY_NOTEBOOK notebook not found: {nb_path}")

    if not task_page_id or not proposal_page_id or not notebook_path:
        return _fail(store, task_item_id, "VERIFY_NOTEBOOK missing task_page_id/proposal_page_id/notebook_path")

    run_mode = (target.get("run_mode") or "PREFIX").upper()
    timeout_sec = int(target.get("timeout_sec") or 300)
    run_id = next_run_id(prefix=run_prefix)
    art = ensure_run_dir(run_id=run_id, task_id=str(task_page_id) or task_id_fallback, notebook_path=str(nb_path))

    # 1) Notebook execution
    up_to_for_sig: Optional[int] = None
    if run_mode == "FULL":
        nb_res = execute_notebook_full(
            notebook_path=str(nb_path),
            artifacts=art,
            timeout_sec=timeout_sec,
        )
        run_type = "EXECUTE_FULL"
    else:
        # Accept multiple aliases (stable contract for callers/LLM):
        # - up_to_cell_index (preferred)
        # - up_to
        # - max_cell_index
        up_to = (
            target.get("up_to_cell_index")
            if target.get("up_to_cell_index") is not None
            else (target.get("up_to") if target.get("up_to") is not None else target.get("max_cell_index"))
        )
        if up_to is None:
            return _fail(
                store,
                task_item_id,
                "VERIFY_NOTEBOOK run_mode=PREFIX requires up_to_cell_index (or alias: up_to / max_cell_index)",
            )

        try:
            up_to_i = int(up_to)
        except Exception:
            return _fail(store, task_item_id, f"VERIFY_NOTEBOOK up_to_cell_index must be int; got={up_to!r}")

        if up_to_i < 0:
            return _fail(store, task_item_id, f"VERIFY_NOTEBOOK up_to_cell_index must be >=0; got={up_to_i}")

        up_to_for_sig = up_to_i
        nb_res = execute_notebook_prefix(
            notebook_path=str(nb_path),
            artifacts=art,
            up_to_cell_index=up_to_i,
            timeout_sec=timeout_sec,
        )
        run_type = "EXECUTE_PREFIX"



    write_run_reports(artifacts=art, result=nb_res)

    # 2) Create RUN record for notebook execution
    payload = run_payload_from_notebook_result(
        run_id=run_id,
        task_page_id=task_page_id,
        proposal_page_ids=[proposal_page_id],
        run_type=run_type,
        notebook_path=str(nb_path),
        nb_result=nb_res,
        artifacts_path=str(art.base_dir),
    )

    # --- HARD GUARD: Notion rich_text limit (commonly 2000 chars per text.content) ---
    def _trim(s: Any, n: int = 1500) -> str:
        t = "" if s is None else str(s)
        if len(t) <= n:
            return t
        return t[:n] + "…(truncated)"

    # Adjust keys to your RUNS_DB schema field names if needed
    for k in ("Error Trace (Short)", "Error Summary", "Stdout", "Stderr"):
        if k in payload and isinstance(payload.get(k), str):
            payload[k] = _trim(payload.get(k), 1900)


    import os, json
    
    def _dump_notion_exc(e: Exception, art_base: str, payload: dict, stage: str = "create_run"):
        p = Path(art_base) / "notion_error.json"
        data = {
            "stage": stage,
            "type": type(e).__name__,
            "str": str(e),
            "repr": repr(e),
            "payload_keys": list(payload.keys()),
            "payload_preview": {k: (str(v)[:500] if isinstance(v, str) else v) for k, v in list(payload.items())[:50]},
        }
    
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                data["status_code"] = getattr(resp, "status_code", None)
                data["text"] = getattr(resp, "text", None)
            except Exception:
                pass
            try:
                data["json"] = resp.json()
            except Exception:
                pass
    
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    
        # ✅ ターミナル出力（トリム）
        def _t(x, n=2000):
            s = "" if x is None else str(x)
            return s if len(s) <= n else s[:n] + "\n...(truncated)"
    
        print(f"[NOTION_RAW] stage={stage} type={type(e).__name__}")
        if data.get("status_code") is not None:
            print(f"[NOTION_RAW] status_code={data.get('status_code')}")
        if "json" in data:
            print("[NOTION_RAW] json:\n", _t(json.dumps(data["json"], ensure_ascii=False, indent=2), 4000))
        elif data.get("text"):
            print("[NOTION_RAW] text:\n", _t(data.get("text"), 4000))
        print("[NOTION_RAW] dumped:", str(p))
    
        return str(p)

    
    try:
        run_page = repos.runs.create_run(**payload)
        run_page_id = run_page["id"]
    
        repos.tasks.link_run(task_page_id=task_page_id, run_page_id=run_page_id, set_latest=True)
        repos.proposals.link_run(proposal_page_id=proposal_page_id, run_page_id=run_page_id, set_last_run=True)
    
    except Exception as e:
        p = _dump_notion_exc(e, str(art.base_dir), payload, stage="create_run_or_link")
        # ✅ Do not fail verification due to Notion logging.
        # Continue with nb_res and mark run_page_id as empty.
        run_page_id = ""
        print(f"[warn] Notion run logging failed (see {p}). Continuing without RUN page.")




    # --------------- FAIL: notebook execution ---------------
    if not nb_res.ok:
        bump_metric(store, "execute_fail_count", 1)

        # ✅ 1) Normalize raw nb result once
        nb_fields = _normalize_nb_fail_fields(nb_res)

        # ✅ 2) Ask error_parser once
        rep = suggest_next_action_from_notebook_result(nb_res)

        # ✅ 3) Derive stable fields
        rep_cell_index = nb_fields.get("failing_cell_index")
        err_sum = str(getattr(rep, "error_summary", "") or nb_fields.get("error_summary", "") or "")
        next_action = str(getattr(rep, "next_action", "") or "")
        category = str(getattr(rep, "category", "") or "UNKNOWN")

        # (optional) special-case config errors (keep your existing ones if you want)
        tb_str = str(nb_fields.get("traceback", "") or "")
        # for planner_target/run_evidence
        actionable_tb = tb_str
        # optional: avoid Notion / state bloat
        if len(actionable_tb) > 8000:
            actionable_tb = actionable_tb[:8000] + "\n…(truncated)"
        if "build_repos() missing" in tb_str and "required keyword-only argument" in tb_str:
            category = "CONFIG_ERROR"
            next_action = "Fix build_repos call signature: pass notion_client=... and resolved_registry=... (keyword-only required)."
        if "build_repos() takes 0 positional arguments but 1 was given" in tb_str:
            category = "CONFIG_ERROR"
            next_action = "Fix build_repos usage: do not pass positional args. Call build_repos() with keyword args only."

        # ✅ 4) Notion: mark proposal failed (best-effort; do not crash verify)
        try:
            repos.proposals.mark_failed(
                proposal_page_id=proposal_page_id,
                run_page_id=run_page_id,
                failure_reason=err_sum,
                next_action=next_action,
                artifacts_path=str(art.base_dir),
            )
        except Exception:
            pass

        # ✅ 5) Queue item bookkeeping
        update_task_item(
            store,
            task_item_id=task_item_id,
            status="FAILED",
            last_error=err_sum,
            patch={
                "result": {
                    "run_page_id": str(run_page_id),
                    "next_action": next_action,
                    "category": category,
                    "error_summary": err_sum,
                    "cell_index": rep_cell_index,
                    "artifacts_path": str(art.base_dir),
                    "extracted": getattr(rep, "extracted", {}) if rep is not None else {},
                    "hints": getattr(rep, "hints", []) if rep is not None else [],
                }
            },
        )

        # ✅ 6) Canonical last_error writer (single place)
        _write_last_error_from_verify(
            store=store,
            has_error=True,
            run_mode=str(run_mode),
            run_page_id=str(run_page_id or ""),
            artifacts_path=str(art.base_dir),
            nb_fields={**nb_fields, "error_summary": err_sum, "failing_cell_index": rep_cell_index},
            rep=rep,
            up_to_for_sig=up_to_for_sig,
        )

        # ---- keep your existing AUTO REPLAN / debounce logic below this line ----
        


        # --------- AUTO REPLAN (debounced + cleanup) ---------
        auto_replan = bool(target.get("auto_replan_on_fail", True))
        debounce_sec = int(target.get("replan_debounce_sec") or 120)
    
        failure_sig = _make_failure_signature(
            proposal_page_id=str(proposal_page_id),
            run_mode=str(run_mode),
            up_to_cell_index=up_to_for_sig,
            category=category,
            next_action=next_action,
            error_summary=err_sum,
            cell_index=rep_cell_index,
        )
    
        should = False
        if auto_replan:
            should = _debounce_should_replan(
                store=store,
                proposal_page_id=str(proposal_page_id),
                failure_sig=failure_sig,
                debounce_sec=debounce_sec,
                max_replans_per_sig=int(target.get("max_replans_per_sig") or 2),
            )
    
        if should:
            cancelled_n = _cleanup_queue_for_task_proposal(
                store=store,
                task_page_id=str(task_page_id),
                proposal_page_id=str(proposal_page_id),
                keep_task_item_id=None,
            )
            bump_metric(store, "queue_cancelled_by_replan", cancelled_n)

            llm_cfg = dict(target.get("llm") or {})
            planner_target = {
                "task_page_id": str(task_page_id),
                "proposal_page_id": str(proposal_page_id),
                "notebook_path": str(notebook_path),
                "run_evidence": {
                    "last_error": {
                        "category": category,
                        "next_action": next_action,
                        "error_summary": err_sum,
                        "cell_index": rep_cell_index,
                        "run_page_id": str(run_page_id),
                        "artifacts_path": str(art.base_dir),
                        "extracted": getattr(rep, "extracted", {}),
                        "traceback": actionable_tb,
                    }
                },
                "hint": {
                    "target_cell_index": rep_cell_index,
                    "run_mode": run_mode,
                    "up_to_cell_index": up_to_for_sig,
                },
                "llm": llm_cfg,
            }

            enqueue(
                store,
                new_task_item(
                    type="LLM_PLAN",
                    intent=f"Replan after VERIFY_NOTEBOOK failure ({category}): {next_action}",
                    assignee="PLANNER",
                    priority=int(target.get("replan_priority") or 95),
                    target=planner_target,
                ),
            )
            bump_metric(store, "llm_replan_enqueued", 1)
        else:
            # Debounced / capped: record that we're stopping auto-replans for this signature
            bump_metric(store, "llm_replan_suppressed", 1)
            try:
                repos.proposals.mark_failed(
                    proposal_page_id=proposal_page_id,
                    run_page_id=run_page_id,
                    failure_reason=err_sum,
                    next_action=f"[AUTO-REPLAN STOPPED] {next_action}",
                    artifacts_path=str(art.base_dir),
                )
            except Exception:
                pass

    
        return LoopResult(
            did_work=True,
            task_item_id=task_item_id,
            step_type="VERIFY_NOTEBOOK",
            ok=False,
            message=(
                f"Notebook execution failed: {category} / {next_action} "
                f"(cell={rep_cell_index}, executed_up_to={up_to_for_sig}, artifacts={art.base_dir})"
            ),
            run_page_id=run_page_id,
            proposal_page_id=proposal_page_id,
        )



    # --------------- PASS: notebook execution ---------------
    bump_metric(store, "execute_pass_count", 1)

    # ✅ Persist schema snapshot into state (best-effort)
    try:
        _save_schema_cache_from_artifacts(
            store=store,
            proposal_page_id=str(proposal_page_id),
            notebook_path=str(nb_path),
            artifacts_dir=str(art.base_dir),
        )
    except Exception:
        pass

    # ✅ advance executed_up_to first (PREFIX pass)
    if str(run_mode).upper() == "PREFIX" and up_to_for_sig is not None:
        try:
            _advance_executed_up_to_on_prefix_pass(
                store=store,
                proposal_page_id=str(proposal_page_id),
                notebook_path=str(nb_path),
                passed_up_to=int(up_to_for_sig),
            )
        except Exception:
            pass

    # ✅ after PREFIX pass bookkeeping (bootstrap verify pass only)
    if str(run_mode).upper() == "PREFIX" and up_to_for_sig is not None and int(up_to_for_sig) >= 2:
        try:
            _maybe_enqueue_post_bootstrap_plan_one_by_one(
                store=store,
                task_page_id=str(task_page_id),
                proposal_page_id=str(proposal_page_id),
                notebook_path=str(notebook_path),
            )
        except Exception:
            pass



    # ✅ D: Clear last_error *after* progress is advanced
    try:
        _write_last_error_from_verify(
            store=store,
            has_error=False,
            run_mode=str(run_mode),
            run_page_id=str(run_page_id or ""),
            artifacts_path=str(art.base_dir),
            nb_fields={},   # ← PASSでは空でOK（fail専用情報を渡さない）
            rep=None,
            up_to_for_sig=up_to_for_sig,
        )
    except Exception:
        pass


    # 3) Optional quality gates
    qg_cfg = dict(target.get("quality_gates") or {})


    # pytest
    if _enabled(qg_cfg, "pytest"):
        cfg = qg_cfg.get("pytest") or {}
        qg = run_pytest(
            artifacts=art,
            cwd=cfg.get("cwd"),
            args=cfg.get("args"),
            timeout_sec=cfg.get("timeout_sec", 900),
        )
        if not qg.ok:
            rep = suggest_next_action_from_quality_gate(qg)
            repos.proposals.mark_failed(
                proposal_page_id=proposal_page_id,
                run_page_id=run_page_id,
                failure_reason=rep.error_summary,
                next_action=rep.next_action,
                artifacts_path=str(art.base_dir),
            )
            update_task_item(
                store,
                task_item_id=task_item_id,
                status="FAILED",
                last_error=rep.error_summary,
                patch={"result": {"run_page_id": run_page_id, "next_action": rep.next_action, "category": rep.category}},
            )

            if bool(target.get("auto_replan_on_quality_fail", False)):
                failure_sig = _make_failure_signature(
                    proposal_page_id=str(proposal_page_id),
                    run_mode="PYTEST",
                    up_to_cell_index=None,
                    category=str(rep.category or "QUALITY_GATE"),
                    next_action=str(rep.next_action or ""),
                    error_summary=str(rep.error_summary or ""),
                    cell_index=None,
                )
                should = _debounce_should_replan(
                    store=store,
                    proposal_page_id=str(proposal_page_id),
                    failure_sig=failure_sig,
                    debounce_sec=int(target.get("replan_debounce_sec") or 120),
                    max_replans_per_sig=int(target.get("max_replans_per_sig") or 2),
                )
                if should:
                    _cleanup_queue_for_task_proposal(
                        store=store,
                        task_page_id=str(task_page_id),
                        proposal_page_id=str(proposal_page_id),
                    )
                    enqueue(
                        store,
                        new_task_item(
                            type="LLM_PLAN",
                            intent=f"Replan after pytest failure: {rep.next_action}",
                            assignee="PLANNER",
                            priority=int(target.get("replan_priority") or 95),
                            target={
                                "task_page_id": str(task_page_id),
                                "proposal_page_id": str(proposal_page_id),
                                "notebook_path": str(notebook_path),
                                "run_evidence": {
                                    "last_error": {
                                        "category": str(rep.category or "QUALITY_GATE"),
                                        "next_action": str(rep.next_action or ""),
                                        "error_summary": str(rep.error_summary or ""),
                                        "cell_index": None,
                                        "run_page_id": str(run_page_id),
                                        "artifacts_path": str(art.base_dir),
                                    }
                                },
                                "llm": dict(target.get("llm") or {}),
                            },
                        ),
                    )

            return LoopResult(
                did_work=True,
                task_item_id=task_item_id,
                step_type="VERIFY_NOTEBOOK",
                ok=False,
                message=f"pytest failed: {rep.next_action}",
                run_page_id=run_page_id,
                proposal_page_id=proposal_page_id,
            )


    
    # ruff
    if _enabled(qg_cfg, "ruff"):
        cfg = qg_cfg.get("ruff") or {}
        qg = run_ruff(
            artifacts=art,
            cwd=cfg.get("cwd"),
            args=cfg.get("args") or ["check", "."],
            timeout_sec=cfg.get("timeout_sec", 300),
        )
        if not qg.ok:
            rep = suggest_next_action_from_quality_gate(qg)
            repos.proposals.mark_failed(
                proposal_page_id=proposal_page_id,
                run_page_id=run_page_id,
                failure_reason=rep.error_summary,
                next_action=rep.next_action,
                artifacts_path=str(art.base_dir),
            )
            update_task_item(
                store,
                task_item_id=task_item_id,
                status="FAILED",
                last_error=rep.error_summary,
                patch={"result": {"run_page_id": run_page_id, "next_action": rep.next_action, "category": rep.category}},
            )

            if bool(target.get("auto_replan_on_quality_fail", False)):
                failure_sig = _make_failure_signature(
                    proposal_page_id=str(proposal_page_id),
                    run_mode="RUFF",
                    up_to_cell_index=None,
                    category=str(rep.category or "QUALITY_GATE"),
                    next_action=str(rep.next_action or ""),
                    error_summary=str(rep.error_summary or ""),
                    cell_index=None,
                )
                should = _debounce_should_replan(
                    store=store,
                    proposal_page_id=str(proposal_page_id),
                    failure_sig=failure_sig,
                    debounce_sec=int(target.get("replan_debounce_sec") or 120),
                    max_replans_per_sig=int(target.get("max_replans_per_sig") or 2),
                )
                if should:
                    _cleanup_queue_for_task_proposal(
                        store=store,
                        task_page_id=str(task_page_id),
                        proposal_page_id=str(proposal_page_id),
                    )
                    enqueue(
                        store,
                        new_task_item(
                            type="LLM_PLAN",
                            intent=f"Replan after ruff failure: {rep.next_action}",
                            assignee="PLANNER",
                            priority=int(target.get("replan_priority") or 95),
                            target={
                                "task_page_id": str(task_page_id),
                                "proposal_page_id": str(proposal_page_id),
                                "notebook_path": str(notebook_path),
                                "run_evidence": {
                                    "last_error": {
                                        "category": str(rep.category or "QUALITY_GATE"),
                                        "next_action": str(rep.next_action or ""),
                                        "error_summary": str(rep.error_summary or ""),
                                        "cell_index": None,
                                        "run_page_id": str(run_page_id),
                                        "artifacts_path": str(art.base_dir),
                                    }
                                },
                                "llm": dict(target.get("llm") or {}),
                            },
                        ),
                    )


            return LoopResult(
                did_work=True,
                task_item_id=task_item_id,
                step_type="VERIFY_NOTEBOOK",
                ok=False,
                message=f"ruff failed: {rep.next_action}",
                run_page_id=run_page_id,
                proposal_page_id=proposal_page_id,
            )

    # ✅ quality gates も含めて全部PASSした後に掃除
    cancelled_n = _cleanup_queue_for_task_proposal(
        store=store,
        task_page_id=str(task_page_id),
        proposal_page_id=str(proposal_page_id),
        keep_task_item_id=task_item_id,
        cancel_types=["VERIFY_NOTEBOOK"],  # ← VERIFY だけ掃除
    )
    bump_metric(store, "queue_cancelled_after_verify_pass", cancelled_n)

    # 4) Mark verified
    repos.proposals.mark_verified(
        proposal_page_id=proposal_page_id,
        run_page_id=run_page_id,
        artifacts_path=str(art.base_dir),
        next_action="Verified (notebook + quality gates passed).",
    )

    update_task_item(
        store,
        task_item_id=task_item_id,
        status="DONE",
        patch={
            "result": {
                "run_page_id": str(run_page_id),
                "artifacts_path": str(art.base_dir),
                "run_mode": str(run_mode),
                "up_to_cell_index": up_to_for_sig,
            }
        },
    )
    mark_done(store, task_item_id=task_item_id)

    return LoopResult(
        did_work=True,
        task_item_id=task_item_id,
        step_type="VERIFY_NOTEBOOK",
        ok=True,
        message=f"Verified proposal {proposal_page_id} (run={run_page_id})",
        run_page_id=run_page_id,
        proposal_page_id=proposal_page_id,
    )



# NOTE:
# _step_apply_patch is assumed to exist in your original file.
# Keep your existing _step_apply_patch implementation below unchanged,
# or paste it here if you want this file to be fully self-contained.
