# src/state/state_store.py
"""
state.json store (single-writer safe-ish) + queue helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Literal, Tuple
import json
import os
import time
import uuid


# -----------------------------
# Exceptions
# -----------------------------

class StateStoreError(RuntimeError):
    pass


class StateLockTimeout(StateStoreError):
    pass


# -----------------------------
# Defaults / constants
# -----------------------------

DEFAULT_SCHEMA_VERSION = "1.0"

TaskItemStatus = Literal["TODO", "DOING", "WAITING", "DONE", "FAILED", "SKIPPED", "CANCELLED"]
TaskItemAssignee = Literal["PLANNER", "IMPLEMENTER", "VERIFIER", "SYSTEM"]

DEFAULT_LOCK_TIMEOUT_SEC = 10.0
DEFAULT_LOCK_POLL_SEC = 0.05
DEFAULT_LOCK_STALE_SEC = 120.0  # stale lock cleanup window


def now_iso_local() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# -----------------------------
# Helpers
# -----------------------------

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, obj: Any) -> None:
    """
    Atomic-ish write:
      - write to unique temp file
      - flush + fsync
      - replace into place
    """
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    data = json.dumps(obj, ensure_ascii=False, indent=2)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    tmp.replace(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise StateStoreError(f"Failed to read/parse state json: {path}: {e}") from e


def _merge_default_state(st: Optional[dict]) -> dict:
    """
    Ensure core keys exist even if file is partially empty/corrupt-ish.
    """
    base = default_state()
    if isinstance(st, dict):
        # shallow merge top-level; sub-objects are replaced as-is (intentional)
        base.update(st)
    return base


def _shallow_deep_merge(dst: dict, src: dict) -> dict:
    """
    Merge up to 2 levels deep for dict values.
    - If both dst[k] and src[k] are dict -> update (and if values are dict too, update once more)
    - Else overwrite
    """
    out = dict(dst)
    for k, v in (src or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            lv1 = dict(out[k])
            for k2, v2 in v.items():
                if isinstance(lv1.get(k2), dict) and isinstance(v2, dict):
                    lv2 = dict(lv1[k2])
                    lv2.update(v2)
                    lv1[k2] = lv2
                else:
                    lv1[k2] = v2
            out[k] = lv1
        else:
            out[k] = v
    return out



# -----------------------------
# Lock (best-effort)
# -----------------------------

@dataclass
class FileLock:
    lock_path: Path
    timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC
    poll_sec: float = DEFAULT_LOCK_POLL_SEC
    stale_sec: float = DEFAULT_LOCK_STALE_SEC

    def _is_stale(self) -> bool:
        try:
            if not self.lock_path.exists():
                return False
            age = time.time() - self.lock_path.stat().st_mtime
            return age >= float(self.stale_sec)
        except Exception:
            return False

    def _break_stale_lock(self) -> None:
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass

    def acquire(self) -> None:
        _ensure_parent(self.lock_path)
        start = time.time()
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"pid={os.getpid()} at={now_iso_local()}\n")
                    f.flush()
                    os.fsync(f.fileno())
                return

            except FileExistsError:
                # stale lock cleanup
                if self._is_stale():
                    self._break_stale_lock()
                    continue

                if (time.time() - start) >= self.timeout_sec:
                    raise StateLockTimeout(f"Timed out acquiring lock: {self.lock_path}")
                time.sleep(self.poll_sec)

    def release(self) -> None:
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


# -----------------------------
# State Store
# -----------------------------

@dataclass
@dataclass
class StateStore:
    path: Path = Path("state/state.json")
    lock_path: Path = Path("state/state.lock")
    lock_timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC
    lock_stale_sec: float = DEFAULT_LOCK_STALE_SEC

    def load(self) -> dict:
        return _merge_default_state(_read_json(self.path))

    # -------------------------
    # Internal: assumes lock already held
    # -------------------------
    def _save_unlocked(self, state: dict) -> None:
        _atomic_write_json(self.path, state)

    def save(self, state: dict) -> None:
        """
        Public save (safe): acquires file lock.
        """
        lock = FileLock(
            self.lock_path,
            timeout_sec=self.lock_timeout_sec,
            stale_sec=self.lock_stale_sec,
        )
        with lock:
            self._save_unlocked(state)

    def ensure_initialized(self, *, template: Optional[dict] = None) -> dict:
        if self.path.exists():
            return self.load()

        st = template if template is not None else default_state()
        self.save(_merge_default_state(st))
        return self.load()

    def update(self, fn: Callable[[dict], dict]) -> dict:
        """
        Transactional update:
          - acquire lock once
          - load -> apply fn -> merge defaults -> save WITHOUT re-locking
        """
        lock = FileLock(
            self.lock_path,
            timeout_sec=self.lock_timeout_sec,
            stale_sec=self.lock_stale_sec,
        )
        with lock:
            st = self.load()
            st2 = fn(st) or {}
            st2 = _merge_default_state(st2)
            self._save_unlocked(st2)   # ✅ critical: avoid nested lock
            return st2



# -----------------------------
# Default state
# -----------------------------

def default_state() -> dict:
    return {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "task": {},
        "run": {},
        "queue": [],
        "memory_refs": {
            "project_charter_path": "memory/project_charter.md",
            "decisions_log_path": "memory/decisions.log",
            "worklog_path": "memory/worklog.md",
        },
        "last_error": {
            "has_error": False,
            "error_summary": "",
            "error_trace_path": "",
        },

        "metrics": {
            "patch_count": 0,
            "execute_pass_count": 0,
            "execute_fail_count": 0,
        },

        # NEW: iterative build progress (used by one_loop / scaffold lock)
        "build": {
            "scaffold_locked": False,   # once scaffold succeeds, set True
            "current_cell": 1,          # next cell to implement/repair
            "max_cell": 0,              # set from plan.structure max cell_index
            "attempts": {},             # e.g. {"1": 2, "2": 1}
        },
    }

def set_build_state(store: StateStore, patch: dict) -> dict:
    """
    Shallow-merge patch into st["build"].
    """
    def _fn(st: dict) -> dict:
        st = _merge_default_state(st)
        cur = dict(st.get("build", {}))
        cur.update(patch or {})
        st["build"] = cur
        return st
    return store.update(_fn)


# -----------------------------
# Queue item helpers
# -----------------------------

def new_task_item(
    *,
    type: str,
    intent: str,
    assignee: TaskItemAssignee,
    priority: int = 10,
    target: Optional[dict] = None,
    acceptance: Optional[List[str]] = None,
    max_attempts: int = 3,
) -> dict:
    return {
        "task_item_id": f"t_{uuid.uuid4().hex[:10]}",
        "type": type,
        "priority": int(priority),
        "assignee": assignee,
        "target": target or {},
        "intent": intent,
        "acceptance": acceptance or [],
        "status": "TODO",
        "attempts": 0,
        "max_attempts": int(max_attempts),
        "created_at": now_iso_local(),
        "updated_at": now_iso_local(),
        "last_error": "",
    }


def sort_queue(queue: List[dict]) -> List[dict]:
    status_rank = {
        "TODO": 0,
        "WAITING": 1,
        "DOING": 2,
        "DONE": 3,
        "SKIPPED": 4,
        "FAILED": 5,
        "CANCELLED": 6,
    }

    def key(it: dict) -> Tuple[int, int, str]:
        sr = status_rank.get(it.get("status", "TODO"), 9)
        pr = int(it.get("priority", 0))
        ca = str(it.get("created_at", ""))
        return (sr, -pr, ca)

    return sorted(queue, key=key)


def enqueue(store: StateStore, item: dict) -> dict:
    def _fn(st: dict) -> dict:
        st = _merge_default_state(st)
        q = list(st.get("queue", []))
        q.append(item)
        st["queue"] = sort_queue(q)
        return st

    return store.update(_fn)


def enqueue_many(store: StateStore, items: List[dict]) -> dict:
    def _fn(st: dict) -> dict:
        st = _merge_default_state(st)
        q = list(st.get("queue", []))
        q.extend(items)
        st["queue"] = sort_queue(q)
        return st

    return store.update(_fn)


def find_task_item(st: dict, task_item_id: str) -> Optional[dict]:
    for it in st.get("queue", []):
        if it.get("task_item_id") == task_item_id:
            return it
    return None


def update_task_item(
    store: StateStore,
    *,
    task_item_id: str,
    status: Optional[TaskItemStatus] = None,
    bump_attempts: bool = False,
    last_error: Optional[str] = None,
    patch: Optional[dict] = None,
) -> dict:
    def _fn(st: dict) -> dict:
        st = _merge_default_state(st)
        q = list(st.get("queue", []))
        updated = []
        for it in q:
            if it.get("task_item_id") != task_item_id:
                updated.append(it)
                continue

            it2 = dict(it)
            if status is not None:
                it2["status"] = status
            if bump_attempts:
                it2["attempts"] = int(it2.get("attempts", 0)) + 1
            if last_error is not None:
                it2["last_error"] = last_error
            if patch:
                it2 = _shallow_deep_merge(it2, patch)

            it2["updated_at"] = now_iso_local()
            updated.append(it2)

        st["queue"] = sort_queue(updated)
        return st

    return store.update(_fn)


def pop_next_todo(store: StateStore, *, assignee: Optional[TaskItemAssignee] = None) -> Optional[dict]:
    """
    Select the next TODO item (optionally filtered by assignee),
    mark it DOING, bump attempts, and return it (does not remove it).
    """
    selected: Optional[dict] = None

    def _fn(st: dict) -> dict:
        nonlocal selected
        st = _merge_default_state(st)
        q = list(st.get("queue", []))
        out = []
        for it in q:
            if selected is None and it.get("status") == "TODO":
                if assignee is None or it.get("assignee") == assignee:
                    it2 = dict(it)
                    it2["status"] = "DOING"
                    it2["attempts"] = int(it2.get("attempts", 0)) + 1
                    it2["updated_at"] = now_iso_local()
                    selected = it2
                    out.append(it2)
                    continue
            out.append(it)
        st["queue"] = sort_queue(out)
        return st

    store.update(_fn)
    return selected


def mark_done(store: StateStore, *, task_item_id: str) -> dict:
    return update_task_item(store, task_item_id=task_item_id, status="DONE")


def mark_failed(store: StateStore, *, task_item_id: str, reason: str) -> dict:
    return update_task_item(store, task_item_id=task_item_id, status="FAILED", last_error=reason)


def auto_fail_if_exhausted(store: StateStore, *, task_item_id: str) -> dict:
    def _fn(st: dict) -> dict:
        st = _merge_default_state(st)
        q = list(st.get("queue", []))
        out = []
        for it in q:
            if it.get("task_item_id") != task_item_id:
                out.append(it)
                continue
            it2 = dict(it)
            attempts = int(it2.get("attempts", 0))
            max_attempts = int(it2.get("max_attempts", 3))
            if attempts >= max_attempts and it2.get("status") in ("TODO", "DOING", "WAITING"):
                it2["status"] = "FAILED"
                it2["last_error"] = it2.get("last_error", "") or "max_attempts reached"
            it2["updated_at"] = now_iso_local()
            out.append(it2)
        st["queue"] = sort_queue(out)
        return st

    return store.update(_fn)


# -----------------------------
# Run/task level updates
# -----------------------------

def set_current_task(store: StateStore, task_obj: dict) -> dict:
    def _fn(st: dict) -> dict:
        st = _merge_default_state(st)
        st["task"] = task_obj
        return st

    return store.update(_fn)


def set_run_info(store: StateStore, run_obj: dict) -> dict:
    def _fn(st: dict) -> dict:
        st = _merge_default_state(st)
        st["run"] = run_obj
        return st

    return store.update(_fn)


def set_last_error(
    store: StateStore,
    *,
    has_error: bool,
    failing_cell_index: Optional[int],
    error_summary: str,
    error_trace_path: str = "",
    executed_up_to: Optional[int] = None,
    **extra,  # ✅ これで executed_up_to などが来ても落ちない
) -> dict:
    def _fn(st: dict) -> dict:
        st = _merge_default_state(st)
        base = {
            "has_error": bool(has_error),
            "failing_cell_index": failing_cell_index,
            "error_summary": error_summary or "",
            "error_trace_path": error_trace_path or "",
            "executed_up_to": executed_up_to, 
            "updated_at": now_iso_local(),
        }
        # extra をそのまま載せる（必要なら allowlist でもOK）
        base.update({k: v for k, v in extra.items() if v is not None})
        st["last_error"] = base
        return st
    return store.update(_fn)



def bump_metric(store: StateStore, key: str, delta: int = 1) -> dict:
    def _fn(st: dict) -> dict:
        st = _merge_default_state(st)
        metrics = dict(st.get("metrics", {}))
        metrics[key] = int(metrics.get(key, 0)) + int(delta)
        st["metrics"] = metrics
        return st

    return store.update(_fn)
