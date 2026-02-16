# src/exec/nb_runner.py
"""
Notebook execution engine (Verifier "body").

This module executes a Jupyter notebook (prefix or full) via nbclient, captures
stdout/stderr, and writes artifacts into the standardized outputs layout.

Key goals
---------
- Deterministic execution (prefix / full)
- Good failure diagnostics: failing cell index + short error summary
- Always writes artifacts (manifest already handled by layout.ensure_run_dir)
- Safe defaults: errors stop execution (allow_errors=False)

Artifacts written (typical)
---------------------------
<artifacts_dir>/
  notebooks/input.ipynb
  notebooks/output.ipynb
  logs/nbclient.log
  logs/stdout.log
  logs/stderr.log
  reports/summary.md (optional; caller)
  reports/failure.md (optional; caller)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, Tuple
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import traceback
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from src.artifacts.layout import RunArtifacts, write_text, append_text


# -----------------------------
# Result types
# -----------------------------

@dataclass
class NotebookRunResult:
    ok: bool
    run_type: str  # EXECUTE_PREFIX | EXECUTE_FULL
    notebook_path: str
    executed_up_to_cell: Optional[int]  # inclusive index if prefix, else last cell index
    failing_cell_index: Optional[int]
    error_summary: str
    error_trace_short: str
    error_traceback: str
    started_at_unix: float
    finished_at_unix: float
    duration_sec: float
    # ---- Evidence tails (for LLM + human) ----
    stdout_tail: str = ""
    stderr_tail: str = ""
    exception_type: str = ""

    # ---- Schema artifacts (Cell02 outputs if present) ----
    schema_snapshot_path: str = ""
    schema_report_path: str = ""
    db_prop_types_path: str = ""

    def to_evidence_text(self) -> str:
        if self.ok:
            return f"{self.run_type}: OK (duration={self.duration_sec:.2f}s)"
        return (
            f"{self.run_type}: FAIL (cell={self.failing_cell_index}, "
            f"duration={self.duration_sec:.2f}s) - {self.error_summary}"
        )


# -----------------------------
# Public API
# -----------------------------

def execute_notebook_full(
    *,
    notebook_path: str | Path,
    artifacts: RunArtifacts,
    kernel_name: str = "python3",
    timeout_sec: int = 300,
    working_dir: Optional[str | Path] = None,
    allow_errors: bool = False,
) -> NotebookRunResult:
    return _execute(
        notebook_path=notebook_path,
        artifacts=artifacts,
        run_type="EXECUTE_FULL",
        prefix_to_cell=None,
        kernel_name=kernel_name,
        timeout_sec=timeout_sec,
        working_dir=working_dir,
        allow_errors=allow_errors,
    )


def execute_notebook_prefix(
    *,
    notebook_path: str | Path,
    artifacts: RunArtifacts,
    up_to_cell_index: int,
    kernel_name: str = "python3",
    timeout_sec: int = 300,
    working_dir: Optional[str | Path] = None,
    allow_errors: bool = False,
    min_prefix_cell_index: Optional[int] = None,
) -> NotebookRunResult:
    if up_to_cell_index < 0:
        raise ValueError("up_to_cell_index must be >= 0")

    return _execute(
        notebook_path=notebook_path,
        artifacts=artifacts,
        run_type="EXECUTE_PREFIX",
        prefix_to_cell=up_to_cell_index,
        kernel_name=kernel_name,
        timeout_sec=timeout_sec,
        working_dir=working_dir,
        allow_errors=allow_errors,
        min_prefix_cell_index=min_prefix_cell_index,
    )



# -----------------------------
# Internals
# -----------------------------

def _execute(
    *,
    notebook_path: str | Path,
    artifacts: RunArtifacts,
    run_type: str,
    prefix_to_cell: Optional[int],
    kernel_name: str,
    timeout_sec: int,
    working_dir: Optional[str | Path],
    allow_errors: bool,
    min_prefix_cell_index: Optional[int] = None,
) -> NotebookRunResult:
    artifacts.ensure_dirs()

    nb_path = Path(notebook_path).expanduser().resolve()
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    # Always-initialized fields (avoid UnboundLocalError)
    exception_type = ""
    ok = True
    failing_cell_index: Optional[int] = None
    error_summary = ""
    error_trace_short = ""
    error_traceback = ""
    stdout_tail = ""
    stderr_tail = ""
    started = time.time()
    finished = started
    duration = 0.0
    executed_up_to: Optional[int] = None

    # Load notebook
    nb = nbformat.read(str(nb_path), as_version=4)

    # Save input notebook snapshot
    nbformat.write(nb, str(artifacts.input_notebook_path))

    original_cells = list(nb.cells)

    # ---- Edge case: empty notebook ----
    if len(original_cells) == 0:
        append_text(
            artifacts.nbclient_log,
            f"\n[{_now_human()}] START {run_type} notebook={nb_path} "
            f"kernel={kernel_name} timeout={timeout_sec}s prefix_to=None (empty notebook)\n",
        )

        # output snapshot == input snapshot (empty)
        nb_out = nbformat.v4.new_notebook(metadata=nb.metadata)
        nb_out.cells = []
        nbformat.write(nb_out, str(artifacts.output_notebook_path))

        append_text(
            artifacts.nbclient_log,
            f"[{_now_human()}] END {run_type} ok=True duration=0.00s (empty notebook)\n",
        )

        # schema artifacts (likely absent, but keep consistent)
        reports_dir = getattr(artifacts, "reports_dir", None)
        def _exists(p: Optional[Path]) -> str:
            try:
                return str(p) if p and p.exists() else ""
            except Exception:
                return ""

        schema_snapshot_path = _exists(reports_dir / "schema_snapshot.json") if reports_dir else ""
        schema_report_path = _exists(reports_dir / "schema_report.md") if reports_dir else ""
        db_prop_types_path = _exists(reports_dir / "db_prop_types.json") if reports_dir else ""

        return NotebookRunResult(
            ok=True,
            run_type=run_type,
            notebook_path=str(nb_path),
            executed_up_to_cell=None,
            failing_cell_index=None,
            error_summary="",
            error_trace_short="",
            error_traceback="",
            started_at_unix=started,
            finished_at_unix=finished,
            duration_sec=0.0,
            stdout_tail="",
            stderr_tail="",
            exception_type="",
            schema_snapshot_path=schema_snapshot_path,
            schema_report_path=schema_report_path,
            db_prop_types_path=db_prop_types_path,
        )

    # ---- Normalize prefix_to_cell ----
    normalized_prefix: Optional[int]
    if prefix_to_cell is None:
        normalized_prefix = None
    else:
        req = int(prefix_to_cell)
        if min_prefix_cell_index is not None:
            try:
                req = max(req, int(min_prefix_cell_index))
            except Exception:
                pass
        # clamp into [0, len-1]
        normalized_prefix = max(0, min(int(req), len(original_cells) - 1))

    # Prefix slicing strategy
    if normalized_prefix is None:
        executed_cells = list(original_cells)
        remaining_cells = []
        executed_up_to = len(executed_cells) - 1 if executed_cells else None
    else:
        executed_cells = list(original_cells[: normalized_prefix + 1])
        remaining_cells = list(original_cells[normalized_prefix + 1 :])
        executed_up_to = normalized_prefix if executed_cells else None

    # Create notebook object for execution
    nb_exec = nbformat.v4.new_notebook(metadata=nb.metadata)
    nb_exec.cells = executed_cells

    append_text(
        artifacts.nbclient_log,
        f"\n[{_now_human()}] START {run_type} notebook={nb_path} "
        f"kernel={kernel_name} timeout={timeout_sec}s prefix_to={normalized_prefix}\n",
    )

    # Capture python-level stdout/stderr
    stdout_buf = StringIO()
    stderr_buf = StringIO()

    started = time.time()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            client = NotebookClient(
                nb_exec,
                kernel_name=kernel_name,
                timeout=timeout_sec,
                allow_errors=allow_errors,
                resources=_resources_for_workdir(working_dir),
            )
            client.execute()

    except CellExecutionError as e:
        ok = False
        exception_type = "CellExecutionError"

        failing_cell_index = _infer_failing_cell_index(nb_exec, e)
        error_summary, error_trace_short = _summarize_cell_error(e)
        error_traceback = _trim_lines(traceback.format_exc(), max_lines=120, max_chars=12000)

        # PREFIX時に cell index が取れないケースの保険
        if failing_cell_index is None:
            failing_cell_index = executed_up_to

        append_text(
            artifacts.nbclient_log,
            f"[{_now_human()}] FAIL CellExecutionError cell={failing_cell_index} summary={error_summary}\n",
        )

    except Exception as e:
        ok = False
        exception_type = type(e).__name__

        error_summary = f"{type(e).__name__}: {str(e)[:300]}"
        error_trace_short = _short_traceback()
        error_traceback = _trim_lines(traceback.format_exc(), max_lines=120, max_chars=12000)

        append_text(
            artifacts.nbclient_log,
            f"[{_now_human()}] FAIL Exception type={exception_type} summary={error_summary}\n",
        )

    finally:
        finished = time.time()
        duration = finished - started

        # Write captured stdout/stderr (append to preserve history)
        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()
        append_text(artifacts.stdout_log, stdout_val)
        append_text(artifacts.stderr_log, stderr_val)

        append_text(
            artifacts.nbclient_log,
            f"[{_now_human()}] END {run_type} ok={ok} duration={duration:.2f}s\n",
        )

        # ---- tails for evidence (keep small & stable) ----
        def _tail(s: str, max_chars: int = 2000) -> str:
            t = s or ""
            return t if len(t) <= max_chars else t[-max_chars:]

        stdout_tail = _tail(stdout_val, 2000)
        stderr_tail = _tail(stderr_val, 2000)

    # Re-attach remaining cells (untouched) for output notebook snapshot
    nb_out = nbformat.v4.new_notebook(metadata=nb.metadata)
    nb_out.cells = list(nb_exec.cells) + remaining_cells
    nbformat.write(nb_out, str(artifacts.output_notebook_path))

    # ---- Schema artifacts (written by fixed Cell02 if present) ----
    reports_dir = getattr(artifacts, "reports_dir", None)

    def _exists(p: Optional[Path]) -> str:
        try:
            return str(p) if p and p.exists() else ""
        except Exception:
            return ""

    schema_snapshot_path = _exists(reports_dir / "schema_snapshot.json") if reports_dir else ""
    schema_report_path = _exists(reports_dir / "schema_report.md") if reports_dir else ""
    db_prop_types_path = _exists(reports_dir / "db_prop_types.json") if reports_dir else ""

    return NotebookRunResult(
        ok=ok,
        run_type=run_type,
        notebook_path=str(nb_path),
        executed_up_to_cell=executed_up_to,
        failing_cell_index=failing_cell_index,
        error_summary=error_summary,
        error_trace_short=error_trace_short,
        error_traceback=error_traceback if not ok else "",
        started_at_unix=started,
        finished_at_unix=finished,
        duration_sec=duration,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        exception_type=exception_type,
        schema_snapshot_path=schema_snapshot_path,
        schema_report_path=schema_report_path,
        db_prop_types_path=db_prop_types_path,
    )



def _resources_for_workdir(working_dir: Optional[str | Path]) -> Dict[str, Any]:
    if working_dir is None:
        return {}
    wd = str(Path(working_dir).expanduser().resolve())
    # nbclient uses resources['metadata']['path'] as the execution cwd
    return {"metadata": {"path": wd}}


def _infer_failing_cell_index(nb_exec, exc: CellExecutionError) -> Optional[int]:
    """
    Try to infer the failing cell index from nbclient's exception.

    nbclient's CellExecutionError often has a `.cell` attribute (the cell object).
    Identity comparison can fail if cells are copied internally, so we try id match first.
    """
    cell = getattr(exc, "cell", None)
    if cell is None:
        return None

    try:
        cell_id = None
        try:
            cell_id = cell.get("id")  # type: ignore[assignment]
        except Exception:
            cell_id = None

        if cell_id:
            for i, c in enumerate(nb_exec.cells):
                try:
                    if c.get("id") == cell_id:
                        return i
                except Exception:
                    continue

        # fallback: identity match (best-effort)
        for i, c in enumerate(nb_exec.cells):
            if c is cell:
                return i
    except Exception:
        return None

    return None


def _summarize_cell_error(exc: CellExecutionError) -> Tuple[str, str]:
    """
    Produce:
      - error_summary: short single-line string
      - error_trace_short: trimmed traceback-ish text
    """
    msg = str(exc)
    summary = _first_nonempty_line(msg)[:300]
    trace = _trim_lines(msg, max_lines=40, max_chars=4000)
    return summary, trace


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _trim_lines(text: str, *, max_lines: int, max_chars: int) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... (trimmed)"]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n... (trimmed by chars)"
    return out


def _short_traceback() -> str:
    tb = traceback.format_exc()
    return _trim_lines(tb, max_lines=60, max_chars=6000)


def _now_human() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


# -----------------------------
# Optional convenience: write simple reports
# -----------------------------

def write_run_reports(
    *,
    artifacts: RunArtifacts,
    result: NotebookRunResult,
    extra_notes: str = "",
) -> None:
    """
    Write minimal human-readable reports into artifacts/reports/.
    """
    if result.ok:
        body = (
            f"# Run Summary\n\n"
            f"- status: PASS\n"
            f"- run_type: {result.run_type}\n"
            f"- notebook: {result.notebook_path}\n"
            f"- duration_sec: {result.duration_sec:.2f}\n"
            f"- executed_up_to_cell: {result.executed_up_to_cell}\n\n"
        )
        if extra_notes:
            body += f"## Notes\n\n{extra_notes}\n"
        write_text(artifacts.summary_report_path, body)
    else:
        body = (
            f"# Run Failure\n\n"
            f"- status: FAIL\n"
            f"- run_type: {result.run_type}\n"
            f"- notebook: {result.notebook_path}\n"
            f"- duration_sec: {result.duration_sec:.2f}\n"
            f"- failing_cell_index: {result.failing_cell_index}\n\n"
            f"## Error Summary\n\n{result.error_summary}\n\n"
            f"## Error Trace (Short)\n\n```\n{result.error_trace_short}\n```\n\n"
            f"## Error Traceback\n\n```\n{result.error_traceback}\n```\n"
        )
        if extra_notes:
            body += f"\n## Notes\n\n{extra_notes}\n"
        write_text(artifacts.failure_report_path, body)
