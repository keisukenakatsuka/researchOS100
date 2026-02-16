# src/artifacts/layout.py
"""
Artifacts layout utilities.

Goal
----
Standardize where every run writes its outputs so that:
- Verifier can always attach "Artifacts Path" to RUNS_DB
- Humans can quickly inspect what happened
- Tools can reliably find logs/diffs/reports across notebooks

Directory convention
--------------------
outputs/
  runs/
    <run_id>/
      meta/
        manifest.json
        timings.json
      logs/
        stdout.log
        stderr.log
        nbclient.log
        quality_gate.log
      diffs/
        patches/
          0001_cell_05.diff
        git/
          status.txt
          diff.patch
      notebooks/
        input.ipynb
        output.ipynb
      reports/
        summary.md
        failure.md
      data/
        *.json/*.csv (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Optional
from datetime import datetime, timezone
import json
import os


# -----------------------------
# Core layout object
# -----------------------------

@dataclass(frozen=True)
class RunArtifacts:
    """
    A resolved layout for a single run_id.

    Use .ensure_dirs() once, then write files into the returned paths.
    """
    run_id: str
    base_dir: Path  # e.g. outputs/runs/<run_id>

    # Top-level buckets
    meta_dir: Path
    logs_dir: Path
    diffs_dir: Path
    notebooks_dir: Path
    reports_dir: Path
    data_dir: Path

    # Common files
    manifest_path: Path
    timings_path: Path

    stdout_log: Path
    stderr_log: Path
    nbclient_log: Path
    quality_gate_log: Path

    git_status_path: Path
    git_diff_path: Path

    input_notebook_path: Path
    output_notebook_path: Path

    summary_report_path: Path
    failure_report_path: Path

    patches_dir: Path

    def ensure_dirs(self) -> None:
        """Create the directory structure if missing."""
        for d in [
            self.base_dir,
            self.meta_dir,
            self.logs_dir,
            self.diffs_dir,
            self.notebooks_dir,
            self.reports_dir,
            self.data_dir,
            self.patches_dir,
            self.diffs_dir / "git",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (paths become strings)."""
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def write_manifest(
        self,
        *,
        task_id: Optional[str] = None,
        notebook_path: Optional[str] = None,
        git: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Write a manifest.json capturing run metadata and layout paths.

        This is meant to be the canonical "what happened" anchor file.
        """
        self.ensure_dirs()
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "created_at": now_iso(),
            "task_id": task_id,
            "notebook_path": notebook_path,
            "git": dict(git) if git else {},
            "layout": self.to_dict(),
        }
        if extra:
            payload["extra"] = dict(extra)

        write_json(self.manifest_path, payload)

    def write_timings(self, timings: Mapping[str, Any]) -> None:
        """
        Write timings.json (e.g., {"execute_full_sec": 12.3, ...}).
        """
        self.ensure_dirs()
        write_json(self.timings_path, dict(timings))


# -----------------------------
# Public API
# -----------------------------

def resolve_run_artifacts(
    *,
    run_id: str,
    outputs_root: str | Path = "outputs",
) -> RunArtifacts:
    """
    Resolve all artifacts paths for a given run_id.

    Parameters
    ----------
    run_id:
        Unique run identifier (recommend: YYYYMMDD_HHMMSS_<rand>).
    outputs_root:
        Root outputs directory (default: outputs/).
    """
    outputs_root = Path(outputs_root)
    base_dir = outputs_root / "runs" / run_id

    meta_dir = base_dir / "meta"
    logs_dir = base_dir / "logs"
    diffs_dir = base_dir / "diffs"
    notebooks_dir = base_dir / "notebooks"
    reports_dir = base_dir / "reports"
    data_dir = base_dir / "data"
    patches_dir = diffs_dir / "patches"

    return RunArtifacts(
        run_id=run_id,
        base_dir=base_dir,
        meta_dir=meta_dir,
        logs_dir=logs_dir,
        diffs_dir=diffs_dir,
        notebooks_dir=notebooks_dir,
        reports_dir=reports_dir,
        data_dir=data_dir,
        patches_dir=patches_dir,
        manifest_path=meta_dir / "manifest.json",
        timings_path=meta_dir / "timings.json",
        stdout_log=logs_dir / "stdout.log",
        stderr_log=logs_dir / "stderr.log",
        nbclient_log=logs_dir / "nbclient.log",
        quality_gate_log=logs_dir / "quality_gate.log",
        git_status_path=diffs_dir / "git" / "status.txt",
        git_diff_path=diffs_dir / "git" / "diff.patch",
        input_notebook_path=notebooks_dir / "input.ipynb",
        output_notebook_path=notebooks_dir / "output.ipynb",
        summary_report_path=reports_dir / "summary.md",
        failure_report_path=reports_dir / "failure.md",
    )


def ensure_run_dir(
    *,
    run_id: str,
    outputs_root: str | Path = "outputs",
    task_id: Optional[str] = None,
    notebook_path: Optional[str] = None,
    git: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> RunArtifacts:
    """
    Convenience: resolve layout, create dirs, and write initial manifest.

    Returns the resolved RunArtifacts.
    """
    art = resolve_run_artifacts(run_id=run_id, outputs_root=outputs_root)
    art.ensure_dirs()
    art.write_manifest(task_id=task_id, notebook_path=notebook_path, git=git, extra=extra)
    return art


def next_run_id(prefix: Optional[str] = None) -> str:
    """
    Generate a reasonably unique run_id.

    Example:
      20260209_073012_8f3a2c
      nb043_20260209_073012_8f3a2c  (if prefix="nb043")

    Uses 6 hex chars from urandom for collision resistance.
    """
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    rnd = os.urandom(3).hex()  # 6 hex chars
    base = f"{ts}_{rnd}"
    return f"{prefix}_{base}" if prefix else base


# -----------------------------
# Helpers
# -----------------------------

def now_iso() -> str:
    """Current time in ISO format with timezone."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
