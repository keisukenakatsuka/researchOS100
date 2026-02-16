# src/exec/quality_gate.py
"""
Quality gate runner (pytest / ruff / arbitrary commands) + artifacts logging.

Role
----
Verifier "tests":
- Run lightweight checks after notebook execution or before merging changes.
- Capture stdout/stderr + exit code + duration.
- Write logs into artifacts layout (logs/quality_gate.log by default).
- Provide a small result object that can be logged to RUNS_DB.

This module does NOT write to Notion by itself; that belongs to verifier/orchestrator
code using repos.py.

Typical usage
-------------
from src.exec.quality_gate import run_pytest, run_ruff
res = run_pytest(artifacts=art, cwd=".", args=["-q"])
if not res.ok: ...

Notes
-----
- This captures subprocess stdout/stderr reliably.
- On Windows, shell=False is fine for standard tools if they are on PATH.
- If you need to run within a specific venv, ensure the orchestrator launches
  in that environment or pass the correct executable via env.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Mapping, Sequence
import subprocess
import time
import os

from src.artifacts.layout import RunArtifacts, write_text, append_text


# -----------------------------
# Result types
# -----------------------------

@dataclass
class CmdResult:
    ok: bool
    command: List[str]
    exit_code: int
    duration_sec: float
    stdout: str
    stderr: str

    def summary_line(self) -> str:
        status = "OK" if self.ok else "FAIL"
        cmd = " ".join(self.command)
        return f"{status} exit={self.exit_code} dur={self.duration_sec:.2f}s cmd={cmd}"


@dataclass
class QualityGateResult:
    """
    A slightly higher-level wrapper used for RUNS_DB evidence.
    """
    ok: bool
    gate_type: str  # PYTEST | RUFF | CMD
    cmd_result: CmdResult
    artifacts_log_path: Optional[str] = None

    def evidence_text(self) -> str:
        return f"{self.gate_type}: {self.cmd_result.summary_line()}"


# -----------------------------
# Core runner
# -----------------------------

def run_command(
    *,
    command: Sequence[str],
    cwd: Optional[str | Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: Optional[int] = None,
) -> CmdResult:
    cmd = [str(x) for x in command]
    started = time.time()

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(cwd).expanduser().resolve()) if cwd else None,
            env=_merge_env(env),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        finished = time.time()
        dur = finished - started
        ok = proc.returncode == 0
        return CmdResult(
            ok=ok,
            command=cmd,
            exit_code=int(proc.returncode),
            duration_sec=dur,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
    except subprocess.TimeoutExpired as e:
        finished = time.time()
        dur = finished - started
        out = ""
        err = f"TimeoutExpired: {e}"
        return CmdResult(
            ok=False,
            command=cmd,
            exit_code=124,  # conventional timeout code
            duration_sec=dur,
            stdout=out,
            stderr=err,
        )
    except Exception as e:
        finished = time.time()
        dur = finished - started
        return CmdResult(
            ok=False,
            command=cmd,
            exit_code=1,
            duration_sec=dur,
            stdout="",
            stderr=f"{type(e).__name__}: {e}",
        )


def run_quality_gate(
    *,
    gate_type: str,
    command: Sequence[str],
    artifacts: Optional[RunArtifacts] = None,
    cwd: Optional[str | Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_sec: Optional[int] = None,
    log_filename: str = "quality_gate.log",
    truncate_stdout_chars: int = 12000,
    truncate_stderr_chars: int = 12000,
) -> QualityGateResult:
    """
    Run a command as a quality gate and write logs to artifacts (if provided).

    gate_type examples: "PYTEST", "RUFF", "CMD"
    """
    cmd_res = run_command(command=command, cwd=cwd, env=env, timeout_sec=timeout_sec)

    log_path: Optional[str] = None
    if artifacts is not None:
        artifacts.ensure_dirs()
        # Prefer dedicated artifacts.quality_gate_log if present, else logs/<log_filename>
        p = getattr(artifacts, "quality_gate_log", None)
        if p is None:
            p = artifacts.logs_dir / log_filename  # type: ignore[attr-defined]
        log_path = str(p)

        header = (
            f"\n[{_now_human()}] QUALITY_GATE {gate_type}\n"
            f"cmd: {' '.join(cmd_res.command)}\n"
            f"cwd: {str(Path(cwd).expanduser().resolve()) if cwd else ''}\n"
            f"exit_code: {cmd_res.exit_code}\n"
            f"duration_sec: {cmd_res.duration_sec:.2f}\n"
        )
        append_text(Path(log_path), header)

        stdout = _truncate(cmd_res.stdout, truncate_stdout_chars)
        stderr = _truncate(cmd_res.stderr, truncate_stderr_chars)

        if stdout:
            append_text(Path(log_path), f"\n--- stdout ---\n{stdout}\n")
        if stderr:
            append_text(Path(log_path), f"\n--- stderr ---\n{stderr}\n")

        append_text(Path(log_path), f"\n[{_now_human()}] END ok={cmd_res.ok}\n")

    return QualityGateResult(
        ok=cmd_res.ok,
        gate_type=gate_type,
        cmd_result=cmd_res,
        artifacts_log_path=log_path,
    )


# -----------------------------
# Convenience wrappers
# -----------------------------

def run_pytest(
    *,
    artifacts: Optional[RunArtifacts] = None,
    cwd: Optional[str | Path] = None,
    args: Optional[List[str]] = None,
    timeout_sec: Optional[int] = 900,
    env: Optional[Mapping[str, str]] = None,
) -> QualityGateResult:
    """
    Run pytest. Defaults to a moderate timeout.
    """
    cmd = ["pytest"]
    if args:
        cmd.extend(args)
    return run_quality_gate(
        gate_type="PYTEST",
        command=cmd,
        artifacts=artifacts,
        cwd=cwd,
        env=env,
        timeout_sec=timeout_sec,
    )


def run_ruff(
    *,
    artifacts: Optional[RunArtifacts] = None,
    cwd: Optional[str | Path] = None,
    args: Optional[List[str]] = None,
    timeout_sec: Optional[int] = 300,
    env: Optional[Mapping[str, str]] = None,
) -> QualityGateResult:
    """
    Run ruff (lint). Example args: ["check", "."] or ["check", "--fix", "."]
    """
    cmd = ["ruff"]
    if args:
        cmd.extend(args)
    return run_quality_gate(
        gate_type="RUFF",
        command=cmd,
        artifacts=artifacts,
        cwd=cwd,
        env=env,
        timeout_sec=timeout_sec,
    )


def run_mypy(
    *,
    artifacts: Optional[RunArtifacts] = None,
    cwd: Optional[str | Path] = None,
    args: Optional[List[str]] = None,
    timeout_sec: Optional[int] = 600,
    env: Optional[Mapping[str, str]] = None,
) -> QualityGateResult:
    """
    Optional: run mypy if you use it. Safe to ignore if not installed.
    """
    cmd = ["mypy"]
    if args:
        cmd.extend(args)
    return run_quality_gate(
        gate_type="MYPY",
        command=cmd,
        artifacts=artifacts,
        cwd=cwd,
        env=env,
        timeout_sec=timeout_sec,
    )


# -----------------------------
# Helpers
# -----------------------------

def _merge_env(env: Optional[Mapping[str, str]]) -> Dict[str, str]:
    merged = dict(os.environ)
    if env:
        for k, v in env.items():
            merged[str(k)] = str(v)
    return merged


def _truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...(truncated)\n"


def _now_human() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")
