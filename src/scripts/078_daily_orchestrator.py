#!/usr/bin/env python
# src/scripts/078_daily_orchestrator.py
"""078 Daily Orchestrator — CLI-based daily collection pipeline.

Replaces notebook-based orchestration (036) with a script that has
clear logging, execution order, and exit codes.

Execution order: 074 → 109 → 031 → 076 → 075 → 077

031 mode: Drive enabled, slides disabled (--no-slides).
  To enable slides, remove --no-slides from 031's extra_args or run
  031 standalone: python -m src.scripts.031_pdf_inbox_processor --limit N

Usage::

    # Full run
    python -m src.scripts.078_daily_orchestrator

    # Dry-run all scripts
    python -m src.scripts.078_daily_orchestrator --dry-run

    # Run specific step only
    python -m src.scripts.078_daily_orchestrator --only 074

    # Verbose logging
    python -m src.scripts.078_daily_orchestrator -v
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("078_daily_orchestrator")


@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""
    name: str
    module: str
    description: str
    required: bool = False
    extra_args: Optional[List[str]] = None


# Pipeline step definitions — order matters
PIPELINE_STEPS: List[StepConfig] = [
    StepConfig(
        name="074",
        module="src.scripts.074_lit_inbox_processor",
        description="LIT Inbox: PDF download + relevance judgment",
    ),
    StepConfig(
        name="109",
        module="src.scripts.109_lit_enrichment",
        description="LIT Enrichment: enrich paper fields (Core Idea, Methods, etc.)",
    ),
    StepConfig(
        name="031",
        module="src.scripts.031_pdf_inbox_processor",
        description="PDF Inbox: process manual PDFs from data/downloads (Drive on, slides off)",
        extra_args=["--no-slides"],
    ),
    StepConfig(
        name="076",
        module="src.scripts.076_session_to_targets",
        description="Session → Targets: extract monitoring targets from 073",
    ),
    StepConfig(
        name="075",
        module="src.scripts.075_smart_news_monitor",
        description="Smart News Monitor: cadence-optimized search",
    ),
    StepConfig(
        name="077",
        module="src.scripts.077_events_context_bridge",
        description="Events Context Bridge: build recall cache",
    ),
]


@dataclass
class StepResult:
    """Result of a single pipeline step execution."""
    name: str
    status: str  # "success" | "failed" | "skipped"
    duration_seconds: float
    error: Optional[str] = None


def run_step(step: StepConfig, *, dry_run: bool = False) -> StepResult:
    """Execute a single pipeline step via subprocess."""
    cmd = [sys.executable, "-m", step.module]
    if dry_run:
        cmd.append("--dry-run")
    if step.extra_args:
        cmd.extend(step.extra_args)

    logger.info("[%s] Starting: %s", step.name, step.description)
    logger.info("[%s] Command: %s", step.name, " ".join(cmd))

    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per step
            cwd=str(_PROJECT_ROOT),
        )

        duration = time.time() - start

        # Stream output
        if proc.stdout:
            for line in proc.stdout.strip().split("\n"):
                logger.info("[%s] %s", step.name, line)
        if proc.stderr:
            for line in proc.stderr.strip().split("\n"):
                if line.strip():
                    logger.warning("[%s] %s", step.name, line)

        if proc.returncode == 0:
            logger.info("[%s] Completed in %.1fs", step.name, duration)
            return StepResult(name=step.name, status="success", duration_seconds=duration)
        else:
            error_msg = f"Exit code {proc.returncode}"
            logger.error("[%s] Failed (%s) in %.1fs", step.name, error_msg, duration)
            return StepResult(
                name=step.name, status="failed",
                duration_seconds=duration, error=error_msg,
            )

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        logger.error("[%s] Timed out after %.1fs", step.name, duration)
        return StepResult(
            name=step.name, status="failed",
            duration_seconds=duration, error="Timeout (600s)",
        )
    except Exception as e:
        duration = time.time() - start
        logger.error("[%s] Exception: %s", step.name, e)
        return StepResult(
            name=step.name, status="failed",
            duration_seconds=duration, error=str(e),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.078_daily_orchestrator",
        description="Daily collection pipeline orchestrator.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to all steps.",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Run only this step (e.g., '074', '075').",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline_start = time.time()
    now = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("078 Daily Orchestrator — %s", now.strftime("%Y-%m-%d %H:%M UTC"))
    if args.dry_run:
        logger.info("Mode: DRY-RUN")
    logger.info("=" * 60)

    # Filter steps if --only specified
    steps = PIPELINE_STEPS
    if args.only:
        steps = [s for s in steps if s.name == args.only]
        if not steps:
            logger.error("Unknown step: %s (available: %s)",
                         args.only, ", ".join(s.name for s in PIPELINE_STEPS))
            sys.exit(1)

    # Execute steps
    results: List[StepResult] = []
    has_failure = False

    for i, step in enumerate(steps, 1):
        logger.info("")
        logger.info("--- Step %d/%d: %s ---", i, len(steps), step.name)
        result = run_step(step, dry_run=args.dry_run)
        results.append(result)

        if result.status == "failed":
            has_failure = True
            if step.required:
                logger.error("[%s] REQUIRED step failed — halting pipeline", step.name)
                break
            else:
                logger.warning("[%s] Non-required step failed — continuing", step.name)

    # Summary
    pipeline_duration = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline Summary")
    logger.info("=" * 60)

    for r in results:
        status_icon = "OK" if r.status == "success" else "FAIL" if r.status == "failed" else "SKIP"
        error_str = f" ({r.error})" if r.error else ""
        logger.info("  [%4s] %s — %.1fs%s", status_icon, r.name, r.duration_seconds, error_str)

    succeeded = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    logger.info("")
    logger.info("Total: %d steps, %d succeeded, %d failed, %.1fs elapsed",
                len(results), succeeded, failed, pipeline_duration)
    logger.info("=" * 60)

    if has_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
