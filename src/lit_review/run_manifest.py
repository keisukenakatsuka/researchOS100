# src/lit_review/run_manifest.py
"""Run manifest management for Block 3–6 pipeline.

Provides a shared schema for tracking step execution status,
outputs, and dependencies across all scripts (079–100).
Designed for future Web UI orchestrator integration.

Usage::

    from src.lit_review.run_manifest import load_manifest, update_step

    manifest = load_manifest(run_dir)
    update_step(run_dir, "093_research_plan_generator", status="completed", outputs=["research_plan.md"])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_FILE = "run_manifest.json"

# Step status model
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_PARTIAL = "partial"

# Script dependency graph (upstream requirements)
SCRIPT_DEPS: Dict[str, List[str]] = {
    "079_rq_paper_matcher": [],
    "080_literature_gap_filler": ["079"],
    "081_query_evidence_extractor": ["079"],
    "082_lit_review_synthesizer": ["081"],
    "083_research_landscape_mapper": ["082"],
    "084_lit_review_writeback": ["081", "082"],
    "085_cross_rq_comparison": ["082"],
    "086_claim_canonicalization": ["082"],
    "087_hypothesis_generator": ["082"],
    "088_assumption_analyzer": ["087"],
    "089_hypothesis_portfolio": ["087", "088"],
    "090_validation_designer": ["089"],
    "091_data_requirements": ["090"],
    "092_method_selector": ["090", "091"],
    "093_research_plan_generator": ["082", "087"],
    "094_paper_outline_generator": ["093"],
    "095_introduction_drafter": ["094"],
    "096_hypotheses_drafter": ["094"],
    "097_methods_drafter": ["094"],
    "098_literature_review_drafter": ["094"],
    "099_research_output_review": ["095", "096", "097"],
    "100_export_bundle": ["099"],
}

# Script output file expectations
SCRIPT_OUTPUTS: Dict[str, List[str]] = {
    "079_rq_paper_matcher": ["rq_context.json", "candidate_papers.json", "candidate_papers.md"],
    "081_query_evidence_extractor": ["evidence.json", "evidence.md"],
    "082_lit_review_synthesizer": ["lit_review.json", "lit_review.md"],
    "083_research_landscape_mapper": ["landscape.json", "landscape.md"],
    "087_hypothesis_generator": ["hypotheses.json", "hypotheses.md"],
    "088_assumption_analyzer": ["assumptions.json", "assumptions.md"],
    "089_hypothesis_portfolio": ["hypothesis_portfolio.json", "hypothesis_portfolio.md"],
    "090_validation_designer": ["validation_designs.json", "validation_designs.md"],
    "091_data_requirements": ["data_requirements.json", "data_requirements.md"],
    "092_method_selector": ["method_selection.json", "method_selection.md"],
    "093_research_plan_generator": ["research_plan.md"],
    "094_paper_outline_generator": ["paper_outline.json", "paper_outline.md"],
    "095_introduction_drafter": ["draft_introduction.md"],
    "096_hypotheses_drafter": ["draft_hypotheses.md"],
    "097_methods_drafter": ["draft_methods.md"],
    "098_literature_review_drafter": ["draft_literature_review.md"],
    "099_research_output_review": ["review_report.json", "review_report.md"],
    "100_export_bundle": ["paper_draft.md", "export_bundle.json"],
}


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    """Load or initialize run manifest."""
    manifest_path = run_dir / MANIFEST_FILE
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    # Initialize from existing files
    rq_title = ""
    rq_path = run_dir / "rq_context.json"
    if rq_path.exists():
        rq_title = json.loads(rq_path.read_text()).get("title", "")

    manifest = {
        "run_id": run_dir.name,
        "rq_title": rq_title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    # Detect completed steps from existing files
    for script_id, expected_outputs in SCRIPT_OUTPUTS.items():
        if all((run_dir / f).exists() for f in expected_outputs):
            manifest["steps"][script_id] = {
                "status": STATUS_COMPLETED,
                "outputs": expected_outputs,
            }

    _update_derived_fields(manifest, run_dir)
    return manifest


def update_step(
    run_dir: Path,
    script_id: str,
    *,
    status: str,
    outputs: Optional[List[str]] = None,
    error: Optional[str] = None,
):
    """Update a step's status in the manifest and save."""
    manifest = load_manifest(run_dir)
    now = datetime.now(timezone.utc).isoformat()

    step = manifest["steps"].get(script_id, {})
    step["status"] = status
    if status == STATUS_RUNNING:
        step["started_at"] = now
    if status in (STATUS_COMPLETED, STATUS_FAILED, STATUS_PARTIAL):
        step["completed_at"] = now
    if outputs is not None:
        step["outputs"] = outputs
    if error:
        step["error"] = error

    manifest["steps"][script_id] = step
    _update_derived_fields(manifest, run_dir)
    _save_manifest(run_dir, manifest)


def _update_derived_fields(manifest: Dict, run_dir: Path):
    """Update latest_step and step classification.

    Classifies steps into three categories for UI orchestrator:
    - runnable: dependencies met, ready to execute
    - optional: dependencies met but step is independent/supplementary
    - blocked: dependencies not yet met
    """
    completed = [
        sid for sid, s in manifest.get("steps", {}).items()
        if s.get("status") == STATUS_COMPLETED
    ]
    completed_short = {s.split("_")[0] for s in completed}

    manifest["latest_step"] = completed[-1] if completed else None

    # Classify all unexecuted steps
    runnable = []
    optional = []
    blocked = []

    # Steps that are supplementary (not on critical path)
    OPTIONAL_STEPS = {"080_literature_gap_filler", "085_cross_rq_comparison", "086_claim_canonicalization", "098_literature_review_drafter"}

    for script_id, deps in SCRIPT_DEPS.items():
        if script_id in [s for s in manifest.get("steps", {}) if manifest["steps"][s].get("status") == STATUS_COMPLETED]:
            continue
        deps_met = all(d in completed_short for d in deps)
        if not deps_met:
            blocked.append(script_id)
        elif script_id in OPTIONAL_STEPS:
            optional.append(script_id)
        else:
            runnable.append(script_id)

    manifest["runnable_steps"] = runnable
    manifest["optional_steps"] = optional
    manifest["blocked_steps"] = blocked
    # Keep next_available_steps for backward compat
    manifest["next_available_steps"] = runnable + optional


def _save_manifest(run_dir: Path, manifest: Dict):
    path = run_dir / MANIFEST_FILE
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------------

# Human-readable script names
SCRIPT_LABELS: Dict[str, str] = {
    "079_rq_paper_matcher": "Paper Matcher",
    "080_literature_gap_filler": "Gap Filler",
    "081_query_evidence_extractor": "Evidence Extractor",
    "082_lit_review_synthesizer": "Lit Review",
    "083_research_landscape_mapper": "Landscape",
    "084_lit_review_writeback": "Writeback",
    "085_cross_rq_comparison": "Cross-RQ",
    "086_claim_canonicalization": "Canonicalization",
    "087_hypothesis_generator": "Hypothesis Gen",
    "088_assumption_analyzer": "Assumptions",
    "089_hypothesis_portfolio": "Portfolio",
    "090_validation_designer": "Validation Design",
    "091_data_requirements": "Data Requirements",
    "092_method_selector": "Method Selector",
    "093_research_plan_generator": "Research Plan",
    "094_paper_outline_generator": "Paper Outline",
    "095_introduction_drafter": "Introduction",
    "096_hypotheses_drafter": "Hypotheses",
    "097_methods_drafter": "Methods",
    "098_literature_review_drafter": "Literature Review",
    "099_research_output_review": "Output Review",
    "100_export_bundle": "Export Bundle",
}

SCRIPT_BLOCKS: Dict[str, str] = {
    "079": "Block 2", "080": "Block 2", "081": "Block 2", "082": "Block 2",
    "083": "Block 2", "084": "Block 2", "085": "Block 2",
    "086": "Block 3", "087": "Block 4",
    "088": "Block 5", "089": "Block 5", "090": "Block 5",
    "091": "Block 5", "092": "Block 5",
    "093": "Block 6",
    "094": "Block 6", "095": "Block 6", "096": "Block 6", "097": "Block 6",
    "098": "Block 6", "099": "Block 6", "100": "Block 6",
}


def list_runs(data_dir: Path) -> List[Dict[str, Any]]:
    """List all runs with summary info for Dashboard."""
    runs = []
    if not data_dir.exists():
        return runs

    for run_dir in sorted(data_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        # Skip non-run dirs
        if run_dir.name in ("spikes", "canonical_claims", "cross_rq"):
            continue
        if not (run_dir / "rq_context.json").exists():
            continue

        manifest = load_manifest(run_dir)
        steps = manifest.get("steps", {})
        completed = sum(1 for s in steps.values() if s.get("status") == STATUS_COMPLETED)
        total = len(SCRIPT_DEPS)

        # Find last update time
        timestamps = []
        for s in steps.values():
            for k in ("completed_at", "started_at"):
                if s.get(k):
                    timestamps.append(s[k])

        runs.append({
            "run_id": manifest.get("run_id", run_dir.name),
            "run_dir": str(run_dir),
            "rq_title": manifest.get("rq_title", ""),
            "latest_step": manifest.get("latest_step", ""),
            "completed_steps": completed,
            "total_steps": total,
            "last_updated": max(timestamps) if timestamps else manifest.get("created_at", ""),
            "status": _run_status(steps),
        })

    return runs


def _run_status(steps: Dict) -> str:
    """Determine overall run status."""
    statuses = [s.get("status") for s in steps.values()]
    if any(s == STATUS_RUNNING for s in statuses):
        return "running"
    if any(s == STATUS_FAILED for s in statuses):
        return "partial"
    if not statuses:
        return "pending"
    return "completed" if all(s == STATUS_COMPLETED for s in statuses) else "in_progress"


def get_artifacts(run_dir: Path) -> List[Dict[str, Any]]:
    """List available artifacts (.md and .json) in a run directory."""
    artifacts = []
    for f in sorted(run_dir.iterdir()):
        if f.suffix in (".md", ".json") and f.name != MANIFEST_FILE:
            artifacts.append({
                "name": f.name,
                "path": str(f),
                "type": "markdown" if f.suffix == ".md" else "json",
                "size": f.stat().st_size,
            })
    return artifacts
