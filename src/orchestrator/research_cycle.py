# src/orchestrator/research_cycle.py
"""Research Cycle Orchestrator — service logic.

Executes the full Research Cycle pipeline (079-106) via subprocess,
with resume support, progress display, and error handling.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIT_DATA_DIR = PROJECT_ROOT / "data" / "lit_review"
QF_DATA_DIR = PROJECT_ROOT / "data" / "question_formation"

# Phase display names (v2 — superset of v1)
PHASE_LABELS = {
    "data_audit": "Phase 0: Data Source Audit",
    "lit_review": "Phase 1: Literature Review",
    "hypothesis": "Phase 2a: Hypothesis Development",
    "deep_literature": "Phase 2a-ext: Literature Deep Dive",
    "data_build": "Phase 2b: Data Collection & Build",
    "output": "Phase 3a: Research Output",
    "empirical": "Phase 3b: Empirical Analysis",
    "next_rq": "Phase 4: Next-Run Generation",
    "visualization": "Phase 5: Visualization",
}

# Phase number mapping for --stop-after-phase
PHASE_NUMBER_MAP = {
    0: ["data_audit"],
    1: ["lit_review", "hypothesis", "deep_literature"],
    2: ["data_build"],
    3: ["output", "empirical"],
    4: ["next_rq"],
    5: ["visualization"],
}


# ------------------------------------------------------------------
# Step configuration
# ------------------------------------------------------------------

@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""
    name: str
    module: str
    description: str
    phase: str
    optional: bool = False
    extra_args: Optional[List[str]] = None


PIPELINE_STEPS: List[StepConfig] = [
    # Phase 1: Literature Review
    StepConfig("079", "src.scripts.079_rq_paper_matcher",
               "Paper Matcher + Evidence + Synthesis (full pipeline)",
               phase="lit_review",
               extra_args=["--full-pipeline", "--min-score", "{min_score}", "--max-papers", "{max_papers}"]),
    StepConfig("080", "src.scripts.080_literature_gap_filler",
               "External paper search (Semantic Scholar + arXiv)",
               phase="lit_review", optional=True),
    StepConfig("083", "src.scripts.083_research_landscape_mapper",
               "Knowledge landscape mapping",
               phase="lit_review"),
    StepConfig("084", "src.scripts.084_lit_review_writeback",
               "KML writeback (Evidence / Claims / Memos)",
               phase="lit_review", optional=True,
               extra_args=["--writeback"]),

    # Phase 2: Hypothesis Development
    StepConfig("087", "src.scripts.087_hypothesis_generator",
               "Hypothesis generation from gaps & blindspots",
               phase="hypothesis"),
    StepConfig("088", "src.scripts.088_assumption_analyzer",
               "Assumption analysis (3 categories)",
               phase="hypothesis"),
    StepConfig("089", "src.scripts.089_hypothesis_portfolio",
               "Hypothesis portfolio scoring (5 axes)",
               phase="hypothesis"),
    StepConfig("089a", "src.scripts.089a_hypothesis_selector",
               "Hypothesis selection (H1/H2 convergence)",
               phase="hypothesis"),
    StepConfig("089b", "src.scripts.089b_hypothesis_review",
               "Hypothesis review (human review or auto-accept)",
               phase="hypothesis"),
    StepConfig("089c", "src.scripts.089c_focused_hypotheses",
               "Focused hypotheses (canonical H1/H2 output)",
               phase="hypothesis"),

    # Phase 2 (cont.): Literature Deep Dive (optional, per hypothesis)
    StepConfig("114", "src.scripts.114_hyp_query_expansion",
               "Hypothesis query expansion (8-12 queries per H)",
               phase="deep_literature", optional=True),
    StepConfig("115", "src.scripts.115_hyp_mass_retrieval",
               "Mass paper retrieval (S2 + arXiv)",
               phase="deep_literature", optional=True),
    StepConfig("116", "src.scripts.116_hyp_dedup_rank",
               "Dedup, relevance ranking, selection (100-150 per H)",
               phase="deep_literature", optional=True),
    StepConfig("117", "src.scripts.117_hyp_clustering",
               "Paper clustering (TF-IDF + k-means)",
               phase="deep_literature", optional=True),
    StepConfig("118", "src.scripts.118_hyp_extraction",
               "Structured extraction (variables/methods/findings)",
               phase="deep_literature", optional=True),
    StepConfig("119", "src.scripts.119_hyp_literature_synthesis",
               "Per-hypothesis literature synthesis",
               phase="deep_literature", optional=True),
    StepConfig("119b", "src.scripts.119b_evidence_sufficiency",
               "Evidence sufficiency check",
               phase="deep_literature", optional=True),

    StepConfig("090", "src.scripts.090_validation_designer",
               "Validation design per hypothesis",
               phase="hypothesis"),
    StepConfig("091", "src.scripts.091_data_requirements",
               "Data requirements & variable operationalization",
               phase="hypothesis"),
    StepConfig("092", "src.scripts.092_method_selector",
               "Method comparison & recommendation",
               phase="hypothesis"),

    # Phase 2 (cont.): Validation & Grounding (optional)
    StepConfig("110", "src.scripts.110_literature_validator",
               "Literature validation (evidence grounding)",
               phase="hypothesis", optional=True),
    StepConfig("111", "src.scripts.111_dataset_registry",
               "Dataset registry (availability assessment)",
               phase="hypothesis", optional=True),
    StepConfig("112", "src.scripts.112_model_prototyper",
               "Model blueprints (baseline design)",
               phase="hypothesis", optional=True),
    StepConfig("113", "src.scripts.113_data_strategy",
               "Data strategy roadmap",
               phase="hypothesis", optional=True),

    # Phase 3: Research Output
    StepConfig("093", "src.scripts.093_research_plan_generator",
               "Research plan generation",
               phase="output"),
    StepConfig("094", "src.scripts.094_paper_outline_generator",
               "Paper outline (shared reference for drafters)",
               phase="output"),
    StepConfig("095", "src.scripts.095_introduction_drafter",
               "Introduction draft",
               phase="output"),
    StepConfig("096", "src.scripts.096_hypotheses_drafter",
               "Hypotheses Development draft",
               phase="output"),
    StepConfig("097", "src.scripts.097_methods_drafter",
               "Research Methodology draft",
               phase="output"),
    StepConfig("098", "src.scripts.098_literature_review_drafter",
               "Literature Review draft",
               phase="output"),
    StepConfig("099", "src.scripts.099_research_output_review",
               "Cross-section quality review",
               phase="output"),
    StepConfig("100", "src.scripts.100_export_bundle",
               "Export bundle (paper_draft.md)",
               phase="output"),

    # Phase 4: Next-Run Generation
    StepConfig("101", "src.scripts.101_rq_generator",
               "Next-generation RQ candidate extraction",
               phase="next_rq"),
    StepConfig("102", "src.scripts.102_rq_evaluator",
               "RQ quality evaluation (4 axes)",
               phase="next_rq"),
    StepConfig("103", "src.scripts.103_rq_prioritizer",
               "RQ portfolio management",
               phase="next_rq"),
    StepConfig("104", "src.scripts.104_rq_evolution_tracker",
               "RQ lineage DAG tracking",
               phase="next_rq"),
    StepConfig("105", "src.scripts.105_rq_writeback",
               "Export promoted RQs as rq_context.json",
               phase="next_rq",
               extra_args=["--export-only"]),

    # Phase 5: Visualization
    StepConfig("106", "src.scripts.106_knowledge_graph",
               "Interactive knowledge graph (HTML)",
               phase="visualization"),
]

# ------------------------------------------------------------------
# v2 conditional steps (only when --data-dir is provided)
# ------------------------------------------------------------------

PHASE_0_STEPS: List[StepConfig] = [
    StepConfig("125", "src.scripts.125_data_source_audit",
               "Data source audit (DV feasibility, field coverage)",
               phase="data_audit"),
    StepConfig("128-pre", "src.scripts.128_export_validator",
               "Export validation (pre-build, sample check)",
               phase="data_audit"),
]

PHASE_2B_STEPS: List[StepConfig] = [
    StepConfig("128", "src.scripts.128_export_validator",
               "Export validation (full data)",
               phase="data_build"),
    StepConfig("130", "src.scripts.130_build_deal_dataset",
               "Build unified deal-level dataset",
               phase="data_build"),
]

PHASE_3B_STEPS: List[StepConfig] = [
    StepConfig("132", "src.scripts.132_regression_runner",
               "Regression analysis (main + robustness + heterogeneity)",
               phase="empirical"),
]

# Expected outputs for 101-105 (in data/question_formation/from_<run_id>/)
QF_OUTPUTS: Dict[str, List[str]] = {
    "101": ["rq_candidates.json", "rq_candidates.md"],
    "102": ["rq_evaluation.json", "rq_evaluation.md"],
    "103": ["rq_portfolio.json", "rq_portfolio.md"],
    "104": [],  # writes to global lineage dir
    "105": [],  # writes to next_runs/ with variable count
}


# ------------------------------------------------------------------
# Step result
# ------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of a single pipeline step execution."""
    name: str
    status: str  # "completed" | "failed" | "skipped"
    duration_seconds: float = 0.0
    error: Optional[str] = None


# ------------------------------------------------------------------
# Execution plan
# ------------------------------------------------------------------

def build_execution_plan(
    *,
    include_optional: bool = False,
    skip_steps: Optional[List[str]] = None,
    stop_after: Optional[str] = None,
) -> List[StepConfig]:
    """Build the list of steps to execute, filtering optional/skip/stop."""
    skip_set = set(skip_steps or [])
    plan: List[StepConfig] = []

    for step in PIPELINE_STEPS:
        if step.optional and not include_optional:
            continue
        if step.name in skip_set:
            continue
        plan.append(step)
        if stop_after and step.name == stop_after:
            break

    return plan


# ------------------------------------------------------------------
# v2 execution plan
# ------------------------------------------------------------------

def build_execution_plan_v2(
    *,
    include_optional: bool = False,
    skip_steps: Optional[List[str]] = None,
    stop_after: Optional[str] = None,
    stop_after_phase: Optional[int] = None,
    has_data_dir: bool = False,
    skip_empirical: bool = False,
) -> List[StepConfig]:
    """Build v2 execution plan with conditional phases.

    When has_data_dir=False, this produces the same plan as build_execution_plan()
    (v1 backward compatibility).
    """
    skip_set = set(skip_steps or [])
    plan: List[StepConfig] = []

    # Collect allowed phases based on stop_after_phase
    allowed_phases = None
    if stop_after_phase is not None:
        allowed_phases = set()
        for pn in range(stop_after_phase + 1):
            for phase_key in PHASE_NUMBER_MAP.get(pn, []):
                allowed_phases.add(phase_key)

    def _add_steps(steps: List[StepConfig]):
        for step in steps:
            if allowed_phases is not None and step.phase not in allowed_phases:
                continue
            if step.optional and not include_optional:
                continue
            if step.name in skip_set:
                continue
            plan.append(step)
            if stop_after and step.name == stop_after:
                return True  # signal to stop
        return False

    # Phase 0 (conditional)
    if has_data_dir:
        if _add_steps(PHASE_0_STEPS):
            return plan

    # Phase 1 + 2a: lit_review + hypothesis + deep_literature only
    phase_1_2a = [s for s in PIPELINE_STEPS
                  if s.phase in ("lit_review", "hypothesis", "deep_literature")]
    if _add_steps(phase_1_2a):
        return plan

    # Phase 2b: Data Collection & Build (conditional)
    if has_data_dir:
        if _add_steps(PHASE_2B_STEPS):
            return plan

    # Phase 3a: Research Output
    phase_3a = [s for s in PIPELINE_STEPS if s.phase == "output"]
    if _add_steps(phase_3a):
        return plan

    # Phase 3b: Empirical Analysis (conditional)
    if has_data_dir and not skip_empirical:
        if _add_steps(PHASE_3B_STEPS):
            return plan

    # Phase 4 + 5: next_rq + visualization
    phase_4_5 = [s for s in PIPELINE_STEPS
                 if s.phase in ("next_rq", "visualization")]
    if _add_steps(phase_4_5):
        return plan

    return plan


# ------------------------------------------------------------------
# Go/No-Go gates
# ------------------------------------------------------------------

@dataclass
class GateCheck:
    """A single gate check item."""
    name: str
    passed: bool
    value: str
    threshold: str
    message: str = ""


@dataclass
class GateResult:
    """Result of a phase gate evaluation."""
    passed: bool
    phase: str
    checks: List[GateCheck] = field(default_factory=list)


def check_phase_0_gate(audit_result, validator_result=None) -> GateResult:
    """Evaluate Phase 0 gate (DV feasibility + field coverage)."""
    checks = []

    # Check 1: At least 1 feasible DV
    feasible = audit_result.feasible_dvs
    checks.append(GateCheck(
        name="Feasible DVs",
        passed=len(feasible) >= 1,
        value=str(len(feasible)),
        threshold=">= 1",
        message=f"Feasible: {', '.join(feasible)}" if feasible
                else "No feasible DVs. Check 125 output for proxy suggestions.",
    ))

    # Check 2: Core field coverage
    for fld, min_rate in [("Company Status", 0.9), ("Deal Date", 0.9)]:
        rate = audit_result.coverage.get(fld, 0)
        checks.append(GateCheck(
            name=f"{fld} coverage",
            passed=rate >= min_rate,
            value=f"{rate:.0%}",
            threshold=f">= {min_rate:.0%}",
            message="" if rate >= min_rate else "Re-check export filters.",
        ))

    # Check 3: Co-investment rate (informational, never blocks)
    if audit_result.co_investment_rate is not None:
        checks.append(GateCheck(
            name="Co-investment rate",
            passed=True,
            value=f"{audit_result.co_investment_rate:.0%}",
            threshold="info",
            message=audit_result.treatment_definition,
        ))

    passed = all(c.passed for c in checks)
    return GateResult(passed=passed, phase="Phase 0", checks=checks)


def check_phase_2_gate(dataset_stats: Dict[str, int]) -> GateResult:
    """Evaluate Phase 2 gate (sample size)."""
    checks = []

    checks.append(GateCheck(
        name="Treatment count",
        passed=dataset_stats.get("treatment", 0) >= 100,
        value=str(dataset_stats.get("treatment", 0)),
        threshold=">= 100",
        message="" if dataset_stats.get("treatment", 0) >= 100
                else "Expand GVC list or add countries.",
    ))

    checks.append(GateCheck(
        name="Control count",
        passed=dataset_stats.get("control", 0) >= 500,
        value=str(dataset_stats.get("control", 0)),
        threshold=">= 500",
        message="" if dataset_stats.get("control", 0) >= 500
                else "Expand PVC list.",
    ))

    passed = all(c.passed for c in checks)
    return GateResult(passed=passed, phase="Phase 2", checks=checks)


def print_gate_result(gate: GateResult) -> None:
    """Print gate evaluation to stdout."""
    status = "PASS" if gate.passed else "FAIL"
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  {gate.phase} Gate -- {status}")
    print(sep)
    for c in gate.checks:
        icon = "ok" if c.passed else "FAIL"
        print(f"  [{icon:>4}] {c.name}: {c.value} (threshold: {c.threshold})")
        if c.message:
            print(f"         -> {c.message}")
    if not gate.passed:
        print(f"\n  Pipeline stopped at {gate.phase}.")
        print(f"  Fix the issues above and re-run.")


# ------------------------------------------------------------------
# Resume checks
# ------------------------------------------------------------------

def should_skip(step: StepConfig, run_id: str, manifest: Dict[str, Any]) -> bool:
    """Determine if a step should be skipped (already completed)."""
    if step.name in QF_OUTPUTS:
        return _should_skip_qf(step, run_id)
    return _should_skip_manifest(step, manifest)


def _should_skip_manifest(step: StepConfig, manifest: Dict[str, Any]) -> bool:
    """Check run_manifest for 079-100, 106 completion."""
    steps_data = manifest.get("steps", {})
    # Find matching key (e.g., "079_rq_paper_matcher" for step.name "079")
    for key, info in steps_data.items():
        if key.startswith(step.name):
            return info.get("status") == "completed"
    return False


def _should_skip_qf(step: StepConfig, run_id: str) -> bool:
    """Check question_formation output files for 101-105 completion."""
    qf_dir = QF_DATA_DIR / f"from_{run_id}"
    expected = QF_OUTPUTS.get(step.name, [])
    if not expected:
        return False  # 104, 105 have variable outputs; always re-run
    return all((qf_dir / f).exists() for f in expected)


# ------------------------------------------------------------------
# run_id extraction
# ------------------------------------------------------------------

def extract_run_id(before_dirs: set) -> Optional[str]:
    """Compare directory listing before/after 079 to find new run_id."""
    if not LIT_DATA_DIR.exists():
        return None
    after_dirs = {
        d.name for d in LIT_DATA_DIR.iterdir()
        if d.is_dir() and d.name not in ("spikes", "canonical_claims", "cross_rq")
    }
    new_dirs = after_dirs - before_dirs
    if len(new_dirs) == 1:
        return new_dirs.pop()

    # Fallback: latest directory with rq_context.json
    candidates = []
    for d in LIT_DATA_DIR.iterdir():
        if d.is_dir() and (d / "rq_context.json").exists() and d.name in (after_dirs - before_dirs | after_dirs):
            candidates.append(d)
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0].name
    return None


def get_lit_dirs() -> set:
    """Get current directory names under data/lit_review/."""
    if not LIT_DATA_DIR.exists():
        return set()
    return {
        d.name for d in LIT_DATA_DIR.iterdir()
        if d.is_dir() and d.name not in ("spikes", "canonical_claims", "cross_rq")
    }


# ------------------------------------------------------------------
# Argument building
# ------------------------------------------------------------------

def build_step_args(
    step: StepConfig,
    run_id: str,
    *,
    rq_text: Optional[str] = None,
    rq_id: Optional[str] = None,
    min_score: int = 65,
    max_papers: int = 20,
    continue_on_error: bool = False,
) -> List[str]:
    """Build subprocess command arguments for a step."""
    args = [sys.executable, "-m", step.module]

    # 079 uses --rq-text / --rq-id (run_id doesn't exist yet for new runs)
    if step.name == "079":
        if rq_text:
            args += ["--rq-text", rq_text]
        elif rq_id:
            args += ["--rq-id", rq_id]
    else:
        args += ["--run-id", run_id]

    # Expand extra_args templates
    if step.extra_args:
        for arg in step.extra_args:
            args.append(arg.format(
                min_score=min_score,
                max_papers=max_papers,
            ))

    # 119b: force through evidence gate when --continue-on-error is set
    if step.name == "119b" and continue_on_error:
        args.append("--force")

    # 100: skip quality gate when --continue-on-error is set
    if step.name == "100" and continue_on_error:
        args.append("--skip-quality-gate")

    return args


# ------------------------------------------------------------------
# Step execution
# ------------------------------------------------------------------

def execute_step(
    step: StepConfig,
    cmd: List[str],
    *,
    timeout: int = 900,
) -> StepResult:
    """Execute a single pipeline step via subprocess."""
    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )

        duration = time.time() - start

        # Log subprocess output
        if proc.stdout:
            for line in proc.stdout.strip().split("\n"):
                if line.strip():
                    logger.debug("[%s] %s", step.name, line)
        if proc.stderr:
            for line in proc.stderr.strip().split("\n"):
                if line.strip():
                    logger.debug("[%s] stderr: %s", step.name, line)

        if proc.returncode == 0:
            return StepResult(name=step.name, status="completed", duration_seconds=duration)
        else:
            error_msg = f"exit code {proc.returncode}"
            # Include last stderr lines for context
            stderr_lines = [l for l in (proc.stderr or "").strip().split("\n") if l.strip()]
            if stderr_lines:
                last_error = stderr_lines[-1][:200]
                error_msg = f"{error_msg}: {last_error}"
            return StepResult(
                name=step.name, status="failed",
                duration_seconds=duration, error=error_msg,
            )

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return StepResult(
            name=step.name, status="failed",
            duration_seconds=duration, error=f"Timeout ({timeout}s)",
        )
    except Exception as e:
        duration = time.time() - start
        return StepResult(
            name=step.name, status="failed",
            duration_seconds=duration, error=str(e),
        )


# ------------------------------------------------------------------
# Pipeline runner
# ------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Overall result of the pipeline run."""
    run_id: str = ""
    rq_text: str = ""
    steps: List[StepResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def completed(self) -> int:
        return sum(1 for s in self.steps if s.status == "completed")

    @property
    def failed(self) -> int:
        return sum(1 for s in self.steps if s.status == "failed")

    @property
    def skipped(self) -> int:
        return sum(1 for s in self.steps if s.status == "skipped")

    @property
    def success(self) -> bool:
        return self.failed == 0


def _format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def run_pipeline(
    plan: List[StepConfig],
    *,
    run_id: str = "",
    rq_text: str = "",
    rq_id: str = "",
    min_score: int = 65,
    max_papers: int = 20,
    continue_on_error: bool = False,
    step_timeout: int = 900,
) -> PipelineResult:
    """Execute the pipeline plan sequentially.

    Returns PipelineResult with per-step results.
    """
    result = PipelineResult(run_id=run_id, rq_text=rq_text)
    pipeline_start = time.time()

    # Load manifest for resume checks
    manifest: Dict[str, Any] = {}
    if run_id:
        run_dir = LIT_DATA_DIR / run_id
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

    # Snapshot dirs before 079 (for run_id extraction)
    before_dirs = get_lit_dirs()

    current_phase = ""
    total = len(plan)

    for i, step in enumerate(plan, 1):
        # Phase header
        if step.phase != current_phase:
            current_phase = step.phase
            phase_label = PHASE_LABELS.get(current_phase, current_phase)
            print(f"\n--- {phase_label} ---\n")

        # Resume check
        if run_id and should_skip(step, run_id, manifest):
            print(f"[{i:2d}/{total}] {step.description} ... skipped (completed)")
            result.steps.append(StepResult(name=step.name, status="skipped"))
            continue

        # Execute
        print(f"[{i:2d}/{total}] {step.description} ... ", end="", flush=True)

        cmd = build_step_args(
            step, run_id,
            rq_text=rq_text, rq_id=rq_id,
            min_score=min_score, max_papers=max_papers,
            continue_on_error=continue_on_error,
        )
        logger.debug("Command: %s", " ".join(cmd))

        step_result = execute_step(step, cmd, timeout=step_timeout)
        result.steps.append(step_result)

        if step_result.status == "completed":
            print(f"done ({_format_duration(step_result.duration_seconds)})")
        else:
            print(f"FAILED ({_format_duration(step_result.duration_seconds)})")
            if step_result.error:
                print(f"        Error: {step_result.error}")

        # Extract run_id after 079
        if step.name == "079" and step_result.status == "completed":
            new_run_id = extract_run_id(before_dirs)
            if new_run_id:
                run_id = new_run_id
                result.run_id = run_id
                print(f"        run_id: {run_id}")

                # Load newly created manifest
                run_dir = LIT_DATA_DIR / run_id
                manifest_path = run_dir / "run_manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
            else:
                print("        WARNING: Could not detect run_id from 079 output")
                if not continue_on_error:
                    break

        # Post-079 validation: check papers found
        if step.name == "079" and step_result.status == "completed" and run_id:
            candidates_path = LIT_DATA_DIR / run_id / "candidate_papers.json"
            if candidates_path.exists():
                candidates = json.loads(candidates_path.read_text())
                papers = candidates.get("papers", [])
                if not papers:
                    print("        WARNING: No papers matched. Pipeline may produce thin results.")

        # Failure handling
        if step_result.status == "failed":
            if continue_on_error:
                print(f"        Continuing despite failure (--continue-on-error)")
            else:
                if run_id:
                    print(f"\n  Resume with: python -m src.scripts.108_research_cycle_orchestrator --run-id {run_id}")
                break

    result.total_duration = time.time() - pipeline_start
    return result


# ------------------------------------------------------------------
# Summary output
# ------------------------------------------------------------------

def print_summary(result: PipelineResult) -> None:
    """Print pipeline execution summary."""
    sep = "=" * 55

    print(f"\n{sep}")
    print("  Summary")
    print(sep)

    status = "completed" if result.success else "partial" if result.completed > 0 else "failed"
    total_steps = len(result.steps)
    executed = result.completed + result.failed

    print(f"  Status:    {status}")
    if result.run_id:
        print(f"  Run ID:    {result.run_id}")
    print(f"  Duration:  {_format_duration(result.total_duration)}")
    print(f"  Steps:     {result.completed}/{executed} completed"
          + (f", {result.skipped} skipped" if result.skipped else "")
          + (f", {result.failed} failed" if result.failed else ""))

    # Key outputs
    if result.run_id:
        run_dir = LIT_DATA_DIR / result.run_id
        outputs = []

        paper_path = run_dir / "paper_draft.md"
        if paper_path.exists():
            text = paper_path.read_text()
            word_count = len(text.split())
            outputs.append(f"  paper_draft.md          {word_count:,} words")

        kg_path = run_dir / "knowledge_graph.html"
        if kg_path.exists():
            outputs.append(f"  knowledge_graph.html    open with: open {kg_path}")

        qf_dir = QF_DATA_DIR / f"from_{result.run_id}" / "next_runs"
        if qf_dir.exists():
            next_rqs = list(qf_dir.glob("rq_context_*.json"))
            if next_rqs:
                outputs.append(f"  rq_candidates           {len(next_rqs)} promoted -> {qf_dir}")

        if outputs:
            print("\n  Outputs:")
            for line in outputs:
                print(f"    {line}")

    # Failed steps detail
    failed = [s for s in result.steps if s.status == "failed"]
    if failed:
        print("\n  Failed steps:")
        for s in failed:
            print(f"    {s.name}: {s.error or 'unknown error'}")

    print(sep)
