# src/lit_review/validation.py
"""Block 5: Validation Designer (090).

Generates concrete research designs for high-priority hypotheses,
including identification strategy, data requirements, risks, and next steps.

Each hypothesis gets a dedicated LLM call for deep design analysis.

Usage::

    from src.lit_review.validation import design_validations

    result = design_validations(run_dir, llm_client=client)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Risk:
    risk_type: str       # internal_validity | external_validity | data | execution
    description: str
    severity: str        # high | medium | low
    mitigation: str


@dataclass
class DataRequirements:
    dependent_variable: str = ""
    independent_variables: List[str] = field(default_factory=list)
    control_variables: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    sample_description: str = ""
    time_period: str = ""
    sample_size_guidance: str = ""
    feasibility_note: str = ""


@dataclass
class ValidationDesign:
    hypothesis_id: str
    hypothesis_statement: str
    recommendation: str

    design_type: str = ""           # experimental | quasi_experimental | observational | qualitative | mixed
    design_description: str = ""

    identification_strategy: str = ""
    identification_rationale: str = ""
    required_assumptions: List[str] = field(default_factory=list)

    data_requirements: DataRequirements = field(default_factory=DataRequirements)

    key_risks: List[Risk] = field(default_factory=list)
    feasibility_assessment: str = ""
    next_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ValidationResult:
    run_id: str
    rq_title: str = ""
    designs_generated: int = 0
    validation_designs: List[ValidationDesign] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rq_title": self.rq_title,
            "designs_generated": self.designs_generated,
            "validation_designs": [d.to_dict() for d in self.validation_designs],
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# Input collection
# ------------------------------------------------------------------

def collect_inputs(run_dir: Path) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {
        "hypotheses": [],
        "portfolio": [],
        "hypothesis_assumptions": [],
        "methods": [],
        "rq_title": "",
    }

    rq_path = run_dir / "rq_context.json"
    if rq_path.exists():
        inputs["rq_title"] = json.loads(rq_path.read_text()).get("title", "")

    hyp_path = run_dir / "hypotheses.json"
    if hyp_path.exists():
        inputs["hypotheses"] = json.loads(hyp_path.read_text()).get("hypotheses", [])

    port_path = run_dir / "hypothesis_portfolio.json"
    if port_path.exists():
        inputs["portfolio"] = json.loads(port_path.read_text()).get("scored_hypotheses", [])

    asmp_path = run_dir / "assumptions.json"
    if asmp_path.exists():
        inputs["hypothesis_assumptions"] = json.loads(asmp_path.read_text()).get("hypothesis_assumptions", [])

    ls_path = run_dir / "landscape.json"
    if ls_path.exists():
        ls = json.loads(ls_path.read_text())
        dims = ls.get("methodological_landscape", {})
        for cat_items in dims.values():
            if isinstance(cat_items, list):
                for m in cat_items:
                    if isinstance(m, dict):
                        inputs["methods"].append(m.get("name", ""))
                    elif isinstance(m, str):
                        inputs["methods"].append(m)

    return inputs


def select_target_hypotheses(
    inputs: Dict[str, Any],
    *,
    max_designs: int = 5,
) -> List[Dict[str, Any]]:
    """Select hypotheses for validation design based on portfolio recommendation."""
    portfolio = inputs.get("portfolio", [])
    hypotheses = inputs.get("hypotheses", [])
    assumptions = inputs.get("hypothesis_assumptions", [])

    # Build lookup by hypothesis_id
    hyp_by_id = {h.get("hypothesis_id", ""): h for h in hypotheses}
    asmp_by_id = {a.get("hypothesis_id", ""): a for a in assumptions}

    targets = []
    for scored in portfolio:
        rec = scored.get("recommendation", "")
        if rec not in ("high_priority", "promising"):
            continue

        hyp_id = scored.get("hypothesis_id", "")
        hyp = hyp_by_id.get(hyp_id, {})
        asmp = asmp_by_id.get(hyp_id, {})

        if not hyp:
            continue

        targets.append({
            "hypothesis_id": hyp_id,
            "hypothesis_statement": hyp.get("hypothesis_statement", ""),
            "strategy": hyp.get("strategy", ""),
            "suggested_test": hyp.get("suggested_test", ""),
            "recommendation": rec,
            "scores": scored.get("scores", {}),
            "overall_vulnerability": scored.get("overall_vulnerability", ""),
            "assumptions": asmp.get("assumptions", []),
            "weakest_assumption": asmp.get("weakest_assumption", ""),
        })

        if len(targets) >= max_designs:
            break

    logger.info("Selected %d target hypotheses (high_priority + promising)", len(targets))
    return targets


# ------------------------------------------------------------------
# LLM design generation
# ------------------------------------------------------------------

def _parse_json_response(text: str) -> Optional[Any]:
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


_DESIGN_SYSTEM = """\
あなたは研究デザインの専門家です。
研究仮説を検証するための具体的な研究デザインを設計してください。

重要な指示:
- 識別戦略は仮説の前提条件と整合する必要があります
- データ要件は具体的に記述してください（変数名、データソース名、期間）
- key risks は internal validity / external validity / data / execution の観点で整理してください
- next steps は実行可能な具体的アクションにしてください
- 研究の実行可能性（feasibility）を現実的に評価してください"""


def generate_single_design(
    target: Dict[str, Any],
    methods_context: List[str],
    rq_title: str,
    *,
    llm_client: Any,
) -> Optional[Dict]:
    """Generate research design for a single hypothesis."""
    # Format assumptions
    asmp_lines = []
    for a in target.get("assumptions", []):
        cat = a.get("category", "?")
        stmt = a.get("statement", "")
        vuln = a.get("vulnerability", "?")
        asmp_lines.append(f"- [{cat}, {vuln}] {stmt}")

    user_msg = (
        f"## RQ: {rq_title}\n\n"
        f"## 仮説\n"
        f"{target['hypothesis_statement']}\n"
        f"- Strategy: {target.get('strategy', '')}\n"
        f"- Suggested test: {target.get('suggested_test', '')}\n"
        f"- Recommendation: {target.get('recommendation', '')}\n"
        f"- Vulnerability: {target.get('overall_vulnerability', '')}\n\n"
        f"## この仮説の前提条件\n" + ("\n".join(asmp_lines) if asmp_lines else "(なし)") + "\n\n"
        f"## 利用可能な方法論\n" + (", ".join(methods_context[:10]) if methods_context else "(なし)") + "\n\n"
        f"## 指示\nこの仮説を検証する研究デザインを設計してください。\n\n"
        f"出力形式 (JSON):\n"
        f'{{\n'
        f'  "design_type": "quasi_experimental | observational | experimental | qualitative | mixed",\n'
        f'  "design_description": "研究デザインの概要（日本語3-5文）",\n'
        f'  "identification_strategy": "DID | IV | PSM | RDD | synthetic_control | case_study | etc.",\n'
        f'  "identification_rationale": "なぜこの識別戦略を選択するか",\n'
        f'  "required_assumptions": ["この識別戦略が要求する前提条件1", "前提条件2"],\n'
        f'  "data_requirements": {{\n'
        f'    "dependent_variable": "従属変数",\n'
        f'    "independent_variables": ["独立変数1"],\n'
        f'    "control_variables": ["統制変数1"],\n'
        f'    "data_sources": ["具体的なデータベース名"],\n'
        f'    "sample_description": "サンプルの特徴",\n'
        f'    "time_period": "必要な期間",\n'
        f'    "sample_size_guidance": "サンプルサイズの目安と根拠",\n'
        f'    "feasibility_note": "データアクセスの現実的評価"\n'
        f'  }},\n'
        f'  "key_risks": [\n'
        f'    {{"risk_type": "internal_validity | external_validity | data | execution", "description": "...", "severity": "high | medium | low", "mitigation": "..."}}\n'
        f'  ],\n'
        f'  "feasibility_assessment": "この研究デザインの実行可能性の総合評価（日本語2-3文）",\n'
        f'  "next_steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."]\n'
        f'}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _DESIGN_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Design LLM call failed for %s: %s", target["hypothesis_id"], e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Design LLM [%s]: in=%d, out=%d tokens",
                target["hypothesis_id"][:20], usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    return _parse_json_response(resp_text)


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------

def _build_design(target: Dict, raw: Dict) -> ValidationDesign:
    dr = raw.get("data_requirements", {})
    risks = [
        Risk(
            risk_type=r.get("risk_type", ""),
            description=r.get("description", ""),
            severity=r.get("severity", "medium"),
            mitigation=r.get("mitigation", ""),
        )
        for r in raw.get("key_risks", [])
    ]

    return ValidationDesign(
        hypothesis_id=target["hypothesis_id"],
        hypothesis_statement=target["hypothesis_statement"],
        recommendation=target.get("recommendation", ""),
        design_type=raw.get("design_type", ""),
        design_description=raw.get("design_description", ""),
        identification_strategy=raw.get("identification_strategy", ""),
        identification_rationale=raw.get("identification_rationale", ""),
        required_assumptions=raw.get("required_assumptions", []),
        data_requirements=DataRequirements(
            dependent_variable=dr.get("dependent_variable", ""),
            independent_variables=dr.get("independent_variables", []),
            control_variables=dr.get("control_variables", []),
            data_sources=dr.get("data_sources", []),
            sample_description=dr.get("sample_description", ""),
            time_period=dr.get("time_period", ""),
            sample_size_guidance=dr.get("sample_size_guidance", ""),
            feasibility_note=dr.get("feasibility_note", ""),
        ),
        key_risks=risks,
        feasibility_assessment=raw.get("feasibility_assessment", ""),
        next_steps=raw.get("next_steps", []),
    )


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def design_validations(
    run_dir: Path,
    *,
    llm_client: Any,
    max_designs: int = 5,
) -> ValidationResult:
    run_id = run_dir.name
    now_iso = datetime.now(timezone.utc).isoformat()

    inputs = collect_inputs(run_dir)
    targets = select_target_hypotheses(inputs, max_designs=max_designs)

    if not targets:
        logger.warning("No target hypotheses for validation design")
        return ValidationResult(run_id=run_id, rq_title=inputs.get("rq_title", ""))

    designs = []
    for i, target in enumerate(targets):
        logger.info("[%d/%d] Designing validation for: %s", i + 1, len(targets), target["hypothesis_statement"][:50])

        raw = generate_single_design(
            target, inputs.get("methods", []), inputs.get("rq_title", ""),
            llm_client=llm_client,
        )

        if raw:
            designs.append(_build_design(target, raw))
        else:
            logger.warning("Design generation failed for %s", target["hypothesis_id"])

    return ValidationResult(
        run_id=run_id,
        rq_title=inputs.get("rq_title", ""),
        designs_generated=len(designs),
        validation_designs=designs,
        metadata={"created_at": now_iso, "model": _MODEL, "max_designs": max_designs},
    )


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: ValidationResult) -> str:
    lines = [
        f"# Validation Designs",
        f"",
        f"## RQ: {result.rq_title}",
        f"",
        f"**Designs generated: {result.designs_generated}**",
        f"",
    ]

    for i, d in enumerate(result.validation_designs, 1):
        lines.extend([
            f"---",
            f"",
            f"## Design {i}: {d.hypothesis_statement[:70]}",
            f"",
            f"- **Recommendation**: {d.recommendation}",
            f"- **Design type**: {d.design_type}",
            f"- **Identification strategy**: {d.identification_strategy}",
            f"",
            f"### Design Description",
            f"",
            d.design_description,
            f"",
            f"### Identification Strategy",
            f"",
            f"**Strategy**: {d.identification_strategy}",
            f"",
            f"**Rationale**: {d.identification_rationale}",
            f"",
        ])

        if d.required_assumptions:
            lines.append(f"**Required Assumptions**:")
            for a in d.required_assumptions:
                lines.append(f"- {a}")
            lines.append(f"")

        # Data Requirements
        dr = d.data_requirements
        lines.extend([
            f"### Data Requirements",
            f"",
            f"- **Dependent variable**: {dr.dependent_variable}",
            f"- **Independent variables**: {', '.join(dr.independent_variables)}",
            f"- **Control variables**: {', '.join(dr.control_variables)}",
            f"- **Data sources**: {', '.join(dr.data_sources)}",
            f"- **Sample**: {dr.sample_description}",
            f"- **Time period**: {dr.time_period}",
            f"- **Sample size**: {dr.sample_size_guidance}",
            f"- **Feasibility**: {dr.feasibility_note}",
            f"",
        ])

        # Feasibility Assessment
        if d.feasibility_assessment:
            lines.extend([
                f"### Feasibility Assessment",
                f"",
                d.feasibility_assessment,
                f"",
            ])

        # Key Risks
        if d.key_risks:
            lines.extend([f"### Key Risks", f""])
            for r in d.key_risks:
                lines.append(f"- **[{r.severity}] {r.risk_type}**: {r.description}")
                if r.mitigation:
                    lines.append(f"  - Mitigation: {r.mitigation}")
            lines.append(f"")

        # Next Steps
        if d.next_steps:
            lines.extend([f"### Next Steps", f""])
            for step in d.next_steps:
                lines.append(f"1. {step}")
            lines.append(f"")

    return "\n".join(lines)
