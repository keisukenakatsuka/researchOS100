# src/lit_review/data_reqs.py
"""Block 5: Data Requirements (091).

Details data requirements from validation designs: variable
operationalization, source evaluation, alternatives, and risks.

Each design gets a dedicated LLM call for deep analysis.

Usage::

    from src.lit_review.data_reqs import plan_data_requirements

    result = plan_data_requirements(run_dir, llm_client=client)
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
class DataSource:
    name: str
    provider: str = ""
    acquisition_difficulty: str = ""  # open | restricted | commercial | unavailable
    cost_estimate: str = ""           # free | low | high | negotiation_required
    coverage: str = ""
    update_frequency: str = ""
    limitations: str = ""


@dataclass
class VariableSpec:
    name: str
    concept: str = ""
    operationalization: str = ""
    variable_type: str = ""          # dependent | independent | control | instrument
    unit_of_analysis: str = ""       # firm | fund | country | region | year
    temporal_resolution: str = ""    # annual | quarterly | monthly | cross_sectional
    primary_source: DataSource = field(default_factory=DataSource)
    alternative_sources: List[DataSource] = field(default_factory=list)
    missing_risk: str = ""           # high | medium | low
    missing_pattern: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DesignDataPlan:
    hypothesis_id: str
    hypothesis_statement: str
    design_type: str = ""
    identification_strategy: str = ""
    variables: List[VariableSpec] = field(default_factory=list)
    collection_priority: List[str] = field(default_factory=list)
    overall_feasibility: str = ""    # high | medium | low
    critical_data_gaps: List[str] = field(default_factory=list)
    recommended_first_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class DataRequirementsResult:
    run_id: str
    rq_title: str = ""
    designs_detailed: int = 0
    data_plans: List[DesignDataPlan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rq_title": self.rq_title,
            "designs_detailed": self.designs_detailed,
            "data_plans": [p.to_dict() for p in self.data_plans],
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# Input collection
# ------------------------------------------------------------------

def collect_inputs(run_dir: Path) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {"validation_designs": [], "data_assumptions": [], "rq_title": ""}

    rq_path = run_dir / "rq_context.json"
    if rq_path.exists():
        inputs["rq_title"] = json.loads(rq_path.read_text()).get("title", "")

    vd_path = run_dir / "validation_designs.json"
    if vd_path.exists():
        inputs["validation_designs"] = json.loads(vd_path.read_text()).get("validation_designs", [])

    asmp_path = run_dir / "assumptions.json"
    if asmp_path.exists():
        ha_list = json.loads(asmp_path.read_text()).get("hypothesis_assumptions", [])
        for ha in ha_list:
            for a in ha.get("assumptions", []):
                if a.get("category") == "data":
                    inputs["data_assumptions"].append({
                        "hypothesis_id": ha.get("hypothesis_id", ""),
                        "statement": a.get("statement", ""),
                        "testability": a.get("testability", ""),
                    })

    logger.info("Collected: %d designs, %d data assumptions",
                len(inputs["validation_designs"]), len(inputs["data_assumptions"]))
    return inputs


# ------------------------------------------------------------------
# LLM data requirement detailing
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


_SYSTEM = """\
あなたはデータ収集と研究方法論の専門家です。
研究デザインに必要なデータの取得計画を具体化してください。

各変数について:
- operationalization: 抽象概念を具体的な測定指標に変換
- unit_of_analysis: 分析単位（firm / fund / country / region 等）
- temporal_resolution: 時間的粒度（annual / quarterly / monthly / cross_sectional）
- primary_source: 第一候補データソース（名前、取得難易度、コスト、カバレッジ、制約）
- alternative_sources: 代替データソース（1-2件）
- missing_risk / missing_pattern: 欠損リスクとパターン

取得難易度は以下の4段階:
- open: 無料公開
- restricted: 登録・申請が必要だが無料
- commercial: 有料ライセンス
- unavailable: 現時点で入手不可能

先行研究で使われている操作化を参照してください。"""


def detail_single_design(
    design: Dict[str, Any],
    data_assumptions: List[Dict],
    rq_title: str,
    *,
    llm_client: Any,
) -> Optional[Dict]:
    """Detail data requirements for a single validation design."""
    dr = design.get("data_requirements", {})
    hyp_id = design.get("hypothesis_id", "")

    # Filter relevant data assumptions
    relevant_asmp = [a for a in data_assumptions if a.get("hypothesis_id") == hyp_id]
    asmp_text = "\n".join(f"- {a['statement']}" for a in relevant_asmp) if relevant_asmp else "(なし)"

    user_msg = (
        f"## RQ: {rq_title}\n\n"
        f"## 仮説: {design.get('hypothesis_statement', '')}\n"
        f"## デザイン: {design.get('design_type', '')} / {design.get('identification_strategy', '')}\n\n"
        f"## 090 のデータ要件概要\n"
        f"- Dependent: {dr.get('dependent_variable', '')}\n"
        f"- Independent: {', '.join(dr.get('independent_variables', []))}\n"
        f"- Control: {', '.join(dr.get('control_variables', []))}\n"
        f"- Sources: {', '.join(dr.get('data_sources', []))}\n"
        f"- Period: {dr.get('time_period', '')}\n"
        f"- Sample size: {dr.get('sample_size_guidance', '')}\n\n"
        f"## Data Assumptions (088)\n{asmp_text}\n\n"
        f"## 指示\n各変数を詳細化してください。\n\n"
        f"出力形式 (JSON):\n"
        f'{{\n'
        f'  "variables": [\n'
        f'    {{\n'
        f'      "name": "変数名",\n'
        f'      "concept": "測定対象の概念",\n'
        f'      "operationalization": "具体的な測定指標",\n'
        f'      "variable_type": "dependent | independent | control | instrument",\n'
        f'      "unit_of_analysis": "firm | fund | country | region",\n'
        f'      "temporal_resolution": "annual | quarterly | monthly | cross_sectional",\n'
        f'      "primary_source": {{"name": "DB名", "provider": "提供元", "acquisition_difficulty": "open|restricted|commercial|unavailable", "cost_estimate": "free|low|high|negotiation_required", "coverage": "カバレッジ", "update_frequency": "年次等", "limitations": "制約"}},\n'
        f'      "alternative_sources": [{{"name": "代替DB", "provider": "提供元", "acquisition_difficulty": "...", "cost_estimate": "...", "coverage": "...", "update_frequency": "...", "limitations": "..."}}],\n'
        f'      "missing_risk": "high | medium | low",\n'
        f'      "missing_pattern": "欠損パターンの説明",\n'
        f'      "notes": "補足"\n'
        f'    }}\n'
        f'  ],\n'
        f'  "collection_priority": ["変数名1(P1)", "変数名2(P2)"],\n'
        f'  "overall_feasibility": "high | medium | low",\n'
        f'  "critical_data_gaps": ["取得困難なデータ"],\n'
        f'  "recommended_first_steps": ["Step 1", "Step 2"]\n'
        f'}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 8192,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Data reqs LLM failed for %s: %s", hyp_id, e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Data reqs LLM [%s]: in=%d, out=%d tokens",
                hyp_id[:20], usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    return _parse_json_response(resp_text)


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------

def _build_plan(design: Dict, raw: Dict) -> DesignDataPlan:
    variables = []
    for rv in raw.get("variables", []):
        ps = rv.get("primary_source", {})
        alts = [
            DataSource(
                name=a.get("name", ""), provider=a.get("provider", ""),
                acquisition_difficulty=a.get("acquisition_difficulty", ""),
                cost_estimate=a.get("cost_estimate", ""),
                coverage=a.get("coverage", ""), update_frequency=a.get("update_frequency", ""),
                limitations=a.get("limitations", ""),
            )
            for a in rv.get("alternative_sources", [])
        ]
        variables.append(VariableSpec(
            name=rv.get("name", ""),
            concept=rv.get("concept", ""),
            operationalization=rv.get("operationalization", ""),
            variable_type=rv.get("variable_type", ""),
            unit_of_analysis=rv.get("unit_of_analysis", ""),
            temporal_resolution=rv.get("temporal_resolution", ""),
            primary_source=DataSource(
                name=ps.get("name", ""), provider=ps.get("provider", ""),
                acquisition_difficulty=ps.get("acquisition_difficulty", ""),
                cost_estimate=ps.get("cost_estimate", ""),
                coverage=ps.get("coverage", ""), update_frequency=ps.get("update_frequency", ""),
                limitations=ps.get("limitations", ""),
            ),
            alternative_sources=alts,
            missing_risk=rv.get("missing_risk", ""),
            missing_pattern=rv.get("missing_pattern", ""),
            notes=rv.get("notes", ""),
        ))

    return DesignDataPlan(
        hypothesis_id=design.get("hypothesis_id", ""),
        hypothesis_statement=design.get("hypothesis_statement", ""),
        design_type=design.get("design_type", ""),
        identification_strategy=design.get("identification_strategy", ""),
        variables=variables,
        collection_priority=raw.get("collection_priority", []),
        overall_feasibility=raw.get("overall_feasibility", "medium"),
        critical_data_gaps=raw.get("critical_data_gaps", []),
        recommended_first_steps=raw.get("recommended_first_steps", []),
    )


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def plan_data_requirements(
    run_dir: Path,
    *,
    llm_client: Any,
    max_designs: int = 5,
) -> DataRequirementsResult:
    run_id = run_dir.name
    now_iso = datetime.now(timezone.utc).isoformat()

    inputs = collect_inputs(run_dir)
    designs = inputs["validation_designs"][:max_designs]

    if not designs:
        logger.warning("No validation designs found")
        return DataRequirementsResult(run_id=run_id, rq_title=inputs.get("rq_title", ""))

    plans = []
    for i, design in enumerate(designs):
        logger.info("[%d/%d] Detailing data for: %s",
                    i + 1, len(designs), design.get("hypothesis_statement", "")[:50])
        raw = detail_single_design(
            design, inputs["data_assumptions"], inputs.get("rq_title", ""),
            llm_client=llm_client,
        )
        if raw:
            plans.append(_build_plan(design, raw))
        else:
            logger.warning("Data requirement detailing failed for %s", design.get("hypothesis_id", ""))

    return DataRequirementsResult(
        run_id=run_id,
        rq_title=inputs.get("rq_title", ""),
        designs_detailed=len(plans),
        data_plans=plans,
        metadata={"created_at": now_iso, "model": _MODEL},
    )


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: DataRequirementsResult) -> str:
    lines = [
        f"# Data Requirements",
        f"",
        f"## RQ: {result.rq_title}",
        f"",
        f"**Designs detailed: {result.designs_detailed}**",
        f"",
    ]

    for i, plan in enumerate(result.data_plans, 1):
        lines.extend([
            f"---",
            f"",
            f"## Design {i}: {plan.hypothesis_statement[:65]}",
            f"",
            f"- **Design**: {plan.design_type} / {plan.identification_strategy}",
            f"- **Overall feasibility**: {plan.overall_feasibility}",
            f"",
        ])

        # Variables table
        lines.append(f"### Variables")
        lines.append(f"")
        lines.append(f"| Variable | Type | Operationalization | Unit | Resolution | Source | Difficulty | Missing |")
        lines.append(f"|----------|------|--------------------|------|------------|--------|------------|---------|")
        for v in plan.variables:
            lines.append(
                f"| {v.name} | {v.variable_type} | {v.operationalization[:40]} | "
                f"{v.unit_of_analysis} | {v.temporal_resolution} | "
                f"{v.primary_source.name[:20]} | {v.primary_source.acquisition_difficulty} | {v.missing_risk} |"
            )
        lines.append(f"")

        # Variable details
        for v in plan.variables:
            lines.append(f"#### {v.name}")
            lines.append(f"")
            lines.append(f"- **Concept**: {v.concept}")
            lines.append(f"- **Operationalization**: {v.operationalization}")
            lines.append(f"- **Unit**: {v.unit_of_analysis} | **Resolution**: {v.temporal_resolution}")
            lines.append(f"- **Primary**: {v.primary_source.name} ({v.primary_source.provider}) — {v.primary_source.acquisition_difficulty}, {v.primary_source.cost_estimate}")
            if v.primary_source.coverage:
                lines.append(f"  - Coverage: {v.primary_source.coverage}")
            if v.primary_source.limitations:
                lines.append(f"  - Limitations: {v.primary_source.limitations}")
            for alt in v.alternative_sources:
                lines.append(f"- **Alt**: {alt.name} ({alt.provider}) — {alt.acquisition_difficulty}, {alt.cost_estimate}")
            lines.append(f"- **Missing risk**: {v.missing_risk} — {v.missing_pattern}")
            lines.append(f"")

        # Priority & gaps
        if plan.collection_priority:
            lines.extend([f"### Collection Priority", f""])
            for j, p in enumerate(plan.collection_priority, 1):
                lines.append(f"{j}. {p}")
            lines.append(f"")

        if plan.critical_data_gaps:
            lines.extend([f"### Critical Data Gaps", f""])
            for g in plan.critical_data_gaps:
                lines.append(f"- {g}")
            lines.append(f"")

        if plan.recommended_first_steps:
            lines.extend([f"### Recommended First Steps", f""])
            for s in plan.recommended_first_steps:
                lines.append(f"1. {s}")
            lines.append(f"")

    return "\n".join(lines)
