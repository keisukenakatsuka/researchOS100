# src/validation/model_blueprint.py
"""Model Prototyper (112 core logic).

Generates baseline implementation blueprints for high-priority hypotheses.
Combines hypothesis + method selection + available datasets into actionable
pseudocode and data pipeline designs.

See design.md Section 4.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class BaselineModel:
    type: str = ""
    specification: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaselineModel:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BlueprintVariable:
    name: str = ""
    role: str = ""  # dependent | independent | control | instrument
    operationalization: str = ""
    data_source_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BlueprintVariable:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Blueprint:
    hypothesis_id: str = ""
    hypothesis_statement: str = ""
    recommended_method: str = ""
    baseline_model: BaselineModel = field(default_factory=BaselineModel)
    variables: List[BlueprintVariable] = field(default_factory=list)
    pseudocode: str = ""
    libraries: List[str] = field(default_factory=list)
    data_pipeline: List[str] = field(default_factory=list)
    minimum_viable_test: str = ""
    limitations: List[str] = field(default_factory=list)
    estimated_complexity: str = ""  # low | medium | high
    next_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_statement": self.hypothesis_statement,
            "recommended_method": self.recommended_method,
            "baseline_model": self.baseline_model.to_dict(),
            "variables": [v.to_dict() for v in self.variables],
            "pseudocode": self.pseudocode,
            "libraries": self.libraries,
            "data_pipeline": self.data_pipeline,
            "minimum_viable_test": self.minimum_viable_test,
            "limitations": self.limitations,
            "estimated_complexity": self.estimated_complexity,
            "next_steps": self.next_steps,
        }


@dataclass
class ModelBlueprints:
    run_id: str = ""
    created_at: str = ""
    blueprints: List[Blueprint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "blueprints": [b.to_dict() for b in self.blueprints],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Model Blueprints",
            f"Run: {self.run_id} | Created: {self.created_at[:10]}",
            f"",
            f"Blueprints generated: {len(self.blueprints)}",
            f"",
        ]

        for i, bp in enumerate(self.blueprints):
            lines.append(f"---")
            lines.append(f"## Blueprint {i + 1}: {bp.hypothesis_statement[:80]}...")
            lines.append(f"")
            lines.append(f"- **Hypothesis ID**: `{bp.hypothesis_id}`")
            lines.append(f"- **Recommended Method**: {bp.recommended_method}")
            lines.append(f"- **Estimated Complexity**: {bp.estimated_complexity}")
            lines.append(f"")

            # Baseline model
            lines.append(f"### Baseline Model")
            lines.append(f"- **Type**: {bp.baseline_model.type}")
            if bp.baseline_model.specification:
                lines.append(f"- **Specification**: `{bp.baseline_model.specification}`")
            lines.append(f"- **Description**: {bp.baseline_model.description}")
            lines.append(f"")

            # Variables
            if bp.variables:
                lines.append(f"### Variables")
                lines.append(f"| Name | Role | Operationalization | Data Sources |")
                lines.append(f"|------|------|-------------------|-------------|")
                for v in bp.variables:
                    sources = ", ".join(f"`{s}`" for s in v.data_source_ids) if v.data_source_ids else "-"
                    lines.append(f"| {v.name} | {v.role} | {v.operationalization[:60]} | {sources} |")
                lines.append(f"")

            # Pseudocode
            if bp.pseudocode:
                lines.append(f"### Pseudocode")
                lines.append(f"```python")
                lines.append(bp.pseudocode)
                lines.append(f"```")
                lines.append(f"")

            # Libraries
            if bp.libraries:
                lines.append(f"### Required Libraries")
                lines.append(f"`{', '.join(bp.libraries)}`")
                lines.append(f"")

            # Data pipeline
            if bp.data_pipeline:
                lines.append(f"### Data Pipeline")
                for step in bp.data_pipeline:
                    lines.append(f"- {step}")
                lines.append(f"")

            # Minimum viable test
            if bp.minimum_viable_test:
                lines.append(f"### Minimum Viable Test")
                lines.append(bp.minimum_viable_test)
                lines.append(f"")

            # Limitations
            if bp.limitations:
                lines.append(f"### Limitations")
                for lim in bp.limitations:
                    lines.append(f"- {lim}")
                lines.append(f"")

            # Next steps
            if bp.next_steps:
                lines.append(f"### Next Steps")
                for step in bp.next_steps:
                    lines.append(f"- {step}")
                lines.append(f"")

        return "\n".join(lines)


# ------------------------------------------------------------------
# Target selection
# ------------------------------------------------------------------

def select_targets(
    portfolio: Dict[str, Any],
    max_count: int = 3,
) -> List[Dict[str, Any]]:
    """Select high_priority and promising hypotheses from portfolio.

    Returns list of scored_hypothesis dicts, sorted by composite_score desc.
    """
    scored = portfolio.get("scored_hypotheses", [])
    targets = [
        h for h in scored
        if h.get("recommendation") in ("high_priority", "promising")
    ]
    targets.sort(key=lambda h: h.get("composite_score", 0), reverse=True)
    selected = targets[:max_count]

    logger.info(
        "Selected %d/%d hypotheses (high_priority + promising, max %d)",
        len(selected), len(targets), max_count,
    )

    return selected


# ------------------------------------------------------------------
# Blueprint generation
# ------------------------------------------------------------------

_BLUEPRINT_SYSTEM = """\
あなたは実証研究の実装を支援するリサーチエンジニアです。
仮説を最小構成で検証するための具体的な実装方針を設計してください。

重要な指示:
- baseline_model は最もシンプルな検証モデルを設計する
- pseudocode は Python で、実際に実行可能に近い形で記述する
- libraries は標準的な Python パッケージのみ使用する
- data_pipeline は具体的なステップに分解する
- minimum_viable_test は「最低限のデータ・工数で何がわかるか」を明示する
- limitations は正直に記載する

出力は必ず JSON 形式で返してください。"""


def generate_blueprint(
    hypothesis: Dict[str, Any],
    method_selection: Optional[Dict[str, Any]],
    available_datasets: List[Dict[str, Any]],
    llm_client: Any,
) -> Blueprint:
    """Generate a model blueprint for a single hypothesis."""

    hyp_id = hypothesis.get("hypothesis_id", "")
    hyp_stmt = hypothesis.get("statement", hypothesis.get("hypothesis_statement", ""))

    # Build method context
    method_text = "（手法推薦なし — 仮説に適した手法を提案してください）"
    primary_method = ""
    if method_selection:
        primary_method = method_selection.get("primary_method", "")
        method_text = (
            f"Primary: {primary_method}\n"
            f"Rationale: {method_selection.get('primary_rationale', '')}\n"
            f"Secondary: {method_selection.get('secondary_method', '')}\n"
            f"Confidence: {method_selection.get('overall_confidence', '')}"
        )

    # Build dataset context
    if available_datasets:
        ds_lines = []
        for ds in available_datasets[:15]:  # limit to avoid token overflow
            ds_lines.append(
                f"- {ds.get('name', '')} ({ds.get('availability_status', '')}, "
                f"cost: {ds.get('cost_tier', '')}) [{ds.get('dataset_id', '')}]"
            )
        dataset_text = "\n".join(ds_lines)
    else:
        dataset_text = "（利用可能データセット情報なし — 111 Dataset Registry を参照してください）"

    user_msg = (
        f"## 仮説\n"
        f"ID: {hyp_id}\n"
        f"{hyp_stmt}\n\n"
        f"## 推薦手法\n{method_text}\n\n"
        f"## 利用可能データ\n{dataset_text}\n\n"
        f"## Instructions\n"
        f"この仮説を最小構成で検証するための実装方針を以下の JSON 形式で出力してください:\n\n"
        f'{{\n'
        f'  "baseline_model": {{\n'
        f'    "type": "モデルの種類（例: two-way fixed effects DID）",\n'
        f'    "specification": "数式表現（例: Y_it = α + βD_it + γX_it + μ_i + λ_t + ε_it）",\n'
        f'    "description": "モデルの概要説明（日本語）"\n'
        f'  }},\n'
        f'  "variables": [\n'
        f'    {{\n'
        f'      "name": "変数名",\n'
        f'      "role": "dependent | independent | control | instrument",\n'
        f'      "operationalization": "操作化の定義",\n'
        f'      "data_source_ids": ["ds__xxx"]\n'
        f'    }}\n'
        f'  ],\n'
        f'  "pseudocode": "Python疑似コード（複数行）",\n'
        f'  "libraries": ["pandas", "statsmodels"],\n'
        f'  "data_pipeline": ["Step 1: ...", "Step 2: ..."],\n'
        f'  "minimum_viable_test": "最低限の検証で何がわかるか",\n'
        f'  "limitations": ["限界1", "限界2"],\n'
        f'  "estimated_complexity": "low | medium | high",\n'
        f'  "next_steps": ["次のアクション1", "次のアクション2"]\n'
        f'}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _BLUEPRINT_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Blueprint LLM call failed for %s: %s", hyp_id, e)
        return Blueprint(
            hypothesis_id=hyp_id,
            hypothesis_statement=hyp_stmt,
            recommended_method=primary_method,
            limitations=[f"LLM call failed: {e}"],
        )

    resp_text = _extract_text(resp)
    parsed = _parse_json(resp_text)
    if not parsed:
        logger.error("Blueprint JSON parse failed for %s. Raw: %s", hyp_id, resp_text[:300])
        return Blueprint(
            hypothesis_id=hyp_id,
            hypothesis_statement=hyp_stmt,
            recommended_method=primary_method,
            limitations=["JSON parse failed"],
        )

    usage = resp.get("usage", {})
    logger.info(
        "Blueprint generated for %s (in=%d, out=%d tokens)",
        hyp_id, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
    )

    # Parse into Blueprint
    bm_raw = parsed.get("baseline_model", {})
    variables = [BlueprintVariable.from_dict(v) for v in parsed.get("variables", [])]

    return Blueprint(
        hypothesis_id=hyp_id,
        hypothesis_statement=hyp_stmt,
        recommended_method=primary_method or parsed.get("recommended_method", ""),
        baseline_model=BaselineModel.from_dict(bm_raw),
        variables=variables,
        pseudocode=parsed.get("pseudocode", ""),
        libraries=parsed.get("libraries", []),
        data_pipeline=parsed.get("data_pipeline", []),
        minimum_viable_test=parsed.get("minimum_viable_test", ""),
        limitations=parsed.get("limitations", []),
        estimated_complexity=parsed.get("estimated_complexity", ""),
        next_steps=parsed.get("next_steps", []),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def find_method_for_hypothesis(
    hypothesis_id: str,
    method_selections: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the method_selection entry for a given hypothesis_id."""
    for ms in method_selections:
        if ms.get("hypothesis_id") == hypothesis_id:
            return ms
    return None


def _extract_text(resp: Dict[str, Any]) -> str:
    for block in resp.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
