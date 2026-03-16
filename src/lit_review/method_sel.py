# src/lit_review/method_sel.py
"""Block 5: Method Selector (092).

Compares 2-3 candidate methods per hypothesis and recommends
primary + secondary (robustness check) methods.

Usage::

    from src.lit_review.method_sel import select_methods

    result = select_methods(run_dir, llm_client=client)
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
class MethodEvaluation:
    method_name: str
    method_category: str = ""     # quasi_experimental | iv | matching | panel | qualitative
    scores: Dict[str, int] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    why_fit: str = ""             # why this method fits the hypothesis
    failure_mode: str = ""        # how this method could fail
    assumption_fit: str = ""
    data_fit: str = ""
    implementation_notes: str = ""


@dataclass
class MethodSelection:
    hypothesis_id: str
    hypothesis_statement: str
    candidates: List[MethodEvaluation] = field(default_factory=list)
    primary_method: str = ""
    primary_rationale: str = ""   # why primary is better than alternatives
    secondary_method: str = ""
    secondary_rationale: str = ""
    overall_confidence: str = ""  # high | medium | low

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MethodSelectionResult:
    run_id: str
    rq_title: str = ""
    selections_generated: int = 0
    method_selections: List[MethodSelection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rq_title": self.rq_title,
            "selections_generated": self.selections_generated,
            "method_selections": [s.to_dict() for s in self.method_selections],
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# Input collection
# ------------------------------------------------------------------

def collect_inputs(run_dir: Path) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {
        "validation_designs": [],
        "data_plans": [],
        "identification_assumptions": [],
        "rq_title": "",
    }

    rq_path = run_dir / "rq_context.json"
    if rq_path.exists():
        inputs["rq_title"] = json.loads(rq_path.read_text()).get("title", "")

    vd_path = run_dir / "validation_designs.json"
    if vd_path.exists():
        inputs["validation_designs"] = json.loads(vd_path.read_text()).get("validation_designs", [])

    dr_path = run_dir / "data_requirements.json"
    if dr_path.exists():
        inputs["data_plans"] = json.loads(dr_path.read_text()).get("data_plans", [])

    asmp_path = run_dir / "assumptions.json"
    if asmp_path.exists():
        for ha in json.loads(asmp_path.read_text()).get("hypothesis_assumptions", []):
            for a in ha.get("assumptions", []):
                if a.get("category") == "identification":
                    inputs["identification_assumptions"].append({
                        "hypothesis_id": ha.get("hypothesis_id", ""),
                        "statement": a.get("statement", ""),
                        "vulnerability": a.get("vulnerability", ""),
                    })

    logger.info("Collected: %d designs, %d data plans, %d id assumptions",
                len(inputs["validation_designs"]), len(inputs["data_plans"]),
                len(inputs["identification_assumptions"]))
    return inputs


# ------------------------------------------------------------------
# LLM method comparison
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
あなたは計量経済学と研究方法論の専門家です。
研究仮説の検証手法を比較評価し、最適な手法を推薦してください。

重要な指示:
- 異なるカテゴリの手法を含めてください（例: quasi-experimental と matching の両方）
- 各手法の強み・弱みを具体的に記述してください
- データ制約を踏まえた実行可能性を評価してください
- Primary method がなぜ他の候補より適しているかを明確に説明してください
- Secondary method は primary の弱点を補完するものを選んでください

5つの評価軸（各1-5）:
- identification_strength: 因果推論の内的妥当性
- data_fit: 利用可能データとの適合度
- feasibility: 実装の実行可能性
- interpretability: 結果の解釈しやすさ
- robustness: 前提条件への頑健性"""


def compare_methods_for_hypothesis(
    design: Dict[str, Any],
    data_plan: Optional[Dict[str, Any]],
    id_assumptions: List[Dict],
    rq_title: str,
    *,
    llm_client: Any,
) -> Optional[Dict]:
    """Compare methods for a single hypothesis."""
    # Build data constraint summary from data_plan
    data_constraints = []
    if data_plan:
        for v in data_plan.get("variables", []):
            ps = v.get("primary_source", {})
            diff = ps.get("acquisition_difficulty", "")
            risk = v.get("missing_risk", "")
            if diff in ("commercial", "unavailable") or risk == "high":
                data_constraints.append(f"{v.get('name', '')}: {diff}, missing={risk}")

    asmp_lines = [a["statement"] for a in id_assumptions]

    user_msg = (
        f"## RQ: {rq_title}\n\n"
        f"## 仮説: {design.get('hypothesis_statement', '')}\n"
        f"## 現在提案の手法: {design.get('identification_strategy', '')}\n"
        f"## デザイン: {design.get('design_type', '')}\n\n"
        f"## Identification Assumptions\n" +
        ("\n".join(f"- {a}" for a in asmp_lines) if asmp_lines else "(なし)") + "\n\n"
        f"## データ制約\n" +
        ("\n".join(f"- {c}" for c in data_constraints) if data_constraints else "(なし)") + "\n\n"
        f"## 指示\nこの仮説を検証する手法を 2–3 件比較してください。\n"
        f"異なるカテゴリの手法を含めてください。\n\n"
        f"出力形式 (JSON):\n"
        f'{{\n'
        f'  "candidates": [\n'
        f'    {{\n'
        f'      "method_name": "手法名",\n'
        f'      "method_category": "quasi_experimental | iv | matching | panel | qualitative",\n'
        f'      "scores": {{"identification_strength": 4, "data_fit": 3, "feasibility": 4, "interpretability": 5, "robustness": 3}},\n'
        f'      "strengths": ["強み1"],\n'
        f'      "weaknesses": ["弱み1"],\n'
        f'      "why_fit": "なぜこの手法がこの仮説に適しているか",\n'
        f'      "failure_mode": "この手法が失敗するシナリオ",\n'
        f'      "assumption_fit": "identification assumptions との整合性",\n'
        f'      "data_fit": "利用可能データとの適合度",\n'
        f'      "implementation_notes": "実装上の注意点"\n'
        f'    }}\n'
        f'  ],\n'
        f'  "primary_method": "推薦する主手法名",\n'
        f'  "primary_rationale": "なぜ primary が他候補より適しているか（具体的に）",\n'
        f'  "secondary_method": "robustness check 用の副手法名",\n'
        f'  "secondary_rationale": "なぜこの副手法が primary の弱点を補完するか",\n'
        f'  "overall_confidence": "high | medium | low"\n'
        f'}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Method selection LLM failed for %s: %s", design.get("hypothesis_id", ""), e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Method LLM [%s]: in=%d, out=%d tokens",
                design.get("hypothesis_id", "")[:20],
                usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    return _parse_json_response(resp_text)


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------

def _build_selection(design: Dict, raw: Dict) -> MethodSelection:
    candidates = []
    for rc in raw.get("candidates", []):
        candidates.append(MethodEvaluation(
            method_name=rc.get("method_name", ""),
            method_category=rc.get("method_category", ""),
            scores=rc.get("scores", {}),
            strengths=rc.get("strengths", []),
            weaknesses=rc.get("weaknesses", []),
            why_fit=rc.get("why_fit", ""),
            failure_mode=rc.get("failure_mode", ""),
            assumption_fit=rc.get("assumption_fit", ""),
            data_fit=rc.get("data_fit", ""),
            implementation_notes=rc.get("implementation_notes", ""),
        ))

    return MethodSelection(
        hypothesis_id=design.get("hypothesis_id", ""),
        hypothesis_statement=design.get("hypothesis_statement", ""),
        candidates=candidates,
        primary_method=raw.get("primary_method", ""),
        primary_rationale=raw.get("primary_rationale", ""),
        secondary_method=raw.get("secondary_method", ""),
        secondary_rationale=raw.get("secondary_rationale", ""),
        overall_confidence=raw.get("overall_confidence", "medium"),
    )


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def select_methods(
    run_dir: Path,
    *,
    llm_client: Any,
    max_designs: int = 5,
) -> MethodSelectionResult:
    run_id = run_dir.name
    now_iso = datetime.now(timezone.utc).isoformat()

    inputs = collect_inputs(run_dir)
    designs = inputs["validation_designs"][:max_designs]

    if not designs:
        return MethodSelectionResult(run_id=run_id, rq_title=inputs.get("rq_title", ""))

    # Build data plan lookup
    dp_by_id = {p.get("hypothesis_id", ""): p for p in inputs.get("data_plans", [])}

    selections = []
    for i, design in enumerate(designs):
        hyp_id = design.get("hypothesis_id", "")
        logger.info("[%d/%d] Selecting methods for: %s",
                    i + 1, len(designs), design.get("hypothesis_statement", "")[:50])

        data_plan = dp_by_id.get(hyp_id)
        id_asmp = [a for a in inputs["identification_assumptions"] if a.get("hypothesis_id") == hyp_id]

        raw = compare_methods_for_hypothesis(
            design, data_plan, id_asmp, inputs.get("rq_title", ""),
            llm_client=llm_client,
        )

        if raw:
            selections.append(_build_selection(design, raw))
        else:
            logger.warning("Method selection failed for %s", hyp_id)

    return MethodSelectionResult(
        run_id=run_id,
        rq_title=inputs.get("rq_title", ""),
        selections_generated=len(selections),
        method_selections=selections,
        metadata={"created_at": now_iso, "model": _MODEL},
    )


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: MethodSelectionResult) -> str:
    lines = [
        f"# Method Selection",
        f"",
        f"## RQ: {result.rq_title}",
        f"",
        f"**Selections generated: {result.selections_generated}**",
        f"",
    ]

    for i, sel in enumerate(result.method_selections, 1):
        lines.extend([
            f"---",
            f"",
            f"## {i}. {sel.hypothesis_statement[:65]}",
            f"",
            f"**Primary**: {sel.primary_method} | **Secondary**: {sel.secondary_method} | "
            f"**Confidence**: {sel.overall_confidence}",
            f"",
        ])

        # Comparison table
        lines.append(f"### Comparison")
        lines.append(f"")
        lines.append(f"| Method | Category | ID Str | Data | Feas | Interp | Robust |")
        lines.append(f"|--------|----------|--------|------|------|--------|--------|")
        for c in sel.candidates:
            s = c.scores
            marker = " ***" if c.method_name == sel.primary_method else (" *" if c.method_name == sel.secondary_method else "")
            lines.append(
                f"| {c.method_name}{marker} | {c.method_category} | "
                f"{s.get('identification_strength', '?')} | {s.get('data_fit', '?')} | "
                f"{s.get('feasibility', '?')} | {s.get('interpretability', '?')} | "
                f"{s.get('robustness', '?')} |"
            )
        lines.append(f"")

        # Per-method details
        for c in sel.candidates:
            role = ""
            if c.method_name == sel.primary_method:
                role = " [PRIMARY]"
            elif c.method_name == sel.secondary_method:
                role = " [SECONDARY]"
            lines.append(f"#### {c.method_name}{role}")
            lines.append(f"")
            lines.append(f"- **Why fit**: {c.why_fit}")
            lines.append(f"- **Failure mode**: {c.failure_mode}")
            lines.append(f"- **Strengths**: {'; '.join(c.strengths)}")
            lines.append(f"- **Weaknesses**: {'; '.join(c.weaknesses)}")
            if c.assumption_fit:
                lines.append(f"- **Assumption fit**: {c.assumption_fit}")
            if c.implementation_notes:
                lines.append(f"- **Implementation**: {c.implementation_notes}")
            lines.append(f"")

        # Recommendation
        lines.extend([
            f"### Recommendation",
            f"",
            f"**Primary ({sel.primary_method})**: {sel.primary_rationale}",
            f"",
            f"**Secondary ({sel.secondary_method})**: {sel.secondary_rationale}",
            f"",
        ])

    return "\n".join(lines)
