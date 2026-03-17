# src/question/evaluator.py
"""102 RQ Evaluator — service logic.

Evaluates RQ candidates on 4 axes with explainable scoring:
  - Specificity:  Is the RQ sufficiently focused?
  - Testability:  Can it be empirically tested?
  - Novelty:      Is it differentiated from existing research?
  - Feasibility:  Is it executable with available resources?

Each axis gets a 1–5 score with justification.

Usage::

    from src.question.evaluator import evaluate_rq_candidates

    result = evaluate_rq_candidates(candidates_path, llm_client=client)
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
_MAX_TOKENS = 8192


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class AxisScore:
    score: int = 0           # 1–5
    justification: str = ""


@dataclass
class RQEvaluation:
    candidate_id: str = ""
    title: str = ""
    specificity: AxisScore = field(default_factory=AxisScore)
    testability: AxisScore = field(default_factory=AxisScore)
    novelty: AxisScore = field(default_factory=AxisScore)
    feasibility: AxisScore = field(default_factory=AxisScore)
    composite_score: float = 0.0
    refinement_suggestions: List[str] = field(default_factory=list)


@dataclass
class EvaluatorResult:
    status: str = "failed"
    parent_run_id: str = ""
    evaluations: List[RQEvaluation] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_run_id": self.parent_run_id,
            "evaluated_at": self.metadata.get("evaluated_at", ""),
            "candidates_evaluated": len(self.evaluations),
            "evaluations": [asdict(e) for e in self.evaluations],
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------
# LLM evaluation
# ------------------------------------------------------------------

def _parse_evaluations(text: str) -> Optional[List[Dict]]:
    """Parse JSON evaluations from LLM response."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "evaluations" in result:
            return result["evaluations"]
        return None
    except json.JSONDecodeError:
        return None


def _evaluate_via_llm(
    candidates: List[Dict[str, Any]],
    parent_rq_title: str,
    llm_client: Any,
) -> List[RQEvaluation]:
    """Evaluate candidates using LLM."""
    # Build candidate summaries
    cand_parts: List[str] = []
    for c in candidates:
        cand_parts.append(
            f"### {c.get('candidate_id', '')}: {c.get('title', '')}\n"
            f"Question: {c.get('question', '')}\n"
            f"Background: {c.get('background', '')}\n"
            f"Gap: {c.get('gap', '')}\n"
            f"Approach: {c.get('approach', '')}\n"
            f"Source: {c.get('source_type', '')} ({c.get('derived_from', '')})"
        )
    cand_text = "\n\n".join(cand_parts)

    system = (
        "あなたは研究評価の専門家です。\n"
        "Research Question 候補を 4 軸で評価してください。\n"
        "各軸は 1–5 のスコアと、そのスコアの根拠（日本語で 1–2 文）を付けてください。\n"
        "出力は JSON 配列のみ。"
    )

    user = (
        f"## 親RQ\n{parent_rq_title}\n\n"
        f"## RQ候補\n{cand_text}\n\n"
        f"## 評価軸\n"
        f"- **Specificity** (1=曖昧すぎ, 3=方向性あり, 5=明確な変数・対象・スコープ)\n"
        f"- **Testability** (1=検証手段不明, 3=実行可能だが困難, 5=既存手法で検証可能)\n"
        f"- **Novelty** (1=既存と同じ, 3=部分的に新規, 5=未踏の切り口)\n"
        f"- **Feasibility** (1=データ/手法不足, 3=努力で可能, 5=既存リソースで実行可能)\n\n"
        f"## 出力形式\n"
        f"```json\n"
        f"[\n"
        f"  {{\n"
        f'    "candidate_id": "rqc_...",\n'
        f'    "specificity": {{ "score": 4, "justification": "..." }},\n'
        f'    "testability": {{ "score": 3, "justification": "..." }},\n'
        f'    "novelty": {{ "score": 5, "justification": "..." }},\n'
        f'    "feasibility": {{ "score": 2, "justification": "..." }},\n'
        f'    "refinement_suggestions": ["提案1", "提案2"]\n'
        f"  }}\n"
        f"]\n"
        f"```\n"
        f"JSON 配列のみ出力。説明文は不要。"
    )

    body = {
        "model": _MODEL,
        "max_tokens": _MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return []

    text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("LLM: in=%d out=%d tokens", usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    raw_evals = _parse_evaluations(text)
    if not raw_evals:
        logger.error("Failed to parse LLM evaluation response")
        return []

    # Build candidate_id → title map
    title_map = {c.get("candidate_id", ""): c.get("title", "") for c in candidates}

    evaluations: List[RQEvaluation] = []
    for raw in raw_evals:
        cid = raw.get("candidate_id", "")
        spec = raw.get("specificity", {})
        test = raw.get("testability", {})
        nov = raw.get("novelty", {})
        feas = raw.get("feasibility", {})

        scores = [
            spec.get("score", 0),
            test.get("score", 0),
            nov.get("score", 0),
            feas.get("score", 0),
        ]
        composite = sum(scores) / len(scores) if scores else 0.0

        evaluations.append(RQEvaluation(
            candidate_id=cid,
            title=title_map.get(cid, ""),
            specificity=AxisScore(score=spec.get("score", 0), justification=spec.get("justification", "")),
            testability=AxisScore(score=test.get("score", 0), justification=test.get("justification", "")),
            novelty=AxisScore(score=nov.get("score", 0), justification=nov.get("justification", "")),
            feasibility=AxisScore(score=feas.get("score", 0), justification=feas.get("justification", "")),
            composite_score=round(composite, 2),
            refinement_suggestions=raw.get("refinement_suggestions", []),
        ))

    return evaluations


def _render_markdown(result: EvaluatorResult) -> str:
    parts: List[str] = []
    parts.append("# RQ Evaluation Report\n")
    parts.append(f"**Parent Run**: {result.parent_run_id}")
    parts.append(f"**Candidates evaluated**: {len(result.evaluations)}\n")

    # Summary table
    parts.append("| # | Title | Spec | Test | Nov | Feas | Composite |")
    parts.append("|---|-------|------|------|-----|------|-----------|")
    sorted_evals = sorted(result.evaluations, key=lambda e: e.composite_score, reverse=True)
    for i, e in enumerate(sorted_evals, 1):
        parts.append(
            f"| {i} | {e.title[:40]} | {e.specificity.score} | {e.testability.score} | "
            f"{e.novelty.score} | {e.feasibility.score} | **{e.composite_score}** |"
        )
    parts.append("")

    # Detailed evaluations
    for e in sorted_evals:
        parts.append(f"## {e.title}")
        parts.append(f"**Composite score**: {e.composite_score}/5.0\n")
        parts.append(f"- **Specificity** ({e.specificity.score}/5): {e.specificity.justification}")
        parts.append(f"- **Testability** ({e.testability.score}/5): {e.testability.justification}")
        parts.append(f"- **Novelty** ({e.novelty.score}/5): {e.novelty.justification}")
        parts.append(f"- **Feasibility** ({e.feasibility.score}/5): {e.feasibility.justification}")
        if e.refinement_suggestions:
            parts.append(f"\n**Refinement suggestions**:")
            for s in e.refinement_suggestions:
                parts.append(f"  - {s}")
        parts.append("")

    return "\n".join(parts) + "\n"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def evaluate_rq_candidates(
    candidates_path: Path,
    *,
    llm_client: Any,
) -> EvaluatorResult:
    """Evaluate RQ candidates from a candidates JSON file."""
    result = EvaluatorResult()
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        data = json.loads(candidates_path.read_text())
        candidates = data.get("candidates", [])
        result.parent_run_id = data.get("parent_run_id", "")
        parent_rq_title = data.get("parent_rq_title", "")

        if not candidates:
            result.error = "No candidates to evaluate"
            return result

        logger.info("Evaluating %d candidates", len(candidates))

        evaluations = _evaluate_via_llm(candidates, parent_rq_title, llm_client)

        if not evaluations:
            result.error = "LLM failed to produce evaluations"
            return result

        result.evaluations = evaluations
        result.status = "generated"
        result.metadata = {
            "evaluated_at": now_iso,
            "model": _MODEL,
        }

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.error("102: %s", e)

    return result
