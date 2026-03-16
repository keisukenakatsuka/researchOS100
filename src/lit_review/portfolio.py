# src/lit_review/portfolio.py
"""Block 5: Hypothesis Portfolio (089).

Scores hypotheses on 5 axes (novelty, testability, vulnerability,
feasibility, strategic_importance) and produces a portfolio view
for research prioritization.

Testability and vulnerability are deterministic (from 087/088 outputs).
Novelty, feasibility, and strategic_importance are LLM-assessed.

Usage::

    from src.lit_review.portfolio import build_portfolio

    result = build_portfolio(run_dir, llm_client=client)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
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
class HypothesisScore:
    hypothesis_id: str
    statement: str
    strategy: str
    scores: Dict[str, int] = field(default_factory=dict)  # axis -> 1-5
    composite_score: float = 0.0
    quadrant: str = ""  # priority | ambitious | incremental | defer
    recommendation: str = ""  # high_priority | promising | needs_refinement | defer
    explanation: str = ""
    strongest_dimension: str = ""
    weakest_dimension: str = ""
    overall_vulnerability: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioResult:
    run_id: str
    rq_title: str = ""
    hypotheses_scored: int = 0
    scored_hypotheses: List[HypothesisScore] = field(default_factory=list)
    portfolio_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        by_rec = Counter(h.recommendation for h in self.scored_hypotheses)
        by_quad = Counter(h.quadrant for h in self.scored_hypotheses)
        return {
            "run_id": self.run_id,
            "rq_title": self.rq_title,
            "hypotheses_scored": self.hypotheses_scored,
            "scored_hypotheses": [h.to_dict() for h in self.scored_hypotheses],
            "portfolio_summary": {
                "by_recommendation": dict(by_rec),
                "by_quadrant": dict(by_quad),
            },
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
        "hypothesis_assumptions": [],
        "theoretical_streams": [],
        "rq_title": "",
        "cross_rq_opportunities": [],
    }

    rq_path = run_dir / "rq_context.json"
    if rq_path.exists():
        inputs["rq_title"] = json.loads(rq_path.read_text()).get("title", "")

    hyp_path = run_dir / "hypotheses.json"
    if hyp_path.exists():
        inputs["hypotheses"] = json.loads(hyp_path.read_text()).get("hypotheses", [])

    asmp_path = run_dir / "assumptions.json"
    if asmp_path.exists():
        inputs["hypothesis_assumptions"] = json.loads(asmp_path.read_text()).get("hypothesis_assumptions", [])

    lr_path = run_dir / "lit_review.json"
    if lr_path.exists():
        lr = json.loads(lr_path.read_text())
        inputs["theoretical_streams"] = lr.get("theoretical_streams", [])

    # Auto-detect cross-RQ
    cross_base = run_dir.parent / "cross_rq"
    if cross_base.exists():
        dirs = sorted(cross_base.iterdir(), reverse=True)
        for d in dirs:
            cmp_path = d / "cross_rq_comparison.json"
            if cmp_path.exists():
                cmp = json.loads(cmp_path.read_text())
                inputs["cross_rq_opportunities"] = cmp.get("cross_rq_opportunities", [])
                break

    logger.info("Collected: %d hypotheses, %d assumption sets, %d streams, %d cross-RQ opps",
                len(inputs["hypotheses"]), len(inputs["hypothesis_assumptions"]),
                len(inputs["theoretical_streams"]), len(inputs["cross_rq_opportunities"]))
    return inputs


# ------------------------------------------------------------------
# Deterministic scoring
# ------------------------------------------------------------------

def _testability_score(testability: str) -> int:
    return {"high": 5, "medium": 3, "low": 1}.get(testability, 3)


def _vulnerability_score(overall_vulnerability: str) -> int:
    # Inverted: low vulnerability = high robustness score
    return {"low": 5, "medium": 3, "high": 1}.get(overall_vulnerability, 3)


def _get_assumption_info(hypothesis_id: str, assumption_sets: List[Dict]) -> Dict[str, Any]:
    for ha in assumption_sets:
        if ha.get("hypothesis_id") == hypothesis_id:
            return ha
    return {}


# ------------------------------------------------------------------
# LLM scoring (novelty, feasibility, strategic_importance)
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


_SCORING_SYSTEM = """\
あなたは研究戦略の専門家です。
研究仮説のポートフォリオを評価し、各仮説の新規性・実現可能性・戦略的重要性をスコアリングしてください。

スコアは 1–5 で以下の基準:
- 5: 非常に高い
- 4: 高い
- 3: 中程度
- 2: 低い
- 1: 非常に低い

各軸の評価基準:
- novelty: 既存研究で未検証の度合い。gap_driven仮説は高め、既存知見の延長は低め
- feasibility: データ・リソースの現実的入手可能性。利用可能なデータセットが明確なら高い
- strategic_importance: RQの核心にどれだけ近いか、cross-RQに波及しうるか、研究ポートフォリオ全体にどれだけ価値があるか"""


def score_hypotheses_llm(
    inputs: Dict[str, Any],
    *,
    llm_client: Any,
) -> Optional[Dict]:
    hyp_lines = []
    for i, h in enumerate(inputs["hypotheses"]):
        stmt = h.get("hypothesis_statement", "")
        strategy = h.get("strategy", "")
        test = h.get("suggested_test", "")
        hyp_lines.append(f"[H{i}] ({strategy}) {stmt}\n  suggested_test: {test}")

    streams = [s.get("name", "") for s in inputs.get("theoretical_streams", [])]
    cross_rq = [o.get("theme", "") for o in inputs.get("cross_rq_opportunities", [])]

    user_msg = (
        f"## RQ: {inputs.get('rq_title', '')}\n\n"
        f"## 理論的文脈: {', '.join(streams) if streams else '(なし)'}\n\n"
        f"## Cross-RQ Opportunities: {', '.join(cross_rq) if cross_rq else '(なし)'}\n\n"
        f"## 仮説 ({len(inputs['hypotheses'])} 件)\n\n" + "\n\n".join(hyp_lines) + "\n\n"
        f"## 指示\n"
        f"各仮説について novelty, feasibility, strategic_importance を 1–5 でスコアリングしてください。\n\n"
        f"出力形式 (JSON):\n"
        f'{{"scores": [\n'
        f'  {{"hypothesis_index": 0, "novelty": 4, "feasibility": 3, "strategic_importance": 5}}\n'
        f']}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 2048,
        "system": _SCORING_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Portfolio LLM call failed: %s", e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Portfolio LLM: in=%d, out=%d tokens", usage.get("input_tokens", 0), usage.get("output_tokens", 0))
    return _parse_json_response(resp_text)


# ------------------------------------------------------------------
# Portfolio assembly
# ------------------------------------------------------------------

def _determine_quadrant(impact: float, feasibility_score: int) -> str:
    high_impact = impact >= 3.5
    high_feasibility = feasibility_score >= 3
    if high_impact and high_feasibility:
        return "priority"
    if high_impact and not high_feasibility:
        return "ambitious"
    if not high_impact and high_feasibility:
        return "incremental"
    return "defer"


def _determine_recommendation(
    composite: float,
    overall_vulnerability: str,
    quadrant: str,
) -> str:
    """Recommendation based on portfolio position + vulnerability."""
    if quadrant == "priority" and overall_vulnerability != "high":
        return "high_priority"
    if quadrant == "priority" and overall_vulnerability == "high":
        return "promising"
    if quadrant == "ambitious" and overall_vulnerability != "high":
        return "promising"
    if quadrant == "ambitious" and overall_vulnerability == "high":
        return "needs_refinement"
    if quadrant == "incremental":
        return "promising" if composite >= 3.0 else "defer"
    return "defer"


def _build_explanation(hs: HypothesisScore) -> str:
    parts = []
    parts.append(f"Strongest: {hs.strongest_dimension} ({hs.scores.get(hs.strongest_dimension, '?')})")
    parts.append(f"Weakest: {hs.weakest_dimension} ({hs.scores.get(hs.weakest_dimension, '?')})")

    if hs.recommendation == "high_priority":
        parts.append("高いインパクトと実現可能性を兼ね備えており、脆弱性も管理可能な範囲にある")
    elif hs.recommendation == "promising":
        parts.append("有望だが、脆弱性の管理または実現可能性の向上が必要")
    elif hs.recommendation == "needs_refinement":
        parts.append("インパクトは高いが、前提条件の脆弱性が大きく、仮説の精緻化が必要")
    else:
        parts.append("現時点では優先度が低い。他の仮説を先行させることを推奨")

    return ". ".join(parts)


def build_portfolio(
    run_dir: Path,
    *,
    llm_client: Any,
) -> PortfolioResult:
    run_id = run_dir.name
    now_iso = datetime.now(timezone.utc).isoformat()

    inputs = collect_inputs(run_dir)
    if not inputs["hypotheses"]:
        return PortfolioResult(run_id=run_id, rq_title=inputs.get("rq_title", ""))

    # LLM scoring (novelty, feasibility, strategic_importance)
    llm_result = score_hypotheses_llm(inputs, llm_client=llm_client)
    llm_scores = {}
    if llm_result:
        for s in llm_result.get("scores", []):
            llm_scores[s.get("hypothesis_index", -1)] = s

    # Build scored hypotheses
    scored = []
    for i, hyp in enumerate(inputs["hypotheses"]):
        hyp_id = hyp.get("hypothesis_id", f"hyp_{i}")
        testability = hyp.get("testability", "medium")

        # Get assumption info
        asmp_info = _get_assumption_info(hyp_id, inputs["hypothesis_assumptions"])
        overall_vuln = asmp_info.get("overall_vulnerability", "medium")

        # Deterministic scores
        test_score = _testability_score(testability)
        vuln_score = _vulnerability_score(overall_vuln)

        # LLM scores
        ls = llm_scores.get(i, {})
        novelty = ls.get("novelty", 3)
        feasibility = ls.get("feasibility", 3)
        strategic = ls.get("strategic_importance", 3)

        all_scores = {
            "novelty": novelty,
            "testability": test_score,
            "vulnerability": vuln_score,
            "feasibility": feasibility,
            "strategic_importance": strategic,
        }

        composite = sum(all_scores.values()) / len(all_scores)

        # Impact = (novelty + strategic_importance) / 2
        impact = (novelty + strategic) / 2
        quadrant = _determine_quadrant(impact, feasibility)
        recommendation = _determine_recommendation(composite, overall_vuln, quadrant)

        # Strongest / weakest
        strongest = max(all_scores, key=all_scores.get)
        weakest = min(all_scores, key=all_scores.get)

        hs = HypothesisScore(
            hypothesis_id=hyp_id,
            statement=hyp.get("hypothesis_statement", ""),
            strategy=hyp.get("strategy", ""),
            scores=all_scores,
            composite_score=round(composite, 1),
            quadrant=quadrant,
            recommendation=recommendation,
            strongest_dimension=strongest,
            weakest_dimension=weakest,
            overall_vulnerability=overall_vuln,
        )
        hs.explanation = _build_explanation(hs)
        scored.append(hs)

    # Sort by composite descending
    scored.sort(key=lambda h: h.composite_score, reverse=True)

    return PortfolioResult(
        run_id=run_id,
        rq_title=inputs.get("rq_title", ""),
        hypotheses_scored=len(scored),
        scored_hypotheses=scored,
        metadata={"created_at": now_iso, "model": _MODEL},
    )


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: PortfolioResult) -> str:
    by_rec = Counter(h.recommendation for h in result.scored_hypotheses)
    by_quad = Counter(h.quadrant for h in result.scored_hypotheses)

    lines = [
        f"# Hypothesis Portfolio",
        f"",
        f"## RQ: {result.rq_title}",
        f"",
        f"## Summary",
        f"",
        f"- Hypotheses scored: {result.hypotheses_scored}",
        f"",
        f"### By Recommendation",
    ]
    for rec in ["high_priority", "promising", "needs_refinement", "defer"]:
        lines.append(f"- {rec}: {by_rec.get(rec, 0)}")
    lines.extend([f"", f"### By Quadrant"])
    for q in ["priority", "ambitious", "incremental", "defer"]:
        lines.append(f"- {q}: {by_quad.get(q, 0)}")
    lines.append(f"")

    # Portfolio Matrix (text-based)
    lines.extend([f"## Portfolio Matrix", f""])
    lines.append(f"```")
    lines.append(f"                    High Feasibility    Low Feasibility")
    lines.append(f"  High Impact  │  PRIORITY            AMBITIOUS")
    for h in result.scored_hypotheses:
        if h.quadrant == "priority":
            lines.append(f"               │  [{h.composite_score}] H: {h.statement[:40]}")
    for h in result.scored_hypotheses:
        if h.quadrant == "ambitious":
            lines.append(f"               │                      [{h.composite_score}] H: {h.statement[:35]}")
    lines.append(f"  ─────────────┼──────────────────────────────────────────")
    lines.append(f"  Low Impact   │  INCREMENTAL          DEFER")
    for h in result.scored_hypotheses:
        if h.quadrant == "incremental":
            lines.append(f"               │  [{h.composite_score}] H: {h.statement[:40]}")
    for h in result.scored_hypotheses:
        if h.quadrant == "defer":
            lines.append(f"               │                      [{h.composite_score}] H: {h.statement[:35]}")
    lines.append(f"```")
    lines.append(f"")

    # Detailed scores
    lines.extend([f"## Detailed Scores", f""])
    lines.append(f"| # | Rec | Comp | Nov | Test | Vuln | Feas | Strat | Hypothesis |")
    lines.append(f"|---|-----|------|-----|------|------|------|-------|------------|")
    for i, h in enumerate(result.scored_hypotheses, 1):
        s = h.scores
        rec_short = h.recommendation[:4]
        lines.append(
            f"| {i} | {rec_short} | {h.composite_score} | "
            f"{s.get('novelty', '?')} | {s.get('testability', '?')} | "
            f"{s.get('vulnerability', '?')} | {s.get('feasibility', '?')} | "
            f"{s.get('strategic_importance', '?')} | {h.statement[:45]} |"
        )
    lines.append(f"")

    # Per-hypothesis detail
    lines.extend([f"## Hypothesis Details", f""])
    for i, h in enumerate(result.scored_hypotheses, 1):
        rec_icon = {"high_priority": "***", "promising": "**", "needs_refinement": "*", "defer": ""}.get(h.recommendation, "")
        lines.append(f"### {i}. [{h.recommendation}] {rec_icon} {h.statement[:65]}")
        lines.append(f"")
        lines.append(f"- **Strategy**: {h.strategy}")
        lines.append(f"- **Quadrant**: {h.quadrant}")
        lines.append(f"- **Composite**: {h.composite_score}")
        lines.append(f"- **Vulnerability**: {h.overall_vulnerability}")
        lines.append(f"- **Scores**: novelty={h.scores.get('novelty')}, testability={h.scores.get('testability')}, "
                     f"vulnerability={h.scores.get('vulnerability')}, feasibility={h.scores.get('feasibility')}, "
                     f"strategic={h.scores.get('strategic_importance')}")
        lines.append(f"- **{h.explanation}**")
        lines.append(f"")

    return "\n".join(lines)
