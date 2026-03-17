# src/question/prioritizer.py
"""103 RQ Prioritizer — service logic.

Manages RQ candidates as a research portfolio, assigning each an
actionable recommendation (promote / refine / defer / merge) with
rationale and portfolio role.

Merge detection uses candidate_id (stable) and title similarity
to identify near-duplicates, never title-string matching alone.

Usage::

    from src.question.prioritizer import prioritize_rq_candidates

    result = prioritize_rq_candidates(evaluation_path, llm_client=client)
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
_MAX_TOKENS = 4096


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class PortfolioEntry:
    """A single RQ in the prioritized portfolio."""
    candidate_id: str = ""
    title: str = ""
    composite_score: float = 0.0
    recommendation: str = ""       # promote | refine | defer | merge
    action_rationale: str = ""
    portfolio_role: str = ""       # near-term | exploratory | long-horizon | methodological
    priority_rank: int = 0
    # merge fields
    merge_target_id: Optional[str] = None   # canonical candidate to merge into
    merge_rationale: str = ""


@dataclass
class PortfolioSummary:
    total_candidates: int = 0
    promote: int = 0
    refine: int = 0
    defer: int = 0
    merge: int = 0


@dataclass
class PrioritizerResult:
    status: str = "failed"
    parent_run_id: str = ""
    portfolio: List[PortfolioEntry] = field(default_factory=list)
    summary: PortfolioSummary = field(default_factory=PortfolioSummary)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_run_id": self.parent_run_id,
            "prioritized_at": self.metadata.get("prioritized_at", ""),
            "portfolio": [asdict(e) for e in self.portfolio],
            "summary": asdict(self.summary),
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------
# LLM-based prioritization
# ------------------------------------------------------------------

def _parse_portfolio(text: str) -> Optional[List[Dict]]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "portfolio" in result:
            return result["portfolio"]
        return None
    except json.JSONDecodeError:
        return None


def _prioritize_via_llm(
    evaluations: List[Dict[str, Any]],
    parent_rq_title: str,
    llm_client: Any,
) -> List[PortfolioEntry]:
    """Use LLM to assign portfolio roles and recommendations."""
    # Build evaluation summary
    eval_parts: List[str] = []
    for e in evaluations:
        cid = e.get("candidate_id", "")
        title = e.get("title", "")
        composite = e.get("composite_score", 0)
        spec = e.get("specificity", {}).get("score", 0)
        test = e.get("testability", {}).get("score", 0)
        nov = e.get("novelty", {}).get("score", 0)
        feas = e.get("feasibility", {}).get("score", 0)
        suggestions = e.get("refinement_suggestions", [])
        eval_parts.append(
            f"### {cid}: {title}\n"
            f"Scores: Spec={spec} Test={test} Nov={nov} Feas={feas} → Composite={composite}\n"
            f"Suggestions: {'; '.join(suggestions) if suggestions else 'none'}"
        )
    eval_text = "\n\n".join(eval_parts)

    system = (
        "あなたは研究ポートフォリオの戦略家です。\n"
        "評価済みのRQ候補に対し、研究ポートフォリオの観点から\n"
        "アクション推奨とポートフォリオ上の役割を割り当ててください。\n"
        "出力は JSON 配列のみ。"
    )

    user = (
        f"## 親RQ\n{parent_rq_title}\n\n"
        f"## 評価済み候補\n{eval_text}\n\n"
        f"## 判定基準\n"
        f"**promote**: composite ≥ 3.5 かつ testability ≥ 3 かつ feasibility ≥ 3。即座に新 run を開始可能\n"
        f"**refine**: novelty が高い (≥4) が specificity または feasibility が低い (≤2)。洗練すれば有望\n"
        f"**defer**: composite < 3.0 または当面優先度が低い。将来再評価\n"
        f"**merge**: 他の候補と研究スコープが 70% 以上重複。merge_target_id に統合先を指定\n\n"
        f"**portfolio_role**:\n"
        f"- near-term: すぐ着手可能。データと手法が揃っている\n"
        f"- exploratory: 新規性が高いが不確実性あり。試行的に着手\n"
        f"- long-horizon: 重要だが長期プロジェクト。データ整備が必要\n"
        f"- methodological: 方法論改善が主目的\n\n"
        f"## 出力形式\n"
        f"```json\n"
        f"[\n"
        f"  {{\n"
        f'    "candidate_id": "rqc_...",\n'
        f'    "recommendation": "promote|refine|defer|merge",\n'
        f'    "action_rationale": "推奨理由（日本語で具体的に）",\n'
        f'    "portfolio_role": "near-term|exploratory|long-horizon|methodological",\n'
        f'    "priority_rank": 1,\n'
        f'    "merge_target_id": null,\n'
        f'    "merge_rationale": ""\n'
        f"  }}\n"
        f"]\n"
        f"```\n"
        f"JSON 配列のみ出力。priority_rank は 1 が最高優先。"
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

    raw = _parse_portfolio(text)
    if not raw:
        logger.error("Failed to parse LLM portfolio response")
        return []

    # Build title map from evaluations
    title_map = {e.get("candidate_id", ""): e.get("title", "") for e in evaluations}
    score_map = {e.get("candidate_id", ""): e.get("composite_score", 0) for e in evaluations}

    entries: List[PortfolioEntry] = []
    for r in raw:
        cid = r.get("candidate_id", "")
        entries.append(PortfolioEntry(
            candidate_id=cid,
            title=title_map.get(cid, ""),
            composite_score=score_map.get(cid, 0),
            recommendation=r.get("recommendation", ""),
            action_rationale=r.get("action_rationale", ""),
            portfolio_role=r.get("portfolio_role", ""),
            priority_rank=r.get("priority_rank", 0),
            merge_target_id=r.get("merge_target_id"),
            merge_rationale=r.get("merge_rationale", ""),
        ))

    return entries


def _render_markdown(result: PrioritizerResult) -> str:
    parts: List[str] = []
    parts.append("# RQ Portfolio\n")
    parts.append(f"**Parent Run**: {result.parent_run_id}")
    s = result.summary
    parts.append(f"**Total**: {s.total_candidates} | promote: {s.promote} | refine: {s.refine} | defer: {s.defer} | merge: {s.merge}\n")

    # Group by recommendation
    for action in ["promote", "refine", "defer", "merge"]:
        entries = [e for e in result.portfolio if e.recommendation == action]
        if not entries:
            continue
        parts.append(f"## {action.upper()} ({len(entries)})\n")
        for e in sorted(entries, key=lambda x: x.priority_rank):
            parts.append(f"### #{e.priority_rank} {e.title}")
            parts.append(f"**ID**: `{e.candidate_id}`")
            parts.append(f"**Score**: {e.composite_score} | **Role**: {e.portfolio_role}")
            parts.append(f"**Rationale**: {e.action_rationale}")
            if e.merge_target_id:
                parts.append(f"**Merge into**: `{e.merge_target_id}` — {e.merge_rationale}")
            parts.append("")

    return "\n".join(parts) + "\n"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def prioritize_rq_candidates(
    evaluation_path: Path,
    *,
    llm_client: Any,
) -> PrioritizerResult:
    """Prioritize evaluated RQ candidates into a portfolio."""
    result = PrioritizerResult()
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        data = json.loads(evaluation_path.read_text())
        evaluations = data.get("evaluations", [])
        result.parent_run_id = data.get("parent_run_id", "")

        # Recover parent_rq_title from candidates file
        candidates_path = evaluation_path.parent / "rq_candidates.json"
        parent_rq_title = ""
        if candidates_path.exists():
            cand_data = json.loads(candidates_path.read_text())
            parent_rq_title = cand_data.get("parent_rq_title", "")

        if not evaluations:
            result.error = "No evaluations to prioritize"
            return result

        entries = _prioritize_via_llm(evaluations, parent_rq_title, llm_client)
        if not entries:
            result.error = "LLM failed to produce portfolio"
            return result

        result.portfolio = entries
        result.summary = PortfolioSummary(
            total_candidates=len(entries),
            promote=sum(1 for e in entries if e.recommendation == "promote"),
            refine=sum(1 for e in entries if e.recommendation == "refine"),
            defer=sum(1 for e in entries if e.recommendation == "defer"),
            merge=sum(1 for e in entries if e.recommendation == "merge"),
        )
        result.status = "generated"
        result.metadata = {"prioritized_at": now_iso, "model": _MODEL}

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.error("103: %s", e)

    return result
