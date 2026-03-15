# src/lit_review/synthesizer.py
"""Lit Review synthesis (082).

Integrates Evidence across papers to produce a structured Lit Review:
- Executive summary
- Theoretical streams
- Empirical findings (established / emerging / contested)
- Open questions (research gaps)
- Research dimensions

Outputs both structured JSON and readable Markdown.

Implementation uses a 2-pass LLM approach:
  Pass 1: Synthesis — theoretical streams, findings classification, gaps
  Pass 2: Dimensions + executive summary
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.lit_review.rq_context import RQContext
from src.lit_review.extractor import Evidence

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class TheoreticalStream:
    name: str
    description: str
    key_concepts: List[str] = field(default_factory=list)
    papers: List[str] = field(default_factory=list)


@dataclass
class Finding:
    statement: str
    supporting_papers: List[str] = field(default_factory=list)
    evidence_summary: str = ""
    paper_count: int = 0
    strength: str = ""


@dataclass
class ContestedPoint:
    topic: str
    positions: List[Dict[str, Any]] = field(default_factory=list)
    nature_of_disagreement: str = ""


@dataclass
class OpenQuestion:
    description: str
    why_unresolved: str = ""
    potential_approach: str = ""


@dataclass
class LitReviewResult:
    rq: Dict[str, Any]
    papers: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    executive_summary: str = ""
    theoretical_streams: List[TheoreticalStream] = field(default_factory=list)
    established: List[Finding] = field(default_factory=list)
    emerging: List[Finding] = field(default_factory=list)
    contested: List[ContestedPoint] = field(default_factory=list)
    open_questions: List[OpenQuestion] = field(default_factory=list)
    research_dimensions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.metadata.get("run_id", ""),
            "rq_context": self.rq,
            "executive_summary": self.executive_summary,
            "theoretical_streams": [asdict(s) for s in self.theoretical_streams],
            "empirical_findings": {
                "established": [asdict(f) for f in self.established],
                "emerging": [asdict(f) for f in self.emerging],
                "contested": [asdict(c) for c in self.contested],
            },
            "open_questions": [asdict(q) for q in self.open_questions],
            "research_dimensions": self.research_dimensions,
            "papers": self.papers,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# LLM helpers
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


def _llm_call(llm_client, system: str, user: str, max_tokens: int = 8192) -> Optional[Dict]:
    body = {
        "model": _MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("LLM call: in=%d, out=%d tokens", usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    parsed = _parse_json_response(resp_text)
    if not parsed:
        logger.error("JSON parse failed. Raw head: %s", resp_text[:300])
    return parsed


# ------------------------------------------------------------------
# Pass 1: Synthesis (theoretical streams + findings + gaps)
# ------------------------------------------------------------------

_SYNTHESIS_SYSTEM = """\
あなたは学術文献レビューの専門家です。
Research Question (RQ) に関連する複数論文からの Evidence を統合し、
構造化された Literature Review を生成してください。

重要な指示:
- 論文ごとの列挙ではなく、RQ に対する知見の統合的な整理を行ってください
- 事実と解釈を区別し、根拠の強度を明示してください
- 複数論文で一致する知見は established、少数の知見は emerging、矛盾する知見は contested として分類してください
- evidence_summary には可能な限り具体的な数値・定量表現を含めてください（例: 「+43.8ポイント」「約6.7%の成長」）
- 研究ギャップは「今後の課題」ではなく、このRQに対して何がまだ直接検証されていないかを具体的に示してください"""

_SYNTHESIS_SCHEMA_INSTRUCTION = """\
以下の JSON 形式で出力してください:
{
  "theoretical_streams": [
    {
      "name": "理論的系譜の名称",
      "description": "この理論的系譜の概要（日本語2-3文）",
      "key_concepts": ["概念1", "概念2"],
      "papers": ["論文タイトル1", "論文タイトル2"]
    }
  ],
  "established_findings": [
    {
      "statement": "確立された知見（日本語で簡潔に）",
      "supporting_papers": ["論文タイトル"],
      "evidence_summary": "この知見を支持する Evidence の要約（日本語2-3文）",
      "paper_count": 3,
      "strength": "strong | moderate"
    }
  ],
  "emerging_findings": [
    {
      "statement": "萌芽的知見（日本語で簡潔に）",
      "supporting_papers": ["論文タイトル"],
      "evidence_summary": "根拠の要約",
      "paper_count": 1,
      "strength": "preliminary | suggestive"
    }
  ],
  "contested_findings": [
    {
      "topic": "論争的トピック",
      "positions": [
        {"statement": "立場A", "papers": ["論文タイトル"]},
        {"statement": "立場B", "papers": ["論文タイトル"]}
      ],
      "nature_of_disagreement": "対立の背景（日本語1-2文）"
    }
  ],
  "open_questions": [
    {
      "description": "未解明の問い（日本語で具体的に）",
      "why_unresolved": "なぜ未解明か",
      "potential_approach": "検証の可能性"
    }
  ]
}"""


def _build_evidence_summary(evidence_items: List[Dict[str, Any]], papers: List[Dict[str, Any]]) -> str:
    """Build a condensed evidence summary for the synthesis prompt."""
    # Group by paper
    by_paper: Dict[str, List[Dict]] = {}
    for e in evidence_items:
        title = e.get("paper_title", "Unknown")
        by_paper.setdefault(title, []).append(e)

    parts = []
    for title, items in by_paper.items():
        lines = [f"### {title}"]
        for e in items:
            dim = e.get("dimension", "?")
            conf = e.get("confidence", 0)
            claim = e.get("claim_or_point", "")
            evidence = e.get("evidence_text", "")
            relevance = e.get("relevance_to_rq", "")
            lines.append(f"- [{dim}, conf={conf}] {claim}")
            if evidence:
                lines.append(f"  根拠: {evidence[:150]}")
            if relevance:
                lines.append(f"  RQ関連: {relevance[:120]}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _run_synthesis(
    rq_context: RQContext,
    evidence_items: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    llm_client: Any,
) -> Optional[Dict]:
    evidence_text = _build_evidence_summary(evidence_items, papers)

    # Truncate if too long
    if len(evidence_text) > 60_000:
        evidence_text = evidence_text[:60_000] + "\n\n[... truncated ...]"

    user_msg = (
        f"## Research Question\n"
        f"{rq_context.to_prompt_text()}\n\n"
        f"## 論文数: {len(papers)}\n"
        f"## Evidence 数: {len(evidence_items)}\n\n"
        f"## Evidence 一覧\n"
        f"{evidence_text}\n\n"
        f"## 指示\n"
        f"上記の Evidence を統合して、RQ に対する Literature Review を構造化してください。\n"
        f"論文ごとの列挙ではなく、RQ の観点からの統合的な整理をしてください。\n\n"
        f"{_SYNTHESIS_SCHEMA_INSTRUCTION}"
    )

    logger.info("Running synthesis pass (evidence text: %d chars)", len(evidence_text))
    return _llm_call(llm_client, _SYNTHESIS_SYSTEM, user_msg, max_tokens=8192)


# ------------------------------------------------------------------
# Pass 2: Dimensions + Executive Summary
# ------------------------------------------------------------------

_SUMMARY_SYSTEM = """\
あなたは学術研究の分析専門家です。
Literature Review の構造化結果を基に、研究のdimensionと executive summary を生成してください。"""


def _run_dimensions_and_summary(
    rq_context: RQContext,
    synthesis_result: Dict,
    evidence_items: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    llm_client: Any,
) -> Optional[Dict]:
    # Build paper info for dimension extraction
    paper_info = []
    for p in papers:
        info = f"- {p.get('title', '')}"
        if p.get("tags"):
            info += f" [Tags: {p['tags']}]"
        if p.get("methods"):
            info += f" [Methods: {p['methods'][:100]}]"
        paper_info.append(info)

    # Count dimensions from evidence
    dim_counts: Dict[str, int] = {}
    for e in evidence_items:
        d = e.get("dimension", "unknown")
        dim_counts[d] = dim_counts.get(d, 0) + 1

    streams_summary = json.dumps(synthesis_result.get("theoretical_streams", []), ensure_ascii=False, indent=2)

    user_msg = (
        f"## Research Question\n{rq_context.to_prompt_text()}\n\n"
        f"## 論文リスト ({len(papers)} 本)\n" + "\n".join(paper_info) + "\n\n"
        f"## Theoretical Streams (既に抽出済み)\n{streams_summary}\n\n"
        f"## Evidence の dimension 分布\n{json.dumps(dim_counts, indent=2)}\n\n"
        f"## 指示\n"
        f"1. 上記の論文群から research dimensions を抽出してください:\n"
        f"   - theoretical_lens: 使用されている理論的枠組み\n"
        f"   - method: 研究手法 (例: panel regression, DID, case study)\n"
        f"   - dataset: データソースの種類 (例: VC investment data, patent data)\n"
        f"   - context: 地理的・時期的文脈 (例: US, Europe, 2000-2020)\n"
        f"   - research_focus: 主要な研究テーマ\n\n"
        f"2. この RQ に対する Literature Review の executive summary を生成してください。\n"
        f"   5〜10文で、何が分かっていて何が未解明かを簡潔にまとめてください。\n\n"
        f"出力形式 (JSON):\n"
        f'{{"research_dimensions": {{"theoretical_lens": [...], "method": [...], '
        f'"dataset": [...], "context": [...], "research_focus": [...]}}, '
        f'"executive_summary": "日本語で5-10文の要約"}}'
    )

    logger.info("Running dimensions + summary pass")
    return _llm_call(llm_client, _SUMMARY_SYSTEM, user_msg, max_tokens=4096)


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------

def _assemble_result(
    rq_context: RQContext,
    synthesis: Dict,
    dims_summary: Dict,
    evidence_items: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    run_id: str,
) -> LitReviewResult:
    """Assemble LitReviewResult from LLM outputs."""

    # Theoretical streams
    streams = []
    for s in synthesis.get("theoretical_streams", []):
        streams.append(TheoreticalStream(
            name=s.get("name", ""),
            description=s.get("description", ""),
            key_concepts=s.get("key_concepts", []),
            papers=s.get("papers", []),
        ))

    # Findings
    established = []
    for f in synthesis.get("established_findings", []):
        established.append(Finding(
            statement=f.get("statement", ""),
            supporting_papers=f.get("supporting_papers", []),
            evidence_summary=f.get("evidence_summary", ""),
            paper_count=f.get("paper_count", 0),
            strength=f.get("strength", ""),
        ))

    emerging = []
    for f in synthesis.get("emerging_findings", []):
        emerging.append(Finding(
            statement=f.get("statement", ""),
            supporting_papers=f.get("supporting_papers", []),
            evidence_summary=f.get("evidence_summary", ""),
            paper_count=f.get("paper_count", 0),
            strength=f.get("strength", ""),
        ))

    contested = []
    for c in synthesis.get("contested_findings", []):
        contested.append(ContestedPoint(
            topic=c.get("topic", ""),
            positions=c.get("positions", []),
            nature_of_disagreement=c.get("nature_of_disagreement", ""),
        ))

    # Open questions
    open_qs = []
    for q in synthesis.get("open_questions", []):
        open_qs.append(OpenQuestion(
            description=q.get("description", ""),
            why_unresolved=q.get("why_unresolved", ""),
            potential_approach=q.get("potential_approach", ""),
        ))

    # Only include papers that have evidence
    evidence_paper_titles = {e.get("paper_title", "") for e in evidence_items}
    filtered_papers = [p for p in papers if p.get("title", "") in evidence_paper_titles]
    if not filtered_papers:
        filtered_papers = papers  # fallback

    return LitReviewResult(
        rq=rq_context.to_dict(),
        papers=[{
            "title": p.get("title", ""),
            "paper_id": p.get("paper_id", ""),
            "relevance_score": p.get("relevance_score", 0),
        } for p in filtered_papers],
        evidence=evidence_items,
        executive_summary=dims_summary.get("executive_summary", ""),
        theoretical_streams=streams,
        established=established,
        emerging=emerging,
        contested=contested,
        open_questions=open_qs,
        research_dimensions=dims_summary.get("research_dimensions", {}),
        metadata={
            "run_id": run_id,
            "papers_count": len(filtered_papers),
            "evidence_count": len(evidence_items),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


# ------------------------------------------------------------------
# Markdown rendering
# ------------------------------------------------------------------

def _render_markdown(result: LitReviewResult) -> str:
    lines = [
        f"# Literature Review: {result.rq.get('title', '')}",
        f"",
        f"## Executive Summary",
        f"",
        result.executive_summary,
        f"",
        f"---",
        f"",
        f"**対象論文**: {result.metadata.get('papers_count', 0)} 本",
        f"**Evidence 数**: {result.metadata.get('evidence_count', 0)} 件",
        f"",
    ]

    # Theoretical Streams
    lines.extend([f"## 1. 研究の理論的系譜 (Theoretical Streams)", f""])
    for i, s in enumerate(result.theoretical_streams, 1):
        lines.append(f"### 1.{i} {s.name}")
        lines.append(f"")
        lines.append(s.description)
        if s.key_concepts:
            lines.append(f"")
            lines.append(f"**主要概念**: {', '.join(s.key_concepts)}")
        if s.papers:
            lines.append(f"")
            lines.append(f"**関連論文**: {', '.join(s.papers[:5])}")
        lines.append(f"")

    # Established Findings
    lines.extend([f"## 2. 確立された知見 (Established Findings)", f""])
    if result.established:
        for i, f in enumerate(result.established, 1):
            lines.append(f"### 2.{i} {f.statement}")
            lines.append(f"")
            if f.evidence_summary:
                lines.append(f"{f.evidence_summary}")
                lines.append(f"")
            lines.append(f"**論文数**: {f.paper_count} | **確度**: {f.strength}")
            if f.supporting_papers:
                lines.append(f"**論文**: {', '.join(f.supporting_papers[:5])}")
            lines.append(f"")
    else:
        lines.extend(["（確立された知見は特定されませんでした）", ""])

    # Emerging Findings
    lines.extend([f"## 3. 萌芽的知見 (Emerging Findings)", f""])
    if result.emerging:
        for i, f in enumerate(result.emerging, 1):
            lines.append(f"### 3.{i} {f.statement}")
            lines.append(f"")
            if f.evidence_summary:
                lines.append(f"{f.evidence_summary}")
                lines.append(f"")
            lines.append(f"**論文数**: {f.paper_count} | **段階**: {f.strength}")
            if f.supporting_papers:
                lines.append(f"**論文**: {', '.join(f.supporting_papers[:5])}")
            lines.append(f"")
    else:
        lines.extend(["（萌芽的知見は特定されませんでした）", ""])

    # Contested Points
    lines.extend([f"## 4. 論争的知見 (Contested Points)", f""])
    if result.contested:
        for i, c in enumerate(result.contested, 1):
            lines.append(f"### 4.{i} {c.topic}")
            lines.append(f"")
            for pos in c.positions:
                papers_str = ", ".join(pos.get("papers", [])[:3])
                lines.append(f"- **{pos.get('statement', '')}** ({papers_str})")
            lines.append(f"")
            if c.nature_of_disagreement:
                lines.append(f"**対立の背景**: {c.nature_of_disagreement}")
                lines.append(f"")
    else:
        lines.extend(["（論争的知見は特定されませんでした）", ""])

    # Open Questions
    lines.extend([f"## 5. 研究ギャップ (Open Questions)", f""])
    if result.open_questions:
        for i, q in enumerate(result.open_questions, 1):
            lines.append(f"### 5.{i} {q.description}")
            lines.append(f"")
            if q.why_unresolved:
                lines.append(f"**未解明の理由**: {q.why_unresolved}")
            if q.potential_approach:
                lines.append(f"**検証の可能性**: {q.potential_approach}")
            lines.append(f"")
    else:
        lines.extend(["（研究ギャップは特定されませんでした）", ""])

    # Research Dimensions
    dims = result.research_dimensions
    if dims:
        lines.extend([f"## 6. Research Dimensions", f""])
        for cat in ["theoretical_lens", "method", "dataset", "context", "research_focus"]:
            items = dims.get(cat, [])
            if items:
                label = cat.replace("_", " ").title()
                lines.append(f"### {label}")
                for item in items:
                    lines.append(f"- {item}")
                lines.append(f"")

    # Paper list
    lines.extend([f"## 7. 論文一覧", f""])
    lines.append(f"| # | Score | Title |")
    lines.append(f"|---|-------|-------|")
    for i, p in enumerate(result.papers, 1):
        lines.append(f"| {i} | {p.get('relevance_score', '')} | {p.get('title', '')} |")
    lines.append(f"")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def synthesize_lit_review(
    *,
    rq_context: RQContext,
    evidence_items: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    llm_client: Any,
    run_id: str = "",
) -> LitReviewResult:
    """Synthesize a structured Lit Review from evidence.

    Two-pass LLM approach:
      Pass 1: Synthesis (theoretical streams, findings, gaps)
      Pass 2: Dimensions + executive summary
    """
    logger.info("Starting synthesis: %d papers, %d evidence items", len(papers), len(evidence_items))

    # Pass 1: Synthesis
    synthesis = _run_synthesis(rq_context, evidence_items, papers, llm_client)
    if not synthesis:
        logger.error("Synthesis pass failed")
        return LitReviewResult(
            rq=rq_context.to_dict(), papers=[], evidence=evidence_items,
            metadata={"run_id": run_id, "error": "synthesis_failed"},
        )

    logger.info(
        "Synthesis pass complete: %d streams, %d established, %d emerging, %d contested, %d gaps",
        len(synthesis.get("theoretical_streams", [])),
        len(synthesis.get("established_findings", [])),
        len(synthesis.get("emerging_findings", [])),
        len(synthesis.get("contested_findings", [])),
        len(synthesis.get("open_questions", [])),
    )

    # Pass 2: Dimensions + Summary
    dims_summary = _run_dimensions_and_summary(
        rq_context, synthesis, evidence_items, papers, llm_client,
    )
    if not dims_summary:
        logger.warning("Dimensions/summary pass failed; continuing without")
        dims_summary = {"research_dimensions": {}, "executive_summary": ""}

    # Assemble
    result = _assemble_result(rq_context, synthesis, dims_summary, evidence_items, papers, run_id)
    logger.info("Lit Review assembled: summary=%d chars", len(result.executive_summary))

    return result
