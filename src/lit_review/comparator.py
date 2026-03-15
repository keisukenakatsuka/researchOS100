# src/lit_review/comparator.py
"""Cross-RQ comparison (085).

Compares multiple RQ Lit Reviews and Landscapes to identify
shared/unique theoretical streams, findings, blind spots,
and cross-cutting research opportunities.

Usage::

    from src.lit_review.comparator import compare_rqs

    result = compare_rqs(run_dirs, llm_client=client)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
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
class RQSummary:
    """Summary of a single RQ's lit review for comparison."""
    run_id: str
    rq_title: str
    rq_background: str = ""
    theoretical_streams: List[Dict] = field(default_factory=list)
    established: List[Dict] = field(default_factory=list)
    emerging: List[Dict] = field(default_factory=list)
    contested: List[Dict] = field(default_factory=list)
    open_questions: List[Dict] = field(default_factory=list)
    hotspots: List[Dict] = field(default_factory=list)
    blindspots: List[Dict] = field(default_factory=list)
    opportunities: List[Dict] = field(default_factory=list)
    dimensions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of cross-RQ comparison."""
    comparison_id: str
    rqs: List[Dict[str, Any]] = field(default_factory=list)
    executive_summary: str = ""
    shared_theoretical_streams: List[Dict] = field(default_factory=list)
    unique_theoretical_streams: List[Dict] = field(default_factory=list)
    shared_findings: List[Dict] = field(default_factory=list)
    divergent_findings: List[Dict] = field(default_factory=list)
    shared_blindspots: List[Dict] = field(default_factory=list)
    unique_blindspots: List[Dict] = field(default_factory=list)
    cross_rq_opportunities: List[Dict] = field(default_factory=list)
    dimension_comparison: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_rq_summary(run_dir: Path) -> RQSummary:
    """Load all relevant data from a run directory."""
    run_id = run_dir.name

    rq_context = json.loads((run_dir / "rq_context.json").read_text())

    lit_review = {}
    lr_path = run_dir / "lit_review.json"
    if lr_path.exists():
        lit_review = json.loads(lr_path.read_text())

    landscape = {}
    ls_path = run_dir / "landscape.json"
    if ls_path.exists():
        landscape = json.loads(ls_path.read_text())

    findings = lit_review.get("empirical_findings", {})

    return RQSummary(
        run_id=run_id,
        rq_title=rq_context.get("title", ""),
        rq_background=rq_context.get("background", ""),
        theoretical_streams=lit_review.get("theoretical_streams", []),
        established=findings.get("established", []),
        emerging=findings.get("emerging", []),
        contested=findings.get("contested", []),
        open_questions=lit_review.get("open_questions", []),
        hotspots=landscape.get("hotspots", []),
        blindspots=landscape.get("blindspots", []),
        opportunities=landscape.get("research_opportunities", []),
        dimensions=lit_review.get("research_dimensions", {}),
    )


# ------------------------------------------------------------------
# LLM comparison
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


_COMPARISON_SYSTEM = """\
あなたは学術研究の比較分析の専門家です。
複数の Research Question (RQ) に関する Literature Review の結果を横断的に比較し、
共通点・差異・横断的な研究余地を特定してください。

重要な指示:
- 単なる並列ではなく、RQ 間の関係性を分析してください
- 共通の理論基盤がある場合はその構造を明示してください
- 一方の RQ で解明されていて他方で未解明な知見は特に重要です
- 横断的な research opportunities は具体的に「理論×手法×データ×文脈」で示してください"""


def _run_comparison(
    summaries: List[RQSummary],
    llm_client: Any,
) -> Optional[Dict]:
    """LLM-based cross-RQ comparison."""
    # Build per-RQ summary texts
    rq_texts = []
    for s in summaries:
        streams = [st.get("name", "") for st in s.theoretical_streams]
        est = [f.get("statement", "") for f in s.established]
        emg = [f.get("statement", "") for f in s.emerging]
        cont = [c.get("topic", "") for c in s.contested]
        gaps = [q.get("description", "") for q in s.open_questions]
        bs = [b.get("area", "") for b in s.blindspots]

        rq_texts.append(
            f"### RQ: {s.rq_title}\n"
            f"背景: {s.rq_background[:200]}\n"
            f"理論的系譜: {', '.join(streams)}\n"
            f"確立された知見: {'; '.join(est)}\n"
            f"萌芽的知見: {'; '.join(emg)}\n"
            f"論争点: {'; '.join(cont)}\n"
            f"研究ギャップ: {'; '.join(gaps)}\n"
            f"Blind spots: {'; '.join(bs)}\n"
        )

    user_msg = (
        f"## 比較対象 RQ ({len(summaries)} 件)\n\n"
        + "\n\n".join(rq_texts) +
        f"\n\n## 指示\n"
        f"上記の RQ 間を横断的に比較し、以下を JSON で出力してください。\n\n"
        f'{{\n'
        f'  "executive_summary": "全体の比較要約（日本語5-8文）",\n'
        f'  "shared_theoretical_streams": [\n'
        f'    {{"stream": "共通する理論名", "rqs": ["RQ1タイトル", "RQ2タイトル"], "note": "共通性の説明"}}\n'
        f'  ],\n'
        f'  "unique_theoretical_streams": [\n'
        f'    {{"stream": "固有の理論名", "rq": "RQタイトル", "note": "なぜこのRQに固有か"}}\n'
        f'  ],\n'
        f'  "shared_findings": [\n'
        f'    {{"finding": "共通する知見", "rqs": ["RQ1", "RQ2"], "strength": "strong|moderate"}}\n'
        f'  ],\n'
        f'  "divergent_findings": [\n'
        f'    {{"topic": "結果が異なるトピック", "rq_positions": [{{"rq": "RQ1", "finding": "RQ1での結論"}}, ...], "explanation": "なぜ異なるか"}}\n'
        f'  ],\n'
        f'  "shared_blindspots": [\n'
        f'    {{"blindspot": "共通のブラインドスポット", "rqs": ["RQ1", "RQ2"]}}\n'
        f'  ],\n'
        f'  "unique_blindspots": [\n'
        f'    {{"blindspot": "固有のブラインドスポット", "rq": "RQ1"}}\n'
        f'  ],\n'
        f'  "cross_rq_opportunities": [\n'
        f'    {{"theme": "横断的研究テーマ", "relevant_rqs": ["RQ1", "RQ2"], "theory": "理論", "method": "手法", "rationale": "なぜ有望か"}}\n'
        f'  ]\n'
        f'}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 8192,
        "system": _COMPARISON_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Comparison LLM call failed: %s", e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Comparison LLM: in=%d, out=%d tokens", usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    return _parse_json_response(resp_text)


# ------------------------------------------------------------------
# Dimension comparison (deterministic)
# ------------------------------------------------------------------

def _compare_dimensions(summaries: List[RQSummary]) -> Dict[str, Any]:
    """Compare research dimensions across RQs."""
    result = {}
    for cat in ["theoretical_lens", "method", "dataset", "context", "research_focus"]:
        per_rq = {}
        for s in summaries:
            items = s.dimensions.get(cat, [])
            label = s.rq_title[:40]
            per_rq[label] = items if isinstance(items, list) else []
        result[cat] = per_rq
    return result


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------

def compare_rqs(
    run_dirs: List[Path],
    *,
    llm_client: Any,
) -> ComparisonResult:
    """Compare multiple RQ lit reviews and landscapes."""
    # Load
    summaries = []
    for rd in run_dirs:
        try:
            s = load_rq_summary(rd)
            summaries.append(s)
            logger.info("Loaded RQ: %s (run=%s)", s.rq_title[:50], s.run_id)
        except Exception as e:
            logger.error("Failed to load run %s: %s", rd.name, e)

    if len(summaries) < 2:
        raise ValueError(f"Need at least 2 RQs to compare, got {len(summaries)}")

    comparison_id = f"compare_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    logger.info("Comparing %d RQs (id=%s)", len(summaries), comparison_id)

    # LLM comparison
    llm_result = _run_comparison(summaries, llm_client)
    if not llm_result:
        logger.error("LLM comparison failed")
        return ComparisonResult(
            comparison_id=comparison_id,
            rqs=[{"run_id": s.run_id, "title": s.rq_title} for s in summaries],
            metadata={"error": "llm_comparison_failed"},
        )

    # Dimension comparison
    dim_comparison = _compare_dimensions(summaries)

    return ComparisonResult(
        comparison_id=comparison_id,
        rqs=[{"run_id": s.run_id, "title": s.rq_title, "background": s.rq_background} for s in summaries],
        executive_summary=llm_result.get("executive_summary", ""),
        shared_theoretical_streams=llm_result.get("shared_theoretical_streams", []),
        unique_theoretical_streams=llm_result.get("unique_theoretical_streams", []),
        shared_findings=llm_result.get("shared_findings", []),
        divergent_findings=llm_result.get("divergent_findings", []),
        shared_blindspots=llm_result.get("shared_blindspots", []),
        unique_blindspots=llm_result.get("unique_blindspots", []),
        cross_rq_opportunities=llm_result.get("cross_rq_opportunities", []),
        dimension_comparison=dim_comparison,
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rq_count": len(summaries),
        },
    )


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: ComparisonResult) -> str:
    rq_labels = [f"RQ{i+1}" for i in range(len(result.rqs))]
    lines = [
        f"# Cross-RQ Comparison",
        f"",
        f"## 対象 RQ",
        f"",
    ]
    for i, rq in enumerate(result.rqs):
        lines.append(f"- **{rq_labels[i]}**: {rq.get('title', '')}")
    lines.append(f"")

    # Executive summary
    if result.executive_summary:
        lines.extend([f"## Executive Summary", f"", result.executive_summary, f""])

    # Shared theoretical streams
    lines.extend([f"## 1. 共通する理論的系譜", f""])
    for s in result.shared_theoretical_streams:
        rqs = ", ".join(s.get("rqs", [])[:3])
        lines.append(f"### {s.get('stream', '')}")
        lines.append(f"")
        lines.append(f"**RQs**: {rqs}")
        if s.get("note"):
            lines.append(f"")
            lines.append(s["note"])
        lines.append(f"")

    # Unique theoretical streams
    if result.unique_theoretical_streams:
        lines.extend([f"## 2. 各 RQ に固有の理論的系譜", f""])
        for s in result.unique_theoretical_streams:
            lines.append(f"- **{s.get('stream', '')}** → {s.get('rq', '')}: {s.get('note', '')}")
        lines.append(f"")

    # Shared findings
    lines.extend([f"## 3. 共通する知見", f""])
    for f in result.shared_findings:
        rqs = ", ".join(f.get("rqs", [])[:3])
        lines.append(f"### {f.get('finding', '')}")
        lines.append(f"")
        lines.append(f"**RQs**: {rqs} | **確度**: {f.get('strength', '')}")
        lines.append(f"")

    # Divergent findings
    if result.divergent_findings:
        lines.extend([f"## 4. RQ 間で結果が異なる知見", f""])
        for d in result.divergent_findings:
            lines.append(f"### {d.get('topic', '')}")
            lines.append(f"")
            for pos in d.get("rq_positions", []):
                lines.append(f"- **{pos.get('rq', '')[:40]}**: {pos.get('finding', '')}")
            if d.get("explanation"):
                lines.append(f"")
                lines.append(f"**背景**: {d['explanation']}")
            lines.append(f"")

    # Shared blindspots
    lines.extend([f"## 5. 共通するブラインドスポット", f""])
    for b in result.shared_blindspots:
        rqs = ", ".join(b.get("rqs", [])[:3])
        lines.append(f"- **{b.get('blindspot', '')}** ({rqs})")
    lines.append(f"")

    # Unique blindspots
    if result.unique_blindspots:
        lines.extend([f"## 6. 各 RQ に固有のブラインドスポット", f""])
        for b in result.unique_blindspots:
            lines.append(f"- **{b.get('blindspot', '')}** → {b.get('rq', '')}")
        lines.append(f"")

    # Cross-RQ opportunities
    lines.extend([f"## 7. 横断的 Research Opportunities", f""])
    for i, opp in enumerate(result.cross_rq_opportunities, 1):
        lines.append(f"### Opportunity {i}: {opp.get('theme', '')}")
        lines.append(f"")
        rqs = ", ".join(opp.get("relevant_rqs", [])[:3])
        lines.append(f"- **関連 RQs**: {rqs}")
        if opp.get("theory"):
            lines.append(f"- **理論**: {opp['theory']}")
        if opp.get("method"):
            lines.append(f"- **手法**: {opp['method']}")
        if opp.get("rationale"):
            lines.append(f"- **根拠**: {opp['rationale']}")
        lines.append(f"")

    # Dimension comparison
    if result.dimension_comparison:
        lines.extend([f"## 8. Research Dimensions 比較", f""])
        for cat, per_rq in result.dimension_comparison.items():
            label = cat.replace("_", " ").title()
            lines.append(f"### {label}")
            lines.append(f"")
            for rq_label, items in per_rq.items():
                if items:
                    lines.append(f"**{rq_label}**: {', '.join(str(x) for x in items[:8])}")
            lines.append(f"")

    return "\n".join(lines)
