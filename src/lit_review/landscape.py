# src/lit_review/landscape.py
"""Research Landscape mapping (083).

Takes the structured JSON output from 082 (lit_review.json) and produces:
- RQ Knowledge Graph structure (nodes + edges)
- Research Landscape Summary with hotspots, blindspots, opportunities
- Normalized research dimensions

Implementation:
  Pass 1: LLM normalizes dimensions + identifies hotspots/blindspots/opportunities
  Pass 2: Build knowledge graph from normalized data (deterministic)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class GraphNode:
    id: str
    node_type: str  # rq | theoretical_stream | method | dataset | finding | gap
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str  # studied_through | employs | uses_data | supports | identifies | has_gap
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LandscapeResult:
    rq_title: str
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    theoretical_landscape: Dict[str, Any] = field(default_factory=dict)
    methodological_landscape: Dict[str, Any] = field(default_factory=dict)
    data_landscape: Dict[str, Any] = field(default_factory=dict)
    context_landscape: Dict[str, Any] = field(default_factory=dict)
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    blindspots: List[Dict[str, Any]] = field(default_factory=list)
    research_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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
# Pass 1: Normalize dimensions + analyze landscape
# ------------------------------------------------------------------

_LANDSCAPE_SYSTEM = """\
あなたは研究分野のランドスケープ分析の専門家です。
Research Question に関連する論文群から抽出された research dimensions と findings を分析し、
研究分野の構造、偏り、空白を特定してください。

重要な指示:
- 同義語や近い表現は統合してください（例: panel regression ≈ fixed effects panel regression）
- Hotspots は「よく研究されている領域」、Blindspots は「研究が不足している領域」です
- Research Opportunities は、Blindspots を踏まえた「有望な研究テーマ候補」です
- Blindspots は抽象的な「今後の課題」ではなく、具体的に何が・なぜ未解明かを述べてください"""


def _run_landscape_analysis(
    lit_review: Dict[str, Any],
    llm_client: Any,
) -> Optional[Dict]:
    """LLM-based landscape analysis with dimension normalization."""
    rq = lit_review.get("rq_context", {})
    dims = lit_review.get("research_dimensions", {})
    streams = lit_review.get("theoretical_streams", [])
    findings = lit_review.get("empirical_findings", {})
    gaps = lit_review.get("open_questions", [])
    papers = lit_review.get("papers", [])

    user_msg = (
        f"## Research Question\n"
        f"{rq.get('title', '')}\n\n"
        f"## 論文数: {len(papers)}\n\n"
        f"## Theoretical Streams (082 で抽出済み)\n"
        f"{json.dumps(streams, ensure_ascii=False, indent=2)}\n\n"
        f"## Research Dimensions (082 で抽出済み)\n"
        f"{json.dumps(dims, ensure_ascii=False, indent=2)}\n\n"
        f"## Empirical Findings\n"
        f"Established: {len(findings.get('established', []))}\n"
        f"Emerging: {len(findings.get('emerging', []))}\n"
        f"Contested: {len(findings.get('contested', []))}\n\n"
        f"## Open Questions\n"
        f"{json.dumps(gaps, ensure_ascii=False, indent=2)}\n\n"
        f"## 指示\n"
        f"上記の情報を統合して、研究ランドスケープを分析してください。\n\n"
        f"出力形式 (JSON):\n"
        f'{{\n'
        f'  "normalized_dimensions": {{\n'
        f'    "theoretical_lens": [\n'
        f'      {{"canonical": "正規化した名称", "variants": ["元の表現1", "元の表現2"], "paper_count": 3, "description": "概要"}}\n'
        f'    ],\n'
        f'    "method": [\n'
        f'      {{"canonical": "正規化した名称", "variants": ["元の表現1"], "paper_count": 2, "category": "quantitative|qualitative|mixed"}}\n'
        f'    ],\n'
        f'    "dataset": [\n'
        f'      {{"canonical": "カテゴリ名", "variants": ["元の表現1"], "paper_count": 1, "category": "vc_investment|patent|survey|government_program|startup_panel|macro_statistics"}}\n'
        f'    ],\n'
        f'    "context": {{\n'
        f'      "geographic": ["国/地域名"],\n'
        f'      "institutional": ["制度的文脈"],\n'
        f'      "sectoral": ["産業セクター"],\n'
        f'      "temporal": ["時期"]\n'
        f'    }}\n'
        f'  }},\n'
        f'  "hotspots": [\n'
        f'    {{"area": "よく研究されている領域", "evidence": "根拠", "strength": "high|medium"}}\n'
        f'  ],\n'
        f'  "blindspots": [\n'
        f'    {{"area": "不足している領域", "what_is_missing": "何が未解明か", "why_missing": "なぜ未解明か", "severity": "critical|significant|moderate"}}\n'
        f'  ],\n'
        f'  "research_opportunities": [\n'
        f'    {{"theme": "研究テーマ", "theory": "使用する理論", "method": "手法", "data": "必要なデータ", "context": "対象文脈", "rationale": "なぜ有望か"}}\n'
        f'  ]\n'
        f'}}'
    )

    logger.info("Running landscape analysis")
    return _llm_call(llm_client, _LANDSCAPE_SYSTEM, user_msg, max_tokens=8192)


# ------------------------------------------------------------------
# Pass 2: Build knowledge graph (deterministic)
# ------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Create a URL-safe slug from text."""
    # Keep alphanumeric + CJK characters
    slug = re.sub(r"[^\w]+", "_", text.strip(), flags=re.UNICODE).strip("_")[:60]
    if not slug:
        # Fallback for edge cases
        slug = str(hash(text) % 10**8)
    return slug


def _build_knowledge_graph(
    lit_review: Dict[str, Any],
    landscape_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Build knowledge graph from lit_review + landscape analysis."""
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    node_ids = set()

    def add_node(id_: str, type_: str, label: str, **attrs):
        if id_ not in node_ids:
            nodes.append(GraphNode(id=id_, node_type=type_, label=label, attributes=attrs))
            node_ids.add(id_)

    rq = lit_review.get("rq_context", {})
    rq_title = rq.get("title", "RQ")
    rq_id = f"rq:{_slugify(rq_title)}"
    add_node(rq_id, "rq", rq_title)

    # Theoretical streams
    for stream in lit_review.get("theoretical_streams", []):
        s_name = stream.get("name", "")
        s_id = f"stream:{_slugify(s_name)}"
        add_node(s_id, "theoretical_stream", s_name,
                 description=stream.get("description", ""),
                 key_concepts=stream.get("key_concepts", []))
        edges.append(GraphEdge(source=rq_id, target=s_id, relation="studied_through"))

        # Link stream to papers
        for paper_title in stream.get("papers", []):
            p_id = f"paper:{_slugify(paper_title)}"
            add_node(p_id, "paper", paper_title)
            edges.append(GraphEdge(source=s_id, target=p_id, relation="supported_by"))

    # Normalized methods
    norm_dims = landscape_data.get("normalized_dimensions", {})
    for method in norm_dims.get("method", []):
        m_name = method.get("canonical", "")
        m_id = f"method:{_slugify(m_name)}"
        count = method.get("paper_count", 1)
        add_node(m_id, "method", m_name,
                 category=method.get("category", ""),
                 paper_count=count)
        edges.append(GraphEdge(source=rq_id, target=m_id, relation="investigated_by", weight=count))

    # Normalized datasets
    for ds in norm_dims.get("dataset", []):
        d_name = ds.get("canonical", "")
        d_id = f"dataset:{_slugify(d_name)}"
        count = ds.get("paper_count", 1)
        add_node(d_id, "dataset", d_name,
                 category=ds.get("category", ""),
                 paper_count=count)
        edges.append(GraphEdge(source=rq_id, target=d_id, relation="uses_data", weight=count))

    # Findings → connect to streams
    findings = lit_review.get("empirical_findings", {})
    for category, items in [("established", findings.get("established", [])),
                             ("emerging", findings.get("emerging", []))]:
        for f in items:
            stmt = f.get("statement", "")
            f_id = f"finding:{_slugify(stmt)}"
            add_node(f_id, "finding", stmt,
                     category=category,
                     paper_count=f.get("paper_count", 0),
                     strength=f.get("strength", ""))
            edges.append(GraphEdge(source=rq_id, target=f_id, relation="has_finding"))

    # Gaps
    for q in lit_review.get("open_questions", []):
        desc = q.get("description", "")
        g_id = f"gap:{_slugify(desc)}"
        add_node(g_id, "gap", desc, why_unresolved=q.get("why_unresolved", ""))
        edges.append(GraphEdge(source=rq_id, target=g_id, relation="has_gap"))

    # Blindspots from landscape analysis
    for bs in landscape_data.get("blindspots", []):
        area = bs.get("area", "")
        bs_id = f"blindspot:{_slugify(area)}"
        add_node(bs_id, "blindspot", area,
                 what_is_missing=bs.get("what_is_missing", ""),
                 severity=bs.get("severity", ""))
        edges.append(GraphEdge(source=rq_id, target=bs_id, relation="has_blindspot"))

    return {
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
        "summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": dict(Counter(n.node_type for n in nodes)),
            "edge_types": dict(Counter(e.relation for e in edges)),
        },
    }


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------

def _assemble_landscapes(norm_dims: Dict) -> Dict[str, Dict]:
    """Structure normalized dimensions into landscape sections."""
    theoretical = {}
    for t in norm_dims.get("theoretical_lens", []):
        theoretical[t.get("canonical", "")] = {
            "variants": t.get("variants", []),
            "paper_count": t.get("paper_count", 0),
            "description": t.get("description", ""),
        }

    methodological = {"quantitative": [], "qualitative": [], "mixed": []}
    for m in norm_dims.get("method", []):
        cat = m.get("category", "quantitative")
        if cat not in methodological:
            cat = "quantitative"
        methodological[cat].append({
            "name": m.get("canonical", ""),
            "paper_count": m.get("paper_count", 0),
        })

    data = {}
    for d in norm_dims.get("dataset", []):
        cat = d.get("category", "other")
        data.setdefault(cat, []).append({
            "name": d.get("canonical", ""),
            "paper_count": d.get("paper_count", 0),
        })

    context = norm_dims.get("context", {})

    return {
        "theoretical": theoretical,
        "methodological": methodological,
        "data": data,
        "context": context,
    }


# ------------------------------------------------------------------
# Markdown rendering
# ------------------------------------------------------------------

def _render_markdown(result: LandscapeResult) -> str:
    lines = [
        f"# Research Landscape: {result.rq_title}",
        f"",
    ]

    # Executive summary
    lines.extend([
        f"## Executive Summary",
        f"",
        f"本ランドスケープ分析は、RQ「{result.rq_title}」に関連する研究分野の構造を整理し、",
        f"理論的系譜・方法論・データソース・研究文脈の分布と偏りを明らかにする。",
        f"",
    ])

    # Knowledge Graph summary
    kg = result.knowledge_graph
    kg_summary = kg.get("summary", {})
    lines.extend([
        f"**Knowledge Graph**: {kg_summary.get('total_nodes', 0)} nodes, "
        f"{kg_summary.get('total_edges', 0)} edges",
        f"",
    ])

    # Theoretical Landscape
    lines.extend([f"## 1. Theoretical Landscape", f""])
    theo = result.theoretical_landscape
    if theo:
        for name, info in theo.items():
            count = info.get("paper_count", 0)
            desc = info.get("description", "")
            lines.append(f"### {name} ({count} papers)")
            if desc:
                lines.append(f"")
                lines.append(desc)
            variants = info.get("variants", [])
            if variants:
                lines.append(f"")
                lines.append(f"*関連表現*: {', '.join(variants)}")
            lines.append(f"")
    else:
        lines.extend(["（データなし）", ""])

    # Methodological Landscape
    lines.extend([f"## 2. Methodological Landscape", f""])
    meth = result.methodological_landscape
    for cat_label, cat_key in [("定量的手法", "quantitative"), ("質的手法", "qualitative"), ("混合手法", "mixed")]:
        items = meth.get(cat_key, [])
        if items:
            lines.append(f"### {cat_label}")
            for m in sorted(items, key=lambda x: -x.get("paper_count", 0)):
                lines.append(f"- {m['name']} ({m.get('paper_count', 0)} papers)")
            lines.append(f"")

    # Data Landscape
    lines.extend([f"## 3. Data Landscape", f""])
    data = result.data_landscape
    if data:
        for cat, items in data.items():
            cat_label = cat.replace("_", " ").title()
            lines.append(f"### {cat_label}")
            for d in items:
                lines.append(f"- {d['name']} ({d.get('paper_count', 0)} papers)")
            lines.append(f"")
    else:
        lines.extend(["（データなし）", ""])

    # Context Landscape
    lines.extend([f"## 4. Context Landscape", f""])
    ctx = result.context_landscape
    for label, key in [("地理的文脈", "geographic"), ("制度的文脈", "institutional"),
                        ("産業セクター", "sectoral"), ("時期", "temporal")]:
        items = ctx.get(key, [])
        if items:
            lines.append(f"### {label}")
            for item in items:
                lines.append(f"- {item}")
            lines.append(f"")

    # Hotspots
    lines.extend([f"## 5. Hotspots（よく研究されている領域）", f""])
    for h in result.hotspots:
        strength = h.get("strength", "")
        lines.append(f"### {h.get('area', '')} [{strength}]")
        lines.append(f"")
        lines.append(h.get("evidence", ""))
        lines.append(f"")

    # Blindspots
    lines.extend([f"## 6. Blindspots（研究が不足している領域）", f""])
    for b in result.blindspots:
        severity = b.get("severity", "")
        lines.append(f"### {b.get('area', '')} [{severity}]")
        lines.append(f"")
        lines.append(f"**何が未解明か**: {b.get('what_is_missing', '')}")
        lines.append(f"")
        lines.append(f"**なぜ未解明か**: {b.get('why_missing', '')}")
        lines.append(f"")

    # Research Opportunities
    lines.extend([f"## 7. Research Opportunities", f""])
    for i, opp in enumerate(result.research_opportunities, 1):
        lines.append(f"### Opportunity {i}: {opp.get('theme', '')}")
        lines.append(f"")
        lines.append(f"- **理論**: {opp.get('theory', '')}")
        lines.append(f"- **手法**: {opp.get('method', '')}")
        lines.append(f"- **データ**: {opp.get('data', '')}")
        lines.append(f"- **文脈**: {opp.get('context', '')}")
        lines.append(f"- **根拠**: {opp.get('rationale', '')}")
        lines.append(f"")

    # Knowledge Graph Summary
    lines.extend([f"## 8. Knowledge Graph Summary", f""])
    if kg_summary:
        lines.append(f"| Type | Count |")
        lines.append(f"|------|-------|")
        for ntype, count in sorted(kg_summary.get("node_types", {}).items()):
            lines.append(f"| {ntype} | {count} |")
        lines.append(f"")
        lines.append(f"### Edge Types")
        lines.append(f"| Relation | Count |")
        lines.append(f"|----------|-------|")
        for etype, count in sorted(kg_summary.get("edge_types", {}).items()):
            lines.append(f"| {etype} | {count} |")
        lines.append(f"")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def build_research_landscape(
    lit_review_json: Dict[str, Any],
    *,
    llm_client: Any,
) -> LandscapeResult:
    """Build a Research Landscape from 082's lit_review.json output.

    Two passes:
      Pass 1 (LLM): Normalize dimensions, identify hotspots/blindspots/opportunities
      Pass 2 (deterministic): Build knowledge graph
    """
    rq_title = lit_review_json.get("rq_context", {}).get("title", "")
    logger.info("Building landscape for RQ: %s", rq_title[:60])

    # Pass 1: LLM analysis
    landscape_data = _run_landscape_analysis(lit_review_json, llm_client)
    if not landscape_data:
        logger.error("Landscape analysis failed")
        return LandscapeResult(rq_title=rq_title, metadata={"error": "analysis_failed"})

    norm_dims = landscape_data.get("normalized_dimensions", {})
    logger.info(
        "Landscape analysis: %d theories, %d methods, %d datasets, %d hotspots, %d blindspots, %d opportunities",
        len(norm_dims.get("theoretical_lens", [])),
        len(norm_dims.get("method", [])),
        len(norm_dims.get("dataset", [])),
        len(landscape_data.get("hotspots", [])),
        len(landscape_data.get("blindspots", [])),
        len(landscape_data.get("research_opportunities", [])),
    )

    # Pass 2: Build knowledge graph
    knowledge_graph = _build_knowledge_graph(lit_review_json, landscape_data)
    logger.info("Knowledge graph: %d nodes, %d edges",
                knowledge_graph["summary"]["total_nodes"],
                knowledge_graph["summary"]["total_edges"])

    # Assemble landscapes
    landscapes = _assemble_landscapes(norm_dims)

    return LandscapeResult(
        rq_title=rq_title,
        knowledge_graph=knowledge_graph,
        theoretical_landscape=landscapes["theoretical"],
        methodological_landscape=landscapes["methodological"],
        data_landscape=landscapes["data"],
        context_landscape=landscapes["context"],
        hotspots=landscape_data.get("hotspots", []),
        blindspots=landscape_data.get("blindspots", []),
        research_opportunities=landscape_data.get("research_opportunities", []),
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "papers_count": len(lit_review_json.get("papers", [])),
            "evidence_count": len(lit_review_json.get("evidence", [])),
        },
    )
