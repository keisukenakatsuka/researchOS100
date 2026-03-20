# src/lit_review/deep_lit/synthesis.py
"""119 Hypothesis Literature Synthesis — service logic.

Generates a structured synthesis per hypothesis from clusters, maps,
and extraction results.

Usage::

    from src.lit_review.deep_lit.synthesis import synthesize_literature

    result = synthesize_literature(hypothesis, clusters, maps, llm_client=client)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.lit_review.deep_lit import _MODEL, parse_json_response

logger = logging.getLogger(__name__)


_SYNTHESIS_SYSTEM = """\
あなたは学術研究の文献レビュー統合の専門家です。
研究仮説に関する大量の文献情報を統合し、構造化された文献レビューを生成してください。

出力は以下の構成で JSON を返してください:
1. executive_summary: 文献全体の概要 (日本語, 200-300語)
2. cluster_syntheses: 各クラスタの貢献要約
3. known_established: 確立された知見 (複数論文で支持)
4. known_contested: 論争中の知見 (方向性が分かれる)
5. unknown_gaps: 未解明の領域
6. implications_for_hypothesis: 仮説への含意
7. implications_for_design: 研究設計への含意"""


def synthesize_literature(
    hypothesis: Dict[str, Any],
    clusters: Dict[str, Any],
    variable_map: Dict[str, Any],
    method_map: Dict[str, Any],
    finding_map: Dict[str, Any],
    *,
    llm_client: Any,
) -> Dict[str, Any]:
    """Generate comprehensive literature synthesis."""
    hypothesis_id = hypothesis.get("hypothesis_id", "")
    stmt = hypothesis.get("hypothesis_statement", "")

    # Build compact context
    context = _build_synthesis_context(hypothesis, clusters, variable_map, method_map, finding_map)

    user_msg = (
        f"{context}\n\n"
        f"## Instructions\n"
        f"Synthesize the above literature information for hypothesis:\n"
        f"{stmt}\n\n"
        f"Output structured JSON with all 7 sections listed in the system prompt.\n"
        f"Write executive_summary, implications_for_hypothesis, implications_for_design in Japanese.\n"
        f"known_established, known_contested, unknown_gaps can be in English or Japanese."
    )

    body = {
        "model": _MODEL,
        "max_tokens": 8192,
        "system": _SYNTHESIS_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Synthesis LLM call failed: %s", e)
        return _empty_synthesis(hypothesis_id, stmt)

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Synthesis: in=%d, out=%d tokens",
                usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    parsed = parse_json_response(resp_text)
    if not parsed:
        logger.warning("Failed to parse synthesis response")
        return _empty_synthesis(hypothesis_id, stmt)

    # Enrich with metadata
    cluster_list = clusters.get("clusters", [])
    total_papers = sum(c.get("paper_count", 0) for c in cluster_list)

    result = {
        "hypothesis_id": hypothesis_id,
        "hypothesis_statement": stmt,
        "total_papers": total_papers,
        "total_clusters": clusters.get("n_clusters", 0),
        "executive_summary": parsed.get("executive_summary", ""),
        "cluster_syntheses": parsed.get("cluster_syntheses", []),
        "known_established": parsed.get("known_established", []),
        "known_contested": parsed.get("known_contested", []),
        "unknown_gaps": parsed.get("unknown_gaps", []),
        "implications_for_hypothesis": parsed.get("implications_for_hypothesis", ""),
        "implications_for_design": parsed.get("implications_for_design", ""),
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": _MODEL,
        },
    }

    return result


def _build_synthesis_context(
    hypothesis: Dict[str, Any],
    clusters: Dict[str, Any],
    variable_map: Dict[str, Any],
    method_map: Dict[str, Any],
    finding_map: Dict[str, Any],
) -> str:
    """Build compact context for synthesis LLM call."""
    parts: List[str] = []

    # Hypothesis
    parts.append(f"## Hypothesis\n{hypothesis.get('hypothesis_statement', '')}")
    parts.append(f"Strategy: {hypothesis.get('strategy', '')}")
    parts.append(f"Rationale: {hypothesis.get('rationale', '')[:300]}")

    # Clusters
    parts.append(f"\n## Clusters ({clusters.get('n_clusters', 0)})")
    for c in clusters.get("clusters", []):
        parts.append(
            f"- {c.get('cluster_name', '')} ({c.get('paper_count', 0)} papers): "
            f"{c.get('description', '')[:100]}"
        )

    # Variables
    parts.append(f"\n## Variables")
    for vtype in ["dependent", "independent", "control"]:
        vars_list = variable_map.get("variables", {}).get(vtype, [])
        if vars_list:
            top = [v.get("name", "") for v in vars_list[:5]]
            parts.append(f"- {vtype}: {', '.join(top)}")

    # Methods
    parts.append(f"\n## Methods")
    for m in method_map.get("methods", [])[:8]:
        parts.append(f"- {m.get('name', '')} ({m.get('paper_count', 0)} papers)")

    # Findings direction
    ds = finding_map.get("direction_summary", {})
    parts.append(
        f"\n## Finding Directions: "
        f"positive={ds.get('positive', 0)}, "
        f"negative={ds.get('negative', 0)}, "
        f"null={ds.get('null', 0)}"
    )

    # Top disagreements
    disagreements = finding_map.get("disagreements", [])
    if disagreements:
        parts.append(f"\n## Disagreements ({len(disagreements)})")
        for d in disagreements[:5]:
            parts.append(f"- {d[:150]}")

    # Common limitations
    limitations = finding_map.get("common_limitations", [])
    if limitations:
        parts.append(f"\n## Common Limitations")
        for l in limitations[:5]:
            parts.append(f"- {l.get('limitation', '')} ({l.get('count', 0)} papers)")

    return "\n".join(parts)


def render_synthesis_md(synthesis: Dict[str, Any]) -> str:
    """Render synthesis as human-readable Markdown."""
    lines = [
        f"# Literature Synthesis",
        f"",
        f"## Hypothesis: {synthesis.get('hypothesis_statement', '')[:80]}",
        f"",
        f"**Papers**: {synthesis.get('total_papers', 0)} | "
        f"**Clusters**: {synthesis.get('total_clusters', 0)}",
        f"",
        f"## Executive Summary",
        f"",
        f"{synthesis.get('executive_summary', '')}",
        f"",
    ]

    # Cluster syntheses
    cs = synthesis.get("cluster_syntheses", [])
    if cs:
        lines.extend([f"## Cluster Syntheses", f""])
        for c in cs:
            if isinstance(c, dict):
                lines.append(f"### {c.get('cluster_id', '')} — {c.get('cluster_name', '')}")
                lines.append(f"{c.get('key_contribution', c.get('synthesis', ''))}")
                lines.append(f"")
            elif isinstance(c, str):
                lines.append(f"- {c}")
        lines.append(f"")

    # Known
    established = synthesis.get("known_established", [])
    if established:
        lines.extend([f"## Established Knowledge", f""])
        for item in established:
            lines.append(f"- {item if isinstance(item, str) else item.get('finding', str(item))}")
        lines.append(f"")

    contested = synthesis.get("known_contested", [])
    if contested:
        lines.extend([f"## Contested Knowledge", f""])
        for item in contested:
            lines.append(f"- {item if isinstance(item, str) else item.get('topic', str(item))}")
        lines.append(f"")

    # Unknown
    gaps = synthesis.get("unknown_gaps", [])
    if gaps:
        lines.extend([f"## Unknown / Gaps", f""])
        for item in gaps:
            lines.append(f"- {item if isinstance(item, str) else item.get('gap', str(item))}")
        lines.append(f"")

    # Implications
    lines.extend([
        f"## Implications for Hypothesis",
        f"",
        f"{synthesis.get('implications_for_hypothesis', '')}",
        f"",
        f"## Implications for Research Design",
        f"",
        f"{synthesis.get('implications_for_design', '')}",
        f"",
    ])

    return "\n".join(lines)


def _empty_synthesis(hypothesis_id: str, stmt: str) -> Dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "hypothesis_statement": stmt,
        "total_papers": 0,
        "total_clusters": 0,
        "executive_summary": "",
        "cluster_syntheses": [],
        "known_established": [],
        "known_contested": [],
        "unknown_gaps": [],
        "implications_for_hypothesis": "",
        "implications_for_design": "",
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": "error",
        },
    }
