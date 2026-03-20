# src/lit_review/deep_lit/query.py
"""114 Hypothesis Query Expansion — service logic.

Generates 8-12 diverse search queries per hypothesis for mass retrieval.

Usage::

    from src.lit_review.deep_lit.query import expand_queries

    result = expand_queries(hypothesis, rq_title, llm_client=client)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.lit_review.deep_lit import _MODEL, parse_json_response

logger = logging.getLogger(__name__)

QUERY_ANGLES = [
    "core_mechanism",
    "key_variables",
    "empirical_methods",
    "competing_explanations",
    "institutional_context",
    "adjacent_fields",
    "review_meta",
    "recent_advances",
]

_SYSTEM = """\
あなたは学術文献検索の専門家です。
研究仮説に基づき、Semantic Scholar と arXiv で網羅的に論文を収集するための
多角的な検索クエリを生成してください。

要件:
- 8〜12 個のクエリを生成
- 英語のクエリを生成 (学術検索API向け)
- 以下のアングルをカバーすること:
  * core_mechanism: 仮説の中核メカニズム
  * key_variables: 主要変数と関係性
  * empirical_methods: 実証研究の手法
  * competing_explanations: 代替仮説・対立説
  * institutional_context: 制度・政策・規制の文脈
  * adjacent_fields: 隣接分野・学際的視点
  * review_meta: レビュー論文・メタ分析
  * recent_advances: 直近3年の最新研究

出力形式 (JSON):
{"queries": [
  {"query_id": "q01", "query_text": "...", "angle": "core_mechanism",
   "source_targets": ["semantic_scholar", "arxiv"]}
]}"""


def expand_queries(
    hypothesis: Dict[str, Any],
    rq_title: str,
    *,
    llm_client: Any,
) -> Dict[str, Any]:
    """Generate search queries for a hypothesis."""
    hyp_id = hypothesis.get("hypothesis_id", "")
    stmt = hypothesis.get("hypothesis_statement", "")
    rationale = hypothesis.get("rationale", "")
    gaps = hypothesis.get("source_gaps", [])
    strategy = hypothesis.get("strategy", "")

    user_msg = (
        f"## Research Question\n{rq_title}\n\n"
        f"## Hypothesis\n{stmt}\n\n"
        f"## Rationale\n{rationale}\n\n"
        f"## Strategy: {strategy}\n"
        f"## Source Gaps: {', '.join(gaps[:5]) if gaps else '(none)'}\n\n"
        f"## Instructions\n"
        f"Generate 8-12 diverse search queries for this hypothesis.\n"
        f"Cover all angle categories. Each query should be specific enough to find "
        f"relevant papers but broad enough to return 50+ results.\n"
        f"Output JSON only."
    )

    body = {
        "model": _MODEL,
        "max_tokens": 2048,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Query expansion LLM call failed: %s", e)
        return _fallback_queries(hypothesis, rq_title)

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Query expansion: in=%d, out=%d tokens",
                usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    parsed = parse_json_response(resp_text)
    if not parsed or "queries" not in parsed:
        logger.warning("Failed to parse query expansion response; using fallback")
        return _fallback_queries(hypothesis, rq_title)

    queries = parsed["queries"]
    # Ensure query_ids are assigned
    for i, q in enumerate(queries):
        if not q.get("query_id"):
            q["query_id"] = f"q{i+1:02d}"
        if not q.get("source_targets"):
            q["source_targets"] = ["semantic_scholar"]

    return {
        "hypothesis_id": hyp_id,
        "hypothesis_statement": stmt,
        "queries": queries,
        "total_queries": len(queries),
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": _MODEL,
        },
    }


def _fallback_queries(hypothesis: Dict[str, Any], rq_title: str) -> Dict[str, Any]:
    """Generate basic queries from hypothesis text when LLM fails."""
    stmt = hypothesis.get("hypothesis_statement", "")
    hyp_id = hypothesis.get("hypothesis_id", "")

    # Extract key terms naively
    queries = [
        {"query_id": "q01", "query_text": stmt[:120], "angle": "core_mechanism",
         "source_targets": ["semantic_scholar", "arxiv"]},
        {"query_id": "q02", "query_text": rq_title, "angle": "key_variables",
         "source_targets": ["semantic_scholar"]},
        {"query_id": "q03", "query_text": f"{stmt[:60]} empirical evidence",
         "angle": "empirical_methods", "source_targets": ["semantic_scholar"]},
        {"query_id": "q04", "query_text": f"{stmt[:60]} review meta-analysis",
         "angle": "review_meta", "source_targets": ["semantic_scholar"]},
    ]

    return {
        "hypothesis_id": hyp_id,
        "hypothesis_statement": stmt,
        "queries": queries,
        "total_queries": len(queries),
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": "fallback",
        },
    }
