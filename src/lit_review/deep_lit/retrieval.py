# src/lit_review/deep_lit/retrieval.py
"""115 Hypothesis Mass Retrieval — service logic.

Executes search queries against Semantic Scholar and arXiv to build
a large raw paper pool per hypothesis.

Usage::

    from src.lit_review.deep_lit.retrieval import retrieve_papers

    result = retrieve_papers(queries, hypothesis_id)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.lit_review.deep_lit import paper_uid, DEFAULT_MAX_PER_QUERY

logger = logging.getLogger(__name__)


def retrieve_papers(
    queries: List[Dict[str, Any]],
    hypothesis_id: str,
    *,
    max_per_query: int = DEFAULT_MAX_PER_QUERY,
) -> Dict[str, Any]:
    """Execute all queries and collect raw papers."""
    all_papers: List[Dict[str, Any]] = []
    s2_total = 0
    arxiv_total = 0
    api_calls = 0

    for q in queries:
        query_text = q.get("query_text", "")
        query_id = q.get("query_id", "")
        targets = q.get("source_targets", ["semantic_scholar"])

        if not query_text:
            continue

        if "semantic_scholar" in targets:
            papers = _search_semantic_scholar(query_text, max_per_query)
            api_calls += 1
            for p in papers:
                p["query_ids"] = [query_id]
                p["query_angles"] = [q.get("angle", "")]
                uid = paper_uid(p)
                p["paper_uid"] = uid
            s2_total += len(papers)
            all_papers.extend(papers)
            time.sleep(1.0)  # Rate limit

        if "arxiv" in targets:
            papers = _search_arxiv(query_text, max_per_query)
            api_calls += 1
            for p in papers:
                p["query_ids"] = [query_id]
                p["query_angles"] = [q.get("angle", "")]
                uid = paper_uid(p)
                p["paper_uid"] = uid
            arxiv_total += len(papers)
            all_papers.extend(papers)
            time.sleep(3.0)  # arXiv rate limit

    logger.info("Retrieved %d raw papers (S2=%d, arXiv=%d) for %s",
                len(all_papers), s2_total, arxiv_total, hypothesis_id)

    return {
        "hypothesis_id": hypothesis_id,
        "retrieval_stats": {
            "queries_executed": len(queries),
            "semantic_scholar_total": s2_total,
            "arxiv_total": arxiv_total,
            "raw_total": len(all_papers),
        },
        "papers": all_papers,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "api_calls": api_calls,
        },
    }


def _search_semantic_scholar(query: str, limit: int) -> List[Dict[str, Any]]:
    """Search Semantic Scholar and normalize results."""
    try:
        from src.search.semantic_scholar import search_papers, normalize_result
        raw = search_papers(query, limit=min(limit, 100))
        return [normalize_result(p) for p in raw]
    except Exception as e:
        logger.warning("Semantic Scholar search failed for '%s': %s", query[:50], e)
        return []


def _search_arxiv(query: str, limit: int) -> List[Dict[str, Any]]:
    """Search arXiv and return normalized results."""
    try:
        from src.search.arxiv import search_arxiv
        return search_arxiv(query, max_results=min(limit, 100))
    except Exception as e:
        logger.warning("arXiv search failed for '%s': %s", query[:50], e)
        return []
