# src/search/semantic_scholar.py
"""Semantic Scholar API client.

Searches academic papers via Semantic Scholar's public API.
Supports keyword search and paper detail retrieval.

Rate limits:
- Without API key: 100 requests per 5 minutes
- With API key: 1 request per second

Usage::

    from src.search.semantic_scholar import search_papers

    results = search_papers("government venture capital ecosystem")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_SEARCH_FIELDS = "paperId,title,authors,year,abstract,citationCount,externalIds,url"


def _get_api_key() -> Optional[str]:
    return os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None


def _headers() -> Dict[str, str]:
    h = {"Accept": "application/json"}
    key = _get_api_key()
    if key:
        h["x-api-key"] = key
    return h


def search_papers(
    query: str,
    *,
    limit: int = 20,
    fields: str = _SEARCH_FIELDS,
    year_range: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search Semantic Scholar for papers.

    Parameters
    ----------
    query: Search query string.
    limit: Max results (API max 100).
    fields: Comma-separated fields to return.
    year_range: Optional year filter, e.g. "2015-2024" or "2020-".

    Returns list of paper dicts with requested fields.
    """
    params: Dict[str, Any] = {
        "query": query,
        "limit": min(limit, 100),
        "fields": fields,
    }
    if year_range:
        params["year"] = year_range

    for attempt in range(3):
        try:
            resp = requests.get(
                f"{_BASE_URL}/paper/search",
                params=params,
                headers=_headers(),
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning("Semantic Scholar rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data", [])
            logger.info("Semantic Scholar: %d results for '%s'", len(papers), query[:50])
            return papers
        except requests.RequestException as e:
            logger.warning("Semantic Scholar search attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    logger.error("Semantic Scholar search failed after retries for '%s'", query[:50])
    return []


def normalize_result(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Semantic Scholar result to common format."""
    authors = paper.get("authors", [])
    author_str = ", ".join(a.get("name", "") for a in authors[:5])
    if len(authors) > 5:
        author_str += " et al."

    ext_ids = paper.get("externalIds", {}) or {}
    arxiv_id = ext_ids.get("ArXiv", "")
    doi = ext_ids.get("DOI", "")

    return {
        "title": paper.get("title", ""),
        "authors": author_str,
        "year": paper.get("year"),
        "abstract": paper.get("abstract", "") or "",
        "citation_count": paper.get("citationCount", 0),
        "source": "semantic_scholar",
        "source_id": paper.get("paperId", ""),
        "arxiv_id": arxiv_id,
        "doi": doi,
        "url": paper.get("url", ""),
    }
