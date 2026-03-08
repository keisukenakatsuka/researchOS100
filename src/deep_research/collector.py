"""068 Multi-Source Collector — service logic.

Collects web sources based on search queries from plan.json.
Executes searches via Google CSE (+ optional NewsAPI), deduplicates
results by URL, fetches content, and produces sources.json.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.deep_research import load_step_output, make_source_id
from src.models.research_plan import ResearchPlan
from src.search.web_fetcher import FetchResult, WebFetcher

logger = logging.getLogger("068_collector")

# -- constants -----------------------------------------------------------

MAX_RESULTS_PER_QUERY = 5
MAX_NEWS_RESULTS_PER_QUERY = 3
_PREVIEW_CHARS = 300

# Canonical source_type values:
#   article  — web articles, blog posts, general content
#   news     — news outlets, press releases
#   paper    — academic papers, preprints
#   report   — government / corporate reports
#   webpage  — generic web pages (e.g. product pages, FAQs)
#   dataset  — data portals, datasets
#   unknown  — no URL or unclassifiable
VALID_SOURCE_TYPES = [
    "article", "news", "paper", "report", "webpage", "dataset", "unknown",
]


# -- helpers -------------------------------------------------------------

def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def _infer_source_type(url: str, search_provider: str = "") -> str:
    """Heuristic source_type from URL domain and search provider."""
    if not url:
        return "unknown"
    domain = _extract_domain(url).lower()
    # Academic
    if any(kw in domain for kw in ("arxiv.org", "scholar.google", "researchgate")):
        return "paper"
    # News
    if any(kw in domain for kw in (
        "reuters.", "bbc.", "nytimes.", "nikkei.", "nhk.",
        "cnn.", "apnews.", "bloomberg.", "techcrunch.",
    )):
        return "news"
    if search_provider == "newsapi":
        return "news"
    # Reports
    if domain.endswith(".go.jp") or domain.endswith(".gov"):
        return "report"
    # Default
    return "article"


# -- pipeline steps ------------------------------------------------------

def search_sources(
    plan: ResearchPlan,
    search_client: Any,
    news_client: Any = None,
) -> List[Dict[str, Any]]:
    """Execute plan.search_queries and return raw search results.

    Uses Google CSE as primary backend.  If *news_client* is provided,
    also queries NewsAPI for additional coverage.

    Returns list of raw result dicts (not yet deduplicated).
    """
    raw: List[Dict[str, Any]] = []

    for query in plan.search_queries:
        # Google CSE (primary)
        try:
            cse_hits = search_client.search(query, num=MAX_RESULTS_PER_QUERY)
            for rank, h in enumerate(cse_hits, start=1):
                raw.append({
                    "title": h.get("title", ""),
                    "url": h.get("link", ""),
                    "snippet": h.get("snippet", ""),
                    "published_at": None,
                    "search_query": query,
                    "search_provider": "google_cse",
                    "search_rank": rank,
                })
            logger.info(
                "CSE: %d results for %r", len(cse_hits), query[:60],
            )
        except Exception as e:
            logger.warning("CSE search failed for %r: %s", query[:60], e)

        # NewsAPI (secondary, optional)
        if news_client is not None:
            try:
                news_hits = news_client.search(
                    query, page_size=MAX_NEWS_RESULTS_PER_QUERY,
                )
                for rank, h in enumerate(news_hits, start=1):
                    raw.append({
                        "title": h.get("title", ""),
                        "url": h.get("url", ""),
                        "snippet": h.get("description", ""),
                        "published_at": h.get("publishedAt"),
                        "search_query": query,
                        "search_provider": "newsapi",
                        "search_rank": rank,
                    })
                logger.info(
                    "NewsAPI: %d results for %r", len(news_hits), query[:60],
                )
            except Exception as e:
                logger.warning("NewsAPI failed for %r: %s", query[:60], e)

    logger.info("Search total: %d raw results", len(raw))
    return raw


def deduplicate(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates.  URL is the primary key; title is fallback."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    unique: List[Dict[str, Any]] = []

    for r in raw_results:
        url = r.get("url", "").strip()
        title = r.get("title", "").strip()

        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        else:
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

        unique.append(r)

    logger.info("Dedup: %d → %d sources", len(raw_results), len(unique))
    return unique


def fetch_contents(
    sources: List[Dict[str, Any]],
    fetcher: WebFetcher,
) -> List[Dict[str, Any]]:
    """Fetch web content for every source in-place.

    Adds ``fetch_status``, ``fetch_error``, ``fetched_text``, and
    ``fetched_text_preview`` keys to each source dict.
    """
    for src in sources:
        url = src.get("url", "")
        if not url:
            src["fetch_status"] = "skipped"
            src["fetch_error"] = "no URL"
            src["fetched_text"] = None
            src["fetched_text_preview"] = None
            src["fetched_char_count"] = 0
            continue

        result: FetchResult = fetcher.fetch(url)
        src["fetch_status"] = result.status
        src["fetch_error"] = result.error
        src["fetched_text"] = result.text
        src["fetched_text_preview"] = (
            result.text[:_PREVIEW_CHARS] if result.text else None
        )
        src["fetched_char_count"] = len(result.text) if result.text else 0

        logger.info(
            "Fetch [%s] %s — %s (%d chars)",
            result.status,
            url[:80],
            result.error or "ok",
            src["fetched_char_count"],
        )

    return sources


def _skip_fetch(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mark all sources as fetch-skipped (for --skip-fetch mode)."""
    for src in sources:
        src["fetch_status"] = "skipped"
        src["fetch_error"] = "fetch skipped by user"
        src["fetched_text"] = None
        src["fetched_text_preview"] = None
        src["fetched_char_count"] = 0
    return sources


def build_output(
    run_id: str,
    plan: ResearchPlan,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the final sources.json structure."""
    records: List[Dict[str, Any]] = []
    for seq, src in enumerate(sources, start=1):
        records.append({
            "source_id": make_source_id(run_id, seq),
            "title": src.get("title", ""),
            "url": src.get("url", ""),
            "source_type": _infer_source_type(
                src.get("url", ""), src.get("search_provider", ""),
            ),
            "domain": _extract_domain(src.get("url", "")),
            "published_at": src.get("published_at"),
            "snippet": src.get("snippet"),
            "search_query": src.get("search_query"),
            "search_provider": src.get("search_provider"),
            "search_rank": src.get("search_rank"),
            "fetch_status": src.get("fetch_status", "skipped"),
            "fetch_error": src.get("fetch_error"),
            "fetched_text": src.get("fetched_text"),
            "fetched_text_preview": src.get("fetched_text_preview"),
            "fetched_char_count": src.get("fetched_char_count", 0),
        })

    return {
        "run_id": run_id,
        "request": plan.request,
        "collected_at": datetime.now().isoformat(),
        "total_sources": len(records),
        "sources": records,
    }


# -- main entry point ----------------------------------------------------

def run(
    run_id: str,
    search_client: Any,
    news_client: Any = None,
    fetcher: Optional[WebFetcher] = None,
    *,
    skip_fetch: bool = False,
) -> Dict[str, Any]:
    """Execute the full 068 collector pipeline.

    1. Load plan.json for *run_id*.
    2. Search (Google CSE + optional NewsAPI).
    3. Deduplicate by URL / title.
    4. Fetch web content (unless *skip_fetch*).
    5. Build and return the sources.json dict.

    Args:
        run_id: Research run identifier.
        search_client: GoogleCSEClient instance.
        news_client: Optional NewsAPIClient instance.
        fetcher: Optional WebFetcher (created internally if None).
        skip_fetch: If True, skip content fetching entirely.

    Returns:
        Dict ready to be saved as ``sources.json``.
    """
    # Load plan
    plan_dict = load_step_output(run_id, "067")
    plan = ResearchPlan.from_dict(plan_dict)

    logger.info(
        "=== 068 Collector: run_id=%s, queries=%d ===",
        run_id,
        len(plan.search_queries),
    )

    # Search
    raw = search_sources(plan, search_client, news_client)

    # Deduplicate
    unique = deduplicate(raw)

    # Fetch
    if skip_fetch:
        logger.info("Fetch skipped by user flag")
        _skip_fetch(unique)
    else:
        own_fetcher = fetcher is None
        if own_fetcher:
            fetcher = WebFetcher()
        try:
            fetch_contents(unique, fetcher)
        finally:
            if own_fetcher:
                fetcher.close()

    # Build output
    result = build_output(run_id, plan, unique)

    logger.info(
        "Collector done: %d sources (%d fetched, %d failed, %d skipped)",
        result["total_sources"],
        sum(1 for s in result["sources"] if s["fetch_status"] == "success"),
        sum(1 for s in result["sources"] if s["fetch_status"] == "failed"),
        sum(1 for s in result["sources"] if s["fetch_status"] == "skipped"),
    )
    return result
