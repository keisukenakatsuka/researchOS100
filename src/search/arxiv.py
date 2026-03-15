# src/search/arxiv.py
"""arXiv API client.

Searches papers via arXiv's Atom feed API.
Free, no API key required. Rate limit: ~3 seconds between requests.

Usage::

    from src.search.arxiv import search_arxiv

    results = search_arxiv("sovereign wealth fund venture capital")
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def search_arxiv(
    query: str,
    *,
    max_results: int = 20,
    categories: Optional[List[str]] = None,
    sort_by: str = "relevance",
) -> List[Dict[str, Any]]:
    """Search arXiv for papers.

    Parameters
    ----------
    query: Search query.
    max_results: Max results (API max 300).
    categories: Optional category filter, e.g. ["q-fin.GN", "econ.GN"].
    sort_by: "relevance" or "lastUpdatedDate" or "submittedDate".

    Returns list of normalized paper dicts.
    """
    search_query = f"all:{query}"
    if categories:
        cat_filter = " OR ".join(f"cat:{c}" for c in categories)
        search_query = f"({search_query}) AND ({cat_filter})"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": min(max_results, 300),
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    for attempt in range(3):
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            papers = _parse_atom_feed(resp.text)
            logger.info("arXiv: %d results for '%s'", len(papers), query[:50])
            return papers
        except requests.RequestException as e:
            logger.warning("arXiv search attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    logger.error("arXiv search failed after retries for '%s'", query[:50])
    return []


def _parse_atom_feed(xml_text: str) -> List[Dict[str, Any]]:
    """Parse arXiv Atom feed XML into paper dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error("arXiv XML parse failed: %s", e)
        return []

    papers = []
    for entry in root.findall("atom:entry", _NS):
        title_el = entry.find("atom:title", _NS)
        summary_el = entry.find("atom:summary", _NS)
        published_el = entry.find("atom:published", _NS)
        id_el = entry.find("atom:id", _NS)

        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
        abstract = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
        published = (published_el.text or "") if published_el is not None else ""
        entry_id = (id_el.text or "") if id_el is not None else ""

        # Extract arXiv ID from URL
        arxiv_id = ""
        m = re.search(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", entry_id)
        if m:
            arxiv_id = m.group(1)

        # Extract year
        year = None
        if published:
            ym = re.match(r"(\d{4})", published)
            if ym:
                year = int(ym.group(1))

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", _NS):
            name_el = author_el.find("atom:name", _NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        author_str = ", ".join(authors[:5])
        if len(authors) > 5:
            author_str += " et al."

        # PDF link
        pdf_link = ""
        for link_el in entry.findall("atom:link", _NS):
            if link_el.get("title") == "pdf":
                pdf_link = link_el.get("href", "")
                break

        papers.append({
            "title": title,
            "authors": author_str,
            "year": year,
            "abstract": abstract,
            "citation_count": 0,
            "source": "arxiv",
            "source_id": arxiv_id,
            "arxiv_id": arxiv_id,
            "doi": "",
            "url": entry_id,
            "pdf_link": pdf_link,
        })

    return papers
