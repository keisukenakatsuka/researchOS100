# src/notion/rq_normalize.py
"""Normalize and filter Notion Research Question (RQ) pages.

Extracted from notebook 042_weekly_rq_status Cell 08.
Pure functions — no API calls, no side effects.

Usage::

    from src.notion.rq_normalize import normalize_rqs, filter_rqs

    records = normalize_rqs(raw_pages)
    high = filter_rqs(records, priorities={"High"})
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from src.notion.properties import extract_property_value
from src.notion.rq_schema import DEFAULT_TARGET_PRIORITIES

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _extract_keywords(text: str, *, max_k: int = 15) -> List[str]:
    """Simple stopword-filtered keyword extraction.

    Mirrors the pattern in ``events_normalize.py`` but allows a larger
    default ``max_k`` since RQ text tends to be longer and more diverse.
    """
    _STOP = frozenset({
        "this", "that", "with", "from", "into", "over", "will", "have",
        "has", "been", "were", "their", "about", "after", "before",
        "also", "than", "then", "them", "they", "what", "when", "where",
        "which", "while", "said", "says", "the", "and", "for", "are",
        "not", "but", "was", "can", "all", "may", "its", "does", "how",
    })
    tokens = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    seen: List[str] = []
    for t in tokens:
        if t in _STOP or t in seen:
            continue
        seen.append(t)
        if len(seen) >= max_k:
            break
    return seen


def _parse_tags(raw: Any) -> List[str]:
    """Parse Tags value from extract_property_value into a list.

    ``extract_property_value`` returns multi_select as a comma-separated
    string.  We split it back into a list for downstream consumption.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


# ----------------------------------------------------------------
# Normalize a single RQ page
# ----------------------------------------------------------------

def normalize_rq(page: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw Notion RQ page into a flat record dict.

    Returns ``None`` if the page lacks a ``Name`` (title).

    Uses :func:`src.notion.properties.extract_property_value` for
    standard types and adds RQ-specific derived fields.
    """
    page_id = page.get("id", "") or ""
    ev = extract_property_value

    title = ev(page, "Name") or ""
    if not title:
        return None

    status = ev(page, "Status") or ""
    priority = ev(page, "Priority") or ""
    tags = _parse_tags(ev(page, "Tags"))

    # Content fields
    rationale = ev(page, "Rationale / Background") or ""
    approach = ev(page, "Proposed Approach") or ""
    gap = ev(page, "Gap Identified") or ""

    # Derive keywords from all text content for downstream matching
    all_text = " ".join([title, rationale, approach, gap, " ".join(tags)])
    keywords = _extract_keywords(all_text)

    return {
        "page_id": page_id,
        "title": title,
        "status": status,
        "priority": priority,
        "tags": tags,
        "rationale": rationale,
        "approach": approach,
        "gap": gap,
        "keywords": keywords,
    }


def normalize_rqs(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of raw Notion RQ pages.

    Pages missing a title are silently skipped (logged as warning).
    """
    records: List[Dict[str, Any]] = []
    skipped = 0
    for page in pages:
        rec = normalize_rq(page)
        if rec is None:
            skipped += 1
            logger.warning("Skipping RQ page %s: missing Name/title",
                           page.get("id", "?"))
            continue
        records.append(rec)
    logger.info("Normalised %d RQs (skipped %d)", len(records), skipped)
    return records


# ----------------------------------------------------------------
# Filter by priority
# ----------------------------------------------------------------

def filter_rqs(
    rqs: List[Dict[str, Any]],
    *,
    priorities: Set[str] | frozenset = DEFAULT_TARGET_PRIORITIES,
) -> List[Dict[str, Any]]:
    """Filter normalised RQ records by priority.

    Parameters
    ----------
    priorities:
        Set of priority values to keep (case-sensitive).
        Default: ``{"High"}``.
    """
    out = [r for r in rqs if r.get("priority") in priorities]
    logger.info(
        "Filtered RQs: %d → %d (priority in %s)",
        len(rqs), len(out), sorted(priorities),
    )
    return out
