# src/deep_research/recall.py
"""Knowledge Recall — search past Evidence / Claims for reuse in planning.

MVP implementation uses keyword-based search against Notion DBs.
Called by the planner before generating a new research plan.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("067_recall")

# -- constants ---------------------------------------------------------------

ENV_ENABLE_RECALL = "ENABLE_RESEARCH_RECALL"

MAX_EVIDENCE = 20
MAX_CLAIMS = 10

# Common stop words to filter out (EN + JA particles)
_STOP_WORDS = frozenset({
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "of", "in", "to", "for", "on", "with", "at", "by", "from",
    "and", "or", "not", "no", "it", "its", "this", "that",
    "about", "what", "how", "who", "which", "when", "where",
    "do", "does", "did", "has", "have", "had",
    # Japanese particles / common words
    "の", "は", "が", "を", "に", "で", "と", "も", "か",
    "て", "た", "する", "した", "している", "です", "ます",
    "から", "まで", "より", "など", "という", "について",
    "最近", "最新", "調べて", "教えて", "まとめて",
    "ついて", "ついて調べて", "ついて教えて",
})

# Minimum keyword length to search
_MIN_KW_LEN = 2


# -- keyword extraction ------------------------------------------------------

def extract_keywords(request: str) -> List[str]:
    """Extract search keywords from a request string.

    Simple approach:
    - Split on whitespace, punctuation, and Japanese particles
    - Remove stop words and short tokens
    - Deduplicate while preserving order
    """
    # First split on whitespace and punctuation
    tokens = re.split(r'[\s、。,.\-/;:?!！？「」（）()\[\]]+', request)

    # Further split Japanese text on common particles
    expanded: List[str] = []
    for token in tokens:
        # If token contains CJK, split on particles
        if any("\u3000" <= c <= "\u9fff" for c in token):
            parts = re.split(r'[のはがをにでともか]', token)
            expanded.extend(parts)
        else:
            expanded.append(token)

    seen: set[str] = set()
    keywords: List[str] = []

    for token in expanded:
        token = token.strip()
        if not token or len(token) < _MIN_KW_LEN:
            continue
        lower = token.lower()
        if lower in _STOP_WORDS:
            continue
        if lower in seen:
            continue
        seen.add(lower)
        keywords.append(token)

    return keywords


# -- recall logic ------------------------------------------------------------

def is_recall_enabled() -> bool:
    """Check if recall is enabled via environment variable."""
    return os.getenv(ENV_ENABLE_RECALL, "false").strip().lower() == "true"


def recall_knowledge(
    request: str,
    *,
    notion_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Search existing Knowledge Memory Layer for related entities.

    Args:
        request: The free-form research request text.
        notion_client: A NotionClient instance. Required when recall is enabled.

    Returns:
        Dict with evidence_ids, claim_ids, evidence_hits, claim_hits.
    """
    empty_result: Dict[str, Any] = {
        "evidence_ids": [],
        "claim_ids": [],
        "evidence_hits": [],
        "claim_hits": [],
        "keywords": [],
    }

    if not is_recall_enabled():
        logger.info("[Recall] disabled (ENABLE_RESEARCH_RECALL != 'true')")
        return empty_result

    if notion_client is None:
        logger.warning("[Recall] enabled but no Notion client — skipping")
        return empty_result

    keywords = extract_keywords(request)
    logger.info("[Recall] keywords: %s", " ".join(keywords))

    if not keywords:
        logger.info("[Recall] no keywords extracted — skipping")
        return empty_result

    # Build repos
    evidence_repo, claims_repo = _build_recall_repos(notion_client)

    # Search Evidence
    evidence_hits: List[Dict[str, Any]] = []
    if evidence_repo:
        try:
            evidence_hits = evidence_repo.search_by_keywords(
                keywords, limit=MAX_EVIDENCE,
            )
        except Exception as e:
            logger.warning("[Recall] evidence search failed: %s", e)

    # Search Claims
    claim_hits: List[Dict[str, Any]] = []
    if claims_repo:
        try:
            claim_hits = claims_repo.search_by_keywords(
                keywords, limit=MAX_CLAIMS,
            )
        except Exception as e:
            logger.warning("[Recall] claims search failed: %s", e)

    evidence_ids = [h["evidence_id"] for h in evidence_hits]
    claim_ids = [h["claim_id"] for h in claim_hits]

    logger.info("[Recall] evidence hits: %d", len(evidence_ids))
    logger.info("[Recall] claim hits: %d", len(claim_ids))

    return {
        "evidence_ids": evidence_ids,
        "claim_ids": claim_ids,
        "evidence_hits": evidence_hits,
        "claim_hits": claim_hits,
        "keywords": keywords,
    }


def _build_recall_repos(notion_client: Any):
    """Build EvidenceRepo and ClaimsRepo for recall search."""
    from src.config import get_db_id
    from src.notion.evidence_repo import EvidenceRepo
    from src.notion.claims_repo import ClaimsRepo
    from src.notion.research_schema import ENV_EVIDENCE_DB_ID, ENV_CLAIMS_DB_ID

    def _resolve_ds_id(db_id: str) -> str:
        try:
            db_meta = notion_client.get_database(database_id=db_id)
            ds_list = db_meta.get("data_sources", [])
            if ds_list:
                return ds_list[0]["id"]
        except Exception:
            pass
        return db_id

    evidence_repo = None
    claims_repo = None

    try:
        ev_db = get_db_id(ENV_EVIDENCE_DB_ID)
        evidence_repo = EvidenceRepo(
            client=notion_client,
            database_id=ev_db,
            data_source_id=_resolve_ds_id(ev_db),
        )
    except Exception as e:
        logger.warning("[Recall] could not build EvidenceRepo: %s", e)

    try:
        cl_db = get_db_id(ENV_CLAIMS_DB_ID)
        claims_repo = ClaimsRepo(
            client=notion_client,
            database_id=cl_db,
            data_source_id=_resolve_ds_id(cl_db),
        )
    except Exception as e:
        logger.warning("[Recall] could not build ClaimsRepo: %s", e)

    return evidence_repo, claims_repo
