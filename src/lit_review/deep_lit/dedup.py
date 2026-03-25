# src/lit_review/deep_lit/dedup.py
"""116 Hypothesis Dedup, Rank, Select — service logic.

Deduplicates raw papers, scores relevance via LLM, and selects
top 100-150 papers per hypothesis.

Usage::

    from src.lit_review.deep_lit.dedup import dedup_rank_select

    result = dedup_rank_select(raw_result, hypothesis, llm_client=client)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.lit_review.deep_lit import (
    _MODEL, normalize_title, paper_uid, parse_json_response,
    SCORING_BATCH_SIZE, DEFAULT_MIN_PAPERS, DEFAULT_MAX_PAPERS,
)

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 0.90


# ------------------------------------------------------------------
# Dedup
# ------------------------------------------------------------------

def dedup_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate papers using 3-level matching.

    Level 1: Exact paper_uid match
    Level 2: Normalized title exact match
    Level 3: Fuzzy title match (>90% SequenceMatcher)

    When merging, prefer the record with longer abstract and higher citation count.
    """
    seen_uids: Dict[str, int] = {}   # uid → index in unique
    seen_titles: Dict[str, int] = {}  # normalized_title → index in unique
    unique: List[Dict[str, Any]] = []
    duplicates = 0

    for p in papers:
        uid = p.get("paper_uid", "")
        norm_title = normalize_title(p.get("title", ""))

        # Level 1: UID match
        if uid and uid in seen_uids:
            _merge_into(unique[seen_uids[uid]], p)
            duplicates += 1
            continue

        # Level 2: Exact title match
        if norm_title and norm_title in seen_titles:
            _merge_into(unique[seen_titles[norm_title]], p)
            duplicates += 1
            continue

        # Level 3: Fuzzy title match
        fuzzy_match = _find_fuzzy_match(norm_title, seen_titles)
        if fuzzy_match is not None:
            _merge_into(unique[fuzzy_match], p)
            duplicates += 1
            continue

        # New unique paper
        idx = len(unique)
        unique.append(p)
        if uid:
            seen_uids[uid] = idx
        if norm_title:
            seen_titles[norm_title] = idx

    logger.info("Dedup: %d raw → %d unique (%d duplicates removed)",
                len(papers), len(unique), duplicates)
    return unique


def _find_fuzzy_match(norm_title: str, seen_titles: Dict[str, int]) -> Optional[int]:
    """Find a fuzzy match in seen titles."""
    if not norm_title or len(norm_title) < 30:
        return None  # Skip fuzzy for very short titles to avoid false positives
    for existing_title, idx in seen_titles.items():
        if len(existing_title) < 30:
            continue
        if SequenceMatcher(None, norm_title, existing_title).ratio() > FUZZY_THRESHOLD:
            return idx
    return None


def _merge_into(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Merge source paper data into target, keeping richer metadata."""
    # Prefer longer abstract
    if len(source.get("abstract", "")) > len(target.get("abstract", "")):
        target["abstract"] = source["abstract"]

    # Prefer higher citation count
    if (source.get("citation_count", 0) or 0) > (target.get("citation_count", 0) or 0):
        target["citation_count"] = source["citation_count"]

    # Merge query angles
    target_angles = set(target.get("query_angles", []))
    target_angles.update(source.get("query_angles", []))
    target["query_angles"] = list(target_angles)

    # Merge query IDs
    target_qids = set(target.get("query_ids", []))
    target_qids.update(source.get("query_ids", []))
    target["query_ids"] = list(target_qids)

    # Fill missing DOI/arXiv
    if not target.get("doi") and source.get("doi"):
        target["doi"] = source["doi"]
    if not target.get("arxiv_id") and source.get("arxiv_id"):
        target["arxiv_id"] = source["arxiv_id"]


# ------------------------------------------------------------------
# Ranking (LLM)
# ------------------------------------------------------------------

_SCORING_SYSTEM = """\
あなたは学術文献の関連性評価の専門家です。
研究仮説に対する各論文の関連度を 0-100 でスコアリングしてください。

スコア基準:
- 90-100: 仮説の直接的な検証・支持・反証に関わる論文
- 70-89: 仮説の理論的基盤・変数・手法に密接に関連
- 50-69: 関連するが間接的
- 30-49: 周辺的に関連
- 0-29: ほぼ無関係"""


def rank_papers(
    papers: List[Dict[str, Any]],
    hypothesis: Dict[str, Any],
    *,
    llm_client: Any,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Score all papers for relevance to hypothesis via LLM batch calls.

    When use_cache is True, cached scores are reused and only uncached
    papers are sent to the LLM. New scores are written back to the cache.
    """
    stmt = hypothesis.get("hypothesis_statement", "")
    hyp_id = hypothesis.get("hypothesis_id", hypothesis.get("id", ""))

    # Load cache
    cache = _load_cache(hyp_id, stmt) if use_cache else {}
    cache_hits = 0

    # Partition: apply cached scores, collect uncached
    uncached: List[int] = []  # indices into papers
    for i, p in enumerate(papers):
        uid = p.get("paper_uid", "")
        if uid and uid in cache:
            p["relevance_score"] = cache[uid]["score"]
            p["relevance_reasoning"] = cache[uid]["reasoning"]
            cache_hits += 1
        else:
            uncached.append(i)

    # Score uncached papers via LLM
    llm_calls = 0
    newly_scored: List[Dict[str, Any]] = []

    for batch_start in range(0, len(uncached), SCORING_BATCH_SIZE):
        batch_indices = uncached[batch_start:batch_start + SCORING_BATCH_SIZE]
        batch = [papers[i] for i in batch_indices]
        scores = _score_batch(batch, stmt, llm_client)
        llm_calls += 1

        for j, idx in enumerate(batch_indices):
            p = papers[idx]
            if scores and j < len(scores):
                p["relevance_score"] = scores[j].get("score", 0)
                p["relevance_reasoning"] = scores[j].get("reasoning", "")
            else:
                p["relevance_score"] = 0
                p["relevance_reasoning"] = "Scoring failed"
            newly_scored.append(p)

    # Update cache with new scores
    if use_cache and newly_scored:
        _update_cache(hyp_id, stmt, newly_scored)

    # Sort by relevance descending
    papers.sort(key=lambda p: p.get("relevance_score", 0), reverse=True)

    # Assign ranks
    for i, p in enumerate(papers, 1):
        p["rank"] = i

    logger.info("Ranked %d papers: %d cached, %d scored in %d LLM calls",
                len(papers), cache_hits, len(newly_scored), llm_calls)
    return papers


def _score_batch(
    batch: List[Dict[str, Any]],
    hypothesis_stmt: str,
    llm_client: Any,
) -> Optional[List[Dict]]:
    """Score a batch of papers."""
    paper_lines = []
    for i, p in enumerate(batch):
        title = p.get("title", "")
        abstract = (p.get("abstract", "") or "")[:300]
        year = p.get("year", "")
        paper_lines.append(f"[P{i}] ({year}) {title}\n  Abstract: {abstract}")

    user_msg = (
        f"## Hypothesis\n{hypothesis_stmt}\n\n"
        f"## Papers ({len(batch)})\n\n"
        + "\n\n".join(paper_lines) + "\n\n"
        f"## Instructions\n"
        f"Score each paper 0-100 for relevance to the hypothesis.\n"
        f'Output JSON: {{"scores": [{{"paper_index": 0, "score": 85, "reasoning": "..."}}]}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _SCORING_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Scoring batch failed: %s", e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    parsed = parse_json_response(resp_text)
    if not parsed:
        return None

    # Sort by paper_index to align with batch order
    scores = parsed.get("scores", [])
    scores.sort(key=lambda s: s.get("paper_index", 0))
    return scores


# ------------------------------------------------------------------
# Selection
# ------------------------------------------------------------------

def select_papers(
    papers: List[Dict[str, Any]],
    *,
    min_papers: int = DEFAULT_MIN_PAPERS,
    max_papers: int = DEFAULT_MAX_PAPERS,
) -> List[Dict[str, Any]]:
    """Select top papers by relevance score."""
    # Already sorted by rank
    for p in papers:
        p["selected"] = False

    selected_count = 0
    for p in papers:
        if selected_count >= max_papers:
            break
        if p.get("relevance_score", 0) > 0:
            p["selected"] = True
            selected_count += 1

    if selected_count < min_papers:
        logger.warning("Only %d papers selected (min=%d). Insufficient unique papers.",
                       selected_count, min_papers)

    logger.info("Selected %d papers (min=%d, max=%d)", selected_count, min_papers, max_papers)
    return papers


# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------

def dedup_rank_select(
    raw_result: Dict[str, Any],
    hypothesis: Dict[str, Any],
    *,
    llm_client: Any,
    min_papers: int = DEFAULT_MIN_PAPERS,
    max_papers: int = DEFAULT_MAX_PAPERS,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Full pipeline: dedup → rank → select."""
    hypothesis_id = raw_result.get("hypothesis_id", "")
    raw_papers = raw_result.get("papers", [])

    # Step 1: Dedup
    unique = dedup_papers(raw_papers)

    # Step 2: Rank
    ranked = rank_papers(unique, hypothesis, llm_client=llm_client, use_cache=use_cache)

    # Step 3: Select
    final = select_papers(ranked, min_papers=min_papers, max_papers=max_papers)

    selected_count = sum(1 for p in final if p.get("selected"))

    return {
        "hypothesis_id": hypothesis_id,
        "dedup_stats": {
            "raw_total": len(raw_papers),
            "duplicates_removed": len(raw_papers) - len(unique),
            "unique_total": len(unique),
            "scored": len(unique),
            "selected": selected_count,
        },
        "papers": final,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": _MODEL,
            "min_papers": min_papers,
            "max_papers": max_papers,
        },
    }


# ------------------------------------------------------------------
# Scoring cache
# ------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache" / "scoring_cache"


def _cache_path(hypothesis_id: str) -> Path:
    """Resolve cache file path for a hypothesis."""
    prefix = hypothesis_id[:12] if hypothesis_id else "unknown"
    return _CACHE_DIR / f"{prefix}.json"


def _stmt_hash(statement: str) -> str:
    """Hash hypothesis statement for invalidation check."""
    return hashlib.sha256(statement.encode()).hexdigest()[:16]


def _load_cache(hypothesis_id: str, statement: str) -> Dict[str, Dict[str, Any]]:
    """Load cached scores for a hypothesis.

    Returns {paper_uid: {"score": int, "reasoning": str}} or empty dict
    if cache is missing or hypothesis statement has changed.
    """
    path = _cache_path(hypothesis_id)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Cache file corrupt, ignoring: %s", path)
        return {}

    # Invalidate if hypothesis statement changed
    if data.get("hypothesis_statement_hash") != _stmt_hash(statement):
        logger.info("Cache invalidated (hypothesis statement changed): %s", path.name)
        return {}

    entries = data.get("entries", {})
    logger.info("Cache loaded: %d entries from %s", len(entries), path.name)
    return entries


def _update_cache(
    hypothesis_id: str,
    statement: str,
    scored_papers: List[Dict[str, Any]],
) -> None:
    """Merge newly scored papers into the cache file."""
    path = _cache_path(hypothesis_id)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}

    stmt_h = _stmt_hash(statement)
    if data.get("hypothesis_statement_hash") != stmt_h:
        # Statement changed — reset cache
        data = {
            "hypothesis_id": hypothesis_id,
            "hypothesis_statement_hash": stmt_h,
            "entries": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    entries = data.setdefault("entries", {})
    for p in scored_papers:
        uid = p.get("paper_uid", "")
        if uid:
            entries[uid] = {
                "score": p.get("relevance_score", 0),
                "reasoning": p.get("relevance_reasoning", ""),
            }

    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("Cache updated: %d total entries in %s", len(entries), path.name)
