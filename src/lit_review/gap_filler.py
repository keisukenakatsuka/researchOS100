# src/lit_review/gap_filler.py
"""Literature Gap Filler (080).

Searches external sources (Semantic Scholar, arXiv) for papers
related to the RQ but not yet in LIT DB, and produces a candidate
list for supplementation.

Usage::

    from src.lit_review.gap_filler import fill_gaps

    result = fill_gaps(rq_context, lit_papers, llm_client=client)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.lit_review.rq_context import RQContext

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class GapCandidate:
    """A paper found via external search."""
    title: str
    authors: str = ""
    year: Optional[int] = None
    abstract: str = ""
    source: str = ""  # "semantic_scholar" | "arxiv"
    source_id: str = ""
    arxiv_id: str = ""
    doi: str = ""
    url: str = ""
    citation_count: int = 0
    relevance_score: int = 0
    reasoning: str = ""
    is_in_lit: bool = False
    decision: str = ""  # "add" | "skip" | "duplicate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GapFillerResult:
    """Result of gap filling."""
    rq_title: str
    search_queries: List[str] = field(default_factory=list)
    total_found: int = 0
    duplicates_removed: int = 0
    scored: int = 0
    recommended: int = 0
    candidates: List[GapCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rq_title": self.rq_title,
            "search_queries": self.search_queries,
            "total_found": self.total_found,
            "duplicates_removed": self.duplicates_removed,
            "scored": self.scored,
            "recommended": self.recommended,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def recommended_papers(self) -> List[GapCandidate]:
        return [c for c in self.candidates if c.decision == "add"]

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# Query generation
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


def generate_search_queries(
    rq_context: RQContext,
    *,
    llm_client: Any,
    num_queries: int = 4,
) -> List[str]:
    """Generate search queries from RQ context using LLM."""
    user_msg = (
        f"Research Question: {rq_context.title}\n"
        f"背景: {rq_context.background}\n"
        f"ギャップ: {rq_context.gap}\n"
        f"キーワード: {', '.join(rq_context.keywords)}\n\n"
        f"上記の RQ に関連する学術論文を検索するための英語クエリを {num_queries} 本生成してください。\n"
        f"各クエリは異なる角度からカバーしてください（理論的、実証的、政策的など）。\n"
        f"クエリは Semantic Scholar / arXiv で検索可能な簡潔な英語にしてください。\n\n"
        f'出力形式 (JSON): {{"queries": ["query1", "query2", ...]}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 512,
        "system": "You are an academic search query generator.",
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
        text = ""
        for block in resp.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")
                break
        parsed = _parse_json_response(text)
        if parsed and "queries" in parsed:
            queries = parsed["queries"][:num_queries]
            logger.info("Generated %d search queries", len(queries))
            return queries
    except Exception as e:
        logger.warning("Query generation failed: %s", e)

    # Fallback: use RQ title + keywords
    fallback = [rq_context.title]
    if rq_context.keywords:
        fallback.append(" ".join(rq_context.keywords[:5]))
    logger.info("Using fallback queries: %d", len(fallback))
    return fallback


# ------------------------------------------------------------------
# External search
# ------------------------------------------------------------------

def _search_all_sources(
    queries: List[str],
    *,
    max_per_query: int = 15,
    sources: List[str] = None,
) -> List[Dict[str, Any]]:
    """Search Semantic Scholar and arXiv with multiple queries."""
    if sources is None:
        sources = ["semantic_scholar", "arxiv"]

    all_results: List[Dict[str, Any]] = []

    for query in queries:
        if "semantic_scholar" in sources:
            try:
                from src.search.semantic_scholar import search_papers, normalize_result
                raw = search_papers(query, limit=max_per_query)
                for r in raw:
                    all_results.append(normalize_result(r))
            except Exception as e:
                logger.warning("Semantic Scholar search failed for '%s': %s", query[:40], e)

        if "arxiv" in sources:
            try:
                from src.search.arxiv import search_arxiv
                results = search_arxiv(query, max_results=max_per_query)
                all_results.extend(results)
            except Exception as e:
                logger.warning("arXiv search failed for '%s': %s", query[:40], e)

    logger.info("Total raw results from all sources: %d", len(all_results))
    return all_results


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    """Normalize title for dedup comparison."""
    return re.sub(r"[^a-z0-9 ]", "", title.lower().strip())


def deduplicate_results(
    external: List[Dict[str, Any]],
    lit_papers: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    """Remove duplicates within external results and against LIT DB.

    Returns (unique_external, duplicates_removed).
    """
    # Build LIT title set
    lit_titles = {_normalize_title(p.get("title", "")) for p in lit_papers}
    lit_titles.discard("")

    seen_titles: set = set()
    seen_ids: set = set()
    unique = []
    dups = 0

    for paper in external:
        title_norm = _normalize_title(paper.get("title", ""))
        if not title_norm:
            dups += 1
            continue

        # Check against LIT DB
        if title_norm in lit_titles:
            dups += 1
            continue

        # Check within external results
        if title_norm in seen_titles:
            dups += 1
            continue

        # Check by source_id / arxiv_id
        sid = paper.get("source_id", "")
        aid = paper.get("arxiv_id", "")
        if sid and sid in seen_ids:
            dups += 1
            continue
        if aid and aid in seen_ids:
            dups += 1
            continue

        seen_titles.add(title_norm)
        if sid:
            seen_ids.add(sid)
        if aid:
            seen_ids.add(aid)
        unique.append(paper)

    logger.info("Dedup: %d → %d unique (%d removed)", len(external), len(unique), dups)
    return unique, dups


# ------------------------------------------------------------------
# Relevance scoring (reuse matcher logic)
# ------------------------------------------------------------------

_SCORING_SYSTEM = """\
あなたは学術研究の関連性を判定する専門家です。
Research Question (RQ) と論文のリストが与えられます。
各論文について、RQ との関連度を 0〜100 のスコアで評価し、理由を簡潔に述べてください。

スコアの目安:
- 80–100: RQ に直接関連
- 60–79: 間接的に関連
- 40–59: 部分的に関連
- 20–39: 弱い関連
- 0–19: ほぼ無関係"""


def score_candidates(
    rq_context: RQContext,
    papers: List[Dict[str, Any]],
    *,
    llm_client: Any,
    batch_size: int = 10,
) -> List[Dict[str, Any]]:
    """Score external papers against RQ. Returns list of {paper_index, relevance_score, reasoning}."""
    all_scores = []
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        lines = []
        for j, p in enumerate(batch):
            idx = i + j
            entry = f"[{idx}] {p.get('title', '')}"
            abstract = p.get("abstract", "")
            if abstract:
                entry += f"\n    Abstract: {abstract[:200]}"
            lines.append(entry)

        user_msg = (
            f"## Research Question\n{rq_context.to_prompt_text()}\n\n"
            f"## 論文リスト\n" + "\n\n".join(lines) + "\n\n"
            f"上記の各論文について、RQ との関連度スコア (0–100) と理由を JSON で返してください。\n\n"
            f'{{"scores": [{{"paper_index": 0, "relevance_score": 75, "reasoning": "理由"}}]}}'
        )

        body = {
            "model": _MODEL,
            "max_tokens": 4096,
            "system": _SCORING_SYSTEM,
            "messages": [{"role": "user", "content": user_msg}],
        }

        try:
            resp = llm_client.messages_create(body=body)
            text = ""
            for block in resp.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    break
            parsed = _parse_json_response(text)
            if parsed and "scores" in parsed:
                all_scores.extend(parsed["scores"])
                logger.info("Scored batch %d–%d: %d results", i, i + len(batch) - 1, len(parsed["scores"]))
        except Exception as e:
            logger.warning("Scoring batch %d failed: %s", i, e)

    return all_scores


# ------------------------------------------------------------------
# Main API
# ------------------------------------------------------------------

def fill_gaps(
    rq_context: RQContext,
    lit_papers: List[Dict[str, Any]],
    *,
    llm_client: Any,
    max_results: int = 50,
    threshold: int = 60,
    sources: Optional[List[str]] = None,
) -> GapFillerResult:
    """Find papers related to RQ but not in LIT DB.

    Parameters
    ----------
    rq_context: The RQ to search for.
    lit_papers: Existing LIT DB papers (for dedup).
    llm_client: Claude client.
    max_results: Max papers to search per source per query.
    threshold: Min relevance score to recommend.
    sources: Search sources (default: ["semantic_scholar", "arxiv"]).
    """
    # Step 1: Generate queries
    queries = generate_search_queries(rq_context, llm_client=llm_client)

    # Step 2: Search
    raw_results = _search_all_sources(queries, max_per_query=max_results // len(queries), sources=sources)

    # Step 3: Deduplicate
    unique, dups_removed = deduplicate_results(raw_results, lit_papers)

    # Step 4: Score
    scores = score_candidates(rq_context, unique, llm_client=llm_client)
    score_map = {s["paper_index"]: s for s in scores}

    # Step 5: Build candidates
    candidates = []
    for i, paper in enumerate(unique):
        s = score_map.get(i)
        score = s.get("relevance_score", 0) if s else 0
        reasoning = s.get("reasoning", "") if s else "(not scored)"
        decision = "add" if score >= threshold else "skip"

        candidates.append(GapCandidate(
            title=paper.get("title", ""),
            authors=paper.get("authors", ""),
            year=paper.get("year"),
            abstract=paper.get("abstract", "")[:500],
            source=paper.get("source", ""),
            source_id=paper.get("source_id", ""),
            arxiv_id=paper.get("arxiv_id", ""),
            doi=paper.get("doi", ""),
            url=paper.get("url", ""),
            citation_count=paper.get("citation_count", 0),
            relevance_score=score,
            reasoning=reasoning,
            decision=decision,
        ))

    candidates.sort(key=lambda c: c.relevance_score, reverse=True)
    recommended = sum(1 for c in candidates if c.decision == "add")

    logger.info("Gap filler: %d found, %d unique, %d scored, %d recommended (threshold=%d)",
                len(raw_results), len(unique), len(scores), recommended, threshold)

    return GapFillerResult(
        rq_title=rq_context.title,
        search_queries=queries,
        total_found=len(raw_results),
        duplicates_removed=dups_removed,
        scored=len(scores),
        recommended=recommended,
        candidates=candidates,
    )


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: GapFillerResult) -> str:
    lines = [
        f"# Literature Gap Filler Results",
        f"",
        f"## RQ: {result.rq_title}",
        f"",
        f"## Search Queries",
        f"",
    ]
    for i, q in enumerate(result.search_queries, 1):
        lines.append(f"{i}. {q}")
    lines.extend([
        f"",
        f"## Summary",
        f"",
        f"- Total found: {result.total_found}",
        f"- Duplicates removed: {result.duplicates_removed}",
        f"- Scored: {result.scored}",
        f"- **Recommended for addition: {result.recommended}**",
        f"",
    ])

    recommended = result.recommended_papers()
    if recommended:
        lines.extend([f"## Recommended Papers", f""])
        lines.append(f"| # | Score | Year | Source | Title |")
        lines.append(f"|---|-------|------|--------|-------|")
        for i, c in enumerate(recommended, 1):
            lines.append(f"| {i} | {c.relevance_score} | {c.year or '?'} | {c.source} | {c.title[:60]} |")
        lines.append(f"")

        for c in recommended:
            lines.extend([
                f"### [{c.relevance_score}] {c.title}",
                f"",
                f"- Authors: {c.authors}",
                f"- Year: {c.year or '?'}",
                f"- Source: {c.source} ({c.source_id})",
                f"- Citations: {c.citation_count}",
                f"- Reasoning: {c.reasoning}",
                f"",
            ])

    skipped = [c for c in result.candidates if c.decision == "skip"]
    if skipped:
        lines.extend([f"## Skipped Papers (below threshold)", f""])
        for c in skipped[:10]:
            lines.append(f"- [{c.relevance_score}] {c.title[:70]} ({c.source})")
        if len(skipped) > 10:
            lines.append(f"- ... and {len(skipped) - 10} more")
        lines.append(f"")

    return "\n".join(lines)
