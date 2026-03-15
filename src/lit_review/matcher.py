# src/lit_review/matcher.py
"""RQ-Paper relevance matching (079).

Scores papers from LIT DB against an RQContext and selects
candidates above a relevance threshold.

Design decisions (from T0.1 spike):
- LLM-only scoring (no keyword pre-filter needed at current scale ~200 papers)
- Batch size 10 papers per LLM call
- Threshold default 50
- Prompt-based JSON output (not structured output API)

Usage::

    from src.lit_review.matcher import match_papers

    result = match_papers(
        rq_context=ctx,
        notion_client=notion,
        resolver=resolver,
        llm_client=llm,
    )
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
_BATCH_SIZE = 10
_DEFAULT_THRESHOLD = 50


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class ScoredPaper:
    """A paper with its relevance score against an RQ."""
    paper_id: str
    title: str
    relevance_score: int
    reasoning: str
    decision: str  # "include" | "exclude"
    # Original metadata carried through
    core_idea: str = ""
    findings: str = ""
    methods: str = ""
    tags: str = ""
    source_uid: str = ""
    pdf_link: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    """Result of matching papers against an RQ."""
    rq: Dict[str, Any]
    threshold: int
    total_papers: int
    included: int
    excluded: int
    scored_papers: List[ScoredPaper] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rq": self.rq,
            "threshold": self.threshold,
            "total_papers": self.total_papers,
            "included": self.included,
            "excluded": self.excluded,
            "scored_papers": [p.to_dict() for p in self.scored_papers],
        }

    def included_papers(self) -> List[ScoredPaper]:
        return [p for p in self.scored_papers if p.decision == "include"]

    def to_markdown(self) -> str:
        """Render a readable summary."""
        lines = [
            f"# RQ Paper Matching Results",
            f"",
            f"## RQ: {self.rq.get('title', '')}",
            f"",
            f"- Threshold: {self.threshold}",
            f"- Total papers scored: {self.total_papers}",
            f"- Included: {self.included}",
            f"- Excluded: {self.excluded}",
            f"",
            f"## Included Papers",
            f"",
            f"| # | Score | Title |",
            f"|---|-------|-------|",
        ]
        for i, p in enumerate(self.included_papers(), 1):
            lines.append(f"| {i} | {p.relevance_score} | {p.title} |")

        lines.extend([
            f"",
            f"## All Papers (by score)",
            f"",
            f"| Score | Decision | Title | Reasoning |",
            f"|-------|----------|-------|-----------|",
        ])
        for p in self.scored_papers:
            reason_short = p.reasoning[:80] + "..." if len(p.reasoning) > 80 else p.reasoning
            lines.append(f"| {p.relevance_score} | {p.decision} | {p.title[:60]} | {reason_short} |")

        return "\n".join(lines)


# ------------------------------------------------------------------
# LIT DB query
# ------------------------------------------------------------------

def query_lit_papers(
    notion_client: Any,
    resolver: Any,
    *,
    filter_decisions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch papers from LIT DB.

    Parameters
    ----------
    filter_decisions:
        Decision values to include. Default: ["READ", "KEEP"].
    """
    from src.config import get_db_id
    from src.notion import extract_property_value as ev

    if filter_decisions is None:
        filter_decisions = ["READ", "KEEP"]

    lit_db_id = get_db_id("NOTION_LIT_DB_ID")
    resolved = resolver.resolve_once(name="LIT_DB", database_id=lit_db_id)

    filt = {"or": [
        {"property": "Decision", "select": {"equals": d}}
        for d in filter_decisions
    ]}

    pages = notion_client.query_data_source(
        data_source_id=resolved.data_source_id,
        filter=filt,
        fetch_all=True,
    )

    records = []
    for page in pages:
        name = ev(page, "Name") or ""
        if not name:
            continue
        records.append({
            "page_id": page.get("id", ""),
            "title": name,
            "core_idea": ev(page, "Core Idea") or "",
            "findings": ev(page, "Findings") or "",
            "methods": ev(page, "Methods") or "",
            "tags": ev(page, "Tags") or "",
            "source_uid": ev(page, "Source UID") or "",
            "pdf_link": ev(page, "PDF Link") or "",
        })

    logger.info("Fetched %d papers from LIT DB (Decision in %s)", len(records), filter_decisions)
    return records


# ------------------------------------------------------------------
# LLM Scoring
# ------------------------------------------------------------------

_SCORING_SYSTEM = """\
あなたは学術研究の関連性を判定する専門家です。
Research Question (RQ) と論文のリストが与えられます。
各論文について、RQ との関連度を 0〜100 のスコアで評価し、理由を簡潔に述べてください。

スコアの目安:
- 80–100: RQ に直接関連。RQ の中心テーマを扱う論文
- 60–79: RQ に間接的に関連。方法論や関連領域で有用
- 40–59: 部分的に関連。一部のコンセプトが共通
- 20–39: 弱い関連。周辺領域
- 0–19: RQ とほぼ無関係"""


def _format_paper_batch(papers: List[Dict[str, Any]], start_idx: int) -> str:
    lines = []
    for i, p in enumerate(papers):
        idx = start_idx + i
        entry = f"[{idx}] {p['title']}"
        if p.get("core_idea"):
            entry += f"\n    Core Idea: {p['core_idea']}"
        if p.get("findings"):
            entry += f"\n    Findings: {p['findings'][:200]}"
        if p.get("methods"):
            entry += f"\n    Methods: {p['methods'][:150]}"
        if p.get("tags"):
            entry += f"\n    Tags: {p['tags']}"
        lines.append(entry)
    return "\n\n".join(lines)


def _parse_json_response(text: str) -> Optional[Any]:
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _score_batch(
    llm_client: Any,
    rq_context: RQContext,
    papers: List[Dict[str, Any]],
    start_idx: int,
) -> List[Dict[str, Any]]:
    """Score a batch of papers. Returns list of {paper_index, relevance_score, reasoning}."""
    rq_text = rq_context.to_prompt_text()
    papers_text = _format_paper_batch(papers, start_idx)

    user_msg = (
        f"## Research Question\n{rq_text}\n\n"
        f"## 論文リスト\n{papers_text}\n\n"
        f"上記の各論文について、RQ との関連度スコア (0–100) と理由を JSON で返してください。\n\n"
        f"以下の JSON 形式で出力してください:\n"
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
    except Exception as e:
        logger.error("LLM scoring failed for batch starting at %d: %s", start_idx, e)
        return []

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    parsed = _parse_json_response(resp_text)
    if not parsed or "scores" not in parsed:
        logger.error("Failed to parse scoring JSON for batch starting at %d", start_idx)
        return []

    usage = resp.get("usage", {})
    logger.info(
        "Scored batch %d–%d: %d results (in=%d, out=%d tokens)",
        start_idx, start_idx + len(papers) - 1,
        len(parsed["scores"]),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    return parsed["scores"]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def score_papers(
    rq_context: RQContext,
    papers: List[Dict[str, Any]],
    *,
    llm_client: Any,
    batch_size: int = _BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """Score all papers against the RQ in batches.

    Returns list of {paper_index, relevance_score, reasoning}.
    """
    all_scores = []
    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        logger.info("Scoring batch %d–%d of %d papers", i, i + len(batch) - 1, len(papers))
        scores = _score_batch(llm_client, rq_context, batch, start_idx=i)
        all_scores.extend(scores)
    return all_scores


def match_papers(
    rq_context: RQContext,
    papers: List[Dict[str, Any]],
    scores: List[Dict[str, Any]],
    *,
    threshold: int = _DEFAULT_THRESHOLD,
) -> MatchResult:
    """Apply threshold to scored papers and produce MatchResult.

    Parameters
    ----------
    rq_context:
        The RQ context used for scoring.
    papers:
        Original paper metadata list (same order as scoring input).
    scores:
        List of {paper_index, relevance_score, reasoning} from score_papers.
    threshold:
        Minimum relevance score for inclusion.
    """
    score_map = {s["paper_index"]: s for s in scores}

    scored_papers = []
    for i, p in enumerate(papers):
        s = score_map.get(i)
        if s is None:
            # Paper was not scored (LLM failure) — exclude
            scored_papers.append(ScoredPaper(
                paper_id=p.get("page_id", ""),
                title=p.get("title", ""),
                relevance_score=-1,
                reasoning="(scoring failed)",
                decision="exclude",
                core_idea=p.get("core_idea", ""),
                findings=p.get("findings", ""),
                methods=p.get("methods", ""),
                tags=p.get("tags", ""),
                source_uid=p.get("source_uid", ""),
                pdf_link=p.get("pdf_link", ""),
            ))
            continue

        score_val = s.get("relevance_score", 0)
        decision = "include" if score_val >= threshold else "exclude"

        scored_papers.append(ScoredPaper(
            paper_id=p.get("page_id", ""),
            title=p.get("title", ""),
            relevance_score=score_val,
            reasoning=s.get("reasoning", ""),
            decision=decision,
            core_idea=p.get("core_idea", ""),
            findings=p.get("findings", ""),
            methods=p.get("methods", ""),
            tags=p.get("tags", ""),
            source_uid=p.get("source_uid", ""),
            pdf_link=p.get("pdf_link", ""),
        ))

    # Sort by score descending
    scored_papers.sort(key=lambda x: x.relevance_score, reverse=True)

    included = sum(1 for p in scored_papers if p.decision == "include")
    excluded = len(scored_papers) - included

    return MatchResult(
        rq=rq_context.to_dict(),
        threshold=threshold,
        total_papers=len(scored_papers),
        included=included,
        excluded=excluded,
        scored_papers=scored_papers,
    )
