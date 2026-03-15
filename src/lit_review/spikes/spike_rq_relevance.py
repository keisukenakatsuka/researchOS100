#!/usr/bin/env python
"""T0.1 Spike: LLM RQ-Paper Relevance Scoring.

Validates that LLM-based batch relevance scoring can reliably
distinguish RQ-relevant papers from irrelevant ones.

Usage:
    python -m src.lit_review.spikes.spike_rq_relevance
    python -m src.lit_review.spikes.spike_rq_relevance --rq-index 0 --limit 15
    python -m src.lit_review.spikes.spike_rq_relevance --list-rqs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env, get_db_id
from src.notion import (
    build_notion_client_from_env,
    NotionDataSourceResolver,
    extract_property_value as ev,
)
from src.notion.rq_normalize import normalize_rqs
from src.llm.claude_client import build_claude_client_from_env

logger = logging.getLogger("spike_rq_relevance")

MODEL = "claude-sonnet-4-20250514"
BATCH_SIZE = 10  # papers per LLM call


# ----------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------

def fetch_rqs(notion_client, resolver) -> List[Dict[str, Any]]:
    """Fetch and normalize all RQs."""
    rq_db_id = get_db_id("NOTION_RQ_DB_ID")
    resolved = resolver.resolve_once(name="RQ_DB", database_id=rq_db_id)
    pages = notion_client.query_data_source(
        data_source_id=resolved.data_source_id,
        fetch_all=True,
    )
    return normalize_rqs(pages)


def fetch_lit_papers(notion_client, resolver, *, limit: int = 30) -> List[Dict[str, Any]]:
    """Fetch papers from LIT DB with Decision=READ or KEEP."""
    lit_db_id = get_db_id("NOTION_LIT_DB_ID")
    resolved = resolver.resolve_once(name="LIT_DB", database_id=lit_db_id)
    filt = {
        "or": [
            {"property": "Decision", "select": {"equals": "READ"}},
            {"property": "Decision", "select": {"equals": "KEEP"}},
        ]
    }
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
            "decision": ev(page, "Decision") or "",
        })
    logger.info("Fetched %d papers (Decision=READ|KEEP) from LIT DB", len(records))
    if limit and len(records) > limit:
        records = records[:limit]
        logger.info("Limiting to %d papers for spike", limit)
    return records


# ----------------------------------------------------------------
# LLM scoring
# ----------------------------------------------------------------

SYSTEM_PROMPT = """\
あなたは学術研究の関連性を判定する専門家です。
Research Question (RQ) と論文のリストが与えられます。
各論文について、RQ との関連度を 0〜100 のスコアで評価し、理由を簡潔に述べてください。

スコアの目安:
- 80–100: RQ に直接関連。RQ の中心テーマを扱う論文
- 60–79: RQ に間接的に関連。方法論や関連領域で有用
- 40–59: 部分的に関連。一部のコンセプトが共通
- 20–39: 弱い関連。周辺領域
- 0–19: RQ とほぼ無関係"""

def _parse_json_response(text: str):
    """Parse JSON from LLM response, tolerating markdown fences."""
    import re as _re
    text = text.strip()
    m = _re.match(r"```(?:json)?\s*(.*?)\s*```", text, _re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def format_rq_text(rq: Dict[str, Any]) -> str:
    """Format RQ for the prompt."""
    parts = [f"タイトル: {rq['title']}"]
    if rq.get("rationale"):
        parts.append(f"背景: {rq['rationale']}")
    if rq.get("approach"):
        parts.append(f"アプローチ: {rq['approach']}")
    if rq.get("gap"):
        parts.append(f"ギャップ: {rq['gap']}")
    if rq.get("keywords"):
        parts.append(f"キーワード: {', '.join(rq['keywords'])}")
    return "\n".join(parts)


def format_paper_batch(papers: List[Dict[str, Any]], start_idx: int) -> str:
    """Format a batch of papers for the prompt."""
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


def score_batch(
    llm_client,
    rq: Dict[str, Any],
    papers: List[Dict[str, Any]],
    start_idx: int,
) -> List[Dict[str, Any]]:
    """Score a batch of papers against the RQ."""
    rq_text = format_rq_text(rq)
    papers_text = format_paper_batch(papers, start_idx)

    user_msg = (
        f"## Research Question\n{rq_text}\n\n"
        f"## 論文リスト\n{papers_text}\n\n"
        f"上記の各論文について、RQ との関連度スコア (0–100) と理由を JSON で返してください。\n\n"
        f"以下の JSON 形式で出力してください:\n"
        f'{{"scores": [{{"paper_index": 0, "relevance_score": 75, "reasoning": "理由"}}]}}'
    )

    body = {
        "model": MODEL,
        "max_tokens": 4096,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return []

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    parsed = _parse_json_response(resp_text)
    if not parsed or "scores" not in parsed:
        logger.error("Failed to parse LLM response as JSON")
        logger.debug("Raw response: %s", resp_text[:500])
        return []

    scores = parsed["scores"]
    usage = resp.get("usage", {})
    logger.info(
        "Batch scored %d papers (tokens: in=%d, out=%d)",
        len(scores),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    return scores


def score_all_papers(
    llm_client,
    rq: Dict[str, Any],
    papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Score all papers in batches."""
    all_scores = []
    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i : i + BATCH_SIZE]
        logger.info("Scoring batch %d–%d of %d", i, i + len(batch) - 1, len(papers))
        scores = score_batch(llm_client, rq, batch, start_idx=i)
        all_scores.extend(scores)
    return all_scores


# ----------------------------------------------------------------
# Report
# ----------------------------------------------------------------

def print_report(rq: Dict[str, Any], papers: List[Dict[str, Any]], scores: List[Dict[str, Any]]):
    """Print a formatted report of scoring results."""
    # Build lookup
    score_map = {s["paper_index"]: s for s in scores}

    print("\n" + "=" * 70)
    print("T0.1 SPIKE REPORT: RQ-Paper Relevance Scoring")
    print("=" * 70)

    print(f"\n## RQ: {rq['title']}")
    if rq.get("rationale"):
        print(f"   背景: {rq['rationale'][:120]}...")
    if rq.get("gap"):
        print(f"   ギャップ: {rq['gap'][:120]}...")

    print(f"\n## 対象論文数: {len(papers)}")
    print(f"## スコア取得数: {len(scores)}")

    # Merge and sort
    results = []
    for i, p in enumerate(papers):
        s = score_map.get(i, {})
        results.append({
            "index": i,
            "title": p["title"],
            "decision": p.get("decision", ""),
            "score": s.get("relevance_score", -1),
            "reasoning": s.get("reasoning", "(no score)"),
        })
    results.sort(key=lambda x: x["score"], reverse=True)

    # Score distribution
    scored = [r for r in results if r["score"] >= 0]
    if scored:
        scores_list = [r["score"] for r in scored]
        print(f"\n## スコア分布")
        print(f"   最高: {max(scores_list)}")
        print(f"   最低: {min(scores_list)}")
        print(f"   平均: {sum(scores_list) / len(scores_list):.1f}")
        print(f"   中央値: {sorted(scores_list)[len(scores_list) // 2]}")

        # Distribution buckets
        buckets = {"80-100": 0, "60-79": 0, "40-59": 0, "20-39": 0, "0-19": 0}
        for s in scores_list:
            if s >= 80: buckets["80-100"] += 1
            elif s >= 60: buckets["60-79"] += 1
            elif s >= 40: buckets["40-59"] += 1
            elif s >= 20: buckets["20-39"] += 1
            else: buckets["0-19"] += 1
        print(f"\n   分布:")
        for bucket, count in buckets.items():
            bar = "█" * count
            print(f"     {bucket}: {count:2d} {bar}")

    # Top papers
    print(f"\n## 上位論文 (Top 5)")
    for r in results[:5]:
        print(f"   [{r['score']:3d}] {r['title'][:70]}")
        print(f"         理由: {r['reasoning'][:100]}")

    # Bottom papers
    print(f"\n## 下位論文 (Bottom 5)")
    for r in results[-5:]:
        print(f"   [{r['score']:3d}] {r['title'][:70]}")
        print(f"         理由: {r['reasoning'][:100]}")

    # Full results table
    print(f"\n## 全結果")
    print(f"{'#':>3} {'Score':>5} {'Decision':>8}  Title")
    print("-" * 70)
    for r in results:
        print(f"{r['index']:3d} {r['score']:5d} {r['decision']:>8}  {r['title'][:55]}")

    return results


def save_results(rq, papers, scores, results, output_dir: Path):
    """Save raw results to JSON for later analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "rq": {
            "page_id": rq.get("page_id", ""),
            "title": rq["title"],
            "rationale": rq.get("rationale", ""),
            "gap": rq.get("gap", ""),
            "keywords": rq.get("keywords", []),
        },
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "papers_count": len(papers),
        "scores_count": len(scores),
        "results": results,
        "raw_scores": scores,
    }
    path = output_dir / "spike_rq_relevance_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    logger.info("Results saved to %s", path)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.lit_review.spikes.spike_rq_relevance",
        description="T0.1 Spike: RQ-Paper Relevance Scoring",
    )
    p.add_argument("--rq-index", type=int, default=0,
                    help="Index of RQ to use (from sorted list). Default: 0 (first High-priority RQ)")
    p.add_argument("--limit", type=int, default=15,
                    help="Max papers to score. Default: 15")
    p.add_argument("--list-rqs", action="store_true",
                    help="List available RQs and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    load_env()

    notion_client = build_notion_client_from_env()
    resolver = NotionDataSourceResolver(notion_client)

    # Fetch RQs
    all_rqs = fetch_rqs(notion_client, resolver)
    if not all_rqs:
        logger.error("No RQs found in RQ DB")
        return

    # Sort: High priority first
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    all_rqs.sort(key=lambda r: (priority_order.get(r.get("priority", ""), 9), r.get("title", "")))

    if args.list_rqs:
        print("\nAvailable RQs:")
        for i, rq in enumerate(all_rqs):
            print(f"  [{i}] ({rq['priority']:>6}) {rq['title'][:80]}")
        return

    # Select RQ
    if args.rq_index >= len(all_rqs):
        logger.error("RQ index %d out of range (have %d RQs)", args.rq_index, len(all_rqs))
        return
    rq = all_rqs[args.rq_index]
    logger.info("Selected RQ [%d]: %s (priority=%s)", args.rq_index, rq["title"], rq["priority"])

    # Fetch papers
    papers = fetch_lit_papers(notion_client, resolver, limit=args.limit)
    if not papers:
        logger.error("No papers found in LIT DB with Decision=READ|KEEP")
        return

    # Score
    llm_client = build_claude_client_from_env()
    scores = score_all_papers(llm_client, rq, papers)

    # Report
    results = print_report(rq, papers, scores)

    # Save
    output_dir = _PROJECT_ROOT / "data" / "lit_review" / "spikes"
    save_results(rq, papers, scores, results, output_dir)


if __name__ == "__main__":
    main()
