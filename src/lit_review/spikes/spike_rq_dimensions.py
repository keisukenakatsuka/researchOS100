#!/usr/bin/env python
"""T0.4 Spike: Research Dimension Extraction.

Validates that LLM can extract structured research dimensions
(theoretical lenses, methods, datasets, contexts) from a set of
RQ-related papers for downstream landscape mapping (083).

Usage:
    python -m src.lit_review.spikes.spike_rq_dimensions
    python -m src.lit_review.spikes.spike_rq_dimensions --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger("spike_rq_dimensions")

MODEL = "claude-sonnet-4-20250514"

DIMENSION_CATEGORIES = [
    "theoretical_lens",
    "method",
    "dataset",
    "context",
    "research_focus",
]


# ----------------------------------------------------------------
# Data fetching (reuse patterns from T0.1/T0.2)
# ----------------------------------------------------------------

def fetch_rq(notion_client, resolver, rq_index: int = 0) -> Dict[str, Any]:
    rq_db_id = get_db_id("NOTION_RQ_DB_ID")
    resolved = resolver.resolve_once(name="RQ_DB", database_id=rq_db_id)
    pages = notion_client.query_data_source(
        data_source_id=resolved.data_source_id, fetch_all=True,
    )
    all_rqs = normalize_rqs(pages)
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    all_rqs.sort(key=lambda r: (priority_order.get(r.get("priority", ""), 9), r.get("title", "")))
    return all_rqs[rq_index]


def fetch_lit_papers(notion_client, resolver) -> List[Dict[str, Any]]:
    lit_db_id = get_db_id("NOTION_LIT_DB_ID")
    resolved = resolver.resolve_once(name="LIT_DB", database_id=lit_db_id)
    filt = {
        "or": [
            {"property": "Decision", "select": {"equals": "READ"}},
            {"property": "Decision", "select": {"equals": "KEEP"}},
        ]
    }
    pages = notion_client.query_data_source(
        data_source_id=resolved.data_source_id, filter=filt, fetch_all=True,
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
            "notes": ev(page, "Notes") or "",
            "tags": ev(page, "Tags") or "",
        })
    return records


def select_papers_by_t01_scores(
    all_papers: List[Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Select top papers using T0.1 relevance scores."""
    t01_path = _PROJECT_ROOT / "data" / "lit_review" / "spikes" / "spike_rq_relevance_results.json"
    if not t01_path.exists():
        logger.warning("T0.1 results not found, using first %d papers", limit)
        return all_papers[:limit]

    t01_data = json.loads(t01_path.read_text())
    t01_results = t01_data.get("results", [])
    score_map = {r["title"]: r["score"] for r in t01_results}

    # Score all papers
    scored = []
    for p in all_papers:
        s = score_map.get(p["title"], -1)
        p_copy = dict(p)
        p_copy["t01_score"] = s
        scored.append(p_copy)

    # Only use papers that were scored in T0.1, sorted by score desc
    scored_only = [x for x in scored if x["t01_score"] > 0]
    scored_only.sort(key=lambda x: -x["t01_score"])

    if len(scored_only) < limit:
        # If not enough T0.1-scored papers, run a quick relevance screen
        # on additional papers using the same RQ
        logger.info(
            "Only %d T0.1-scored papers available (need %d). Using all scored papers.",
            len(scored_only), limit,
        )
        return scored_only

    return scored_only[:limit]


def get_paper_text(paper: Dict[str, Any]) -> str:
    """Get best available text for a paper (metadata only for this spike)."""
    parts = [f"Title: {paper['title']}"]
    if paper.get("core_idea"):
        parts.append(f"Core Idea: {paper['core_idea']}")
    if paper.get("findings"):
        parts.append(f"Findings: {paper['findings']}")
    if paper.get("methods"):
        parts.append(f"Methods: {paper['methods']}")
    if paper.get("notes"):
        parts.append(f"Notes: {paper['notes']}")
    if paper.get("tags"):
        parts.append(f"Tags: {paper['tags']}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------
# LLM Dimension Extraction
# ----------------------------------------------------------------

DIMENSION_SYSTEM = """\
あなたは学術研究の分析専門家です。
Research Question (RQ) の観点から、複数の論文を横断的に分析し、
研究分野を構成する research dimensions を抽出してください。

抽出する dimension:

1. theoretical_lens: この研究が依拠する理論的枠組み・概念フレームワーク
   例: institutional theory, network theory, resource-based view, signaling theory
   - 具体的な理論名を記載してください
   - 「VC research」のような一般的すぎる記述は避けてください

2. method: 使用されている研究手法・分析手法
   例: panel regression, difference-in-differences, network analysis, case study, instrumental variables
   - 具体的な手法名を記載してください

3. dataset: 使用されているデータセット・データソース
   例: Crunchbase, VentureXpert, USPTO patents, OECD statistics, survey data (N=xxx)
   - データセット名がある場合はそのまま記載してください
   - 名前がない場合は「survey of N=200 entrepreneurs in country X」のように特定してください

4. context: 研究の地理的・制度的・産業的文脈
   例: United States, European Union, emerging markets, biotech sector, post-2008 crisis
   - 地理、産業、時期、制度環境を含めてください

5. research_focus: この論文の主要な研究テーマ・焦点
   例: government VC effectiveness, ecosystem formation, cross-border knowledge transfer
   - RQ との関係を意識した記述にしてください"""


def _parse_json_response(text: str) -> Optional[Any]:
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_dimensions_batch(
    llm_client,
    rq: Dict[str, Any],
    papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract research dimensions from a batch of papers."""
    # Build paper descriptions
    paper_texts = []
    for i, p in enumerate(papers):
        text = get_paper_text(p)
        paper_texts.append(f"### 論文 [{i}]: {p['title']}\n{text}")

    rq_desc = f"タイトル: {rq['title']}"
    if rq.get("rationale"):
        rq_desc += f"\n背景: {rq['rationale']}"

    user_msg = (
        f"## Research Question\n{rq_desc}\n\n"
        f"## 論文リスト ({len(papers)} 本)\n\n"
        f"{chr(10).join(paper_texts)}\n\n"
        f"## 指示\n"
        f"上記の各論文について、以下の research dimensions を抽出してください。\n"
        f"各 dimension は具体的に記述し、一般的すぎる記述は避けてください。\n\n"
        f"出力形式 (JSON):\n"
        f'{{"papers": [\n'
        f'  {{\n'
        f'    "paper_index": 0,\n'
        f'    "title": "論文タイトル",\n'
        f'    "theoretical_lens": ["theory1", "theory2"],\n'
        f'    "method": ["method1", "method2"],\n'
        f'    "dataset": ["dataset1"],\n'
        f'    "context": ["context1", "context2"],\n'
        f'    "research_focus": ["focus1"]\n'
        f'  }}\n'
        f']}}'
    )

    body = {
        "model": MODEL,
        "max_tokens": 8192,
        "system": DIMENSION_SYSTEM,
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
    if not parsed or "papers" not in parsed:
        logger.error("Failed to parse dimensions JSON")
        logger.debug("Raw: %s", resp_text[:500])
        return []

    usage = resp.get("usage", {})
    logger.info(
        "Extracted dimensions for %d papers (tokens: in=%d, out=%d)",
        len(parsed["papers"]),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    return parsed["papers"]


# ----------------------------------------------------------------
# Analysis & Report
# ----------------------------------------------------------------

def aggregate_dimensions(
    paper_dims: List[Dict[str, Any]],
) -> Dict[str, Counter]:
    """Aggregate dimension frequencies across all papers."""
    agg: Dict[str, Counter] = {cat: Counter() for cat in DIMENSION_CATEGORIES}
    for pd in paper_dims:
        for cat in DIMENSION_CATEGORIES:
            items = pd.get(cat, [])
            if isinstance(items, list):
                for item in items:
                    # Normalize: lowercase, strip
                    normalized = str(item).strip().lower()
                    agg[cat][normalized] += 1
    return agg


def print_report(
    rq: Dict[str, Any],
    papers: List[Dict[str, Any]],
    paper_dims: List[Dict[str, Any]],
    agg: Dict[str, Counter],
):
    print("\n" + "=" * 70)
    print("T0.4 SPIKE REPORT: Research Dimension Extraction")
    print("=" * 70)

    print(f"\n## RQ: {rq['title']}")
    print(f"## 対象論文数: {len(papers)}")

    # Frequency tables
    for cat in DIMENSION_CATEGORIES:
        counts = agg[cat]
        if not counts:
            print(f"\n## {cat}: (no items)")
            continue
        print(f"\n## {cat}")
        for item, count in counts.most_common(15):
            bar = "█" * count
            print(f"   {count:2d} {bar}  {item}")

    # Per-paper view
    print(f"\n{'─' * 70}")
    print("## Per-Paper Dimensions")
    print(f"{'─' * 70}")

    for pd in paper_dims:
        idx = pd.get("paper_index", "?")
        title = pd.get("title", "?")
        print(f"\n[{idx}] {title[:65]}")
        for cat in DIMENSION_CATEGORIES:
            items = pd.get(cat, [])
            if items:
                print(f"    {cat:20s}: {', '.join(str(x) for x in items)}")

    # Landscape summary
    print(f"\n{'─' * 70}")
    print("## Research Landscape Summary")
    print(f"{'─' * 70}")

    print("\n### Theoretical Streams")
    for lens, count in agg["theoretical_lens"].most_common(5):
        papers_with = [
            pd.get("title", "?")[:40]
            for pd in paper_dims
            if lens in [str(x).strip().lower() for x in pd.get("theoretical_lens", [])]
        ]
        print(f"   {lens} ({count} papers)")
        for p in papers_with:
            print(f"      └─ {p}")

    print("\n### Methodological Patterns")
    for method, count in agg["method"].most_common(5):
        print(f"   {method}: {count} papers")

    print("\n### Data Availability")
    for ds, count in agg["dataset"].most_common(5):
        print(f"   {ds}: {count} papers")

    print("\n### Geographic/Contextual Coverage")
    for ctx, count in agg["context"].most_common(5):
        print(f"   {ctx}: {count} papers")


def save_results(rq, papers, paper_dims, agg, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert Counters to dicts for JSON
    agg_json = {cat: dict(counts.most_common()) for cat, counts in agg.items()}

    out = {
        "rq": {
            "page_id": rq.get("page_id", ""),
            "title": rq["title"],
        },
        "model": MODEL,
        "papers_count": len(papers),
        "paper_dimensions": paper_dims,
        "dimension_frequency": agg_json,
    }
    path = output_dir / "spike_rq_dimensions_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    logger.info("Results saved to %s", path)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.lit_review.spikes.spike_rq_dimensions",
        description="T0.4 Spike: Research Dimension Extraction",
    )
    p.add_argument("--rq-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=10,
                    help="Number of papers to analyze (default: 10)")
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

    # Fetch RQ
    rq = fetch_rq(notion_client, resolver, args.rq_index)
    logger.info("Selected RQ: %s", rq["title"])

    # Fetch and select papers
    all_papers = fetch_lit_papers(notion_client, resolver)
    papers = select_papers_by_t01_scores(all_papers, limit=args.limit)
    logger.info("Selected %d papers for dimension extraction", len(papers))

    for p in papers:
        logger.info("  [score=%s] %s", p.get("t01_score", "?"), p["title"][:60])

    # Extract dimensions (single batch for 10 papers)
    llm_client = build_claude_client_from_env()

    # Split into batches of 5 to keep prompt manageable
    BATCH = 5
    all_dims = []
    for i in range(0, len(papers), BATCH):
        batch = papers[i : i + BATCH]
        logger.info("Processing batch %d–%d", i, i + len(batch) - 1)
        dims = extract_dimensions_batch(llm_client, rq, batch)
        # Fix paper_index to be global
        for d in dims:
            d["paper_index"] = i + d.get("paper_index", 0)
        all_dims.extend(dims)

    if not all_dims:
        logger.error("No dimensions extracted")
        return

    # Aggregate
    agg = aggregate_dimensions(all_dims)

    # Report
    print_report(rq, papers, all_dims, agg)

    # Save
    output_dir = _PROJECT_ROOT / "data" / "lit_review" / "spikes"
    save_results(rq, papers, all_dims, agg, output_dir)


if __name__ == "__main__":
    main()
