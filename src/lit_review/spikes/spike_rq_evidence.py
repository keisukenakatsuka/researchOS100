#!/usr/bin/env python
"""T0.2 Spike: Query-Focused Evidence Extraction.

Validates that LLM can extract RQ-focused Evidence from papers
with sufficient quality for downstream synthesis (082).

Usage:
    python -m src.lit_review.spikes.spike_rq_evidence
    python -m src.lit_review.spikes.spike_rq_evidence --paper-indices 0,2,4
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
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

logger = logging.getLogger("spike_rq_evidence")

MODEL = "claude-sonnet-4-20250514"
_LIT_INBOX_DIR = _PROJECT_ROOT / "data" / "downloads" / "lit_inbox"

DIMENSIONS = [
    "mechanism",   # how something works / causal pathway
    "outcome",     # measured results / effects
    "condition",   # under what circumstances
    "method",      # research methodology used
    "dataset",     # data sources used
    "limitation",  # acknowledged limitations
    "implication", # implications for policy/practice/theory
]


# ----------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------

def fetch_rq(notion_client, resolver, rq_index: int = 0) -> Dict[str, Any]:
    """Fetch a specific RQ by index."""
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
    """Fetch papers from LIT DB with Decision=READ or KEEP, including text fields."""
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
            "source_uid": ev(page, "Source UID") or "",
            "pdf_link": ev(page, "PDF Link") or "",
        })
    return records


def get_paper_text(paper: Dict[str, Any]) -> tuple[str, str]:
    """Get the best available text for a paper.

    Returns (text, source) where source is 'pdf' or 'metadata'.
    """
    # Try PDF first
    source_uid = paper.get("source_uid", "")
    if source_uid:
        safe_name = re.sub(r"[^\w\-.]", "_", source_uid)
        pdf_path = _LIT_INBOX_DIR / f"{safe_name}.pdf"
        if pdf_path.exists():
            try:
                from src.pdf.metadata import extract_pdf_text_for_llm
                text = extract_pdf_text_for_llm(pdf_path)
                if text and len(text) > 200:
                    return text, "pdf"
            except Exception as e:
                logger.warning("PDF extraction failed for %s: %s", safe_name, e)

    # Fallback to LIT DB metadata
    parts = []
    if paper.get("core_idea"):
        parts.append(f"Core Idea: {paper['core_idea']}")
    if paper.get("findings"):
        parts.append(f"Findings: {paper['findings']}")
    if paper.get("methods"):
        parts.append(f"Methods: {paper['methods']}")
    if paper.get("notes"):
        parts.append(f"Notes: {paper['notes']}")

    if parts:
        return "\n\n".join(parts), "metadata"
    return f"Title: {paper['title']}", "title_only"


# ----------------------------------------------------------------
# LLM Evidence Extraction
# ----------------------------------------------------------------

EXTRACTION_SYSTEM = """\
あなたは学術文献のレビュー専門家です。
Research Question (RQ) の観点から、論文の内容を分析し、RQ に関連する Evidence を構造化して抽出してください。

重要な指示:
- 論文の一般的な要約ではなく、RQ に直接関係する知見のみを抽出してください
- 各 Evidence は具体的で、他の論文と比較可能な粒度にしてください
- RQ と無関係な記述は含めないでください
- 各 Evidence に dimension（分類）を付与してください

dimension の分類:
- mechanism: 因果メカニズム、作用経路（「〜を通じて〜が生じる」）
- outcome: 測定された結果、効果、パフォーマンス指標
- condition: 効果が成立する条件、モデレーター、境界条件
- method: 使用された研究手法、分析手法
- dataset: 使用されたデータセット、サンプル
- limitation: 著者が認めた限界、未検証事項
- implication: 政策・実務・理論への示唆

confidence は 0.0–1.0 で以下の基準:
- 0.8–1.0: 実証的に裏付けられた強い Evidence
- 0.5–0.7: 示唆的だが追加検証が必要
- 0.2–0.4: 理論的推論や限定的なデータに基づく"""


def build_extraction_prompt(rq: Dict[str, Any], paper: Dict[str, Any], text: str) -> str:
    """Build the user prompt for evidence extraction."""
    rq_parts = [f"タイトル: {rq['title']}"]
    if rq.get("rationale"):
        rq_parts.append(f"背景: {rq['rationale']}")
    if rq.get("gap"):
        rq_parts.append(f"ギャップ: {rq['gap']}")

    return (
        f"## Research Question\n"
        f"{chr(10).join(rq_parts)}\n\n"
        f"## 論文\n"
        f"タイトル: {paper['title']}\n"
        f"Tags: {paper.get('tags', '')}\n\n"
        f"## 論文の内容\n"
        f"{text[:80_000]}\n\n"
        f"## 指示\n"
        f"上記の論文から、RQ に関連する Evidence を抽出してください。\n"
        f"各 Evidence について、以下の JSON 形式で出力してください。\n"
        f"一般的な要約ではなく、RQ の視点から見た具体的な知見を抽出してください。\n"
        f"1論文あたり 3〜8 件程度の Evidence を目安にしてください。\n\n"
        f"出力形式:\n"
        f'{{"evidence_items": [\n'
        f'  {{\n'
        f'    "claim_or_point": "この論文が示している主張や知見（日本語で簡潔に）",\n'
        f'    "evidence_text": "論文中の根拠となる具体的な記述や数値（可能な限り原文に近い形で）",\n'
        f'    "relevance_to_rq": "この Evidence が RQ にどう関係するか（日本語で1-2文）",\n'
        f'    "dimension": "mechanism | outcome | condition | method | dataset | limitation | implication のいずれか",\n'
        f'    "confidence": 0.0\n'
        f'  }}\n'
        f']}}'
    )


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, tolerating markdown fences."""
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_evidence(
    llm_client,
    rq: Dict[str, Any],
    paper: Dict[str, Any],
    text: str,
) -> Dict[str, Any]:
    """Extract RQ-focused evidence from a single paper."""
    user_msg = build_extraction_prompt(rq, paper, text)

    body = {
        "model": MODEL,
        "max_tokens": 4096,
        "system": EXTRACTION_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("LLM call failed for '%s': %s", paper["title"][:50], e)
        return {"evidence_items": [], "error": str(e)}

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    parsed = _parse_json_response(resp_text)
    if not parsed or "evidence_items" not in parsed:
        logger.error("Failed to parse evidence JSON for '%s'", paper["title"][:50])
        logger.debug("Raw response: %s", resp_text[:500])
        return {"evidence_items": [], "error": "JSON parse failed", "raw": resp_text[:500]}

    usage = resp.get("usage", {})
    logger.info(
        "Extracted %d evidence items from '%s' (tokens: in=%d, out=%d)",
        len(parsed["evidence_items"]),
        paper["title"][:50],
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    return parsed


# ----------------------------------------------------------------
# Report
# ----------------------------------------------------------------

def print_report(
    rq: Dict[str, Any],
    results: List[Dict[str, Any]],
):
    """Print a formatted report of extraction results."""
    print("\n" + "=" * 70)
    print("T0.2 SPIKE REPORT: Query-Focused Evidence Extraction")
    print("=" * 70)

    print(f"\n## RQ: {rq['title']}")

    total_items = 0
    dimension_counts: Dict[str, int] = {}
    confidence_values: List[float] = []

    for r in results:
        items = r.get("extraction", {}).get("evidence_items", [])
        total_items += len(items)
        for item in items:
            dim = item.get("dimension", "unknown")
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
            conf = item.get("confidence", 0)
            if isinstance(conf, (int, float)):
                confidence_values.append(float(conf))

    print(f"\n## 統計")
    print(f"   対象論文数: {len(results)}")
    print(f"   総 Evidence 数: {total_items}")
    print(f"   論文あたり平均: {total_items / len(results):.1f}" if results else "   N/A")

    if confidence_values:
        print(f"   Confidence 平均: {sum(confidence_values)/len(confidence_values):.2f}")
        print(f"   Confidence 範囲: {min(confidence_values):.2f} – {max(confidence_values):.2f}")

    if dimension_counts:
        print(f"\n## Dimension 分布")
        for dim in DIMENSIONS + ["unknown"]:
            count = dimension_counts.get(dim, 0)
            if count:
                bar = "█" * count
                print(f"   {dim:12s}: {count:2d} {bar}")

    for r in results:
        print(f"\n{'─' * 70}")
        print(f"### {r['title']}")
        print(f"    テキストソース: {r['text_source']}, テキスト長: {r['text_length']:,} chars")
        items = r.get("extraction", {}).get("evidence_items", [])
        print(f"    抽出 Evidence 数: {len(items)}")

        for i, item in enumerate(items):
            print(f"\n    [{i+1}] {item.get('dimension', '?'):12s} (conf={item.get('confidence', '?')})")
            print(f"        主張: {item.get('claim_or_point', '')[:100]}")
            print(f"        根拠: {item.get('evidence_text', '')[:120]}")
            print(f"        RQ関連: {item.get('relevance_to_rq', '')[:120]}")


def save_results(rq, results, output_dir: Path):
    """Save raw results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "rq": {
            "page_id": rq.get("page_id", ""),
            "title": rq["title"],
            "rationale": rq.get("rationale", ""),
            "gap": rq.get("gap", ""),
        },
        "model": MODEL,
        "papers_count": len(results),
        "results": results,
    }
    path = output_dir / "spike_rq_evidence_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    logger.info("Results saved to %s", path)


# ----------------------------------------------------------------
# Paper selection
# ----------------------------------------------------------------

def select_diverse_papers(
    papers: List[Dict[str, Any]],
    t01_scores: Optional[List[Dict[str, Any]]] = None,
    indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Select diverse papers for the spike.

    If indices are provided, use those directly.
    Otherwise, pick papers from T0.1 results with diverse characteristics.
    """
    if indices:
        return [papers[i] for i in indices if i < len(papers)]

    # Load T0.1 results if available
    t01_path = _PROJECT_ROOT / "data" / "lit_review" / "spikes" / "spike_rq_relevance_results.json"
    if t01_path.exists():
        t01_data = json.loads(t01_path.read_text())
        t01_results = t01_data.get("results", [])
        # Map title -> score
        score_map = {r["title"]: r["score"] for r in t01_results}
    else:
        score_map = {}

    # Score papers and pick diverse set
    scored = []
    for i, p in enumerate(papers):
        s = score_map.get(p["title"], -1)
        scored.append((i, p, s))

    # Sort by score descending
    scored.sort(key=lambda x: x[2], reverse=True)

    # Pick: top, mid, low from scored papers
    selected = []
    scored_only = [x for x in scored if x[2] > 0]

    if len(scored_only) >= 5:
        # Top (direct relevance)
        selected.append(scored_only[0])
        selected.append(scored_only[1])
        # Mid (indirect relevance)
        mid = len(scored_only) // 2
        selected.append(scored_only[mid])
        # Lower (peripheral)
        selected.append(scored_only[-2])
        selected.append(scored_only[-1])
    elif scored_only:
        selected = scored_only[:5]
    else:
        selected = [(i, p, 0) for i, p in enumerate(papers[:5])]

    result = []
    for idx, paper, score in selected:
        paper_copy = dict(paper)
        paper_copy["t01_score"] = score
        paper_copy["original_index"] = idx
        result.append(paper_copy)

    return result


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.lit_review.spikes.spike_rq_evidence",
        description="T0.2 Spike: Query-Focused Evidence Extraction",
    )
    p.add_argument("--rq-index", type=int, default=0)
    p.add_argument("--paper-indices", type=str, default=None,
                    help="Comma-separated paper indices to use (from LIT DB order)")
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

    # Fetch papers
    all_papers = fetch_lit_papers(notion_client, resolver)
    logger.info("Total LIT papers: %d", len(all_papers))

    # Select papers
    indices = None
    if args.paper_indices:
        indices = [int(x.strip()) for x in args.paper_indices.split(",")]

    selected = select_diverse_papers(all_papers, indices=indices)
    logger.info("Selected %d papers for extraction", len(selected))

    for p in selected:
        logger.info("  [score=%s] %s", p.get("t01_score", "?"), p["title"][:60])

    # Extract evidence
    llm_client = build_claude_client_from_env()
    results = []

    for paper in selected:
        text, text_source = get_paper_text(paper)
        logger.info(
            "Processing '%s' (source=%s, len=%d)",
            paper["title"][:50], text_source, len(text),
        )

        extraction = extract_evidence(llm_client, rq, paper, text)

        results.append({
            "paper_id": paper["page_id"],
            "title": paper["title"],
            "t01_score": paper.get("t01_score", -1),
            "text_source": text_source,
            "text_length": len(text),
            "extraction": extraction,
        })

    # Report
    print_report(rq, results)

    # Save
    output_dir = _PROJECT_ROOT / "data" / "lit_review" / "spikes"
    save_results(rq, results, output_dir)


if __name__ == "__main__":
    main()
