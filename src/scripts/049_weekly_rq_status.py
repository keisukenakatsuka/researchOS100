#!/usr/bin/env python
# src/scripts/049_weekly_rq_status.py
"""Weekly RQ revision proposals — categorized, contextual, in Japanese.

Pipeline:
1. Fetch all Research Questions from Notion
2. Load this week's papers (047) and events (048) output JSON
3. LLM generates categorized revision proposals per RQ (in Japanese)
4. Each revision becomes a separate Notion page in WEEKLY_RQ_UPDATE_DB
5. Categories: "Rationale / Background", "Gap Identified", "Proposed Approach"

LLM is mandatory. Write-back is default (use --no-write for debug).

Usage::

    python -m src.scripts.049_weekly_rq_status --run
    python -m src.scripts.049_weekly_rq_status --run --no-write -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    RunMetadata,
    get_db_id,
    get_output_dir,
    get_week_context,
    load_env,
    setup_logging,
)
from src.llm.openai_client import OpenAIClient, build_openai_client_from_env
from src.notion import (
    NotionDataSourceResolver,
    build_notion_client_from_env,
)
from src.notion.rq_normalize import filter_rqs, normalize_rqs
from src.notion.truncation import TruncationTracker
from src.notion.weekly_updates_repo import WeeklyRQUpdateRepo

logger = logging.getLogger("049_weekly_rq_status")

SCRIPT_NAME = "049_weekly_rq_status"

# ================================================================
# LLM revision proposal generation
# ================================================================

RQ_REVISION_SYSTEM_PROMPT = """\
あなたはスタートアップエコシステム、ベンチャーキャピタルの動態、起業政策に\
特化した研究インテリジェンスアナリストです。

各Research Question (RQ) について、今週の論文とイベントを分析し、\
具体的な改訂提案を生成してください。

各RQには以下の情報が含まれています:
- title: RQのタイトル
- rationale: 背景・根拠
- gap: 特定されたギャップ
- approach: 提案されたアプローチ

各RQについて、具体的な修正が必要な箇所を特定し、以下のカテゴリで\
改訂提案を作成してください:

カテゴリ:
1. "Rationale / Background" — 背景・根拠の修正
2. "Gap Identified" — ギャップの修正・追加
3. "Proposed Approach" — アプローチの修正・追加

各改訂提案について:
- proposed_text: 修正案（日本語）— 元のRQ内容を具体的に参照し、\
どこをどう修正すべきか明確に記述
- reason: その理由（日本語）— 今週のエビデンス（論文・イベント）に\
基づいて、なぜこの修正が必要か説明
- linked_paper_indices: 根拠となる論文のインデックス
- linked_event_indices: 根拠となるイベントのインデックス
- confidence: 0.0-1.0 この提案の信頼度

重要なルール:
- 汎用的・一般的な提案は不可。元のRQ内容を具体的に参照すること
- エビデンスに基づかない提案は不可
- confidence < 0.5 の提案は除外
- 修正不要のRQには revisions: [] を返すこと

JSON形式で返答してください:
{
  "rq_revisions": [
    {
      "rq_index": 0,
      "revisions": [
        {
          "category": "Gap Identified",
          "proposed_text": "（修正案を日本語で記述）",
          "reason": "（その理由を日本語で記述）",
          "linked_paper_indices": [1, 3],
          "linked_event_indices": [0],
          "confidence": 0.8
        }
      ]
    }
  ]
}
"""


def generate_revision_proposals(
    llm: OpenAIClient,
    rqs: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Use LLM to generate categorized revision proposals per RQ.

    Returns a flat list of revision dicts, each with rq_id, rq_title, etc.
    """
    rqs_for_prompt = []
    for i, rq in enumerate(rqs):
        rqs_for_prompt.append({
            "index": i,
            "title": rq.get("title", ""),
            "rationale": (rq.get("rationale") or rq.get("background") or "")[:500],
            "gap": (rq.get("gap") or "")[:500],
            "approach": (rq.get("approach") or "")[:500],
            "tags": rq.get("tags", []),
        })

    papers_for_prompt = []
    for i, p in enumerate(papers[:50]):
        papers_for_prompt.append({
            "index": i,
            "title": p.get("Name", ""),
            "core_idea": (p.get("Core Idea") or "")[:200],
        })

    events_for_prompt = []
    for i, ev in enumerate(events[:50]):
        events_for_prompt.append({
            "index": i,
            "title": ev.get("title", ""),
            "event_type": ev.get("event_type", ""),
            "summary": (ev.get("summary_text") or "")[:200],
        })

    user_prompt = (
        f"以下の{len(rqs_for_prompt)}件のRQについて改訂提案を生成してください。\n\n"
        f"今週の論文 ({len(papers_for_prompt)}件):\n"
        f"{json.dumps(papers_for_prompt, ensure_ascii=False)}\n\n"
        f"今週のイベント ({len(events_for_prompt)}件):\n"
        f"{json.dumps(events_for_prompt, ensure_ascii=False)}\n\n"
        f"RQ一覧:\n{json.dumps(rqs_for_prompt, indent=2, ensure_ascii=False)}"
    )

    result = llm.call_json(system=RQ_REVISION_SYSTEM_PROMPT, user=user_prompt)

    # Flatten into individual revision records
    all_revisions: List[Dict[str, Any]] = []
    rq_revisions = result.parsed.get("rq_revisions", [])

    for rq_rev in rq_revisions:
        rq_idx = rq_rev.get("rq_index", -1)
        if rq_idx < 0 or rq_idx >= len(rqs):
            logger.warning("Invalid rq_index %d (have %d RQs)", rq_idx, len(rqs))
            continue

        rq = rqs[rq_idx]
        revisions = rq_rev.get("revisions") or []

        for rev in revisions:
            confidence = rev.get("confidence", 0.0)
            if confidence < 0.5:
                logger.debug(
                    "Skipping low-confidence revision for RQ %d (%s): %.2f",
                    rq_idx, rq.get("title", "")[:40], confidence,
                )
                continue

            # Build evidence lines
            evidence_lines: list[str] = []
            for pi in rev.get("linked_paper_indices", []):
                if 0 <= pi < len(papers):
                    evidence_lines.append(f"[paper] {papers[pi].get('Name', '?')}")
            for ei in rev.get("linked_event_indices", []):
                if 0 <= ei < len(events):
                    evidence_lines.append(f"[event] {events[ei].get('title', '?')}")

            all_revisions.append({
                "rq_id": rq["page_id"],
                "rq_title": rq["title"],
                "priority": rq.get("priority", ""),
                "status": rq.get("status", ""),
                "tags": rq.get("tags", []),
                "category": rev.get("category", ""),
                "proposed_text": rev.get("proposed_text", ""),
                "reason": rev.get("reason", ""),
                "confidence": confidence,
                "evidence_lines": evidence_lines,
                "evidence_count": len(evidence_lines),
            })

    logger.info(
        "LLM: %d revision proposals for %d RQs (from %d RQ revision sets)",
        len(all_revisions), len(rqs), len(rq_revisions),
    )
    return all_revisions


# ================================================================
# Data loading
# ================================================================

def fetch_all_rqs(client, data_source_id: str) -> List[dict]:
    pages = client.query_data_source(
        data_source_id=data_source_id,
        sorts=[{"timestamp": "created_time", "direction": "descending"}],
        fetch_all=True,
    )
    logger.info("Fetched %d RQ pages from Notion.", len(pages))
    return pages


def load_evidence_json(
    week_id: str, script_name: str, filename: str, *, base: str = "outputs",
) -> List[Dict[str, Any]]:
    path = Path(base) / "weekly" / week_id / script_name / filename
    if not path.exists():
        logger.warning("Evidence file not found: %s", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    logger.info("Loaded %d records from %s", len(data), path)
    return data


# ================================================================
# Output writers
# ================================================================

def write_revisions_json(revisions: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(revisions, indent=2, ensure_ascii=False, default=str) + "\n")
    logger.info("Wrote %d revision proposals to %s", len(revisions), path)


def write_summary_md(
    revisions: List[Dict[str, Any]], week_id: str, *,
    rqs_count: int, papers_count: int, events_count: int, path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Group by RQ
    by_rq: Dict[str, List[Dict[str, Any]]] = {}
    for rev in revisions:
        rq_title = rev["rq_title"]
        by_rq.setdefault(rq_title, []).append(rev)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Weekly RQ Revision Proposals \u2014 {week_id}\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- **RQs tracked:** {rqs_count}\n")
        f.write(f"- **RQs with revisions:** {len(by_rq)}\n")
        f.write(f"- **Total revision proposals:** {len(revisions)}\n")
        f.write(f"- **Evidence pool:** {papers_count} papers, {events_count} events\n\n")

        for rq_title, rq_revs in by_rq.items():
            f.write(f"## {rq_title}\n\n")
            for i, rev in enumerate(rq_revs, 1):
                f.write(f"### {i}. [{rev['category']}]\n\n")
                f.write(f"- **\u4fee\u6b63\u6848:** {rev['proposed_text'][:200]}\n")
                f.write(f"- **\u7406\u7531:** {rev['reason'][:200]}\n")
                f.write(f"- **Confidence:** {rev['confidence']:.2f}\n")
                if rev.get("evidence_lines"):
                    f.write(f"- **Evidence:** {'; '.join(rev['evidence_lines'][:3])}\n")
                f.write("\n")

        f.write(f"---\n\n*Generated by {SCRIPT_NAME}*\n")
    logger.info("Wrote summary to %s", path)


# ================================================================
# CLI
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Weekly RQ status: LLM revision proposals (categorized, Japanese).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", default=False)
    mode.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--priority", default="High")
    p.add_argument("--output-base", default="outputs")
    p.add_argument("--write", action="store_true", default=False,
                    help="Persist to Notion (default: off).")
    p.add_argument("--limit", type=int, default=0,
                    help="Max revisions to write (0 = all).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    """Main pipeline. Returns result dict for orchestrator."""
    args = build_parser().parse_args(argv)
    is_live = args.run
    write_enabled = args.write

    result: Dict[str, Any] = {
        "ok": False, "week_id": "", "output_dir": "",
        "summary": {}, "errors": [], "rq_update_page_ids": [],
    }

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    load_env()

    wk = get_week_context()
    result["week_id"] = wk.week_id
    rq_db_id = get_db_id("NOTION_RQ_DB_ID")
    priorities = frozenset(p.strip() for p in args.priority.split(",") if p.strip())

    date_to_utc = wk.now_utc
    date_from_utc = date_to_utc - timedelta(days=args.days)
    date_from_iso = date_from_utc.isoformat(timespec="seconds")
    date_to_iso = date_to_utc.isoformat(timespec="seconds")

    out_dir = get_output_dir(SCRIPT_NAME, wk.week_id, base=args.output_base, create=is_live)
    result["output_dir"] = str(out_dir)

    logger.info("=== %s ===", SCRIPT_NAME)
    logger.info("Week: %s  |  Write: %s", wk.week_id, "ON" if write_enabled else "OFF")

    if not is_live:
        logger.info("[DRY-RUN] Pass --run to execute.")
        result["ok"] = True
        return result

    # ---- Build clients ----
    llm = build_openai_client_from_env()
    client = build_notion_client_from_env()

    # ---- Fetch RQs ----
    resolver = NotionDataSourceResolver(client)
    resolved = resolver.resolve_once(name="RQ_DB", database_id=rq_db_id)
    raw_pages = fetch_all_rqs(client, resolved.data_source_id)
    all_rqs = normalize_rqs(raw_pages)
    target_rqs = filter_rqs(all_rqs, priorities=priorities)

    # ---- Load evidence ----
    papers = load_evidence_json(wk.week_id, "047_weekly_papers_review", "papers.json",
                                 base=args.output_base)
    events = load_evidence_json(wk.week_id, "048_weekly_events_digest", "events.json",
                                 base=args.output_base)

    # ---- LLM revision proposals ----
    logger.info("Generating revision proposals for %d RQs via OpenAI ...", len(target_rqs))
    revisions = generate_revision_proposals(llm, target_rqs, papers, events)
    logger.info("OpenAI: %d revision proposals generated", len(revisions))

    # ---- Write outputs ----
    write_revisions_json(revisions, out_dir / "rq_revisions.json")
    write_summary_md(revisions, wk.week_id, rqs_count=len(target_rqs),
                      papers_count=len(papers), events_count=len(events),
                      path=out_dir / "summary.md")

    # ---- Notion writeback ----
    notion_write_count = 0
    trunc_tracker = TruncationTracker()

    if write_enabled:
        rq_update_db_id = get_db_id("NOTION_WEEKLY_RQ_UPDATE_DB_ID")
        rq_update_resolver = NotionDataSourceResolver(client)
        rq_update_resolved = rq_update_resolver.resolve_once(
            name="WEEKLY_RQ_UPDATE_DB", database_id=rq_update_db_id,
        )
        repo = WeeklyRQUpdateRepo(
            client=client,
            database_id=rq_update_resolved.database_id,
            data_source_id=rq_update_resolved.data_source_id,
        )
        repo.validate_schema()

        rows_to_write = revisions[:args.limit] if args.limit > 0 else revisions
        notion_fail_count = 0
        for idx, rev in enumerate(rows_to_write):
            try:
                key, props = repo.build_rq_revision_properties(
                    revision=rev, week_id=wk.week_id,
                    revision_index=idx, tracker=trunc_tracker,
                )
                page = repo.upsert_row(key=key, properties=props)
                result["rq_update_page_ids"].append(page.get("id", ""))
                notion_write_count += 1
            except Exception as e:
                notion_fail_count += 1
                rq_title = rev.get("rq_title", "?")
                err = f"Failed to write RQ revision {idx} ({rq_title}): {e}"
                logger.error(err)
                result["errors"].append(err)

        logger.info(
            "Notion: %d revision rows upserted to WEEKLY_RQ_UPDATE_DB "
            "(%d failed)",
            notion_write_count, notion_fail_count,
        )

    # ---- Metadata ----
    meta = RunMetadata.build(
        notebook=SCRIPT_NAME, week_id=wk.week_id,
        date_from=date_from_iso, date_to=date_to_iso,
        counts={
            "rqs_fetched": len(raw_pages), "rqs_filtered": len(target_rqs),
            "revisions_generated": len(revisions),
            "papers_loaded": len(papers), "events_loaded": len(events),
        },
        extra={
            "priorities": sorted(priorities),
            "llm_usage": llm.usage_summary(),
            "write_enabled": write_enabled,
            "notion_rows_upserted": notion_write_count,
        },
    )
    meta.save(out_dir / "run_metadata.json")

    result["summary"] = {
        "rqs_tracked": len(target_rqs),
        "revisions_generated": len(revisions),
    }
    result["ok"] = True
    logger.info("=== Done: %d revisions for %d RQs \u2192 %s ===",
                len(revisions), len(target_rqs), out_dir)
    logger.info(llm.usage_summary())
    return result


if __name__ == "__main__":
    r = main()
    raise SystemExit(0 if r["ok"] else 1)
