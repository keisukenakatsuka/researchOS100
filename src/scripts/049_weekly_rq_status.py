#!/usr/bin/env python
# src/scripts/049_weekly_rq_status.py
"""Weekly RQ status — fetch RQs, link evidence via LLM, surface gaps.

Pipeline:
1. Fetch all Research Questions from Notion
2. Load this week's papers (047) and events (048) output JSON
3. LLM-enhanced evidence linking per RQ
4. Generate per-RQ update proposals with gap/approach suggestions
5. Write proposals to WEEKLY_RQ_UPDATE_DB

LLM is mandatory. Write-back is default (use --no-write for debug).

Usage::

    python -m src.scripts.049_weekly_rq_status --run
    python -m src.scripts.049_weekly_rq_status --run --no-write -v
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

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
from src.notion.rq_schema import RQ_ALL_PROPERTIES
from src.notion.truncation import TruncationTracker
from src.notion.weekly_updates_repo import WeeklyRQUpdateRepo

logger = logging.getLogger("049_weekly_rq_status")

SCRIPT_NAME = "049_weekly_rq_status"

# ================================================================
# LLM evidence linking
# ================================================================

RQ_EVIDENCE_SYSTEM_PROMPT = """\
You are a research intelligence analyst specialising in startup ecosystems, \
venture capital dynamics, and entrepreneurship policy.

For each Research Question, identify which papers and events from this week \
are relevant.  For each RQ, return:
- linked_paper_indices: indices of relevant papers (from the papers list)
- linked_event_indices: indices of relevant events (from the events list)
- update_summary: 1-2 sentences on what this week's evidence means for the RQ
- suggested_gap_update: if the evidence fills a gap, suggest updated gap text (or empty)
- suggested_approach_update: if the evidence suggests a method change (or empty)
- confidence: 0.0-1.0 confidence that evidence is meaningfully relevant

Return a JSON object:
{
  "rq_updates": [
    {
      "rq_index": 0,
      "linked_paper_indices": [1, 3],
      "linked_event_indices": [0, 5],
      "update_summary": "...",
      "suggested_gap_update": "",
      "suggested_approach_update": "",
      "confidence": 0.7
    }
  ]
}
"""


def link_evidence_llm(
    llm: OpenAIClient,
    rqs: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Use LLM to link RQs to evidence and generate update proposals."""

    rqs_for_prompt = []
    for i, rq in enumerate(rqs):
        rqs_for_prompt.append({
            "index": i,
            "title": rq.get("title", ""),
            "gap": (rq.get("gap") or "")[:200],
            "approach": (rq.get("approach") or "")[:200],
            "tags": rq.get("tags", []),
        })

    papers_for_prompt = []
    for i, p in enumerate(papers[:50]):
        papers_for_prompt.append({
            "index": i,
            "title": p.get("Name", ""),
            "core_idea": (p.get("Core Idea") or "")[:150],
        })

    events_for_prompt = []
    for i, ev in enumerate(events[:50]):
        events_for_prompt.append({
            "index": i,
            "title": ev.get("title", ""),
            "event_type": ev.get("event_type", ""),
            "summary": (ev.get("summary_text") or "")[:150],
        })

    user_prompt = (
        f"Link evidence to these {len(rqs_for_prompt)} RQs.\n\n"
        f"Papers ({len(papers_for_prompt)}):\n{json.dumps(papers_for_prompt, ensure_ascii=False)}\n\n"
        f"Events ({len(events_for_prompt)}):\n{json.dumps(events_for_prompt, ensure_ascii=False)}\n\n"
        f"RQs:\n{json.dumps(rqs_for_prompt, indent=2, ensure_ascii=False)}"
    )

    result = llm.call_json(system=RQ_EVIDENCE_SYSTEM_PROMPT, user=user_prompt)

    # Build enriched status records
    statuses = []
    updates_by_idx = {}
    for u in result.parsed.get("rq_updates", []):
        updates_by_idx[u.get("rq_index", -1)] = u

    for i, rq in enumerate(rqs):
        update = updates_by_idx.get(i, {})

        # Map paper/event indices to actual records
        related_papers = []
        for pi in update.get("linked_paper_indices", []):
            if 0 <= pi < len(papers):
                related_papers.append({
                    "id": papers[pi].get("notion_page_id", ""),
                    "title": papers[pi].get("Name", ""),
                    "score": 1,
                })

        related_events = []
        for ei in update.get("linked_event_indices", []):
            if 0 <= ei < len(events):
                related_events.append({
                    "id": events[ei].get("page_id", events[ei].get("notion_page_id", "")),
                    "title": events[ei].get("title", ""),
                    "score": 1,
                    "event_type": events[ei].get("event_type", ""),
                })

        statuses.append({
            "rq_id": rq["page_id"],
            "rq_title": rq["title"],
            "priority": rq.get("priority", ""),
            "status": rq.get("status", ""),
            "tags": rq.get("tags", []),
            "related_papers": related_papers,
            "related_events": related_events,
            "evidence_count": len(related_papers) + len(related_events),
            "open_gaps": rq.get("gap", ""),
            "current_approach": rq.get("approach", ""),
            "update_summary": update.get("update_summary", ""),
            "suggested_gap_update": update.get("suggested_gap_update", ""),
            "suggested_approach_update": update.get("suggested_approach_update", ""),
            "llm_confidence": update.get("confidence", 0.0),
        })

    statuses.sort(key=lambda s: (-s["evidence_count"], s["rq_title"]))
    return statuses


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

def write_rq_status_json(statuses: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(statuses, indent=2, ensure_ascii=False, default=str) + "\n")
    logger.info("Wrote %d RQ statuses to %s", len(statuses), path)


def write_summary_md(
    statuses: List[Dict[str, Any]], week_id: str, *,
    papers_count: int, events_count: int, path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with_evidence = [s for s in statuses if s["evidence_count"] > 0]
    without_evidence = [s for s in statuses if s["evidence_count"] == 0]

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Weekly RQ Status — {week_id}\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- **RQs tracked:** {len(statuses)}\n")
        f.write(f"- **RQs with evidence:** {len(with_evidence)}\n")
        f.write(f"- **Evidence pool:** {papers_count} papers, {events_count} events\n\n")

        if with_evidence:
            f.write("## RQs with New Evidence\n\n")
            for i, s in enumerate(with_evidence, 1):
                f.write(f"### {i}. {s['rq_title']}\n\n")
                f.write(f"- **Priority:** {s['priority']}  |  **Evidence:** {s['evidence_count']}\n")
                if s.get("update_summary"):
                    f.write(f"- **Update:** {s['update_summary']}\n")
                if s.get("suggested_gap_update"):
                    f.write(f"- **Gap update:** {s['suggested_gap_update']}\n")
                f.write("\n")

        if without_evidence:
            f.write("## RQs Without New Evidence\n\n")
            for s in without_evidence:
                f.write(f"- {s['rq_title']}\n")
            f.write("\n")

        f.write(f"---\n\n*Generated by {SCRIPT_NAME}*\n")
    logger.info("Wrote summary to %s", path)


# ================================================================
# CLI
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Weekly RQ status: LLM evidence linking + proposals.",
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
                    help="Max rows to write (0 = all).")
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

    # ---- LLM evidence linking ----
    logger.info("Linking evidence to %d RQs via OpenAI ...", len(target_rqs))
    statuses = link_evidence_llm(llm, target_rqs, papers, events)
    with_evidence = sum(1 for s in statuses if s["evidence_count"] > 0)
    logger.info("OpenAI: %d RQs with evidence, %d without",
                with_evidence, len(statuses) - with_evidence)

    # ---- Write outputs ----
    write_rq_status_json(statuses, out_dir / "rq_status.json")
    write_summary_md(statuses, wk.week_id, papers_count=len(papers),
                      events_count=len(events), path=out_dir / "summary.md")

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

        rows_to_write = statuses[:args.limit] if args.limit > 0 else statuses
        for rq_rec in rows_to_write:
            key, props = repo.build_rq_properties(
                rq_record=rq_rec, week_id=wk.week_id, tracker=trunc_tracker,
            )
            page = repo.upsert_row(key=key, properties=props)
            result["rq_update_page_ids"].append(page.get("id", ""))
            notion_write_count += 1

        logger.info("Notion: %d rows upserted to WEEKLY_RQ_UPDATE_DB", notion_write_count)

    # ---- Metadata ----
    meta = RunMetadata.build(
        notebook=SCRIPT_NAME, week_id=wk.week_id,
        date_from=date_from_iso, date_to=date_to_iso,
        counts={
            "rqs_fetched": len(raw_pages), "rqs_filtered": len(target_rqs),
            "rqs_with_evidence": with_evidence,
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
        "rqs_tracked": len(statuses),
        "rqs_with_evidence": with_evidence,
    }
    result["ok"] = True
    logger.info("=== Done: %d RQs → %s ===", len(statuses), out_dir)
    logger.info(llm.usage_summary())
    return result


if __name__ == "__main__":
    r = main()
    raise SystemExit(0 if r["ok"] else 1)
