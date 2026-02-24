#!/usr/bin/env python
# src/scripts/050_weekly_targets_review.py
"""Weekly targets review — fetch, score, LLM shifts, external search, output.

Pipeline:
1. Fetch all monitoring targets from Notion
2. Load 048 events output for the same week
3. Compute per-target signal/noise metrics
4. Generate keep/review proposals with priority/cadence suggestions
5. LLM structural shift detection per target
6. External search validation for potential drop candidates
7. Produce keyword tuning suggestions (rule-based)
8. Write a next-week monitoring policy
9. Write proposals to WEEKLY_TARGET_UPDATE_DB (skip "keep" rows)

LLM is mandatory. Write-back is default (use --no-write for debug).
"keep" is NOT an update — only non-keep proposals are written to Notion.

Usage::

    python -m src.scripts.050_weekly_targets_review --run
    python -m src.scripts.050_weekly_targets_review --run --no-write -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    RunMetadata,
    get_db_id,
    get_iso_week_context,
    get_output_dir,
    load_env,
    setup_logging,
)
from src.llm.openai_client import OpenAIClient, build_openai_client_from_env
from src.notion import (
    NotionDataSourceResolver,
    build_notion_client_from_env,
)
from src.notion.targets_normalize import (
    compute_target_metrics,
    filter_targets,
    normalize_targets,
)
from src.notion.targets_schema import (
    HIGH_NOISE_THRESHOLD,
    KEYWORD_MIN_EVENT_COUNT,
    MIN_SIGNAL_THRESHOLD,
)
from src.search.google_cse import GoogleCSEClient, GoogleCSEError, build_google_cse_from_env
from src.search.newsapi import NewsAPIClient, NewsAPIError, build_newsapi_from_env
from src.targets.proposals import propose_target_actions
from src.notion.truncation import TruncationTracker
from src.notion.weekly_updates_repo import WeeklyTargetUpdateRepo

logger = logging.getLogger("050_weekly_targets_review")

SCRIPT_NAME = "050_weekly_targets_review"
JST = ZoneInfo("Asia/Tokyo")


# ================================================================
# LLM structural shift detection
# ================================================================

SHIFT_SYSTEM_PROMPT = """\
You are a research intelligence analyst specialising in startup ecosystems, \
venture capital dynamics, and entrepreneurship policy.

Given a list of monitoring targets with their rule-based scores and recent \
events, identify structural shifts — meaningful changes in positioning, \
relevance, or strategic significance that go beyond surface-level metrics.

For each target, return:
- structural_shift: boolean — true if a meaningful shift has occurred
- shift_description: 1-2 sentences describing the shift (empty if no shift)
- action: "keep" / "upgrade" / "deprioritize" / "drop_candidate"
- reason: 1 sentence explaining the action recommendation

Return a JSON object:
{
  "target_shifts": [
    {
      "target_index": 0,
      "structural_shift": true,
      "shift_description": "...",
      "action": "upgrade",
      "reason": "..."
    }
  ]
}
"""


def detect_structural_shifts_llm(
    llm: OpenAIClient,
    proposals: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Use LLM to detect structural shifts and enrich proposals."""

    targets_for_prompt = []
    for i, p in enumerate(proposals[:50]):
        targets_for_prompt.append({
            "index": i,
            "name": p.get("target_name", ""),
            "type": p.get("target_type", ""),
            "action_rule": p.get("action", ""),
            "signal_score": round(p.get("signal_score", 0), 3),
            "noise_score": round(p.get("noise_score", 0), 3),
            "number_of_events": p.get("number_of_events", 0),
            "reason_rule": (p.get("reason") or "")[:150],
        })

    events_for_prompt = []
    for i, ev in enumerate(events[:30]):
        events_for_prompt.append({
            "index": i,
            "title": ev.get("title", "")[:100],
            "event_type": ev.get("event_type", ""),
            "summary": (ev.get("summary_text") or "")[:150],
        })

    user_prompt = (
        f"Detect structural shifts for these {len(targets_for_prompt)} targets.\n\n"
        f"Recent events ({len(events_for_prompt)}):\n"
        f"{json.dumps(events_for_prompt, ensure_ascii=False)}\n\n"
        f"Targets:\n{json.dumps(targets_for_prompt, indent=2, ensure_ascii=False)}"
    )

    result = llm.call_json(system=SHIFT_SYSTEM_PROMPT, user=user_prompt)

    shifts_by_idx = {}
    for s in result.parsed.get("target_shifts", []):
        shifts_by_idx[s.get("target_index", -1)] = s

    shift_count = 0
    for i, p in enumerate(proposals):
        shift = shifts_by_idx.get(i, {})
        p["structural_shift"] = shift.get("structural_shift", False)
        p["shift_description"] = shift.get("shift_description", "")
        p["llm_action"] = shift.get("action", p.get("action", "keep"))
        p["llm_reason"] = shift.get("reason", "")
        if shift.get("structural_shift"):
            shift_count += 1

    logger.info("LLM: %d structural shifts detected in %d targets",
                shift_count, len(proposals))
    return proposals


# ================================================================
# External search validation for potential drop candidates
# ================================================================

VALIDATION_SYSTEM_PROMPT = """\
You are a research intelligence analyst. Given a monitoring target and \
search results from Google and NewsAPI, determine whether this target \
is still worth monitoring.

Analyze the search results and decide:
1. Is there recent activity or relevance that our internal data missed?
2. Can the monitoring be improved with better keywords or source URLs?
3. Should this target be kept, reviewed further, or dropped?

For each target, return:
- final_action: "keep" / "review" / "drop_candidate"
- reason: 1-2 sentences explaining the decision
- suggested_keywords: list of 3-5 improved search keywords
- suggested_source_urls: list of 1-3 URLs where this entity is covered
- evidence_summary: what the external search revealed

Return a JSON object:
{
  "validations": [
    {
      "target_index": 0,
      "final_action": "keep",
      "reason": "...",
      "suggested_keywords": ["keyword1", "keyword2"],
      "suggested_source_urls": ["https://..."],
      "evidence_summary": "..."
    }
  ]
}
"""


def validate_drop_candidates_with_search(
    llm: OpenAIClient,
    cse: GoogleCSEClient | None,
    news: NewsAPIClient | None,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate potential drop candidates using external search + LLM.

    For each candidate flagged with needs_external_validation:
    1. Google CSE search for the target name
    2. NewsAPI search for recent news
    3. LLM analysis to decide final action + suggest improvements

    Updates candidates in-place with final_action and suggestions.
    """
    to_validate = [
        (i, c) for i, c in enumerate(candidates)
        if c.get("needs_external_validation", False)
    ]

    if not to_validate:
        logger.info("No candidates need external validation")
        return candidates

    logger.info("Validating %d potential drop candidates with external search ...",
                len(to_validate))

    # Gather search results for each candidate
    search_context = []
    for idx, (orig_idx, cand) in enumerate(to_validate):
        target_name = cand.get("target_name", "")
        cse_results: list[dict] = []
        news_results: list[dict] = []

        if cse:
            try:
                cse_results = cse.search(target_name, num=5)
            except GoogleCSEError as e:
                logger.warning("Google CSE failed for %r: %s", target_name, e)

        if news:
            try:
                news_results = news.search(target_name, days_back=30, page_size=5)
            except NewsAPIError as e:
                logger.warning("NewsAPI failed for %r: %s", target_name, e)

        search_context.append({
            "target_index": idx,
            "name": target_name,
            "type": cand.get("target_type", ""),
            "current_keywords": (cand.get("keyword_suggestions", {})
                                  .get("current_keywords", [])),
            "signal_score": round(cand.get("signal_score", 0), 3),
            "noise_score": round(cand.get("noise_score", 0), 3),
            "rule_reason": (cand.get("reason") or "")[:200],
            "google_results": [
                {"title": r["title"][:100], "snippet": r["snippet"][:200]}
                for r in cse_results[:5]
            ],
            "news_results": [
                {"title": r["title"][:100], "source": r["source"],
                 "date": r["publishedAt"][:10] if r.get("publishedAt") else ""}
                for r in news_results[:5]
            ],
        })

    # LLM analysis
    user_prompt = (
        f"Validate these {len(search_context)} potential drop candidates.\n\n"
        + json.dumps(search_context, indent=2, ensure_ascii=False)
    )

    result = llm.call_json(system=VALIDATION_SYSTEM_PROMPT, user=user_prompt)

    validations_by_idx = {}
    for v in result.parsed.get("validations", []):
        validations_by_idx[v.get("target_index", -1)] = v

    # Apply results back to candidates
    for idx, (orig_idx, cand) in enumerate(to_validate):
        validation = validations_by_idx.get(idx, {})
        final_action = validation.get("final_action", "review")

        cand["action"] = final_action
        cand["reason"] = validation.get("reason", cand.get("reason", ""))
        cand["suggested_keywords"] = validation.get("suggested_keywords", [])
        cand["suggested_source_urls"] = validation.get("suggested_source_urls", [])
        cand["external_evidence_summary"] = validation.get("evidence_summary", "")

    validated_actions: Dict[str, int] = defaultdict(int)
    for _, cand in to_validate:
        validated_actions[cand["action"]] += 1
    logger.info("External validation results: %s", dict(validated_actions))

    return candidates


# ================================================================
# Core pipeline steps
# ================================================================

def fetch_all_targets(client, data_source_id: str) -> List[dict]:
    """Fetch all Monitoring Target pages. Filtering happens in Python."""
    pages = client.query_data_source(
        data_source_id=data_source_id,
        sorts=[{"timestamp": "created_time", "direction": "descending"}],
        fetch_all=True,
    )
    logger.info("Fetched %d target pages from Notion.", len(pages))
    return pages


def load_evidence_json(
    week_id: str,
    script_name: str,
    filename: str,
    *,
    base: str = "outputs",
) -> List[Dict[str, Any]]:
    """Load a JSON array from a sibling script's output directory."""
    path = Path(base) / "weekly" / week_id / script_name / filename
    if not path.exists():
        logger.warning("Evidence file not found: %s (will proceed without)", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        logger.warning("Expected list in %s, got %s", path, type(data).__name__)
        return []
    logger.info("Loaded %d records from %s", len(data), path)
    return data


# ================================================================
# Output writers
# ================================================================

def write_targets_review_json(proposals: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(proposals, indent=2, ensure_ascii=False, default=str) + "\n",
    )
    logger.info("Wrote %d target reviews to %s", len(proposals), path)


def write_keyword_suggestions_json(proposals: List[dict], path: Path) -> None:
    """Extract keyword suggestions from proposals into a dedicated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suggestions = []
    for p in proposals:
        kw = p.get("keyword_suggestions", {})
        if not kw:
            continue
        suggestions.append(kw)
    path.write_text(
        json.dumps(suggestions, indent=2, ensure_ascii=False, default=str) + "\n",
    )
    logger.info("Wrote keyword suggestions for %d targets to %s", len(suggestions), path)


def write_summary_md(
    proposals: List[Dict[str, Any]],
    week_id: str,
    *,
    events_count: int,
    path: Path,
) -> None:
    """Write human-readable keep/drop/review summary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    keep = [p for p in proposals if p["action"] == "keep"]
    drop = [p for p in proposals if p["action"] == "drop_candidate"]
    review = [p for p in proposals if p["action"] == "review"]

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Weekly Targets Review \u2014 {week_id}\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- **Targets reviewed:** {len(proposals)}\n")
        f.write(f"- **Keep:** {len(keep)}\n")
        f.write(f"- **Drop candidates:** {len(drop)}\n")
        f.write(f"- **Review needed:** {len(review)}\n")
        f.write(f"- **Events pool:** {events_count}\n\n")

        # Structural shifts
        shifts = [p for p in proposals if p.get("structural_shift")]
        if shifts:
            f.write("## Structural Shifts (LLM)\n\n")
            for p in shifts:
                f.write(f"### {p['target_name']} ({p['target_type']})\n\n")
                f.write(f"- **Shift:** {p.get('shift_description', '')}\n")
                f.write(f"- **LLM action:** {p.get('llm_action', '')} \u2014 {p.get('llm_reason', '')}\n\n")

        if drop:
            f.write("## Drop Candidates\n\n")
            for p in drop:
                f.write(f"### {p['target_name']} ({p['target_type']})\n\n")
                f.write(f"- **Current:** priority={p['current_priority']}, cadence={p['current_cadence']}\n")
                f.write(f"- **Signal:** {p['signal_score']:.2f}  |  **Noise:** {p['noise_score']:.2f}\n")
                f.write(f"- **Events this week:** {p['number_of_events']}\n")
                f.write(f"- **Reason:** {p['reason']}\n")
                if p.get("suggested_keywords"):
                    f.write(f"- **Suggested keywords:** {', '.join(p['suggested_keywords'][:5])}\n")
                if p.get("suggested_source_urls"):
                    f.write(f"- **Suggested URLs:** {', '.join(p['suggested_source_urls'][:3])}\n")
                f.write("\n")

        if review:
            f.write("## Review Needed\n\n")
            for p in review:
                f.write(f"### {p['target_name']} ({p['target_type']})\n\n")
                f.write(f"- **Current:** priority={p['current_priority']}, cadence={p['current_cadence']}\n")
                f.write(f"- **Proposed:** priority={p['proposed_priority']}, cadence={p['proposed_cadence']}\n")
                f.write(f"- **Signal:** {p['signal_score']:.2f}  |  **Noise:** {p['noise_score']:.2f}\n")
                f.write(f"- **Events this week:** {p['number_of_events']}\n")
                f.write(f"- **Reason:** {p['reason']}\n\n")

        if keep:
            f.write("## Keep\n\n")
            f.write("| Target | Type | Priority | Cadence | Signal | Events | Notes |\n")
            f.write("|--------|------|----------|---------|--------|--------|-------|\n")
            for p in keep:
                name_short = p["target_name"][:40] + ("\u2026" if len(p["target_name"]) > 40 else "")
                change_note = ""
                if p["proposed_priority"] != p["current_priority"]:
                    change_note += f"pri\u2192{p['proposed_priority']} "
                if p["proposed_cadence"] != p["current_cadence"]:
                    change_note += f"cad\u2192{p['proposed_cadence']}"
                change_note = change_note.strip() or "\u2014"
                f.write(
                    f"| {name_short} | {p['target_type']} | {p['current_priority']} "
                    f"| {p['current_cadence']} | {p['signal_score']:.2f} "
                    f"| {p['number_of_events']} | {change_note} |\n"
                )
            f.write("\n")

        f.write("---\n\n")
        f.write(f"*Generated by {SCRIPT_NAME}*\n")

    logger.info("Wrote summary to %s", path)


def write_policy_md(
    proposals: List[Dict[str, Any]],
    week_id: str,
    *,
    path: Path,
) -> None:
    """Write the next-week monitoring policy."""
    path.parent.mkdir(parents=True, exist_ok=True)

    keep = [p for p in proposals if p["action"] == "keep"]
    drop = [p for p in proposals if p["action"] == "drop_candidate"]
    review = [p for p in proposals if p["action"] == "review"]

    priority_changes = [p for p in proposals
                        if p["proposed_priority"] != p["current_priority"]]
    cadence_changes = [p for p in proposals
                       if p["proposed_cadence"] != p["current_cadence"]]
    kw_targets = [p for p in proposals
                  if p.get("keyword_suggestions", {}).get("keywords_to_add")]

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Next-Week Monitoring Policy \u2014 {week_id}\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Continue monitoring:** {len(keep)} targets\n")
        f.write(f"- **Drop candidates:** {len(drop)} (disable or remove)\n")
        f.write(f"- **Needs review:** {len(review)}\n")
        f.write(f"- **Priority adjustments proposed:** {len(priority_changes)}\n")
        f.write(f"- **Cadence adjustments proposed:** {len(cadence_changes)}\n")
        f.write(f"- **Keyword tuning suggestions:** {len(kw_targets)}\n\n")

        if drop:
            f.write("## Recommended to Drop\n\n")
            for p in drop:
                f.write(f"- **{p['target_name']}** ({p['target_type']}): {p['reason']}\n")
            f.write("\n")

        if priority_changes:
            f.write("## Priority Adjustments\n\n")
            for p in priority_changes:
                f.write(
                    f"- **{p['target_name']}**: {p['current_priority']} \u2192 "
                    f"{p['proposed_priority']} ({p['reason']})\n"
                )
            f.write("\n")

        if cadence_changes:
            f.write("## Cadence Adjustments\n\n")
            for p in cadence_changes:
                f.write(
                    f"- **{p['target_name']}**: {p['current_cadence']} \u2192 "
                    f"{p['proposed_cadence']} ({p['reason']})\n"
                )
            f.write("\n")

        if kw_targets:
            f.write("## Keyword Tuning\n\n")
            for p in kw_targets:
                kw = p["keyword_suggestions"]
                f.write(f"### {p['target_name']}\n\n")
                adds = kw.get("keywords_to_add", [])
                if adds:
                    f.write("**Suggested additions:**\n\n")
                    for a in adds[:5]:
                        sample = a["sample_events"][0][:60] if a.get("sample_events") else ""
                        f.write(f"  - `{a['keyword']}` ({a['count']} events, e.g. \"{sample}\")\n")
                    f.write("\n")
                stale = kw.get("keywords_stale", [])
                if stale:
                    f.write("**Stale candidates (0 appearances this week):**\n\n")
                    for s in stale[:5]:
                        f.write(f"  - `{s['keyword']}` \u2014 {s['reason']}\n")
                    f.write("\n")
                excl = kw.get("keywords_to_exclude", [])
                if excl:
                    f.write("**Noise-correlated tokens (consider excluding):**\n\n")
                    for e in excl[:3]:
                        f.write(f"  - `{e['keyword']}` \u2014 {e['reason']}\n")
                    f.write("\n")

        if review:
            f.write("## Needs Manual Review\n\n")
            for p in review:
                f.write(f"- **{p['target_name']}** ({p['target_type']}): {p['reason']}\n")
            f.write("\n")

        f.write("## Continue Watching (no changes)\n\n")
        no_change = [p for p in keep
                     if p["proposed_priority"] == p["current_priority"]
                     and p["proposed_cadence"] == p["current_cadence"]]
        if no_change:
            for p in no_change:
                f.write(f"- {p['target_name']} ({p['target_type']}, {p['current_priority']})\n")
        else:
            f.write("_(all 'keep' targets have proposed changes above)_\n")
        f.write("\n")

        f.write("---\n\n")
        f.write(f"*Generated by {SCRIPT_NAME}*\n")

    logger.info("Wrote policy to %s", path)


# ================================================================
# CLI
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Weekly targets review: score, LLM shifts, external search, propose, output.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", default=False)
    mode.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--min-signal", type=float, default=MIN_SIGNAL_THRESHOLD)
    p.add_argument("--max-noise", type=float, default=HIGH_NOISE_THRESHOLD)
    p.add_argument("--keyword-min-count", type=int, default=KEYWORD_MIN_EVENT_COUNT)
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
        "summary": {}, "errors": [], "target_update_page_ids": [],
    }

    # ---- Setup ----
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    load_env()

    wk = get_iso_week_context(tz=JST)
    result["week_id"] = wk.week_id
    targets_db_id = get_db_id("NOTION_MONITORING_TARGETS_DB_ID")

    now_jst = wk.now_utc
    date_from_jst = now_jst - timedelta(days=args.days)
    date_from_iso = date_from_jst.isoformat(timespec="seconds")
    date_to_iso = now_jst.isoformat(timespec="seconds")

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

    # Build search clients (graceful — warn if missing, don't fail)
    cse: GoogleCSEClient | None = None
    news: NewsAPIClient | None = None
    try:
        cse = build_google_cse_from_env()
    except GoogleCSEError as e:
        logger.warning("Google CSE unavailable: %s — external validation limited", e)
    try:
        news = build_newsapi_from_env()
    except NewsAPIError as e:
        logger.warning("NewsAPI unavailable: %s — external validation limited", e)

    # ---- Fetch targets ----
    resolver = NotionDataSourceResolver(client)
    resolved = resolver.resolve_once(name="MONITORING_TARGETS_DB", database_id=targets_db_id)

    raw_pages = fetch_all_targets(client, resolved.data_source_id)
    all_targets = normalize_targets(raw_pages)
    active_targets = filter_targets(all_targets)

    # ---- Load evidence ----
    events = load_evidence_json(wk.week_id, "048_weekly_events_digest", "events.json",
                                 base=args.output_base)

    # ---- Compute metrics ----
    enriched = compute_target_metrics(active_targets, events, reference_time=now_jst)

    # ---- Generate proposals (rule-based) ----
    proposals = propose_target_actions(
        enriched, events,
        min_signal=args.min_signal, max_noise=args.max_noise,
        keyword_min_count=args.keyword_min_count,
    )

    # ---- LLM structural shift detection ----
    logger.info("Detecting structural shifts in %d targets via OpenAI ...", len(proposals))
    proposals = detect_structural_shifts_llm(llm, proposals, events)

    # ---- External search validation for drop candidates ----
    if cse or news:
        proposals = validate_drop_candidates_with_search(llm, cse, news, proposals)
    else:
        logger.warning("No search clients available — skipping external validation")

    action_counts: Dict[str, int] = defaultdict(int)
    for p in proposals:
        action_counts[p["action"]] += 1
    shift_count = sum(1 for p in proposals if p.get("structural_shift"))

    # ---- Write outputs ----
    write_targets_review_json(proposals, out_dir / "targets_review.json")
    write_keyword_suggestions_json(proposals, out_dir / "keyword_suggestions.json")
    write_summary_md(proposals, wk.week_id, events_count=len(events),
                      path=out_dir / "summary.md")
    write_policy_md(proposals, wk.week_id, path=out_dir / "policy.md")

    # ---- Notion writeback (skip "keep" rows) ----
    notion_write_count = 0
    trunc_tracker = TruncationTracker()

    if write_enabled:
        tgt_update_db_id = get_db_id("NOTION_WEEKLY_TARGET_UPDATE_DB_ID")
        tgt_update_resolver = NotionDataSourceResolver(client)
        tgt_update_resolved = tgt_update_resolver.resolve_once(
            name="WEEKLY_TARGET_UPDATE_DB", database_id=tgt_update_db_id,
        )
        repo = WeeklyTargetUpdateRepo(
            client=client,
            database_id=tgt_update_resolved.database_id,
            data_source_id=tgt_update_resolved.data_source_id,
        )
        repo.validate_schema()

        # Filter: "keep" is NOT an update — only write non-keep proposals
        non_keep = [p for p in proposals if p["action"] != "keep"]
        rows_to_write = non_keep[:args.limit] if args.limit > 0 else non_keep
        logger.info("Writing %d non-keep proposals to Notion (skipping %d keep rows)",
                     len(rows_to_write), len(proposals) - len(non_keep))

        notion_fail_count = 0
        for tgt_rec in rows_to_write:
            try:
                key, props = repo.build_target_review_properties(
                    target_record=tgt_rec, week_id=wk.week_id, tracker=trunc_tracker,
                )
                page = repo.upsert_row(key=key, properties=props)
                result["target_update_page_ids"].append(page.get("id", ""))
                notion_write_count += 1
            except Exception as e:
                notion_fail_count += 1
                tgt_name = tgt_rec.get("target_name", "?")
                err = f"Failed to write target review ({tgt_name}): {e}"
                logger.error(err)
                result["errors"].append(err)

        logger.info(
            "Notion: %d rows upserted to WEEKLY_TARGET_UPDATE_DB "
            "(%d failed)",
            notion_write_count, notion_fail_count,
        )

    # ---- Metadata ----
    search_usage = []
    if cse:
        search_usage.append(cse.usage_summary())
    if news:
        search_usage.append(news.usage_summary())

    meta = RunMetadata.build(
        notebook=SCRIPT_NAME, week_id=wk.week_id,
        date_from=date_from_iso, date_to=date_to_iso,
        counts={
            "targets_fetched": len(raw_pages),
            "targets_normalised": len(all_targets),
            "targets_active": len(active_targets),
            "targets_keep": action_counts.get("keep", 0),
            "targets_drop_candidate": action_counts.get("drop_candidate", 0),
            "targets_review": action_counts.get("review", 0),
            "structural_shifts": shift_count,
            "events_loaded": len(events),
        },
        extra={
            "min_signal": args.min_signal,
            "max_noise": args.max_noise,
            "data_source_id": resolved.data_source_id,
            "timezone": "Asia/Tokyo",
            "llm_usage": llm.usage_summary(),
            "search_usage": "; ".join(search_usage),
            "write_enabled": write_enabled,
            "notion_rows_upserted": notion_write_count,
        },
    )
    meta.save(out_dir / "run_metadata.json")

    result["summary"] = {
        "targets_reviewed": len(proposals),
        "structural_shifts": shift_count,
    }
    result["ok"] = True
    logger.info("=== Done: %d targets \u2192 %s ===", len(proposals), out_dir)
    logger.info(llm.usage_summary())
    return result


if __name__ == "__main__":
    r = main()
    raise SystemExit(0 if r["ok"] else 1)
