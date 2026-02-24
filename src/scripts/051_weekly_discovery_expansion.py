#!/usr/bin/env python
# src/scripts/051_weekly_discovery_expansion.py
"""Weekly discovery & expansion — find NEW candidate entities from recent data.

Pipeline:
1. Load 048 events.json (primary) and optionally 047 papers.json
2. Extract candidate entity names via capitalized-phrase + frequency heuristics
3. Score candidates (frequency x diversity x growth)
4. Categorise each as VC / Startup / Policy / People / Other / Unknown
5. LLM classification + noise-rejection (mandatory)
6. Filter out already-tracked targets and generic keywords
7. External search enrichment (Google CSE + NewsAPI + LLM) for accepted candidates
8. Write results to WEEKLY_TARGET_ADDITIONAL_DB

LLM is mandatory. Write-back is default (use --no-write for debug).
Proposes SPECIFIC named entities — not abstract themes or generic keywords.

Usage::

    python -m src.scripts.051_weekly_discovery_expansion --run
    python -m src.scripts.051_weekly_discovery_expansion --run --no-write -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
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
from src.discovery.entities import (
    extract_entities_from_events,
    extract_entities_from_papers,
    merge_raw_entities,
)
from src.discovery.scoring import (
    DEFAULT_TOP_K,
    MIN_FINAL_SCORE,
    MIN_MENTION_COUNT,
    filter_already_tracked,
    score_candidates,
)
from src.notion import (
    NotionDataSourceResolver,
    build_notion_client_from_env,
)
from src.notion.truncation import TruncationTracker
from src.notion.weekly_target_additional_repo import WeeklyTargetAdditionalRepo

logger = logging.getLogger("051_weekly_discovery_expansion")

SCRIPT_NAME = "051_weekly_discovery_expansion"
JST = ZoneInfo("Asia/Tokyo")

DEFAULT_LLM_TOP_N = 100


# ================================================================
# Generic keyword exclusion (specificity filter)
# ================================================================

GENERIC_KEYWORDS: Set[str] = {
    # English generic terms
    "startup", "startups", "data", "innovation", "technology", "digital",
    "business", "market", "growth", "investment", "fund", "funds",
    "research", "development", "analysis", "strategy", "management",
    "platform", "solution", "solutions", "service", "services",
    "industry", "sector", "ecosystem", "venture", "capital",
    "company", "companies", "enterprise", "enterprises",
    "investor", "investors", "founder", "founders",
    "ai", "ml", "saas", "fintech", "biotech", "deeptech",
    "funding", "round", "series", "valuation", "ipo",
    "report", "survey", "trend", "trends", "forecast",
    # Japanese generic terms
    "スタートアップ", "データ", "イノベーション", "テクノロジー",
    "ビジネス", "マーケット", "投資", "ファンド", "研究",
    "開発", "分析", "戦略", "経営", "プラットフォーム",
    "ソリューション", "サービス", "産業", "エコシステム",
    "ベンチャー", "キャピタル", "企業", "起業家", "創業者",
}


def _is_generic_candidate(name: str) -> bool:
    """Check if a candidate name is too generic to be actionable."""
    normalized = name.strip().lower()
    if normalized in GENERIC_KEYWORDS:
        return True
    # Also check if the entire name is a single generic token
    tokens = normalized.split()
    if len(tokens) == 1 and tokens[0] in GENERIC_KEYWORDS:
        return True
    return False


# ================================================================
# LLM classification (with specificity enforcement)
# ================================================================

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a research intelligence analyst. For each candidate entity, \
perform three tasks:

1. **Classify** into one of: VC, Startup, Policy, People, Other, Unknown. \
Use only the evidence provided (event/paper titles).
2. **Rationale**: Write ONE sentence explaining why this entity is notable, \
grounded strictly in the provided evidence snippets. Do NOT invent facts.
3. **Noise check**: If this is clearly noise (a generic word, date, acronym \
fragment, abstract theme, etc.), set reject=true with a short reason.

CRITICAL — specificity requirement:
- Only ACCEPT candidates that are SPECIFIC NAMED ENTITIES:
  - VC firms: "Sequoia Capital", "a16z", "Coral Capital" (NOT "venture capital firms")
  - Startups: "Stripe", "OpenAI", "SmartHR" (NOT "AI startups" or "SaaS companies")
  - Policy: "経済産業省", "NIST", "DARPA" (NOT "government agencies")
  - People: "Sam Altman", "孫正義" (NOT "tech leaders" or "founders")
- REJECT abstract themes, categories, and generic descriptors.

Return a JSON object with key "results" containing an array of objects, \
one per candidate, each with:
  - candidate_name: string
  - type_llm: string (VC/Startup/Policy/People/Other/Unknown)
  - rationale_llm: string (one sentence)
  - reject: boolean
  - reject_reason: string or null
"""


def classify_candidates_llm(
    llm: OpenAIClient,
    candidates: List[Dict[str, Any]],
    *,
    top_n: int = DEFAULT_LLM_TOP_N,
) -> List[Dict[str, Any]]:
    """Run LLM classification on shortlist and merge results.

    Returns the filtered candidates list (noise-rejected removed).
    """
    shortlist = []
    for c in candidates[:top_n]:
        shortlist.append({
            "candidate_name": c["candidate_name"],
            "type_rule": c["type"],
            "mention_count": c["mention_count"],
            "source_count": c["source_count"],
            "event_types": c["event_types"],
            "sample_event_titles": c["evidence"]["sample_event_titles"],
            "sample_paper_titles": c["evidence"]["sample_paper_titles"],
        })

    user_prompt = (
        "Classify and annotate the following candidates:\n\n"
        + json.dumps(shortlist, indent=2, ensure_ascii=False)
    )

    result = llm.call_json(system=CLASSIFICATION_SYSTEM_PROMPT, user=user_prompt)

    llm_results = result.parsed.get("results", [])
    logger.info("LLM returned %d results for %d candidates",
                len(llm_results), len(shortlist))

    # Merge LLM results into candidates
    llm_by_name: Dict[str, dict] = {}
    for r in llm_results:
        llm_by_name[r.get("candidate_name", "").lower().strip()] = r

    final_candidates = []
    rejected = 0
    for c in candidates:
        llm_r = llm_by_name.get(c["candidate_name_normalized"])
        if llm_r:
            c["llm_override"] = {
                "type_llm": llm_r.get("type_llm", c["type"]),
                "rationale_llm": llm_r.get("rationale_llm", ""),
                "reject": llm_r.get("reject", False),
                "reject_reason": llm_r.get("reject_reason"),
            }
            if not llm_r.get("reject", False):
                final_candidates.append(c)
            else:
                rejected += 1
                logger.debug("LLM rejected: %s (%s)", c["candidate_name"],
                             llm_r.get("reject_reason", ""))
        else:
            # Not in LLM shortlist — keep as-is
            final_candidates.append(c)

    logger.info("LLM: accepted %d, rejected %d candidates",
                len(final_candidates), rejected)
    return final_candidates


# ================================================================
# External search enrichment
# ================================================================

SEARCH_ENRICHMENT_SYSTEM_PROMPT = """\
You are a research analyst. For each candidate entity, I have gathered \
search results from Google and news sources.

For each candidate, produce:
1. **search_keywords**: 3-5 specific search keywords/phrases to track this \
entity. Include the entity's full name and key variations. Be specific, \
not generic.
2. **source_urls**: 1-3 authoritative URLs to monitor for this entity \
(official site, Crunchbase, LinkedIn, etc.). Only include URLs that appear \
in the search results.
3. **source_type**: "HTML" | "RSS" | "NEWS" — recommended monitoring method.
4. **priority**: "High" | "Medium" — based on relevance and activity level.
5. **still_relevant**: boolean — is this entity genuinely relevant to \
startup ecosystem / VC / policy research based on the search evidence?

Return a JSON object with key "results" containing an array of objects, \
one per candidate, each with:
  - candidate_name: string
  - search_keywords: string[] (3-5 items)
  - source_urls: string[] (1-3 items)
  - source_type: string
  - priority: string
  - still_relevant: boolean
  - relevance_note: string (one sentence explaining relevance decision)
"""


def enrich_candidates_with_search(
    llm: OpenAIClient,
    candidates: List[Dict[str, Any]],
    *,
    cse: Any = None,
    news: Any = None,
    max_candidates: int = 30,
) -> List[Dict[str, Any]]:
    """Enrich accepted candidates with external search data.

    For each candidate:
    1. Google CSE: search for entity name → extract source URLs
    2. NewsAPI: search for recent mentions → confirm relevance
    3. LLM: propose Search Keywords and Source URLs
    4. Filter: reject generic/abstract entities, keep specific named entities

    Parameters
    ----------
    candidates : list
        Post-LLM-classification candidates (noise already rejected).
    cse : GoogleCSEClient or None
        Google CSE client. If None, skip Google search.
    news : NewsAPIClient or None
        NewsAPI client. If None, skip news search.
    max_candidates : int
        Max candidates to send through search enrichment.
    """
    if not candidates:
        return candidates

    to_enrich = candidates[:max_candidates]
    remaining = candidates[max_candidates:]

    # Gather search results for each candidate
    search_context = []
    for c in to_enrich:
        name = c["candidate_name"]
        ctx: Dict[str, Any] = {"candidate_name": name, "google_results": [], "news_results": []}

        if cse:
            try:
                ctx["google_results"] = cse.search(name, num=3)
            except Exception as e:
                logger.warning("Google CSE search failed for %r: %s", name, e)

        if news:
            try:
                ctx["news_results"] = news.search(name, days_back=30, page_size=3)
            except Exception as e:
                logger.warning("NewsAPI search failed for %r: %s", name, e)

        search_context.append(ctx)

    # LLM enrichment
    user_prompt = (
        "Enrich the following candidates with search keywords and source URLs.\n\n"
        "Search results per candidate:\n"
        + json.dumps(search_context, indent=2, ensure_ascii=False)
    )

    result = llm.call_json(system=SEARCH_ENRICHMENT_SYSTEM_PROMPT, user=user_prompt)
    llm_enrichments = result.parsed.get("results", [])
    logger.info("LLM enrichment returned %d results", len(llm_enrichments))

    # Merge enrichments back into candidates
    enrich_by_name: Dict[str, dict] = {}
    for r in llm_enrichments:
        enrich_by_name[r.get("candidate_name", "").lower().strip()] = r

    enriched = []
    dropped = 0
    for c in to_enrich:
        norm_name = c["candidate_name_normalized"]
        enr = enrich_by_name.get(norm_name)
        if enr:
            if not enr.get("still_relevant", True):
                dropped += 1
                logger.debug("Search enrichment dropped: %s (%s)",
                             c["candidate_name"], enr.get("relevance_note", ""))
                continue
            c["search_keywords"] = enr.get("search_keywords", [])
            c["source_urls"] = enr.get("source_urls", [])
            c["source_type"] = enr.get("source_type", "HTML")
            c["priority"] = enr.get("priority", "Medium")
        else:
            c["search_keywords"] = []
            c["source_urls"] = []
            c["source_type"] = "HTML"
            c["priority"] = "Medium"
        enriched.append(c)

    logger.info("Search enrichment: %d enriched, %d dropped for irrelevance",
                len(enriched), dropped)
    return enriched + remaining


# ================================================================
# Data loading helpers
# ================================================================

def load_evidence_json(
    week_id: str,
    script_name: str,
    filename: str,
    *,
    base: str = "outputs",
) -> List[Dict[str, Any]]:
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


def load_existing_target_names(
    week_id: str,
    *,
    base: str = "outputs",
) -> Set[str]:
    path = Path(base) / "weekly" / week_id / "050_weekly_targets_review" / "targets_review.json"
    if not path.exists():
        logger.info("050 targets_review.json not found — skipping overlap filter")
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        names = set()
        for t in data:
            name = (t.get("target_name") or "").strip().lower()
            if name:
                names.add(name)
        logger.info("Loaded %d existing target names from 050 output", len(names))
        return names
    except Exception as e:
        logger.warning("Failed to load 050 targets: %s", e)
        return set()


def load_prev_week_candidates(
    week_id: str,
    *,
    base: str = "outputs",
) -> Optional[Dict[str, int]]:
    try:
        parts = week_id.split("-W")
        year = int(parts[0])
        week = int(parts[1])
        if week > 1:
            prev_week_id = f"{year}-W{week - 1:02d}"
        else:
            prev_week_id = f"{year - 1}-W52"
    except (ValueError, IndexError):
        return None

    path = Path(base) / "weekly" / prev_week_id / SCRIPT_NAME / "candidates.json"
    if not path.exists():
        logger.info("No previous-week candidates — growth_score will be null")
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            c["candidate_name_normalized"]: c["mention_count"]
            for c in data
            if "candidate_name_normalized" in c
        }
    except Exception as e:
        logger.warning("Failed to load prev-week candidates: %s", e)
        return None


# ================================================================
# Output writers
# ================================================================

def write_candidates_json(candidates: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False, default=str) + "\n",
    )
    logger.info("Wrote %d candidates to %s", len(candidates), path)


def write_summary_md(
    candidates: List[Dict[str, Any]],
    week_id: str,
    *,
    events_count: int,
    papers_count: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_type[c["type"]].append(c)

    tracked = sum(1 for c in candidates if c.get("already_tracked"))
    new_only = [c for c in candidates if not c.get("already_tracked")]

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Weekly Discovery & Expansion — {week_id}\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- **Total candidates:** {len(candidates)}\n")
        f.write(f"- **New (not yet tracked):** {len(new_only)}\n")
        f.write(f"- **Already tracked:** {tracked}\n")
        f.write(f"- **Events analysed:** {events_count}\n")
        f.write(f"- **Papers analysed:** {papers_count}\n\n")

        type_order = ["VC", "Startup", "Policy", "People", "Other", "Unknown"]
        for typ in type_order:
            group = by_type.get(typ, [])
            if not group:
                continue

            f.write(f"## {typ}\n\n")
            f.write("| # | Candidate | Score | Mentions | Sources | Notable |\n")
            f.write("|---|-----------|-------|----------|---------|----------|\n")
            for i, c in enumerate(group, 1):
                name = c["candidate_name"][:40]
                if c.get("already_tracked"):
                    name += " *(tracked)*"
                aliases = c.get("aliases", [])
                alias_str = ""
                if aliases:
                    alias_str = " aka " + ", ".join(a[:25] for a in aliases[:3])
                why_short = c["why_notable"][:80]
                f.write(
                    f"| {i} | {name}{alias_str} | {c['final_score']:.2f} "
                    f"| {c['mention_count']} | {c['source_count']} "
                    f"| {why_short} |\n"
                )
            f.write("\n")

        f.write("---\n\n")
        f.write(f"*Generated by {SCRIPT_NAME}*\n")

    logger.info("Wrote summary to %s", path)


# ================================================================
# CLI
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Weekly discovery: extract, score, LLM classify, search enrich candidates.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", default=False)
    mode.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--week-id", default=None,
                    help="Override week_id (e.g. '2026-W08').")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--min-count", type=int, default=MIN_MENTION_COUNT)
    p.add_argument("--min-score", type=float, default=MIN_FINAL_SCORE)
    p.add_argument("--with-papers", action="store_true", default=True)
    p.add_argument("--no-papers", action="store_true", default=False)
    p.add_argument("--llm-top-n", type=int, default=DEFAULT_LLM_TOP_N,
                    help=f"Number of candidates to send to LLM (default: {DEFAULT_LLM_TOP_N}).")
    p.add_argument("--output-base", default="outputs")
    p.add_argument("--write", action="store_true", default=False,
                    help="Enable Notion write-back.")
    p.add_argument("--limit", type=int, default=0,
                    help="Max rows to write (0 = all).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    """Main pipeline. Returns result dict for orchestrator."""
    args = build_parser().parse_args(argv)
    is_live = args.run
    use_papers = args.with_papers and not args.no_papers
    write_enabled = args.write

    result: Dict[str, Any] = {
        "ok": False, "week_id": "", "output_dir": "",
        "summary": {}, "errors": [], "discovery_page_ids": [],
    }

    # ---- Setup ----
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    load_env()

    if args.week_id:
        from src.config import WeekContext
        parts = args.week_id.split("-W")
        iso_year, iso_week = int(parts[0]), int(parts[1])
        jan4 = datetime(iso_year, 1, 4, tzinfo=JST)
        start_of_week1 = jan4 - timedelta(days=jan4.isoweekday() - 1)
        week_start = start_of_week1 + timedelta(weeks=iso_week - 1)
        week_end = week_start + timedelta(days=7) - timedelta(seconds=1)
        now = datetime.now(JST)
        wk = WeekContext(
            now_utc=now, week_start=week_start,
            start_date=week_start.strftime("%Y-%m-%d"),
            end_date=week_end.strftime("%Y-%m-%d"),
            date_from_iso=week_start.isoformat(timespec="seconds"),
            date_to_iso=week_end.isoformat(timespec="seconds"),
            iso_year=iso_year, iso_week=iso_week,
            week_id=args.week_id,
        )
    else:
        wk = get_iso_week_context(tz=JST)

    result["week_id"] = wk.week_id

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

    # ---- Build LLM client ----
    llm = build_openai_client_from_env()

    # ---- Build search clients (graceful — warn if missing) ----
    cse_client = None
    news_client = None

    try:
        from src.search.google_cse import build_google_cse_from_env
        cse_client = build_google_cse_from_env()
        logger.info("Google CSE client ready")
    except Exception as e:
        logger.warning("Google CSE unavailable (will skip): %s", e)

    try:
        from src.search.newsapi import build_newsapi_from_env
        news_client = build_newsapi_from_env()
        logger.info("NewsAPI client ready")
    except Exception as e:
        logger.warning("NewsAPI unavailable (will skip): %s", e)

    # ---- Load events (primary input) ----
    events = load_evidence_json(wk.week_id, "048_weekly_events_digest", "events.json",
                                 base=args.output_base)

    # ---- Load papers (optional) ----
    papers: List[Dict[str, Any]] = []
    if use_papers:
        papers = load_evidence_json(wk.week_id, "047_weekly_papers_review", "papers.json",
                                     base=args.output_base)

    # ---- Extract entities ----
    raw_from_events = extract_entities_from_events(events)
    logger.info("Extracted %d raw mentions from %d events", len(raw_from_events), len(events))

    raw_from_papers: List[Dict[str, Any]] = []
    if papers:
        raw_from_papers = extract_entities_from_papers(papers)
        logger.info("Extracted %d raw mentions from %d papers", len(raw_from_papers), len(papers))

    # ---- Merge ----
    all_raw = raw_from_events + raw_from_papers
    merged = merge_raw_entities(all_raw, min_count=1)
    logger.info("Merged into %d unique entity candidates", len(merged))

    # ---- Previous week for growth ----
    prev_week = load_prev_week_candidates(wk.week_id, base=args.output_base)

    # ---- Score & rank ----
    scored = score_candidates(
        merged, prev_week_entities=prev_week,
        min_count=args.min_count, min_score=args.min_score, top_k=args.top_k,
    )
    logger.info("Scored: %d candidates above thresholds", len(scored))

    # ---- Filter already-tracked ----
    existing_names = load_existing_target_names(wk.week_id, base=args.output_base)
    if existing_names:
        scored = filter_already_tracked(scored, existing_names)
        tracked_count = sum(1 for c in scored if c.get("already_tracked"))
        logger.info("Marked %d candidates as already tracked", tracked_count)

    # ---- Filter generic keywords ----
    pre_generic_count = len(scored)
    scored = [c for c in scored if not _is_generic_candidate(c.get("candidate_name", ""))]
    generic_filtered = pre_generic_count - len(scored)
    if generic_filtered > 0:
        logger.info("Filtered %d generic candidates", generic_filtered)

    # ---- Write pre-LLM candidates ----
    write_candidates_json(scored, out_dir / "candidates.json")

    # ---- LLM classification (mandatory) ----
    logger.info("Classifying %d candidates via OpenAI ...", len(scored))
    scored = classify_candidates_llm(llm, scored, top_n=args.llm_top_n)

    # ---- Remove already-tracked from final output (keep only new) ----
    new_candidates = [c for c in scored if not c.get("already_tracked")]
    logger.info("New candidates after filtering: %d (removed %d already-tracked)",
                len(new_candidates), len(scored) - len(new_candidates))

    # ---- External search enrichment ----
    if cse_client or news_client:
        logger.info("Enriching %d candidates with external search ...", len(new_candidates))
        new_candidates = enrich_candidates_with_search(
            llm, new_candidates,
            cse=cse_client, news=news_client,
            max_candidates=30,
        )
    else:
        logger.warning("No search clients available — skipping external enrichment")
        for c in new_candidates:
            c["search_keywords"] = []
            c["source_urls"] = []
            c["source_type"] = "HTML"
            c["priority"] = "Medium"

    # ---- Write final candidates ----
    write_candidates_json(new_candidates, out_dir / "candidates_final.json")

    # ---- Tally by type ----
    type_counts: Dict[str, int] = defaultdict(int)
    for c in new_candidates:
        type_counts[c["type"]] += 1

    # ---- Write summary ----
    write_summary_md(new_candidates, wk.week_id, events_count=len(events),
                      papers_count=len(papers), path=out_dir / "summary.md")

    # ---- Notion writeback (to WEEKLY_TARGET_ADDITIONAL_DB) ----
    notion_write_count = 0
    trunc_tracker = TruncationTracker()

    if write_enabled:
        add_db_id = get_db_id("NOTION_WEEKLY_TARGET_ADDITIONAL_DB_ID")
        client = build_notion_client_from_env()
        add_resolver = NotionDataSourceResolver(client)
        add_resolved = add_resolver.resolve_once(
            name="WEEKLY_TARGET_ADDITIONAL_DB", database_id=add_db_id,
        )
        repo = WeeklyTargetAdditionalRepo(
            client=client,
            database_id=add_resolved.database_id,
            data_source_id=add_resolved.data_source_id,
        )
        repo.validate_schema()

        rows_to_write = new_candidates[:args.limit] if args.limit > 0 else new_candidates
        for cand in rows_to_write:
            try:
                key, props = repo.build_proposal_properties(
                    candidate=cand, week_id=wk.week_id, tracker=trunc_tracker,
                )
                page = repo.upsert_row(key=key, properties=props)
                result["discovery_page_ids"].append(page.get("id", ""))
                notion_write_count += 1
            except Exception as e:
                err = f"Failed to write candidate {cand.get('candidate_name', '?')}: {e}"
                logger.warning(err)
                result["errors"].append(err)

        logger.info("Notion: %d rows upserted to WEEKLY_TARGET_ADDITIONAL_DB", notion_write_count)

    # ---- Metadata ----
    meta = RunMetadata.build(
        notebook=SCRIPT_NAME, week_id=wk.week_id,
        date_from=date_from_iso, date_to=date_to_iso,
        counts={
            "events_loaded": len(events),
            "papers_loaded": len(papers),
            "raw_mentions_events": len(raw_from_events),
            "raw_mentions_papers": len(raw_from_papers),
            "unique_merged": len(merged),
            "candidates_scored": len(scored),
            "candidates_new": len(new_candidates),
            "generic_filtered": generic_filtered,
        },
        extra={
            "top_k": args.top_k,
            "with_papers": use_papers,
            "llm_top_n": args.llm_top_n,
            "timezone": "Asia/Tokyo",
            "type_distribution": dict(type_counts),
            "llm_usage": llm.usage_summary(),
            "search_cse_available": cse_client is not None,
            "search_news_available": news_client is not None,
            "search_cse_usage": cse_client.usage_summary() if cse_client else "N/A",
            "search_news_usage": news_client.usage_summary() if news_client else "N/A",
            "write_enabled": write_enabled,
            "notion_rows_upserted": notion_write_count,
        },
    )
    meta.save(out_dir / "run_metadata.json")

    result["summary"] = {
        "candidates_final": len(new_candidates),
        "candidates_new": len(new_candidates),
    }
    result["ok"] = True
    logger.info("=== Done: %d candidates → %s ===", len(new_candidates), out_dir)
    logger.info(llm.usage_summary())
    return result


if __name__ == "__main__":
    r = main()
    raise SystemExit(0 if r["ok"] else 1)
