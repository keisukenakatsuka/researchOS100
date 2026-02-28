#!/usr/bin/env python
# src/scripts/060_morning_commit.py
"""Morning Commit — finalize today's top 3 and commit items.

Pipeline:
1. Resolve source date (latest prior close log) for 058/059 inputs
2. Load 058 (provisional top 3) and 059 (next-day prep) from source date
3. Re-evaluate provisional top 3
4. Ask for energy level + time budget (interactive or flags)
5. Finalize top 3 commit items with definitions of done
6. Output to data/daily/morning_commit/YYYY-MM-DD/ (commit date)
7. Upsert Notion Daily Log for commit date with source date annotation

When --date is omitted, the commit date is today and the source date is
resolved from Notion by querying the Daily Logs database for the most
recent LogDate strictly before today.  When --date is provided, both
commit date and source date are set to that value (legacy behaviour).

Usage::

    # Interactive morning commit (auto-resolves source date)
    python -m src.scripts.060_morning_commit

    # Non-interactive with explicit settings (--date = both commit & source)
    python -m src.scripts.060_morning_commit --date 2026-02-20 \\
        --non-interactive --energy Medium --time-budget-hrs 6

    # Specific date, interactive
    python -m src.scripts.060_morning_commit --date 2026-02-20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env, setup_logging, RunMetadata, get_iso_week_context
from src.daily.models import (
    CloseStructured, CommitItem, MorningCommit, NextDayPrep,
)
from src.daily.io import (
    CLOSE_STRUCTURED_DIR, NEXT_DAY_PREP_DIR, MORNING_COMMIT_DIR,
    daily_output_dir, save_json, load_json,
)

logger = logging.getLogger("060_morning_commit")

SCRIPT_NAME = "060_morning_commit"
JST = ZoneInfo("Asia/Tokyo")

def _resolve_source_date_from_notion(commit_date: str) -> Optional[str]:
    """Query Notion Daily Logs for the most recent LogDate strictly before *commit_date*.

    Uses ``NOTION_Daily_Logs_ID`` as the authoritative source of truth:

    * Filter:  ``LogDate`` *before* ``commit_date``
    * Sort:    ``LogDate`` descending
    * Limit:   1 page

    Returns the ``LogDate`` (ISO string) of the matched page, or ``None``
    when no qualifying Daily Log exists.
    """
    from src.config import get_db_id
    from src.notion.client import (
        build_notion_client_from_env,
        NotionDataSourceResolver,
    )

    client = build_notion_client_from_env(log_requests=False, log_responses=False)
    db_id = get_db_id("NOTION_Daily_Logs_ID")

    resolver = NotionDataSourceResolver(client=client)
    resolved = resolver.resolve_once(name="daily_logs", database_id=db_id)
    logger.debug(
        "Notion resolver: db_id=%s  data_source_id=%s",
        db_id[:8], resolved.data_source_id[:8],
    )

    pages = client.query_data_source(
        data_source_id=resolved.data_source_id,
        filter={
            "property": "LogDate",
            "date": {"before": commit_date},
        },
        sorts=[{"property": "LogDate", "direction": "descending"}],
        page_size=1,
        fetch_all=False,
    )

    if not pages:
        return None

    # Extract LogDate → date → start from the Notion page object
    log_date_prop = pages[0].get("properties", {}).get("LogDate", {})
    date_obj = log_date_prop.get("date") or {}
    source_date = (date_obj.get("start") or "")[:10]  # "YYYY-MM-DD"

    if not source_date:
        logger.warning(
            "Notion returned a page but LogDate.start is empty: page_id=%s",
            pages[0].get("id", "???"),
        )
        return None

    logger.info(
        "Resolved source_date=%s from Notion (commit_date=%s)",
        source_date, commit_date,
    )
    return source_date


def _load_previous_outputs(date_iso: str):
    """Load 058 and 059 outputs. Either may be missing."""
    structured = None
    prep = None

    s_path = CLOSE_STRUCTURED_DIR / date_iso / "close_structured.json"
    if s_path.exists():
        structured = CloseStructured.from_dict(load_json(s_path))
        logger.info("Loaded 058 output for %s", date_iso)

    p_path = NEXT_DAY_PREP_DIR / date_iso / "next_day_prep.json"
    if p_path.exists():
        prep = NextDayPrep.from_dict(load_json(p_path))
        logger.info("Loaded 059 output for %s", date_iso)

    return structured, prep


def _prompt_energy_level() -> str:
    """Prompt for energy level."""
    try:
        print("\nHow's your energy this morning?")
        print("  1. Low")
        print("  2. Medium")
        print("  3. High")
        raw = input("Choice (1/2/3, default=2): ").strip()
        return {"1": "Low", "2": "Medium", "3": "High"}.get(raw, "Medium")
    except EOFError:
        return "Medium"


def _prompt_time_budget() -> float:
    """Prompt for available hours."""
    try:
        raw = input("Available hours today (default=6): ").strip()
        if not raw:
            return 6.0
        return float(raw)
    except (ValueError, EOFError):
        return 6.0


def _prompt_edit_top3(provisional: list) -> list:
    """Allow user to edit the provisional top 3."""
    print("\nProvisional Top 3:")
    for i, item in enumerate(provisional, 1):
        print(f"  {i}. {item}")

    try:
        raw = input("\nAccept? (y/n, default=y): ").strip().lower()
        if raw in ("", "y", "yes"):
            return provisional

        print("Enter your top 3 (one per line, empty line to finish):")
        edited = []
        for i in range(3):
            line = input(f"  {i+1}. ").strip()
            if not line:
                break
            edited.append(line)
        return edited if edited else provisional
    except EOFError:
        return provisional


def _build_commits(
    final_top3: list,
    energy_level: str,
    time_budget_hrs: float,
    date_iso: str,
) -> list:
    """Build commit items from final top 3."""
    if final_top3:
        minutes_each = int((time_budget_hrs * 60) / len(final_top3))
    else:
        minutes_each = 0

    commits = []
    for i, text in enumerate(final_top3, 1):
        commit = CommitItem(
            title=text[:100],
            rank=i,
            status="Planned",
            why=f"Top {i} priority for {date_iso}",
            definition_of_done=f"Complete: {text[:80]}",
            planned_time_block=f"Block {i}",
            estimated_minutes=minutes_each,
            order=i,
            value_domains=[],
            notes="",
        )
        commits.append(commit)
    return commits


def run_pipeline(
    *,
    date_override: Optional[str] = None,
    energy_level: Optional[str] = None,
    time_budget_hrs: Optional[float] = None,
    non_interactive: bool = False,
    verbose: bool = False,
) -> dict:
    """Execute the morning commit pipeline.

    Parameters
    ----------
    date_override : str | None
        When provided, acts as both commit date *and* source date
        (legacy behaviour).  When ``None``, commit date is today and
        source date is auto-resolved from the latest prior Daily Log
        in Notion (authoritative source of truth).
    non_interactive : bool
        If True, skip all interactive prompts. Uses provided flags or defaults.
    """
    load_env()
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    now_jst = datetime.now(tz=JST)
    wk = get_iso_week_context(tz=JST)

    # ── 0. Resolve commit_date & source_date ──
    if date_override:
        # --date provided: both dates are the same (legacy behaviour)
        commit_date = date_override
        source_date = date_override
        logger.info(
            "Using --date override: commit_date=%s source_date=%s",
            commit_date, source_date,
        )
    else:
        # No --date: commit for today, source from latest prior Daily Log in Notion
        commit_date = now_jst.date().isoformat()
        source_date = _resolve_source_date_from_notion(commit_date)
        if source_date is None:
            msg = (
                f"No prior Daily Log found in Notion before commit_date={commit_date}. "
                f"Run 058_daily_close_structuring first, or use --date to specify a date."
            )
            logger.error(msg)
            print(f"\nERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        logger.info(
            "Auto-resolved: commit_date=%s  source_date=%s",
            commit_date, source_date,
        )

    logger.info(
        "Starting %s commit_date=%s source_date=%s non_interactive=%s",
        SCRIPT_NAME, commit_date, source_date, non_interactive,
    )

    # ── 1. Load previous outputs (from source_date) ──
    structured, prep = _load_previous_outputs(source_date)

    # ── 2. Get provisional top 3 ──
    provisional_top3 = []
    if structured:
        provisional_top3 = structured.provisional_top3[:3]

    if not provisional_top3:
        if prep and prep.follow_ups:
            provisional_top3 = [f.title for f in prep.follow_ups[:3]]

    if not provisional_top3:
        # No data at all for the resolved source_date — abort rather
        # than creating a useless committed page.
        msg = (
            f"Source date {source_date} has no provisional top 3 "
            f"(058) and no follow-ups (059). Nothing to commit."
        )
        logger.error(msg)
        print(f"\nERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    # ── 3. Collect energy + time budget ──
    if non_interactive:
        energy_level = energy_level or "Medium"
        time_budget_hrs = time_budget_hrs if time_budget_hrs is not None else 6.0
        final_top3 = provisional_top3[:3]
        logger.info("Non-interactive: energy=%s, hours=%.1f", energy_level, time_budget_hrs)
    else:
        if energy_level is None:
            energy_level = _prompt_energy_level()
        if time_budget_hrs is None:
            time_budget_hrs = _prompt_time_budget()
        final_top3 = _prompt_edit_top3(provisional_top3)

    # ── 4. Build commits (tagged with commit_date) ──
    commits = _build_commits(final_top3, energy_level, time_budget_hrs, commit_date)

    # ── 5. Build model ──
    morning = MorningCommit(
        date=commit_date,
        energy_level=energy_level,
        time_budget_hrs=time_budget_hrs,
        final_top3=final_top3,
        commits=commits,
    )

    # ── 6. Save output (under commit_date) ──
    out_dir = daily_output_dir(MORNING_COMMIT_DIR, commit_date)
    morning_dict = morning.to_dict()
    morning_dict["source_date"] = source_date
    out_path = save_json(out_dir / "morning_commit.json", morning_dict)
    logger.info("Saved morning_commit.json -> %s", out_path)

    # ── 7. Run metadata ──
    meta = RunMetadata.build(
        notebook=SCRIPT_NAME,
        week_id=wk.week_id,
        counts={
            "commits": len(commits),
            "energy_level": energy_level,
            "time_budget_hrs": time_budget_hrs,
            "source_date": source_date,
        },
    )
    meta.save(out_dir / "run_metadata.json")

    # ── 8. Notion upsert — committed layer (target = commit_date) ──
    from src.notion.daily_schema import build_daily_log_properties
    from src.notion.daily_upsert import safe_truncate, upsert_daily_log

    # Annotate Final Top 3 with source date when it differs from commit date
    top3_lines = "\n".join(f"{i+1}. {t}" for i, t in enumerate(final_top3))
    if source_date != commit_date:
        top3_lines = f"[Source Close Date: {source_date}]\n{top3_lines}"

    schedule_blocks = "\n".join(
        f"{c.rank}. [{c.planned_time_block}] {c.title} ({c.estimated_minutes}min)"
        for c in commits
    )
    notion_props = build_daily_log_properties(
        title=f"Daily Log {commit_date}",
        date=commit_date,
        final_top3=safe_truncate(top3_lines),
        schedule_time_blocks=safe_truncate(schedule_blocks),
        stage="committed",
    )
    notion_result = upsert_daily_log(
        date_iso=commit_date,
        properties=notion_props,
        log_label="060_committed",
    )

    result = {
        "commit_date": commit_date,
        "source_date": source_date,
        "output_dir": str(out_dir),
        "energy_level": energy_level,
        "time_budget_hrs": time_budget_hrs,
        "final_top3": final_top3,
        "commits": len(commits),
    }
    if notion_result:
        result["notion"] = notion_result

    source_note = ""
    if source_date != commit_date:
        source_note = f" (source: {source_date})"
    print(f"\nMorning commit saved -> {out_dir}{source_note}")
    print(f"  Energy: {energy_level} | Time: {time_budget_hrs}h | Commits: {len(commits)}")
    print(f"  Final Top 3:")
    for i, item in enumerate(final_top3, 1):
        print(f"    {i}. {item}")
    if notion_result.get("ok"):
        print(f"  Notion: {notion_result['action']} -> {notion_result.get('page_url', '')}")
    else:
        print(f"  Notion: FAILED -> {notion_result.get('error', 'unknown')}")

    return result


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="060 Morning Commit — finalize top 3 and commit items",
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Override date (YYYY-MM-DD). Sets both commit and source date.")
    parser.add_argument("--energy", type=str, default=None, choices=["Low", "Medium", "High"],
                        help="Energy level")
    parser.add_argument("--time-budget-hrs", type=float, default=None, dest="time_budget_hrs",
                        help="Available hours today")
    parser.add_argument("--non-interactive", action="store_true", dest="non_interactive",
                        help="Skip all interactive prompts (use flags or defaults)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    result = run_pipeline(
        date_override=args.date,
        energy_level=args.energy,
        time_budget_hrs=args.time_budget_hrs,
        non_interactive=args.non_interactive,
        verbose=args.verbose,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
