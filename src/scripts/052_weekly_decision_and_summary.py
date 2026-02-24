#!/usr/bin/env python
# src/scripts/052_weekly_decision_and_summary.py
"""Weekly orchestrator — run 047→051, synthesise strategic digest via LLM.

Running ``python -m src.scripts.052_weekly_decision_and_summary --run``
executes the full pipeline:

1. 047 Weekly Papers Review (LLM classification, digest bootstrap)
2. 048 Weekly Events Digest (LLM theme identification)
3. 049 Weekly RQ Status (LLM evidence linking)
4. 050 Weekly Targets Review (LLM structural shifts)
5. 051 Weekly Discovery (LLM classification)
6. Strategic synthesis via LLM (Executive Summary, Macro Shift, Signals)
7. Update WEEKLY_DIGESTS_DB with synthesis + relation links
8. Generate local summary artefacts via decision_engine

LLM is mandatory. Write-back is default (use --no-write for debug).

Usage::

    python -m src.scripts.052_weekly_decision_and_summary --run
    python -m src.scripts.052_weekly_decision_and_summary --run --no-write -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    RunMetadata,
    get_db_id,
    get_optional_db_id,
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
from src.notion.weekly_digests_repo import WeeklyDigestsRepo
from src.notion.truncation import TruncationTracker
from src.weekly.decision_engine import (
    assess_rq_progress,
    build_next_week_actions,
    compile_weekly_summary,
    draft_worldview_sections,
)

logger = logging.getLogger("052_weekly_decision_and_summary")

SCRIPT_NAME = "052_weekly_decision_and_summary"
OUTPUT_DIR_NAME = "final"


# ================================================================
# Sub-pipeline orchestration
# ================================================================

def _run_sub(name: str, main_fn, argv: List[str]) -> Dict[str, Any]:
    """Run a sub-script's main() and return its result dict.

    If the sub-script raises, catch and return an error result.
    """
    logger.info("--- Running %s ---", name)
    try:
        result = main_fn(argv)
        if not isinstance(result, dict):
            result = {"ok": True, "summary": {}}
        ok = result.get("ok", True)
        logger.info("--- %s %s ---", name, "OK" if ok else "FAILED")
        return result
    except Exception as e:
        logger.error("--- %s FAILED: %s ---", name, e, exc_info=True)
        return {"ok": False, "errors": [str(e)], "summary": {}}


def _import_main(module_name: str):
    """Import main() from a script module whose name starts with a digit."""
    import importlib
    mod = importlib.import_module(f"src.scripts.{module_name}")
    return mod.main


def run_pipeline(
    *,
    write: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run 047→051 in sequence, collecting results."""
    run_047 = _import_main("047_weekly_papers_review")
    run_048 = _import_main("048_weekly_events_digest")
    run_049 = _import_main("049_weekly_rq_status")
    run_050 = _import_main("050_weekly_targets_review")
    run_051 = _import_main("051_weekly_discovery_expansion")

    base_argv = ["--run"]
    if write:
        base_argv.append("--write")
    if verbose:
        base_argv.append("-v")

    results = {}
    results["047"] = _run_sub("047_weekly_papers_review", run_047, base_argv[:])
    results["048"] = _run_sub("048_weekly_events_digest", run_048, base_argv[:])
    results["049"] = _run_sub("049_weekly_rq_status", run_049, base_argv[:])
    results["050"] = _run_sub("050_weekly_targets_review", run_050, base_argv[:])
    results["051"] = _run_sub("051_weekly_discovery_expansion", run_051, base_argv[:])
    return results


# ================================================================
# Strategic synthesis via LLM
# ================================================================

SYNTHESIS_SYSTEM_PROMPT = """\
You are a research intelligence analyst specialising in startup ecosystems, \
venture capital dynamics, and entrepreneurship policy.

Given aggregated weekly data (papers, events, themes, RQ updates, target \
shifts, discovery candidates), produce a strategic synthesis with four fields:

1. **executive_summary**: 3-5 sentences capturing the week's most important \
structural understanding. Think regime-level, not news-level.
2. **macro_shift**: 2-3 sentences on the dominant macro shift (capital flows, \
policy direction, innovation regime, competitive dynamics).
3. **opportunity_signals**: 2-3 bullet points on emerging opportunities \
(growth areas, VC strategy, policy leverage, technology convergence).
4. **risk_signals**: 2-3 bullet points on early warnings (bubbles, \
distortions, misallocation, fragility, regulatory risk).

Tone: analytical, structural, policy-aware, NOT journalistic.
Perspective: a researcher studying startup ecosystems and venture capital.

Return a JSON object:
{
  "executive_summary": "...",
  "macro_shift": "...",
  "opportunity_signals": "...",
  "risk_signals": "..."
}
"""


def _load_themes_from_048(
    sub_results: Dict[str, Dict[str, Any]],
    week_id: str,
    *,
    base: str = "outputs",
) -> List[Dict[str, Any]]:
    """Load themes from 048's sub_results or fallback to themes.json on disk.

    Returns a list of theme dicts (each with at least 'name' and 'summary').
    Themes are ALWAYS sourced from 048 (event-based), never from 047.
    """
    # Prefer in-memory sub_results (when pipeline ran in this process)
    themes = sub_results.get("048", {}).get("themes", [])
    if themes:
        logger.debug("Loaded %d themes from 048 sub_results (in-memory)", len(themes))
        return themes

    # Fallback: load from 048's output file (e.g. --skip-pipeline mode)
    themes_path = Path(base) / "weekly" / week_id / "048_weekly_events_digest" / "themes.json"
    if themes_path.exists():
        try:
            data = json.loads(themes_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                logger.info("Loaded %d themes from %s (file fallback)", len(data), themes_path)
                return data
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Failed to parse %s: %s", themes_path, e)

    logger.warning("No themes available from 048 (sub_results empty, %s not found)", themes_path)
    return []


def synthesize_strategic_digest(
    llm: OpenAIClient,
    *,
    sub_results: Dict[str, Dict[str, Any]],
    week_id: str,
    out_dir: Path,
) -> Dict[str, str]:
    """Generate strategic synthesis from aggregated sub-pipeline data."""

    # Load themes from 048 (in-memory or file fallback)
    themes_raw = _load_themes_from_048(sub_results, week_id)

    # Collect summaries from each sub-pipeline
    data_for_prompt = {
        "week_id": week_id,
        "papers": sub_results.get("047", {}).get("summary", {}),
        "events": sub_results.get("048", {}).get("summary", {}),
        "themes": [
            {"name": t.get("name", ""), "summary": t.get("summary", "")[:100]}
            for t in themes_raw
        ],
        "rq_updates": sub_results.get("049", {}).get("summary", {}),
        "target_review": sub_results.get("050", {}).get("summary", {}),
        "discovery": sub_results.get("051", {}).get("summary", {}),
    }

    # Also load the actual output files for richer context
    base = "outputs"
    base_dir = Path(base) / "weekly" / week_id

    rq_revisions_path = base_dir / "049_weekly_rq_status" / "rq_revisions.json"
    if rq_revisions_path.exists():
        rq_data = json.loads(rq_revisions_path.read_text(encoding="utf-8"))
        if isinstance(rq_data, list):
            data_for_prompt["rq_details"] = [
                {
                    "rq_title": r.get("rq_title", ""),
                    "category": r.get("category", ""),
                    "reason": (r.get("reason") or "")[:100],
                    "proposed_text": (r.get("proposed_text") or "")[:100],
                }
                for r in rq_data[:15]
            ]

    user_prompt = (
        f"Produce a strategic synthesis for week {week_id}.\n\n"
        f"Aggregated data:\n{json.dumps(data_for_prompt, indent=2, ensure_ascii=False)}"
    )

    result = llm.call_json(system=SYNTHESIS_SYSTEM_PROMPT, user=user_prompt)

    synthesis = {
        "executive_summary": result.parsed.get("executive_summary", ""),
        "macro_shift": result.parsed.get("macro_shift", ""),
        "opportunity_signals": result.parsed.get("opportunity_signals", ""),
        "risk_signals": result.parsed.get("risk_signals", ""),
    }

    # Save synthesis to file
    (out_dir / "synthesis.json").write_text(
        json.dumps(synthesis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Strategic synthesis generated (%d chars total)",
                sum(len(v) for v in synthesis.values()))

    return synthesis


# ================================================================
# Local artefact generation (decision_engine)
# ================================================================

def _load_json(path: Path, *, required: bool = False) -> list | dict:
    if not path.exists():
        if required:
            logger.error("Required file missing: %s", path)
            return []
        logger.warning("Optional file not found: %s", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error("Failed to parse %s: %s", path, e)
        return []
    return data


def _load_candidates(week_id: str, *, base: str = "outputs") -> list:
    dir_051 = Path(base) / "weekly" / week_id / "051_weekly_discovery_expansion"
    final_path = dir_051 / "candidates_final.json"
    if final_path.exists():
        return _load_json(final_path)
    return _load_json(dir_051 / "candidates.json")


def generate_local_artefacts(week_id: str, out_dir: Path) -> Dict[str, Any]:
    """Generate local summary artefacts using decision_engine."""
    base = "outputs"
    base_dir = Path(base) / "weekly" / week_id

    rq_statuses = _load_json(base_dir / "049_weekly_rq_status" / "rq_revisions.json")
    papers = _load_json(base_dir / "047_weekly_papers_review" / "papers.json")
    events = _load_json(base_dir / "048_weekly_events_digest" / "events.json")
    themes = _load_json(base_dir / "048_weekly_events_digest" / "themes.json")
    target_reviews = _load_json(base_dir / "050_weekly_targets_review" / "targets_review.json")
    candidates = _load_candidates(week_id, base=base)

    if not isinstance(rq_statuses, list):
        rq_statuses = []
    if not isinstance(papers, list):
        papers = []
    if not isinstance(events, list):
        events = []
    if not isinstance(themes, list):
        themes = []
    if not isinstance(target_reviews, list):
        target_reviews = []
    if not isinstance(candidates, list):
        candidates = []

    # RQ progress assessment
    rq_assessments = assess_rq_progress(
        rq_statuses,
        papers=papers or None,
        events=events or None,
    )

    # Next-week actions
    next_actions = build_next_week_actions(
        rq_assessments,
        target_reviews=target_reviews or None,
        candidates=candidates or None,
    )

    # Worldview narrative
    worldview_md = draft_worldview_sections(
        rq_assessments,
        target_reviews=target_reviews or None,
        candidates=candidates or None,
        events=events or None,
        week_id=week_id,
    )

    # Weekly summary
    summary_md = compile_weekly_summary(
        rq_assessments,
        next_actions,
        week_id=week_id,
        papers_count=len(papers),
        events_count=len(events),
        targets_count=len(target_reviews),
        candidates_count=len(candidates),
    )

    # Write artefacts
    (out_dir / "weekly_worldview.md").write_text(worldview_md, encoding="utf-8")
    (out_dir / "rq_decisions.json").write_text(
        json.dumps(rq_assessments, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "next_week_actions.json").write_text(
        json.dumps(next_actions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "weekly_summary.md").write_text(summary_md, encoding="utf-8")

    logger.info("Local artefacts: worldview, rq_decisions, next_week_actions, summary")
    return {
        "rq_assessments": len(rq_assessments),
        "papers": len(papers),
        "events": len(events),
        "themes": len(themes),
        "targets": len(target_reviews),
        "candidates": len(candidates),
    }


# ================================================================
# CLI
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Orchestrate 047-051, LLM synthesis, Notion update.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", default=False)
    mode.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--no-write", action="store_true", default=False,
                    help="Disable Notion write-back (passed to all sub-scripts).")
    p.add_argument("--skip-pipeline", action="store_true", default=False,
                    help="Skip 047-051 (assume outputs exist). Useful for re-running synthesis.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    """Main orchestrator. Returns result dict."""
    args = build_parser().parse_args(argv)
    is_live = args.run
    write_enabled = not args.no_write

    result: Dict[str, Any] = {
        "ok": False, "week_id": "", "output_dir": "",
        "summary": {}, "errors": [], "sub_results": {},
    }

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    load_env()

    wk = get_iso_week_context()
    result["week_id"] = wk.week_id

    out_dir = get_output_dir(OUTPUT_DIR_NAME, wk.week_id, create=is_live)
    result["output_dir"] = str(out_dir)

    logger.info("=== %s (orchestrator) ===", SCRIPT_NAME)
    logger.info("Week: %s  |  Write: %s", wk.week_id, "ON" if write_enabled else "OFF")

    if not is_live:
        logger.info("[DRY-RUN] Pass --run to execute full pipeline.")
        result["ok"] = True
        return result

    # ---- Step 1: Run 047→051 pipeline ----
    sub_results: Dict[str, Dict[str, Any]] = {}
    if not args.skip_pipeline:
        logger.info("=== Running 047→051 pipeline ===")
        sub_results = run_pipeline(
            write=write_enabled, verbose=args.verbose,
        )
        result["sub_results"] = {
            k: {"ok": v.get("ok"), "summary": v.get("summary", {})}
            for k, v in sub_results.items()
        }

        # Check for failures
        failed = [k for k, v in sub_results.items() if not v.get("ok", False)]
        if failed:
            for f in failed:
                err = f"Sub-pipeline {f} failed"
                result["errors"].append(err)
                logger.warning(err)
    else:
        logger.info("Skipping 047→051 pipeline (--skip-pipeline)")

    # ---- Step 2: Strategic synthesis via LLM ----
    logger.info("=== Strategic Synthesis ===")
    llm = build_openai_client_from_env()

    synthesis = synthesize_strategic_digest(
        llm,
        sub_results=sub_results,
        week_id=wk.week_id,
        out_dir=out_dir,
    )

    # ---- Step 3: Update WEEKLY_DIGESTS_DB ----
    if write_enabled:
        digests_db_id = get_optional_db_id("NOTION_WEEKLY_DIGESTS_DB_ID")
        if digests_db_id:
            try:
                client = build_notion_client_from_env()
                digests_resolver = NotionDataSourceResolver(client)
                digests_resolved = digests_resolver.resolve_once(
                    name="WEEKLY_DIGESTS_DB", database_id=digests_db_id,
                )
                repo = WeeklyDigestsRepo(
                    client=client,
                    database_id=digests_resolved.database_id,
                    data_source_id=digests_resolved.data_source_id,
                )
                repo.validate_schema()

                trunc = TruncationTracker()

                # Collect page IDs from sub-results
                paper_page_ids = sub_results.get("047", {}).get("paper_page_ids", [])
                theme_page_ids = sub_results.get("048", {}).get("theme_page_ids", [])
                rq_update_page_ids = sub_results.get("049", {}).get("rq_update_page_ids", [])
                # Merge 050 target updates + 051 discovery page IDs
                # (both write to WEEKLY_TARGET_UPDATE_DB)
                target_update_page_ids = (
                    sub_results.get("050", {}).get("target_update_page_ids", [])
                    + sub_results.get("051", {}).get("discovery_page_ids", [])
                )

                repo.update_digest_synthesis(
                    week_id=wk.week_id,
                    executive_summary=synthesis.get("executive_summary", ""),
                    macro_shift=synthesis.get("macro_shift", ""),
                    opportunity_signals=synthesis.get("opportunity_signals", ""),
                    risk_signals=synthesis.get("risk_signals", ""),
                    paper_page_ids=paper_page_ids or None,
                    theme_page_ids=theme_page_ids or None,
                    rq_update_page_ids=rq_update_page_ids or None,
                    target_update_page_ids=target_update_page_ids or None,
                    tracker=trunc,
                )
                logger.info("Notion: updated digest %s with synthesis + relations", wk.week_id)
            except Exception as e:
                err = f"Digest synthesis update failed: {e}"
                logger.warning(err)
                result["errors"].append(err)
        else:
            logger.info("NOTION_WEEKLY_DIGESTS_DB_ID not set — skipping digest update")

    # ---- Step 4: Generate local artefacts ----
    logger.info("=== Local Artefacts ===")
    artefact_counts = generate_local_artefacts(wk.week_id, out_dir)

    # ---- Metadata ----
    meta = RunMetadata.build(
        notebook=SCRIPT_NAME, week_id=wk.week_id,
        date_from=wk.date_from_iso, date_to=wk.date_to_iso,
        counts=artefact_counts,
        extra={
            "llm_usage": llm.usage_summary(),
            "write_enabled": write_enabled,
            "pipeline_skipped": args.skip_pipeline,
            "sub_pipeline_results": {
                k: v.get("ok", False) for k, v in sub_results.items()
            },
        },
    )
    meta.save(out_dir / "run_metadata.json")

    result["summary"] = {
        "synthesis_generated": bool(synthesis.get("executive_summary")),
        **artefact_counts,
    }

    # Determine overall success: only OK if no critical sub-pipeline failures
    # and no digest write errors.
    has_errors = bool(result["errors"])
    if has_errors:
        logger.warning(
            "052 completed with %d error(s): %s",
            len(result["errors"]), result["errors"],
        )
    result["ok"] = not has_errors
    logger.info("=== Done: %s → %s (ok=%s) ===", wk.week_id, out_dir, result["ok"])
    logger.info(llm.usage_summary())
    return result


if __name__ == "__main__":
    r = main()
    raise SystemExit(0 if r["ok"] else 1)
