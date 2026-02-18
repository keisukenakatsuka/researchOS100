#!/usr/bin/env python
# src/scripts/054_values_scale_setup.py
"""Values Scale Setup — generate, refine, and persist 12 value domain definitions.

Pipeline:
1. Generate deterministic seed data for all 12 value domains
2. (Optional) Run LLM refinement via OpenAI router  (--llm)
3. (Optional) Show structured diff of LLM suggestions (--diff, requires --llm)
4. (Optional) Apply LLM suggestions to domains        (--apply, requires --llm)
5. Write JSON output to outputs/weekly/{week_id}/054_values_scale_setup/
6. (Optional) Upsert to ROS_Values_Codex in Notion    (--write)

Two Notion databases:
- ROS_Values_Codex  (env: NOTION_ROS_Values_Codex_ID) — master definitions
- ROS_Alignment_Log (env: NOTION_ROS_Alignment_Log_ID) — reflections (append-only)

Value Evaluation Scale (Alignment Log only):
- importance_score (1–5): How important is this domain?
- alignment_score  (1–5): How consistently am I living this value?
- gap_score        (computed): importance − alignment
- significant_gap  (computed): True when gap >= 2

Codex layer is definitions only — NO numeric scores.

Voice Language:
- Default language for voice I/O is Japanese (ja).
- Override with --lang en for English.
- Also configurable via VOICE_LANGUAGE env var.

Usage::

    # Seed only (no LLM, no Notion)
    python -m src.scripts.054_values_scale_setup --run

    # Seed + LLM refinement (show diff, don't apply)
    python -m src.scripts.054_values_scale_setup --run --llm --diff

    # Seed + LLM + apply suggestions + write to Notion
    python -m src.scripts.054_values_scale_setup --run --llm --apply --write

    # Dry-run: validate schemas, show what would happen
    python -m src.scripts.054_values_scale_setup --run --dry-run --write

    # Override quarter
    python -m src.scripts.054_values_scale_setup --run --quarter 2026-Q2

    # Override voice language (default: ja)
    python -m src.scripts.054_values_scale_setup --run --lang en
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
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
from src.notion.client import (
    NotionDataSourceResolver,
    build_notion_client_from_env,
)
from src.notion.truncation import TruncationTracker
from src.notion.values_repo import ValuesCodexRepo
from src.values.generator import generate_value_record
from src.values.schema import (
    ValueRecord,
    _refinement_result_to_dict,
)
from src.values.voice import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    VoiceConfig,
)

logger = logging.getLogger("054_values_scale_setup")

SCRIPT_NAME = "054_values_scale_setup"
JST = ZoneInfo("Asia/Tokyo")


# ================================================================
# Quarter calculation
# ================================================================

def _current_quarter(now: datetime) -> str:
    """Derive quarter string from datetime, e.g. '2026-Q1'."""
    q = (now.month - 1) // 3 + 1
    return f"{now.year}-Q{q}"


# ================================================================
# Main pipeline
# ================================================================

def run_pipeline(
    *,
    quarter: str | None = None,
    write_notion: bool = False,
    enable_llm: bool = False,
    show_diff: bool = False,
    apply_suggestions: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    language: str | None = None,
) -> dict:
    """Execute the values scale setup pipeline.

    Parameters
    ----------
    quarter : str or None
        Review quarter (e.g. "2026-Q1"). Auto-detected if None.
    write_notion : bool
        If True, upsert rows to Notion ROS_Values_Codex.
    enable_llm : bool
        If True, run LLM refinement via OpenAI router.
    show_diff : bool
        If True, print the structured diff to stdout.
    apply_suggestions : bool
        If True, apply LLM suggestions to the record before writing.
        Requires --llm to have any effect.
    dry_run : bool
        If True, do everything except Notion writes.
    verbose : bool
        Enable debug logging.
    language : str or None
        Voice I/O language (ISO 639-1). Defaults to "ja" (Japanese).
        Also reads from VOICE_LANGUAGE env var if not specified.

    Returns
    -------
    dict
        Summary with counts and output paths.
    """
    load_env()
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    now_jst = datetime.now(tz=JST)
    wk = get_iso_week_context(tz=JST)

    if quarter is None:
        quarter = _current_quarter(now_jst)

    # ---- Resolve voice language (CLI > env > default=ja) ----
    if language is None:
        language = os.getenv("VOICE_LANGUAGE", "").strip() or DEFAULT_LANGUAGE
    voice_config = VoiceConfig(language=language)

    logger.info(
        "Starting %s for quarter=%s week=%s (llm=%s, apply=%s, write=%s, dry_run=%s, lang=%s)",
        SCRIPT_NAME, quarter, wk.week_id, enable_llm, apply_suggestions,
        write_notion, dry_run, voice_config.language,
    )

    # ---- 1. Generate deterministic seed data ----
    record = generate_value_record(
        review_quarter=quarter,
        version="2.0",
        notes=f"Initial seed generated {now_jst.date().isoformat()}",
    )
    logger.info("Generated %d value domains (seed)", len(record.domains))

    # ---- 2. Optional LLM refinement ----
    refinement_results = []
    if enable_llm:
        refinement_results = _run_refinement(record, apply=apply_suggestions)
        if apply_suggestions and refinement_results:
            record = _apply_refinements(record, refinement_results)
            logger.info("Applied LLM suggestions — source=Hybrid for modified domains")

    # ---- 3. Build output directory ----
    out_dir = get_output_dir(SCRIPT_NAME, wk.week_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 4. Write JSON output ----
    record_dict = record.to_dict()
    record_dict["voice_config"] = {
        "language": voice_config.language,
        "language_display": voice_config.display_name,
    }
    values_path = out_dir / "values.json"
    values_path.write_text(json.dumps(record_dict, indent=2, ensure_ascii=False))
    logger.info("Wrote values.json to %s", values_path)

    # ---- 5. Write per-domain JSON (for easy review) ----
    domains_dir = out_dir / "domains"
    domains_dir.mkdir(exist_ok=True)
    for domain_dict in record_dict["domains"]:
        domain_path = domains_dir / f"{domain_dict['domain_id']}.json"
        domain_path.write_text(json.dumps(domain_dict, indent=2, ensure_ascii=False))
    logger.info("Wrote %d per-domain JSON files", len(record.domains))

    # ---- 6. Write refinement results (if any) ----
    if refinement_results:
        refine_path = out_dir / "refinement.json"
        refine_data = [_refinement_result_to_dict(r) for r in refinement_results]
        refine_path.write_text(json.dumps(refine_data, indent=2, ensure_ascii=False))
        logger.info("Wrote refinement.json to %s", refine_path)

        if show_diff:
            from src.values.refine import format_diff
            diff_text = format_diff(refinement_results)
            print(diff_text)
            diff_path = out_dir / "diff.txt"
            diff_path.write_text(diff_text)

    # ---- 7. Write summary markdown ----
    _write_summary_markdown(out_dir, record, quarter, now_jst, refinement_results, voice_config)

    # ---- 8. Notion write-back (opt-in) ----
    notion_results = []
    if write_notion and not dry_run:
        logger.info("Notion write-back enabled — upserting to ROS_Values_Codex")
        notion_results = _write_to_notion(
            record=record,
            quarter=quarter,
            now_jst=now_jst,
        )
    elif write_notion and dry_run:
        logger.info("Dry-run mode — Notion writes skipped")
    else:
        logger.info("Notion write-back disabled (use --write to enable)")

    # ---- 9. Run metadata ----
    total_suggestions = sum(len(r.suggestions) for r in refinement_results)
    total_violations = sum(len(r.violations) for r in refinement_results)
    run_meta = RunMetadata.build(
        notebook=SCRIPT_NAME,
        week_id=wk.week_id,
        date_from=wk.date_from_iso,
        date_to=wk.date_to_iso,
        counts={
            "domains_generated": len(record.domains),
            "llm_enabled": enable_llm,
            "suggestions_accepted": total_suggestions,
            "suggestions_rejected": total_violations,
            "suggestions_applied": total_suggestions if apply_suggestions else 0,
            "notion_rows_written": len(notion_results),
            "quarter": quarter,
            "dry_run": dry_run,
            "voice_language": voice_config.language,
        },
    )
    run_meta.save(out_dir / "run_metadata.json")
    logger.info("Pipeline complete. Output: %s", out_dir)

    return {
        "quarter": quarter,
        "domains_generated": len(record.domains),
        "llm_enabled": enable_llm,
        "suggestions_accepted": total_suggestions,
        "suggestions_rejected": total_violations,
        "suggestions_applied": total_suggestions if apply_suggestions else 0,
        "notion_rows_written": len(notion_results),
        "dry_run": dry_run,
        "voice_language": voice_config.language,
        "voice_language_display": voice_config.display_name,
        "output_dir": str(out_dir),
    }


# ================================================================
# LLM refinement (routed to OpenAI)
# ================================================================

def _run_refinement(record: ValueRecord, *, apply: bool) -> list:
    """Run LLM refinement on all domains via the router."""
    from src.llm.router import build_router_from_env
    from src.values.refine import refine_all_domains

    router = build_router_from_env()
    results = refine_all_domains(record.domains, router=router)

    total = sum(len(r.suggestions) for r in results)
    logger.info(
        "LLM refinement complete: %d total suggestions across %d domains",
        total, len(results),
    )
    logger.info("Router usage: %s", router.usage_summary)
    return results


def _apply_refinements(record: ValueRecord, results: list) -> ValueRecord:
    """Apply refinement suggestions to the record, returning a new record."""
    from src.values.refine import apply_suggestions

    refined_domains = []
    for domain in record.domains:
        matching = [r for r in results if r.domain_id == domain.domain_id]
        if matching and matching[0].has_changes:
            refined_domains.append(apply_suggestions(domain, matching[0]))
        else:
            refined_domains.append(domain)

    return ValueRecord(
        version=record.version,
        review_quarter=record.review_quarter,
        domains=tuple(refined_domains),
        notes=record.notes + " | LLM refinement applied",
    )


# ================================================================
# Notion write-back
# ================================================================

def _write_to_notion(
    *,
    record: ValueRecord,
    quarter: str,
    now_jst: datetime,
) -> list[dict]:
    """Upsert all domain rows to Notion ROS_Values_Codex."""
    client = build_notion_client_from_env()
    codex_db_id = get_db_id("NOTION_ROS_Values_Codex_ID")

    resolver = NotionDataSourceResolver(client=client)
    data_source_id = resolver.resolve(codex_db_id)
    logger.info("Resolved data_source_id for ROS_Values_Codex: %s", data_source_id)

    repo = ValuesCodexRepo(
        client=client,
        database_id=codex_db_id,
        data_source_id=data_source_id,
    )
    repo.ensure_schema()

    tracker = TruncationTracker()
    now_iso = now_jst.isoformat(timespec="seconds")

    results = []
    for domain in record.domains:
        key, props = repo.build_domain_properties(
            domain=domain,
            review_quarter=quarter,
            now_iso=now_iso,
            tracker=tracker,
        )
        result = repo.upsert_domain(key=key, properties=props)
        results.append(result)
        logger.info("Upserted domain: %s (key=%s)", domain.domain_label, key)

    if tracker.had_truncations:
        logger.warning(
            "Truncated fields during Notion write: %s",
            [e["field"] for e in tracker.report()],
        )

    return results


# ================================================================
# Summary markdown
# ================================================================

def _write_summary_markdown(
    out_dir: Path,
    record: ValueRecord,
    quarter: str,
    now_jst: datetime,
    refinement_results: list,
    voice_config: VoiceConfig | None = None,
) -> None:
    """Write a human-readable summary of the run."""
    total_suggestions = sum(len(r.suggestions) for r in refinement_results)
    vc = voice_config or VoiceConfig()

    lines = [
        f"# Values Scale Setup — {quarter}",
        "",
        f"Generated: {now_jst.isoformat(timespec='seconds')}",
        f"Domains: {len(record.domains)}",
        f"Version: {record.version}",
        f"LLM Suggestions: {total_suggestions}",
        f"Voice Language: {vc.display_name} ({vc.language})",
        "",
        "## Value Evaluation Scale (Alignment Log)",
        "",
        "| Dimension | Range | Description |",
        "|-----------|-------|-------------|",
        "| Importance Score | 1–5 | How important is this domain? |",
        "| Alignment Score | 1–5 | How consistently am I living it? |",
        "| Gap Score | computed | Importance − Alignment |",
        "| Significant Gap | bool | True when gap ≥ 2 |",
        "",
        "## Domains",
        "",
    ]
    for domain in record.domains:
        lines.append(f"### {domain.domain_label}")
        lines.append(f"**Source:** {domain.source} | **Version:** {domain.version} | **Revision:** {domain.revision}")
        lines.append(f"**Definition:** {domain.value_definition}")
        lines.append(f"**Translation:** {domain.behavioral_translation}")
        lines.append("**Behaviors:**")
        for i, b in enumerate(domain.example_behaviors, 1):
            freq = f" [{b.frequency_hint}]" if b.frequency_hint else ""
            lines.append(f"  {i}. {b.description}{freq}")
        if domain.reflection_questions:
            lines.append("**Reflection Questions:**")
            for i, q in enumerate(domain.reflection_questions, 1):
                lines.append(f"  {i}. {q}")
        if domain.micro_habits:
            lines.append("**Micro Habits:**")
            for i, h in enumerate(domain.micro_habits, 1):
                lines.append(f"  {i}. {h}")
        if domain.misalignment_description:
            lines.append(f"**Misalignment:** {domain.misalignment_description}")
        if domain.change_notes:
            lines.append(f"**Change Notes:** {domain.change_notes}")
        lines.append("")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info("Wrote summary.md to %s", summary_path)


# ================================================================
# CLI
# ================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="054 Values Scale Setup — generate, refine, and persist value domains",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Actually run the pipeline (default: shows help)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        help="Enable Notion write-back to ROS_Values_Codex",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="Enable LLM refinement via OpenAI (generates suggestions)",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        default=False,
        help="Print structured diff of LLM suggestions to stdout",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Apply LLM suggestions to domains (requires --llm)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run pipeline without Notion writes (validate schemas only)",
    )
    parser.add_argument(
        "--quarter",
        type=str,
        default=None,
        help="Review quarter (e.g. 2026-Q1). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        choices=list(SUPPORTED_LANGUAGES),
        help=(
            "Voice I/O language (default: ja). "
            "Also configurable via VOICE_LANGUAGE env var."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if not args.run:
        parser.print_help()
        print("\n  Pass --run to execute the pipeline.")
        sys.exit(0)

    if args.apply and not args.llm:
        parser.error("--apply requires --llm")

    if args.diff and not args.llm:
        parser.error("--diff requires --llm")

    result = run_pipeline(
        quarter=args.quarter,
        write_notion=args.write,
        enable_llm=args.llm,
        show_diff=args.diff,
        apply_suggestions=args.apply,
        dry_run=args.dry_run,
        verbose=args.verbose,
        language=args.lang,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
