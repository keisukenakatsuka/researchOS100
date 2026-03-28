#!/usr/bin/env python
"""087 Hypothesis Generator — Block 4 entry point.

Generates testable research hypotheses from Canonical Claims,
Open Questions, Blindspots, and Cross-RQ Opportunities.

Usage::

    python -m src.scripts.087_hypothesis_generator --run-id <id>
    python -m src.scripts.087_hypothesis_generator --run-id <id> --canon-id <canon_id>
    python -m src.scripts.087_hypothesis_generator --run-id <id> --writeback
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env, get_db_id, is_notion_writeback_enabled
from src.lit_review.hypothesis import (
    generate_hypotheses_pipeline,
    collect_inputs,
    write_hypotheses,
)

logger = logging.getLogger("087_hypothesis_generator")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def _find_latest_canon_dir() -> Path | None:
    """Find the latest canonicalization result directory."""
    canon_base = _DATA_DIR / "canonical_claims"
    if not canon_base.exists():
        return None
    dirs = sorted(canon_base.iterdir(), reverse=True)
    for d in dirs:
        if (d / "canonicalization_result.json").exists():
            return d
    return None


def _find_latest_cross_rq_dir() -> Path | None:
    """Find the latest cross-RQ comparison directory."""
    cross_base = _DATA_DIR / "cross_rq"
    if not cross_base.exists():
        return None
    dirs = sorted(cross_base.iterdir(), reverse=True)
    for d in dirs:
        if (d / "cross_rq_comparison.json").exists():
            return d
    return None


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.087_hypothesis_generator",
        description="087 Hypothesis Generator — generate testable hypotheses from Claims + Gaps",
    )
    p.add_argument("--run-id", type=str, required=True, help="Run ID (for lit_review.json + landscape.json)")
    p.add_argument("--canon-id", type=str, default=None,
                    help="Canonicalization ID (auto-detect latest if omitted)")
    p.add_argument("--cross-rq-id", type=str, default=None,
                    help="Cross-RQ comparison ID (auto-detect latest if omitted)")
    p.add_argument("--writeback", action="store_true", help="Write hypotheses to Claims DB")
    p.add_argument("--dry-run", action="store_true", help="Show inputs without generating")
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

    run_dir = _DATA_DIR / args.run_id
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        return

    # Resolve canon dir
    canon_dir = None
    if args.canon_id:
        canon_dir = _DATA_DIR / "canonical_claims" / args.canon_id
    else:
        canon_dir = _find_latest_canon_dir()
    if canon_dir:
        logger.info("Using canonical claims from: %s", canon_dir.name)

    # Resolve cross-RQ dir
    cross_rq_dir = None
    if args.cross_rq_id:
        cross_rq_dir = _DATA_DIR / "cross_rq" / args.cross_rq_id
    else:
        cross_rq_dir = _find_latest_cross_rq_dir()
    if cross_rq_dir:
        logger.info("Using cross-RQ comparison from: %s", cross_rq_dir.name)

    logger.info("=== 087 Hypothesis Generator === run_id=%s", args.run_id)

    if args.dry_run:
        inputs = collect_inputs(run_dir, canon_dir=canon_dir, cross_rq_dir=cross_rq_dir)
        print(f"\nDRY RUN — inputs for hypothesis generation:")
        print(f"  RQ: {inputs.get('rq_title', '?')}")
        print(f"  Canonical claims: {len(inputs['canonical_claims'])}")
        print(f"  Open questions: {len(inputs['open_questions'])}")
        print(f"  Blindspots: {len(inputs['blindspots'])}")
        print(f"  Cross-RQ opportunities: {len(inputs['cross_rq_opportunities'])}")
        return

    # Generate
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = generate_hypotheses_pipeline(
        run_dir, llm_client=llm_client,
        canon_dir=canon_dir, cross_rq_dir=cross_rq_dir,
    )

    # Save
    json_path = run_dir / "hypotheses.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved hypotheses.json")

    md_path = run_dir / "hypotheses.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved hypotheses.md")

    # Print summary
    by_strat = Counter(h.strategy for h in result.hypotheses)
    by_test = Counter(h.testability for h in result.hypotheses)

    print(f"\n{'=' * 60}")
    print(f"087 Hypothesis Generator — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {result.rq_title}")
    print(f"Total hypotheses: {len(result.hypotheses)}")
    print(f"")
    print(f"By strategy: {dict(by_strat)}")
    print(f"By testability: {dict(by_test)}")
    print(f"")

    for i, h in enumerate(result.hypotheses, 1):
        print(f"  H{i} [{h.strategy}, {h.testability}]: {h.hypothesis_statement[:70]}")
    print(f"")

    # Writeback
    if args.writeback:
        if not is_notion_writeback_enabled():
            logger.error("ENABLE_NOTION_WRITEBACK is not 'true'")
            print(f"Writeback skipped: set ENABLE_NOTION_WRITEBACK=true")
        elif not result.hypotheses:
            print(f"No hypotheses to write")
        else:
            from src.notion import build_notion_client_from_env, NotionDataSourceResolver
            from src.notion.claims_repo import ClaimsRepo
            from src.notion.research_schema import ENV_CLAIMS_DB_ID
            from datetime import datetime, timezone

            notion_client = build_notion_client_from_env()
            resolver = NotionDataSourceResolver(notion_client)
            cl_db_id = get_db_id(ENV_CLAIMS_DB_ID)
            cl_resolved = resolver.resolve_once(name="CLAIMS_DB", database_id=cl_db_id)
            claims_repo = ClaimsRepo(
                client=notion_client, database_id=cl_db_id,
                data_source_id=cl_resolved.data_source_id,
            )

            now_iso = datetime.now(timezone.utc).isoformat()
            print(f"Writing {len(result.hypotheses)} hypotheses to Claims DB...")
            wb = write_hypotheses(result.hypotheses, claims_repo=claims_repo, now_iso=now_iso)
            print(f"  Written: {len(wb['page_ids'])}")
            if wb["errors"]:
                print(f"  Errors: {len(wb['errors'])}")
    elif result.hypotheses:
        print(f"To write to Claims DB, re-run with --writeback")

    print(f"\nOutputs: {run_dir}")
    print(f"  hypotheses.json")
    print(f"  hypotheses.md")

    logger.info("=== 087 Done ===")


if __name__ == "__main__":
    main()
