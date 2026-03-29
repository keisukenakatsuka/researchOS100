#!/usr/bin/env python
"""088 Assumption Analyzer — Block 5 entry point.

Analyzes hypotheses to surface implicit assumptions across
theoretical, identification, and data categories.

Usage::

    python -m src.scripts.088_assumption_analyzer --run-id <id>
    python -m src.scripts.088_assumption_analyzer --run-id <id> --writeback
    python -m src.scripts.088_assumption_analyzer --run-id <id> --dry-run
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
from src.lit_review.assumption import (
    analyze_assumptions_pipeline,
    collect_inputs,
    write_assumptions,
)

logger = logging.getLogger("088_assumption_analyzer")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.088_assumption_analyzer",
        description="088 Assumption Analyzer — surface implicit assumptions in hypotheses",
    )
    p.add_argument("--run-id", type=str, required=True, help="Run ID (with hypotheses.json)")
    p.add_argument("--writeback", action="store_true", help="Write critical+significant assumptions to Claims DB")
    p.add_argument("--dry-run", action="store_true", help="Show inputs without analyzing")
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
    if not (run_dir / "hypotheses.json").exists():
        logger.error("hypotheses.json not found in %s (run 087 first)", args.run_id)
        return

    logger.info("=== 088 Assumption Analyzer === run_id=%s", args.run_id)

    if args.dry_run:
        inputs = collect_inputs(run_dir)
        print(f"\nDRY RUN — inputs:")
        print(f"  RQ: {inputs.get('rq_title', '?')}")
        print(f"  Hypotheses: {len(inputs['hypotheses'])}")
        print(f"  Theoretical streams: {len(inputs['theoretical_streams'])}")
        print(f"  Methods: {len(inputs['methods'])}")
        for i, h in enumerate(inputs["hypotheses"]):
            print(f"  H{i}: {h.get('hypothesis_statement', '')[:65]}")
        return

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = analyze_assumptions_pipeline(run_dir, llm_client=llm_client)

    # Save
    json_path = run_dir / "assumptions.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved assumptions.json")

    md_path = run_dir / "assumptions.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved assumptions.md")

    # Summary
    all_a = result.all_assumptions()
    by_cat = Counter(a.category for a in all_a)
    by_test = Counter(a.testability for a in all_a)
    by_vuln = Counter(a.vulnerability for a in all_a)

    print(f"\n{'=' * 60}")
    print(f"088 Assumption Analyzer — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {result.rq_title}")
    print(f"Hypotheses analyzed: {result.hypotheses_analyzed}")
    print(f"Total assumptions: {result.total_assumptions}")
    print(f"")
    print(f"By category:      {dict(by_cat)}")
    print(f"By testability:   {dict(by_test)}")
    print(f"By vulnerability: {dict(by_vuln)}")
    print(f"")

    # Vulnerability map
    print(f"Vulnerability Map:")
    for ha in result.hypothesis_assumptions:
        print(f"  [{ha.overall_vulnerability:6s}] {ha.hypothesis_statement[:60]}")
        if ha.weakest_assumption:
            print(f"          weakest: {ha.weakest_assumption[:60]}")
    print(f"")

    # Writeback
    if args.writeback:
        if not is_notion_writeback_enabled():
            logger.error("ENABLE_NOTION_WRITEBACK is not 'true'")
            print(f"Writeback skipped: set ENABLE_NOTION_WRITEBACK=true")
        elif not all_a:
            print(f"No assumptions to write")
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
            crit_sig = sum(1 for a in all_a if a.vulnerability in ("critical", "significant"))
            print(f"Writing {crit_sig} assumptions (critical+significant) to Claims DB...")
            wb = write_assumptions(all_a, claims_repo=claims_repo, now_iso=now_iso)
            print(f"  Written: {len(wb['page_ids'])}")
            print(f"  Skipped (minor): {wb['skipped_minor']}")
            if wb["errors"]:
                print(f"  Errors: {len(wb['errors'])}")
    elif all_a:
        crit_sig = sum(1 for a in all_a if a.vulnerability in ("critical", "significant"))
        print(f"To write {crit_sig} assumptions to Claims DB, re-run with --writeback")

    print(f"\nOutputs: {run_dir}")
    print(f"  assumptions.json")
    print(f"  assumptions.md")

    logger.info("=== 088 Done ===")


if __name__ == "__main__":
    main()
