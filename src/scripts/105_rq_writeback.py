#!/usr/bin/env python
"""105 RQ Writeback — Block 1 optional Notion write + next-run export.

Writes promoted RQs to Notion (optional) and exports rq_context.json
files for next research runs (always).

Exit codes:
  0: completed (writeback or export)
  1: fatal error

Usage::

    python -m src.scripts.105_rq_writeback --run-id <parent_run_id>
    python -m src.scripts.105_rq_writeback --run-id <parent_run_id> --export-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env
from src.question.writer import writeback_promoted, export_promoted_for_next_run

logger = logging.getLogger("105_rq_writeback")

_QF_DATA_DIR = _PROJECT_ROOT / "data" / "question_formation"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.105_rq_writeback",
        description="105 RQ Writeback — Notion write (optional) + next-run export",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--export-only", action="store_true", help="Export rq_context files only, skip Notion")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    load_env()

    out_dir = _QF_DATA_DIR / f"from_{args.run_id}"
    portfolio_path = out_dir / "rq_portfolio.json"
    candidates_path = out_dir / "rq_candidates.json"

    if not portfolio_path.exists():
        logger.error("Portfolio not found: %s. Run 103 first.", portfolio_path)
        sys.exit(1)
    if not candidates_path.exists():
        logger.error("Candidates not found: %s. Run 101 first.", candidates_path)
        sys.exit(1)

    logger.info("=== 105 RQ Writeback === parent_run=%s", args.run_id)

    # Always export promoted contexts as rq_context files
    contexts = export_promoted_for_next_run(portfolio_path, candidates_path)

    # Save individual rq_context files for next runs
    next_run_dir = out_dir / "next_runs"
    next_run_dir.mkdir(exist_ok=True)

    for i, ctx in enumerate(contexts):
        meta = ctx.get("_block1_metadata", {})
        cid = meta.get("candidate_id", "unknown")
        ctx_path = next_run_dir / f"rq_context_{cid}.json"
        ctx_path.write_text(json.dumps(ctx, ensure_ascii=False, indent=2))

    # Notion writeback (unless --export-only)
    if args.export_only:
        result_status = "export_only"
        notion_entries = []
    else:
        result = writeback_promoted(portfolio_path, candidates_path)
        result_status = result.status
        notion_entries = result.entries

        # Save writeback result
        wb_path = out_dir / "rq_writeback.json"
        wb_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"105 RQ Writeback — COMPLETED ({result_status})")
    print(f"{'=' * 60}")
    print(f"Promoted candidates: {len(contexts)}\n")

    for i, ctx in enumerate(contexts, 1):
        meta = ctx.get("_block1_metadata", {})
        cid = meta.get("candidate_id", "")
        title = ctx.get("title", "")[:50]
        role = meta.get("portfolio_role", "")

        # Notion status
        notion_status = "skipped"
        for e in notion_entries:
            if e.candidate_id == cid:
                notion_status = e.status
                break

        print(f"  {i}. {title}")
        print(f"     ID: {cid} | Role: {role} | Notion: {notion_status}")
        print(f"     Next-run: next_runs/rq_context_{cid}.json")
        print()

    print(f"Outputs: {out_dir}")
    print(f"  next_runs/  ({len(contexts)} rq_context files)")
    if not args.export_only:
        print(f"  rq_writeback.json")

    print(f"\nTo start a new run with promoted RQ:")
    if contexts:
        meta = contexts[0].get("_block1_metadata", {})
        cid = meta.get("candidate_id", "xxx")
        print(f"  cp {next_run_dir}/rq_context_{cid}.json data/lit_review/<new_run_id>/rq_context.json")
        print(f"  python -m src.scripts.079_rq_paper_matcher --run-id <new_run_id>")

    sys.exit(0)


if __name__ == "__main__":
    main()
