#!/usr/bin/env python
"""084 Lit Review Writeback — Write Block 3 outputs to KML.

Writes Evidence, Memos (Lit Review + Landscape), and Research Run
to the Knowledge Memory Layer (Notion DBs).

Usage::

    # Dry run (show what would be written)
    python -m src.scripts.084_lit_review_writeback --run-id <id> --dry-run

    # Actual writeback (requires ENABLE_NOTION_WRITEBACK=true)
    python -m src.scripts.084_lit_review_writeback --run-id <id> --writeback
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env
from src.lit_review.writer import preflight_check, write_to_notion

logger = logging.getLogger("084_lit_review_writeback")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.084_lit_review_writeback",
        description="084 Lit Review Writeback — write Block 3 outputs to KML",
    )
    p.add_argument("--run-id", type=str, required=True, help="Run ID to write")
    p.add_argument("--writeback", action="store_true",
                    help="Actually write to Notion (requires ENABLE_NOTION_WRITEBACK=true)")
    p.add_argument("--dry-run", action="store_true",
                    help="Show what would be written without writing")
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

    logger.info("=== 084 Lit Review Writeback === run_id=%s", args.run_id)

    # Check required files
    required = ["rq_context.json", "evidence.json"]
    for fname in required:
        if not (run_dir / fname).exists():
            logger.error("Required file missing: %s", fname)
            return

    # Preflight
    pf = preflight_check()

    if args.dry_run or not args.writeback:
        # Dry run mode
        if not args.writeback and not args.dry_run:
            print("\nNo --writeback or --dry-run specified. Showing dry run.")
            print("To write: add --writeback")
            print("To preview: add --dry-run\n")

        result = write_to_notion(run_dir, dry_run=True)

        if not pf["writeback_enabled"]:
            print(f"\nWARNING: ENABLE_NOTION_WRITEBACK is not set to 'true'")
            print(f"Set ENABLE_NOTION_WRITEBACK=true in notebooks/env.txt to enable writeback")
        if pf["missing_db_ids"]:
            print(f"\nWARNING: Missing DB IDs: {pf['missing_db_ids']}")
        return

    # Actual writeback
    if not pf["ok"]:
        if not pf["writeback_enabled"]:
            logger.error("ENABLE_NOTION_WRITEBACK is not 'true'. Set it in notebooks/env.txt")
            return
        if pf["missing_db_ids"]:
            logger.error("Missing DB IDs: %s", pf["missing_db_ids"])
            return

    logger.info("Starting writeback...")
    result = write_to_notion(run_dir, dry_run=False)

    # Print result
    print(f"\n{'=' * 60}")
    print(f"084 Lit Review Writeback — Complete")
    print(f"{'=' * 60}")
    print(f"Run ID: {args.run_id}")
    print(f"Status: {result.get('status', '?')}")
    print(f"")
    print(f"Evidence written: {result.get('evidence_written', 0)}")
    print(f"Evidence failed:  {result.get('evidence_failed', 0)}")
    claims_by_cat = result.get('claims_by_category', {})
    print(f"Claims written:   {result.get('claims_written', 0)}")
    if claims_by_cat:
        for cat, count in sorted(claims_by_cat.items()):
            print(f"  {cat}: {count}")
    print(f"Claims failed:    {result.get('claims_failed', 0)}")
    print(f"Memos written:    {result.get('memos_written', 0)}")
    print(f"Research Run:     {'OK' if result.get('run_page_id') else 'FAILED'}")

    errors = result.get("errors", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    logger.info("=== 084 Done ===")


if __name__ == "__main__":
    main()
