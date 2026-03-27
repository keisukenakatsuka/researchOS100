#!/usr/bin/env python
"""086 Claim Canonicalization — deduplicate Claims across runs.

Groups semantically identical Claims from multiple Block 3 runs
and generates canonical Claims for the KML Claims DB.

Usage::

    python -m src.scripts.086_claim_canonicalization --run-ids id1,id2 --dry-run
    python -m src.scripts.086_claim_canonicalization --run-ids id1,id2 --writeback
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

from src.config import load_env, get_db_id, is_notion_writeback_enabled
from src.lit_review.canonicalizer import (
    canonicalize,
    collect_claims,
    write_canonical_claims,
)

logger = logging.getLogger("086_claim_canonicalization")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.086_claim_canonicalization",
        description="086 Claim Canonicalization — deduplicate Claims across runs",
    )
    p.add_argument("--run-ids", type=str, required=True,
                    help="Comma-separated run IDs to canonicalize")
    p.add_argument("--writeback", action="store_true",
                    help="Write canonical Claims to Claims DB")
    p.add_argument("--dry-run", action="store_true",
                    help="Show collected claims without canonicalizing")
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

    run_ids = [rid.strip() for rid in args.run_ids.split(",")]
    run_dirs = [_DATA_DIR / rid for rid in run_ids]

    for rd in run_dirs:
        if not rd.exists():
            logger.error("Run directory not found: %s", rd)
            return
        if not (rd / "lit_review.json").exists():
            logger.error("lit_review.json not found in %s", rd.name)
            return

    logger.info("=== 086 Claim Canonicalization === %d runs", len(run_ids))

    if args.dry_run:
        claims = collect_claims(run_dirs)
        print(f"\nDRY RUN — collected {len(claims)} claims from {len(run_ids)} runs:")
        for c in claims:
            print(f"  [{c.category:12s}] ({c.run_id[:20]}) {c.statement[:60]}")
        return

    # Run canonicalization
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = canonicalize(run_dirs, llm_client=llm_client)

    # Save outputs
    out_dir = _DATA_DIR / "canonical_claims" / result.canonicalization_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "canonicalization_result.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved canonicalization_result.json")

    md_path = out_dir / "canonicalization_result.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved canonicalization_result.md")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"086 Claim Canonicalization — Complete")
    print(f"{'=' * 60}")
    print(f"ID: {result.canonicalization_id}")
    print(f"Input: {result.total_input_claims} claims from {len(result.input_run_ids)} runs")
    print(f"Groups: {result.groups_formed}")
    print(f"Singletons: {result.singletons}")
    print(f"Canonical claims: {result.canonical_claims_total}")
    print(f"")

    from collections import Counter
    conf_counts = Counter(cc.confidence for cc in result.canonical_claims)
    print(f"Confidence: {dict(conf_counts)}")

    grouped = [cc for cc in result.canonical_claims if len(cc.member_claim_ids) >= 2]
    if grouped:
        print(f"\nGrouped claims ({len(grouped)}):")
        for cc in grouped:
            print(f"  [{cc.confidence}] {cc.canonical_statement[:65]}")
            print(f"    runs={len(cc.supporting_runs)}, members={len(cc.member_claim_ids)}")

    # Writeback
    if args.writeback:
        if not is_notion_writeback_enabled():
            logger.error("ENABLE_NOTION_WRITEBACK is not 'true'")
            print(f"\nWriteback skipped: set ENABLE_NOTION_WRITEBACK=true")
        elif not result.canonical_claims:
            print(f"\nNo canonical claims to write")
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
            print(f"\nWriting {len(result.canonical_claims)} canonical claims to Claims DB...")
            wb = write_canonical_claims(
                result.canonical_claims, claims_repo=claims_repo, now_iso=now_iso,
            )
            print(f"  Written: {len(wb['page_ids'])}")
            if wb["errors"]:
                print(f"  Errors: {len(wb['errors'])}")
    elif result.canonical_claims:
        print(f"\nTo write to Claims DB, re-run with --writeback")

    print(f"\nOutputs: {out_dir}")

    logger.info("=== 086 Done ===")


if __name__ == "__main__":
    main()
