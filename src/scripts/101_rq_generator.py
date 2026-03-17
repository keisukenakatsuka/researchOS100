#!/usr/bin/env python
"""101 RQ Generator — Block 1 entry point.

Generates Research Question candidates from a completed research run.

Exit codes:
  0: candidates generated
  1: fatal error

Usage::

    python -m src.scripts.101_rq_generator --run-id <parent_run_id>
    python -m src.scripts.101_rq_generator --run-id <parent_run_id> --dry-run
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
from src.question.generator import generate_rq_candidates, _extract_seeds, _deduplicate_seeds, _render_markdown

logger = logging.getLogger("101_rq_generator")

_LIT_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"
_QF_DATA_DIR = _PROJECT_ROOT / "data" / "question_formation"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.101_rq_generator",
        description="101 RQ Generator — generate RQ candidates from research run",
    )
    p.add_argument("--run-id", type=str, required=True, help="Parent research run ID")
    p.add_argument("--dry-run", action="store_true", help="Show seeds without generating")
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

    run_dir = _LIT_DATA_DIR / args.run_id
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    logger.info("=== 101 RQ Generator === parent_run=%s", args.run_id)

    if args.dry_run:
        seeds, ctx = _extract_seeds(run_dir)
        seeds = _deduplicate_seeds(seeds)
        rq_title = ctx.get("rq_context", {}).get("title", "(unknown)")
        print(f"\nDRY RUN — RQ seeds from run {args.run_id}")
        print(f"Parent RQ: {rq_title}")
        print(f"\nSeeds extracted: {len(seeds)}")
        by_type = {}
        for s in seeds:
            by_type.setdefault(s.source_type, []).append(s)
        for stype, items in by_type.items():
            print(f"\n  [{stype}] ({len(items)} seeds)")
            for s in items:
                print(f"    - {s.description[:80]}")
        print(f"\nWould generate: rq_candidates.json + rq_candidates.md")
        sys.exit(0)

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = generate_rq_candidates(run_dir, llm_client=llm_client)

    if result.status == "generated":
        # Save outputs
        out_dir = _QF_DATA_DIR / f"from_{args.run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "rq_candidates.json"
        json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

        md_text = _render_markdown(result)
        md_path = out_dir / "rq_candidates.md"
        md_path.write_text(md_text)

        print(f"\n{'=' * 60}")
        print(f"101 RQ Generator — COMPLETED")
        print(f"{'=' * 60}")
        print(f"Parent RQ: {result.parent_rq_title[:60]}...")
        print(f"Seeds extracted: {result.seeds_extracted}")
        print(f"Candidates generated: {len(result.candidates)}")
        print()
        for i, c in enumerate(result.candidates, 1):
            print(f"  {i}. [{c.source_type}] {c.title}")
            print(f"     {c.question[:80]}...")
        print(f"\nOutputs: {out_dir}")
        print(f"  rq_candidates.json")
        print(f"  rq_candidates.md")
        sys.exit(0)
    else:
        print(f"Error: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
