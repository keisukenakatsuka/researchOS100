#!/usr/bin/env python
"""085 Cross-RQ Comparison — compare multiple RQ lit reviews.

Compares theoretical streams, findings, blind spots, and research
opportunities across multiple RQ runs.

Usage::

    python -m src.scripts.085_cross_rq_comparison --run-ids id1,id2
    python -m src.scripts.085_cross_rq_comparison --run-ids id1,id2,id3 --dry-run
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
from src.lit_review.comparator import compare_rqs, load_rq_summary

logger = logging.getLogger("085_cross_rq_comparison")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.085_cross_rq_comparison",
        description="085 Cross-RQ Comparison",
    )
    p.add_argument("--run-ids", type=str, required=True,
                    help="Comma-separated run IDs to compare")
    p.add_argument("--dry-run", action="store_true")
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

    # Validate
    for rd in run_dirs:
        if not rd.exists():
            logger.error("Run directory not found: %s", rd)
            return
        if not (rd / "lit_review.json").exists():
            logger.error("lit_review.json not found in %s (run 082 first)", rd.name)
            return

    logger.info("=== 085 Cross-RQ Comparison === %d RQs", len(run_ids))

    if args.dry_run:
        print(f"\nDRY RUN — would compare {len(run_ids)} RQs:")
        for rd in run_dirs:
            s = load_rq_summary(rd)
            print(f"  [{s.run_id}] {s.rq_title[:60]}")
            print(f"    Streams: {len(s.theoretical_streams)}, "
                  f"Established: {len(s.established)}, "
                  f"Emerging: {len(s.emerging)}, "
                  f"Contested: {len(s.contested)}, "
                  f"Blindspots: {len(s.blindspots)}")
        return

    # Run comparison
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = compare_rqs(run_dirs, llm_client=llm_client)

    # Save
    out_dir = _DATA_DIR / "cross_rq" / result.comparison_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "cross_rq_comparison.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved cross_rq_comparison.json")

    md_path = out_dir / "cross_rq_comparison.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved cross_rq_comparison.md")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"085 Cross-RQ Comparison — Complete")
    print(f"{'=' * 60}")
    print(f"Comparison ID: {result.comparison_id}")
    print(f"RQs compared: {len(result.rqs)}")
    for rq in result.rqs:
        print(f"  - {rq.get('title', '')[:60]}")
    print(f"")
    print(f"Shared theoretical streams: {len(result.shared_theoretical_streams)}")
    for s in result.shared_theoretical_streams:
        print(f"  - {s.get('stream', '')}")
    print(f"Unique theoretical streams: {len(result.unique_theoretical_streams)}")
    print(f"Shared findings: {len(result.shared_findings)}")
    print(f"Divergent findings: {len(result.divergent_findings)}")
    print(f"Shared blindspots: {len(result.shared_blindspots)}")
    print(f"Unique blindspots: {len(result.unique_blindspots)}")
    print(f"Cross-RQ opportunities: {len(result.cross_rq_opportunities)}")
    print(f"")

    if result.executive_summary:
        print(f"Executive Summary:")
        print(f"  {result.executive_summary[:300]}...")

    print(f"\nOutputs: {out_dir}")
    print(f"  cross_rq_comparison.json")
    print(f"  cross_rq_comparison.md")

    logger.info("=== 085 Done ===")


if __name__ == "__main__":
    main()
