#!/usr/bin/env python
"""116 Hypothesis Dedup & Rank — deduplicate and rank papers per hypothesis.

For each hypothesis, deduplicates raw papers, scores relevance, and selects
a ranked subset within the configured paper count bounds.

Usage::

    python -m src.scripts.116_hyp_dedup_rank --run-id <id>
    python -m src.scripts.116_hyp_dedup_rank --run-id <id> --min-papers 80 --max-papers 120
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
from src.lit_review.deep_lit import load_hypotheses_for_deep_lit, hyp_lit_dir

logger = logging.getLogger("116_hyp_dedup_rank")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.116_hyp_dedup_rank",
        description="116 Hypothesis Dedup & Rank — deduplicate and rank papers",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--min-papers", type=int, default=100,
                   help="Minimum papers to retain per hypothesis (default: 100)")
    p.add_argument("--max-papers", type=int, default=150,
                   help="Maximum papers to retain per hypothesis (default: 150)")
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

    run_dir = _DATA_DIR / args.run_id
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        return

    hypotheses = load_hypotheses_for_deep_lit(run_dir)
    if not hypotheses:
        logger.error("No focused hypotheses found in %s", args.run_id)
        return

    logger.info("=== 116 Hypothesis Dedup & Rank === run_id=%s", args.run_id)
    logger.info("Hypotheses to process: %d", len(hypotheses))
    logger.info("Paper bounds: min=%d, max=%d", args.min_papers, args.max_papers)

    if args.dry_run:
        print(f"\nDRY RUN — inputs:")
        print(f"  Hypotheses: {len(hypotheses)}")
        print(f"  Min papers: {args.min_papers}")
        print(f"  Max papers: {args.max_papers}")
        for h in hypotheses:
            hid = h.get("hypothesis_id", h.get("id", "unknown"))
            rpath = hyp_lit_dir(run_dir, hid) / "hyp_raw_papers.json"
            if rpath.exists():
                rdata = json.loads(rpath.read_text())
                n = len(rdata.get("papers", []))
                print(f"    - {hid}: {n} raw papers")
            else:
                print(f"    - {hid}: MISSING hyp_raw_papers.json")
        return

    from src.llm.claude_client import build_claude_client_from_env
    from src.lit_review.deep_lit.dedup import dedup_rank_select

    llm_client = build_claude_client_from_env()

    all_outputs = []
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)

        raw_path = h_dir / "hyp_raw_papers.json"
        if not raw_path.exists():
            logger.warning("Skipping %s — hyp_raw_papers.json not found", hid)
            continue

        logger.info("Dedup & ranking for hypothesis: %s", hid)
        raw_papers = json.loads(raw_path.read_text())

        ranked = dedup_rank_select(
            raw_papers=raw_papers,
            hypothesis=h,
            llm_client=llm_client,
            min_papers=args.min_papers,
            max_papers=args.max_papers,
        )

        out_path = h_dir / "hyp_papers_ranked.json"
        out_path.write_text(json.dumps(ranked, ensure_ascii=False, indent=2))
        n_ranked = len(ranked.get("papers", []))
        logger.info("Saved %s (%d papers)", out_path.name, n_ranked)
        all_outputs.append(f"hyp_literature/{hid}/hyp_papers_ranked.json")

    # Update manifest
    try:
        from src.lit_review.run_manifest import update_step
        update_step(run_dir, "116_hyp_dedup_rank",
                    status="completed",
                    outputs=all_outputs)
    except Exception as e:
        logger.warning("Manifest update failed: %s", e)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"116 Hypothesis Dedup & Rank — Complete")
    print(f"{'=' * 60}")
    print(f"Hypotheses processed: {len(all_outputs)}")
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)
        rpath = h_dir / "hyp_papers_ranked.json"
        if rpath.exists():
            rdata = json.loads(rpath.read_text())
            n = len(rdata.get("papers", []))
            raw_path = h_dir / "hyp_raw_papers.json"
            n_raw = 0
            if raw_path.exists():
                n_raw = len(json.loads(raw_path.read_text()).get("papers", []))
            print(f"  {hid}: {n_raw} raw -> {n} ranked")
    print(f"\nOutputs: {run_dir}")
    for o in all_outputs:
        print(f"  {o}")

    logger.info("=== 116 Done ===")


if __name__ == "__main__":
    main()
