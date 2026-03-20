#!/usr/bin/env python
"""115 Hypothesis Mass Retrieval — retrieve papers for each hypothesis.

For each hypothesis, loads expanded queries and retrieves candidate papers
from academic search APIs.

Usage::

    python -m src.scripts.115_hyp_mass_retrieval --run-id <id>
    python -m src.scripts.115_hyp_mass_retrieval --run-id <id> --max-per-query 50
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

logger = logging.getLogger("115_hyp_mass_retrieval")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.115_hyp_mass_retrieval",
        description="115 Hypothesis Mass Retrieval — retrieve papers per hypothesis",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--max-per-query", type=int, default=100,
                   help="Maximum papers to retrieve per query (default: 100)")
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

    logger.info("=== 115 Hypothesis Mass Retrieval === run_id=%s", args.run_id)
    logger.info("Hypotheses to process: %d", len(hypotheses))
    logger.info("Max per query: %d", args.max_per_query)

    if args.dry_run:
        print(f"\nDRY RUN — inputs:")
        print(f"  Hypotheses: {len(hypotheses)}")
        print(f"  Max per query: {args.max_per_query}")
        for h in hypotheses:
            hid = h.get("hypothesis_id", h.get("id", "unknown"))
            qpath = hyp_lit_dir(run_dir, hid) / "hyp_queries.json"
            status = "ready" if qpath.exists() else "MISSING hyp_queries.json"
            print(f"    - {hid}: {status}")
        return

    from src.lit_review.deep_lit.retrieval import retrieve_papers

    all_outputs = []
    total_papers = 0
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)

        queries_path = h_dir / "hyp_queries.json"
        if not queries_path.exists():
            logger.warning("Skipping %s — hyp_queries.json not found", hid)
            continue

        logger.info("Retrieving papers for hypothesis: %s", hid)
        queries_data = json.loads(queries_path.read_text())

        raw_papers = retrieve_papers(
            queries=queries_data,
            max_per_query=args.max_per_query,
        )

        out_path = h_dir / "hyp_raw_papers.json"
        out_path.write_text(json.dumps(raw_papers, ensure_ascii=False, indent=2))
        n_papers = len(raw_papers.get("papers", []))
        total_papers += n_papers
        logger.info("Saved %s (%d papers)", out_path.name, n_papers)
        all_outputs.append(f"hyp_literature/{hid}/hyp_raw_papers.json")

    # Update manifest
    try:
        from src.lit_review.run_manifest import update_step
        update_step(run_dir, "115_hyp_mass_retrieval",
                    status="completed",
                    outputs=all_outputs)
    except Exception as e:
        logger.warning("Manifest update failed: %s", e)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"115 Hypothesis Mass Retrieval — Complete")
    print(f"{'=' * 60}")
    print(f"Hypotheses processed: {len(all_outputs)}")
    print(f"Total papers retrieved: {total_papers}")
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)
        rpath = h_dir / "hyp_raw_papers.json"
        if rpath.exists():
            rdata = json.loads(rpath.read_text())
            n = len(rdata.get("papers", []))
            print(f"  {hid}: {n} papers")
    print(f"\nOutputs: {run_dir}")
    for o in all_outputs:
        print(f"  {o}")

    logger.info("=== 115 Done ===")


if __name__ == "__main__":
    main()
