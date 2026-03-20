#!/usr/bin/env python
"""114 Hypothesis Query Expansion — generate search queries per hypothesis.

For each focused hypothesis, expands the hypothesis statement and RQ context
into a set of academic search queries for comprehensive literature retrieval.

Usage::

    python -m src.scripts.114_hyp_query_expansion --run-id <id>
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

logger = logging.getLogger("114_hyp_query_expansion")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.114_hyp_query_expansion",
        description="114 Hypothesis Query Expansion — generate search queries per hypothesis",
    )
    p.add_argument("--run-id", type=str, required=True)
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

    logger.info("=== 114 Hypothesis Query Expansion === run_id=%s", args.run_id)
    logger.info("Hypotheses to process: %d", len(hypotheses))

    # Load RQ context
    rq_context_path = run_dir / "rq_context.json"
    rq_title = ""
    if rq_context_path.exists():
        rq_context = json.loads(rq_context_path.read_text())
        rq_title = rq_context.get("title", "")
        logger.info("RQ title: %s", rq_title)
    else:
        logger.warning("rq_context.json not found, proceeding without RQ title")

    if args.dry_run:
        print(f"\nDRY RUN — inputs:")
        print(f"  RQ: {rq_title or '?'}")
        print(f"  Hypotheses: {len(hypotheses)}")
        for h in hypotheses:
            hid = h.get("hypothesis_id", "?")
            stmt = h.get("hypothesis_statement", h.get("statement", ""))[:60]
            print(f"    - {hid}: {stmt}")
        return

    from src.llm.claude_client import build_claude_client_from_env
    from src.lit_review.deep_lit.query import expand_queries

    llm_client = build_claude_client_from_env()

    all_outputs = []
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        logger.info("Expanding queries for hypothesis: %s", hid)

        out_dir = hyp_lit_dir(run_dir, hid)
        out_dir.mkdir(parents=True, exist_ok=True)

        queries = expand_queries(
            hypothesis=h,
            rq_title=rq_title,
            llm_client=llm_client,
        )

        out_path = out_dir / "hyp_queries.json"
        out_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2))
        logger.info("Saved %s (%d queries)", out_path.name, len(queries.get("queries", [])))
        all_outputs.append(f"hyp_literature/{hid}/hyp_queries.json")

    # Update manifest
    try:
        from src.lit_review.run_manifest import update_step
        update_step(run_dir, "114_hyp_query_expansion",
                    status="completed",
                    outputs=all_outputs)
    except Exception as e:
        logger.warning("Manifest update failed: %s", e)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"114 Hypothesis Query Expansion — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {rq_title}")
    print(f"Hypotheses processed: {len(hypotheses)}")
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        out_dir = hyp_lit_dir(run_dir, hid)
        qpath = out_dir / "hyp_queries.json"
        if qpath.exists():
            qdata = json.loads(qpath.read_text())
            n = len(qdata.get("queries", []))
            print(f"  {hid}: {n} queries")
    print(f"\nOutputs: {run_dir}")
    for o in all_outputs:
        print(f"  {o}")

    logger.info("=== 114 Done ===")


if __name__ == "__main__":
    main()
