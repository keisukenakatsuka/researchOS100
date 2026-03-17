#!/usr/bin/env python
"""104 RQ Evolution Tracker — Block 1 lineage tracking.

Updates the RQ lineage DAG with parent → child edges from a prioritized portfolio.

Exit codes:
  0: lineage updated
  1: fatal error

Usage::

    python -m src.scripts.104_rq_evolution_tracker --run-id <parent_run_id>
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
from src.question.tracker import update_lineage

logger = logging.getLogger("104_rq_evolution_tracker")

_LIT_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"
_QF_DATA_DIR = _PROJECT_ROOT / "data" / "question_formation"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.104_rq_evolution_tracker",
        description="104 RQ Evolution Tracker — update lineage DAG",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    load_env()

    out_dir = _QF_DATA_DIR / f"from_{args.run_id}"
    portfolio_path = out_dir / "rq_portfolio.json"
    rq_context_path = _LIT_DATA_DIR / args.run_id / "rq_context.json"

    if not portfolio_path.exists():
        logger.error("Portfolio not found: %s. Run 103 first.", portfolio_path)
        sys.exit(1)
    if not rq_context_path.exists():
        logger.error("rq_context.json not found: %s", rq_context_path)
        sys.exit(1)

    logger.info("=== 104 RQ Evolution Tracker === parent_run=%s", args.run_id)

    lineage_dir = _QF_DATA_DIR / "lineage"
    result = update_lineage(portfolio_path, rq_context_path, lineage_dir=lineage_dir)

    if result.status == "generated":
        print(f"\n{'=' * 60}")
        print(f"104 RQ Evolution Tracker — COMPLETED")
        print(f"{'=' * 60}")
        print(f"Nodes added: {result.nodes_added} (total: {result.total_nodes})")
        print(f"Edges added: {result.edges_added} (total: {result.total_edges})")
        print(f"\nLineage: {lineage_dir}")
        print(f"  rq_lineage.json")
        print(f"  rq_lineage.md")
        sys.exit(0)
    else:
        print(f"Error: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
