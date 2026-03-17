#!/usr/bin/env python
"""107 RQ Lineage Visualizer — interactive lineage tree.

Generates an interactive HTML visualization of the RQ lineage DAG.

Exit codes:
  0: graph generated
  1: fatal error

Usage::

    python -m src.scripts.107_rq_lineage_graph
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

from src.graph.lineage_viz import render_lineage

logger = logging.getLogger("107_rq_lineage_graph")

_LINEAGE_DIR = _PROJECT_ROOT / "data" / "question_formation" / "lineage"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.107_rq_lineage_graph",
        description="107 RQ Lineage Visualizer — interactive lineage tree",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    lineage_path = _LINEAGE_DIR / "rq_lineage.json"
    if not lineage_path.exists():
        logger.error("Lineage file not found: %s", lineage_path)
        logger.error("Run 104_rq_evolution_tracker first")
        sys.exit(1)

    logger.info("=== 107 RQ Lineage Visualizer ===")

    output_path = _LINEAGE_DIR / "rq_lineage_graph.html"
    summary = render_lineage(lineage_path, output_path)

    print(f"\n{'=' * 60}")
    print(f"107 RQ Lineage Visualizer — COMPLETED")
    print(f"{'=' * 60}")
    print(f"Nodes: {summary['nodes']}")
    print(f"Edges: {summary['edges']}")
    print(f"\nOutputs:")
    print(f"  {summary['html_path']}")
    print(f"  {summary['data_path']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
