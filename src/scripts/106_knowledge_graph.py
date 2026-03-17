#!/usr/bin/env python
"""106 Knowledge Graph Builder — interactive graph visualization.

Generates interactive HTML graphs from research run artifacts.
No LLM — purely deterministic graph construction + rendering.

Exit codes:
  0: graphs generated
  1: fatal error

Usage::

    python -m src.scripts.106_knowledge_graph --run-id <id>
    python -m src.scripts.106_knowledge_graph --run-id <id> --layer landscape
    python -m src.scripts.106_knowledge_graph --run-id <id> --layer evidence
    python -m src.scripts.106_knowledge_graph --run-id <id> --dry-run
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

from src.graph.builder import build_all_graphs

logger = logging.getLogger("106_knowledge_graph")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.106_knowledge_graph",
        description="106 Knowledge Graph Builder — interactive graph visualization",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--layer", type=str, choices=["landscape", "evidence", "combined", "all"], default="all",
                    help="Which graph layer to build (default: all)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    run_dir = _DATA_DIR / args.run_id
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    logger.info("=== 106 Knowledge Graph Builder === run_id=%s", args.run_id)

    if args.dry_run:
        print(f"\nDRY RUN — available data:")
        for fname in ["landscape.json", "evidence.json", "lit_review.json", "hypotheses.json"]:
            exists = (run_dir / fname).exists()
            icon = "\u2713" if exists else "\u2717"
            print(f"  {icon} {fname}")
        print(f"\nWould generate: knowledge_graph.html, evidence_chain.html, graph_data.json")
        sys.exit(0)

    layers = ["landscape", "evidence", "combined"] if args.layer == "all" else [args.layer]
    result = build_all_graphs(run_dir, layers=layers)

    if result.status == "generated":
        print(f"\n{'=' * 60}")
        print(f"106 Knowledge Graph Builder — COMPLETED")
        print(f"{'=' * 60}")

        for layer in result.layers_built:
            nodes = result.node_counts.get(layer, 0)
            edges = result.edge_counts.get(layer, 0)
            print(f"\n  Layer: {layer}")
            print(f"    Nodes: {nodes}")
            print(f"    Edges: {edges}")

        print(f"\nOutputs: {run_dir}")
        for f in result.html_files:
            print(f"  {Path(f).name}")
        sys.exit(0)
    else:
        print(f"Error: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
