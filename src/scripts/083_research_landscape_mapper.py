#!/usr/bin/env python
"""083 Research Landscape Mapper — Block 3 landscape analysis.

Reads lit_review.json from 082 and generates:
- Research landscape analysis (theoretical/methodological/data/context)
- RQ Knowledge Graph structure
- Hotspots, blindspots, and research opportunities

Usage::

    python -m src.scripts.083_research_landscape_mapper --run-id <id>
    python -m src.scripts.083_research_landscape_mapper --input data/lit_review/.../lit_review.json
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
from src.lit_review.landscape import build_research_landscape

logger = logging.getLogger("083_research_landscape_mapper")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


# ------------------------------------------------------------------
# Input / Output
# ------------------------------------------------------------------

def load_lit_review(
    run_id: str | None = None,
    input_path: str | None = None,
) -> tuple[Path, dict]:
    if run_id:
        run_dir = _DATA_DIR / run_id
    elif input_path:
        run_dir = Path(input_path).parent
    else:
        raise ValueError("Either --run-id or --input is required")

    lit_review_path = run_dir / "lit_review.json"
    if input_path:
        lit_review_path = Path(input_path)
    if not lit_review_path.exists():
        raise FileNotFoundError(f"lit_review.json not found: {lit_review_path}")

    data = json.loads(lit_review_path.read_text())
    logger.info("Loaded lit_review.json: %d papers, %d evidence items",
                len(data.get("papers", [])), len(data.get("evidence", [])))
    return run_dir, data


def save_outputs(run_dir: Path, result):
    json_path = run_dir / "landscape.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved landscape.json")

    md_path = run_dir / "landscape.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved landscape.md")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.083_research_landscape_mapper",
        description="083 Research Landscape Mapper",
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--run-id", type=str, help="Run ID from 079–082")
    input_group.add_argument("--input", type=str, help="Path to lit_review.json")

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

    run_dir, lit_review = load_lit_review(run_id=args.run_id, input_path=args.input)
    logger.info("=== 083 Research Landscape Mapper === run_dir=%s", run_dir)

    rq_title = lit_review.get("rq_context", {}).get("title", "?")
    logger.info("RQ: %s", rq_title)

    if args.dry_run:
        dims = lit_review.get("research_dimensions", {})
        print(f"\nDRY RUN — would analyze landscape for:")
        print(f"  RQ: {rq_title}")
        print(f"  Papers: {len(lit_review.get('papers', []))}")
        print(f"  Theoretical streams: {len(lit_review.get('theoretical_streams', []))}")
        for cat, items in dims.items():
            print(f"  {cat}: {len(items)} items")
        return

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = build_research_landscape(lit_review, llm_client=llm_client)

    save_outputs(run_dir, result)

    # Print summary
    kg = result.knowledge_graph
    kg_summary = kg.get("summary", {})

    print(f"\n{'=' * 60}")
    print(f"083 Research Landscape Mapper — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {result.rq_title}")
    print(f"")
    print(f"Knowledge Graph: {kg_summary.get('total_nodes', 0)} nodes, {kg_summary.get('total_edges', 0)} edges")
    for ntype, count in sorted(kg_summary.get("node_types", {}).items()):
        print(f"  {ntype}: {count}")
    print(f"")

    print(f"Theoretical landscape: {len(result.theoretical_landscape)} theories")
    for name in list(result.theoretical_landscape.keys())[:5]:
        print(f"  - {name}")

    print(f"\nHotspots: {len(result.hotspots)}")
    for h in result.hotspots:
        print(f"  [{h.get('strength', '')}] {h.get('area', '')}")

    print(f"\nBlindspots: {len(result.blindspots)}")
    for b in result.blindspots:
        print(f"  [{b.get('severity', '')}] {b.get('area', '')}")

    print(f"\nResearch Opportunities: {len(result.research_opportunities)}")
    for opp in result.research_opportunities:
        print(f"  - {opp.get('theme', '')}")

    print(f"\nOutputs: {run_dir}")
    print(f"  landscape.json")
    print(f"  landscape.md")

    logger.info("=== 083 Done ===")


if __name__ == "__main__":
    main()
