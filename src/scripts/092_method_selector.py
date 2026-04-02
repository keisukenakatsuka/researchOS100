#!/usr/bin/env python
"""092 Method Selector — compare and recommend research methods.

Compares 2-3 candidate methods per hypothesis and recommends
primary + secondary (robustness check) methods.

Usage::

    python -m src.scripts.092_method_selector --run-id <id>
    python -m src.scripts.092_method_selector --run-id <id> --max-designs 3
    python -m src.scripts.092_method_selector --run-id <id> --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env
from src.lit_review.method_sel import select_methods, collect_inputs

logger = logging.getLogger("092_method_selector")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.092_method_selector",
        description="092 Method Selector — compare and recommend research methods",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--max-designs", type=int, default=5)
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
    if not (run_dir / "validation_designs.json").exists():
        logger.error("validation_designs.json not found (run 090 first)")
        return

    logger.info("=== 092 Method Selector === run_id=%s", args.run_id)

    if args.dry_run:
        inputs = collect_inputs(run_dir)
        designs = inputs["validation_designs"][:args.max_designs]
        print(f"\nDRY RUN — would compare methods for {len(designs)} designs:")
        for d in designs:
            print(f"  [{d.get('identification_strategy', '?')}] {d.get('hypothesis_statement', '')[:55]}")
        return

    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = select_methods(run_dir, llm_client=llm_client, max_designs=args.max_designs)

    # Save
    json_path = run_dir / "method_selection.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved method_selection.json")

    md_path = run_dir / "method_selection.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved method_selection.md")

    # Summary
    primary_methods = Counter(s.primary_method for s in result.method_selections)
    secondary_methods = Counter(s.secondary_method for s in result.method_selections)

    print(f"\n{'=' * 60}")
    print(f"092 Method Selector — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {result.rq_title}")
    print(f"Selections: {result.selections_generated}")
    print(f"")
    print(f"Primary methods: {dict(primary_methods)}")
    print(f"Secondary methods: {dict(secondary_methods)}")
    print(f"")

    for i, sel in enumerate(result.method_selections, 1):
        n_candidates = len(sel.candidates)
        print(f"  {i}. {sel.hypothesis_statement[:50]}")
        print(f"     Primary: {sel.primary_method} | Secondary: {sel.secondary_method} | "
              f"Conf: {sel.overall_confidence} | Candidates: {n_candidates}")

    print(f"\nOutputs: {run_dir}")
    print(f"  method_selection.json")
    print(f"  method_selection.md")

    logger.info("=== 092 Done ===")


if __name__ == "__main__":
    main()
