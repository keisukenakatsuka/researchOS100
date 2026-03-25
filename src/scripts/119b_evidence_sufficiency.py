#!/usr/bin/env python
"""119b Evidence Sufficiency Gate — check if literature supports hypothesis.

Rule-based check on synthesis outputs. No LLM calls.

Exit codes:
  0: all hypotheses sufficient or weak (with --force)
  1: insufficient evidence found (without --force)

Usage::

    python -m src.scripts.119b_evidence_sufficiency --run-id <id>
    python -m src.scripts.119b_evidence_sufficiency --run-id <id> --force
    python -m src.scripts.119b_evidence_sufficiency --run-id <id> --dry-run
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

logger = logging.getLogger("119b_evidence_sufficiency")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.119b_evidence_sufficiency",
        description="119b Evidence Sufficiency Gate — check hypothesis evidence support",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--force", action="store_true",
                   help="Proceed even if evidence is insufficient (annotate only)")
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
        sys.exit(1)

    hypotheses = load_hypotheses_for_deep_lit(run_dir)
    if not hypotheses:
        logger.error("No focused hypotheses found in %s", args.run_id)
        sys.exit(1)

    logger.info("=== 119b Evidence Sufficiency Gate === run_id=%s", args.run_id)

    from src.lit_review.deep_lit.sufficiency import check_sufficiency

    results = []
    has_insufficient = False

    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)

        # Load synthesis
        syn_path = h_dir / "hyp_literature_synthesis.json"
        if not syn_path.exists():
            logger.warning("Skipping %s — no synthesis found", hid)
            continue

        synthesis = json.loads(syn_path.read_text())

        if args.dry_run:
            est_count = len(synthesis.get("known_established", []))
            gap_count = len(synthesis.get("unknown_gaps", []))
            print(f"  {hid}: {est_count} established, {gap_count} gaps → would check")
            continue

        result = check_sufficiency(synthesis, h)
        results.append(result)

        # Save per-hypothesis result
        out_path = h_dir / "evidence_sufficiency.json"
        out_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        logger.info("Saved evidence_sufficiency.json for %s: %s", hid[:12], result.result)

        if result.result == "insufficient":
            has_insufficient = True

    if args.dry_run:
        sys.exit(0)

    # Update manifest
    try:
        from src.lit_review.run_manifest import update_step
        outputs = [f"hyp_literature/{r.hypothesis_id}/evidence_sufficiency.json" for r in results]
        update_step(run_dir, "119b_evidence_sufficiency",
                    status="completed", outputs=outputs)
    except Exception as e:
        logger.warning("Manifest update failed: %s", e)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"119b Evidence Sufficiency Gate")
    print(f"{'=' * 60}")
    for r in results:
        icon = {"sufficient": "✓", "weak": "⚠", "insufficient": "✗"}.get(r.result, "?")
        print(f"  {icon} {r.hypothesis_id[:20]}: {r.result} ({r.recommendation})")
        if r.consensus_support.get("relevant_findings"):
            for f in r.consensus_support["relevant_findings"][:2]:
                print(f"      support: {f[:80]}")
        if r.gap_concerns.get("critical_gaps"):
            for g in r.gap_concerns["critical_gaps"][:2]:
                print(f"      gap: {g[:80]}")
        if r.suggested_queries:
            print(f"      queries: {len(r.suggested_queries)} suggested")

    if has_insufficient and not args.force:
        print(f"\n  BLOCKED: insufficient evidence detected.")
        print(f"  To proceed anyway: --force")
        print(f"  Review evidence_sufficiency.json for details.")
        sys.exit(1)
    elif has_insufficient and args.force:
        print(f"\n  WARNING: proceeding with insufficient evidence (--force)")

    logger.info("=== 119b Done ===")


if __name__ == "__main__":
    main()
