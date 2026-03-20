#!/usr/bin/env python
"""118 Hypothesis Extraction — extract variables, methods, and findings per hypothesis.

For each hypothesis, processes ranked papers and cluster structure to extract
structured information: variables, methods, and findings mapped to the hypothesis.

Usage::

    python -m src.scripts.118_hyp_extraction --run-id <id>
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

logger = logging.getLogger("118_hyp_extraction")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.118_hyp_extraction",
        description="118 Hypothesis Extraction — extract variables, methods, findings",
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

    logger.info("=== 118 Hypothesis Extraction === run_id=%s", args.run_id)
    logger.info("Hypotheses to process: %d", len(hypotheses))

    if args.dry_run:
        print(f"\nDRY RUN — inputs:")
        print(f"  Hypotheses: {len(hypotheses)}")
        for h in hypotheses:
            hid = h.get("hypothesis_id", h.get("id", "unknown"))
            h_dir = hyp_lit_dir(run_dir, hid)
            ranked_ok = (h_dir / "hyp_papers_ranked.json").exists()
            clusters_ok = (h_dir / "hyp_clusters.json").exists()
            status_parts = []
            if not ranked_ok:
                status_parts.append("MISSING hyp_papers_ranked.json")
            if not clusters_ok:
                status_parts.append("MISSING hyp_clusters.json")
            status = ", ".join(status_parts) if status_parts else "ready"
            print(f"    - {hid}: {status}")
        return

    from src.llm.claude_client import build_claude_client_from_env
    from src.lit_review.deep_lit.extraction import extract_and_map

    llm_client = build_claude_client_from_env()

    all_outputs = []
    output_files = [
        "hyp_extraction.json",
        "hyp_variable_map.json",
        "hyp_method_map.json",
        "hyp_finding_map.json",
    ]

    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)

        ranked_path = h_dir / "hyp_papers_ranked.json"
        clusters_path = h_dir / "hyp_clusters.json"

        if not ranked_path.exists():
            logger.warning("Skipping %s — hyp_papers_ranked.json not found", hid)
            continue
        if not clusters_path.exists():
            logger.warning("Skipping %s — hyp_clusters.json not found", hid)
            continue

        logger.info("Extracting for hypothesis: %s", hid)
        ranked_data = json.loads(ranked_path.read_text())
        clusters_data = json.loads(clusters_path.read_text())

        result = extract_and_map(
            papers=ranked_data,
            clusters=clusters_data,
            hypothesis=h,
            llm_client=llm_client,
        )

        # Save 4 output files
        for fname in output_files:
            key = fname.replace("hyp_", "").replace(".json", "")
            data = result.get(key, {})
            out_path = h_dir / fname
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            logger.info("Saved %s", out_path.name)
            all_outputs.append(f"hyp_literature/{hid}/{fname}")

    # Update manifest
    try:
        from src.lit_review.run_manifest import update_step
        update_step(run_dir, "118_hyp_extraction",
                    status="completed",
                    outputs=all_outputs)
    except Exception as e:
        logger.warning("Manifest update failed: %s", e)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"118 Hypothesis Extraction — Complete")
    print(f"{'=' * 60}")
    n_hyp = len([h for h in hypotheses
                 if (hyp_lit_dir(run_dir, h.get("hypothesis_id", h.get("id", "unknown"))) / "hyp_extraction.json").exists()])
    print(f"Hypotheses processed: {n_hyp}")
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)
        ext_path = h_dir / "hyp_extraction.json"
        if ext_path.exists():
            print(f"  {hid}:")
            for fname in output_files:
                fpath = h_dir / fname
                if fpath.exists():
                    fdata = json.loads(fpath.read_text())
                    n_items = len(fdata) if isinstance(fdata, list) else len(fdata.get("items", fdata.get("entries", [])))
                    print(f"    {fname}: {n_items} items")
    print(f"\nOutputs: {run_dir}")
    for o in all_outputs:
        print(f"  {o}")

    logger.info("=== 118 Done ===")


if __name__ == "__main__":
    main()
