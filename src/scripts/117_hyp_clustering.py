#!/usr/bin/env python
"""117 Hypothesis Paper Clustering — cluster ranked papers per hypothesis.

For each hypothesis, groups ranked papers into thematic clusters to
facilitate structured extraction and synthesis.

Usage::

    python -m src.scripts.117_hyp_clustering --run-id <id>
    python -m src.scripts.117_hyp_clustering --run-id <id> --max-clusters 8
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

logger = logging.getLogger("117_hyp_clustering")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.117_hyp_clustering",
        description="117 Hypothesis Paper Clustering — cluster ranked papers",
    )
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--max-clusters", type=int, default=10,
                   help="Maximum number of clusters per hypothesis (default: 10)")
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

    logger.info("=== 117 Hypothesis Paper Clustering === run_id=%s", args.run_id)
    logger.info("Hypotheses to process: %d", len(hypotheses))
    logger.info("Max clusters: %d", args.max_clusters)

    if args.dry_run:
        print(f"\nDRY RUN — inputs:")
        print(f"  Hypotheses: {len(hypotheses)}")
        print(f"  Max clusters: {args.max_clusters}")
        for h in hypotheses:
            hid = h.get("hypothesis_id", h.get("id", "unknown"))
            rpath = hyp_lit_dir(run_dir, hid) / "hyp_papers_ranked.json"
            if rpath.exists():
                rdata = json.loads(rpath.read_text())
                n = len(rdata.get("papers", []))
                print(f"    - {hid}: {n} ranked papers")
            else:
                print(f"    - {hid}: MISSING hyp_papers_ranked.json")
        return

    from src.llm.claude_client import build_claude_client_from_env
    from src.lit_review.deep_lit.clustering import cluster_papers

    llm_client = build_claude_client_from_env()

    all_outputs = []
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)

        ranked_path = h_dir / "hyp_papers_ranked.json"
        if not ranked_path.exists():
            logger.warning("Skipping %s — hyp_papers_ranked.json not found", hid)
            continue

        logger.info("Clustering papers for hypothesis: %s", hid)
        ranked_data = json.loads(ranked_path.read_text())

        clusters = cluster_papers(
            ranked_data,
            llm_client=llm_client,
            max_clusters=args.max_clusters,
        )

        out_path = h_dir / "hyp_clusters.json"
        out_path.write_text(json.dumps(clusters, ensure_ascii=False, indent=2))
        n_clusters = len(clusters.get("clusters", []))
        logger.info("Saved %s (%d clusters)", out_path.name, n_clusters)
        all_outputs.append(f"hyp_literature/{hid}/hyp_clusters.json")

    # Update manifest
    try:
        from src.lit_review.run_manifest import update_step
        update_step(run_dir, "117_hyp_clustering",
                    status="completed",
                    outputs=all_outputs)
    except Exception as e:
        logger.warning("Manifest update failed: %s", e)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"117 Hypothesis Paper Clustering — Complete")
    print(f"{'=' * 60}")
    print(f"Hypotheses processed: {len(all_outputs)}")
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)
        cpath = h_dir / "hyp_clusters.json"
        if cpath.exists():
            cdata = json.loads(cpath.read_text())
            clusters_list = cdata.get("clusters", [])
            n = len(clusters_list)
            print(f"  {hid}: {n} clusters")
            for cl in clusters_list:
                cl_name = cl.get("name", cl.get("label", "?"))
                cl_size = len(cl.get("papers", cl.get("paper_ids", [])))
                print(f"    - {cl_name}: {cl_size} papers")
    print(f"\nOutputs: {run_dir}")
    for o in all_outputs:
        print(f"  {o}")

    logger.info("=== 117 Done ===")


if __name__ == "__main__":
    main()
