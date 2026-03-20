#!/usr/bin/env python
"""119 Hypothesis Literature Synthesis — synthesize literature per hypothesis.

For each hypothesis, combines cluster structure with extracted variables,
methods, and findings to produce a structured literature synthesis and
a human-readable markdown report.

Usage::

    python -m src.scripts.119_hyp_literature_synthesis --run-id <id>
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

logger = logging.getLogger("119_hyp_literature_synthesis")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.119_hyp_literature_synthesis",
        description="119 Hypothesis Literature Synthesis — synthesize literature per hypothesis",
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

    logger.info("=== 119 Hypothesis Literature Synthesis === run_id=%s", args.run_id)
    logger.info("Hypotheses to process: %d", len(hypotheses))

    if args.dry_run:
        print(f"\nDRY RUN — inputs:")
        print(f"  Hypotheses: {len(hypotheses)}")
        for h in hypotheses:
            hid = h.get("hypothesis_id", h.get("id", "unknown"))
            h_dir = hyp_lit_dir(run_dir, hid)
            required = ["hyp_clusters.json", "hyp_variable_map.json",
                        "hyp_method_map.json", "hyp_finding_map.json"]
            missing = [f for f in required if not (h_dir / f).exists()]
            if missing:
                print(f"    - {hid}: MISSING {', '.join(missing)}")
            else:
                print(f"    - {hid}: ready")
        return

    from src.llm.claude_client import build_claude_client_from_env
    from src.lit_review.deep_lit.synthesis import synthesize_literature, render_synthesis_md

    llm_client = build_claude_client_from_env()

    all_outputs = []
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)

        clusters_path = h_dir / "hyp_clusters.json"
        variable_map_path = h_dir / "hyp_variable_map.json"
        method_map_path = h_dir / "hyp_method_map.json"
        finding_map_path = h_dir / "hyp_finding_map.json"

        required_paths = [clusters_path, variable_map_path, method_map_path, finding_map_path]
        missing = [p.name for p in required_paths if not p.exists()]
        if missing:
            logger.warning("Skipping %s — missing: %s", hid, ", ".join(missing))
            continue

        logger.info("Synthesizing literature for hypothesis: %s", hid)
        clusters_data = json.loads(clusters_path.read_text())
        variable_map = json.loads(variable_map_path.read_text())
        method_map = json.loads(method_map_path.read_text())
        finding_map = json.loads(finding_map_path.read_text())

        synthesis = synthesize_literature(
            clusters=clusters_data,
            variable_map=variable_map,
            method_map=method_map,
            finding_map=finding_map,
            hypothesis=h,
            llm_client=llm_client,
        )

        # Save JSON
        json_path = h_dir / "hyp_literature_synthesis.json"
        json_path.write_text(json.dumps(synthesis, ensure_ascii=False, indent=2))
        logger.info("Saved %s", json_path.name)
        all_outputs.append(f"hyp_literature/{hid}/hyp_literature_synthesis.json")

        # Render and save Markdown
        md_content = render_synthesis_md(synthesis, hypothesis=h)
        md_path = h_dir / "hyp_literature_synthesis.md"
        md_path.write_text(md_content)
        logger.info("Saved %s", md_path.name)
        all_outputs.append(f"hyp_literature/{hid}/hyp_literature_synthesis.md")

    # Update manifest
    try:
        from src.lit_review.run_manifest import update_step
        update_step(run_dir, "119_hyp_literature_synthesis",
                    status="completed",
                    outputs=all_outputs)
    except Exception as e:
        logger.warning("Manifest update failed: %s", e)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"119 Hypothesis Literature Synthesis — Complete")
    print(f"{'=' * 60}")
    n_hyp = len([h for h in hypotheses
                 if (hyp_lit_dir(run_dir, h.get("hypothesis_id", h.get("id", "unknown"))) / "hyp_literature_synthesis.json").exists()])
    print(f"Hypotheses synthesized: {n_hyp}")
    for h in hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "unknown"))
        h_dir = hyp_lit_dir(run_dir, hid)
        syn_path = h_dir / "hyp_literature_synthesis.json"
        if syn_path.exists():
            syn_data = json.loads(syn_path.read_text())
            n_sections = len(syn_data.get("sections", syn_data.get("themes", [])))
            print(f"  {hid}: {n_sections} sections")
    print(f"\nOutputs: {run_dir}")
    for o in all_outputs:
        print(f"  {o}")

    logger.info("=== 119 Done ===")


if __name__ == "__main__":
    main()
