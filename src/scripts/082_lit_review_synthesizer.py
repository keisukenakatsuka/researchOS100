#!/usr/bin/env python
"""082 Lit Review Synthesizer — Block 3 synthesis.

Reads evidence from 081's output and synthesizes a structured
Literature Review with theoretical streams, findings classification,
research gaps, and executive summary.

Usage::

    python -m src.scripts.082_lit_review_synthesizer --run-id <id>
    python -m src.scripts.082_lit_review_synthesizer --run-id <id> --min-score 70
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env
from src.lit_review.rq_context import RQContext
from src.lit_review.synthesizer import synthesize_lit_review

logger = logging.getLogger("082_lit_review_synthesizer")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


# ------------------------------------------------------------------
# Input loading
# ------------------------------------------------------------------

def load_inputs(
    run_id: str | None = None,
    input_path: str | None = None,
) -> tuple[Path, RQContext, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load RQ context, evidence, and paper list.

    Returns (run_dir, rq_context, evidence_items, papers).
    """
    if run_id:
        run_dir = _DATA_DIR / run_id
    elif input_path:
        run_dir = Path(input_path).parent
    else:
        raise ValueError("Either --run-id or --input is required")

    # RQ context
    rq_path = run_dir / "rq_context.json"
    if not rq_path.exists():
        raise FileNotFoundError(f"rq_context.json not found in {run_dir}")
    rq_context = RQContext.from_dict(json.loads(rq_path.read_text()))

    # Evidence
    evidence_path = run_dir / "evidence.json"
    if input_path:
        evidence_path = Path(input_path)
    if not evidence_path.exists():
        raise FileNotFoundError(f"evidence.json not found: {evidence_path}")
    evidence_data = json.loads(evidence_path.read_text())
    evidence_items = evidence_data.get("evidence_items", [])

    # Papers — from candidate_papers.json (included only)
    candidates_path = run_dir / "candidate_papers.json"
    papers = []
    if candidates_path.exists():
        candidates = json.loads(candidates_path.read_text())
        papers = [p for p in candidates.get("scored_papers", []) if p.get("decision") == "include"]

    # If papers not available from candidates, derive from evidence
    if not papers:
        seen = set()
        for e in evidence_items:
            title = e.get("paper_title", "")
            if title and title not in seen:
                seen.add(title)
                papers.append({
                    "title": title,
                    "paper_id": e.get("paper_id", ""),
                    "relevance_score": 0,
                })

    logger.info("Loaded %d evidence items, %d papers", len(evidence_items), len(papers))
    return run_dir, rq_context, evidence_items, papers


def filter_inputs(
    evidence_items: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    *,
    min_score: int = 0,
    max_papers: int = 0,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter papers and corresponding evidence."""
    if min_score > 0:
        papers = [p for p in papers if p.get("relevance_score", 0) >= min_score]
        logger.info("After min_score=%d: %d papers", min_score, len(papers))

    if max_papers > 0 and len(papers) > max_papers:
        papers = papers[:max_papers]
        logger.info("Limited to %d papers", max_papers)

    # Filter evidence to match remaining papers
    paper_titles = {p.get("title", "") for p in papers}
    evidence_items = [e for e in evidence_items if e.get("paper_title", "") in paper_titles]
    logger.info("Filtered evidence: %d items for %d papers", len(evidence_items), len(papers))

    return evidence_items, papers


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def save_outputs(run_dir: Path, result):
    """Save lit review to JSON and Markdown."""
    # lit_review.json
    json_path = run_dir / "lit_review.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved lit_review.json")

    # lit_review.md
    md_path = run_dir / "lit_review.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved lit_review.md")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.082_lit_review_synthesizer",
        description="082 Lit Review Synthesizer",
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--run-id", type=str, help="Run ID from 079/081")
    input_group.add_argument("--input", type=str, help="Path to evidence.json")

    p.add_argument("--min-score", type=int, default=0)
    p.add_argument("--max-papers", type=int, default=0)
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

    # Load
    run_dir, rq_context, evidence_items, papers = load_inputs(
        run_id=args.run_id, input_path=args.input,
    )
    logger.info("=== 082 Lit Review Synthesizer === run_dir=%s", run_dir)
    logger.info("RQ: %s", rq_context.title)

    # Filter
    evidence_items, papers = filter_inputs(
        evidence_items, papers,
        min_score=args.min_score, max_papers=args.max_papers,
    )

    if args.dry_run:
        print(f"\nDRY RUN — would synthesize from {len(evidence_items)} evidence items across {len(papers)} papers")
        return

    if not evidence_items:
        logger.error("No evidence items to synthesize")
        return

    # Synthesize
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = synthesize_lit_review(
        rq_context=rq_context,
        evidence_items=evidence_items,
        papers=papers,
        llm_client=llm_client,
        run_id=run_dir.name,
    )

    # Save
    save_outputs(run_dir, result)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"082 Lit Review Synthesizer — Complete")
    print(f"{'=' * 60}")
    print(f"Run: {run_dir.name}")
    print(f"RQ: {rq_context.title}")
    print(f"")
    print(f"Theoretical streams: {len(result.theoretical_streams)}")
    for s in result.theoretical_streams:
        print(f"  - {s.name} ({len(s.papers)} papers)")
    print(f"Established findings: {len(result.established)}")
    print(f"Emerging findings: {len(result.emerging)}")
    print(f"Contested points: {len(result.contested)}")
    print(f"Open questions: {len(result.open_questions)}")
    print(f"")
    print(f"Executive Summary:")
    print(f"  {result.executive_summary[:300]}...")
    print(f"")
    print(f"Outputs: {run_dir}")
    print(f"  lit_review.json")
    print(f"  lit_review.md")

    logger.info("=== 082 Done ===")


if __name__ == "__main__":
    main()
