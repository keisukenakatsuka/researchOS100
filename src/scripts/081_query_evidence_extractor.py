#!/usr/bin/env python
"""081 Query Evidence Extractor — Block 3 Evidence extraction.

Reads candidate papers from 079's output and extracts RQ-focused
Evidence from each paper.

Usage::

    # Using run_id from 079
    python -m src.scripts.081_query_evidence_extractor --run-id <id>

    # With filters
    python -m src.scripts.081_query_evidence_extractor --run-id <id> --max-papers 20 --min-score 65

    # Direct input file
    python -m src.scripts.081_query_evidence_extractor --input data/lit_review/.../candidate_papers.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env
from src.lit_review.rq_context import RQContext
from src.lit_review.extractor import (
    batch_extract,
    get_paper_text,
    Evidence,
    ExtractionResult,
)

logger = logging.getLogger("081_query_evidence_extractor")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


# ------------------------------------------------------------------
# Input loading
# ------------------------------------------------------------------

def load_inputs(
    run_id: str | None = None,
    input_path: str | None = None,
) -> tuple[Path, RQContext, List[Dict[str, Any]]]:
    """Load RQ context and candidate papers from 079's output.

    Returns (run_dir, rq_context, included_papers).
    """
    if run_id:
        run_dir = _DATA_DIR / run_id
    elif input_path:
        run_dir = Path(input_path).parent
    else:
        raise ValueError("Either --run-id or --input is required")

    # Load RQ context
    rq_path = run_dir / "rq_context.json"
    if not rq_path.exists():
        raise FileNotFoundError(f"rq_context.json not found in {run_dir}")
    rq_data = json.loads(rq_path.read_text())
    rq_context = RQContext.from_dict(rq_data)

    # Load candidate papers
    candidates_path = run_dir / "candidate_papers.json"
    if input_path:
        candidates_path = Path(input_path)
    if not candidates_path.exists():
        raise FileNotFoundError(f"candidate_papers.json not found: {candidates_path}")

    candidates_data = json.loads(candidates_path.read_text())
    all_papers = candidates_data.get("scored_papers", [])

    # Filter to included only
    included = [p for p in all_papers if p.get("decision") == "include"]
    logger.info("Loaded %d included papers from %s", len(included), candidates_path.name)

    return run_dir, rq_context, included


def filter_papers(
    papers: List[Dict[str, Any]],
    *,
    min_score: int = 0,
    max_papers: int = 0,
) -> List[Dict[str, Any]]:
    """Filter and limit papers for processing."""
    # Already sorted by score desc from 079
    if min_score > 0:
        papers = [p for p in papers if p.get("relevance_score", 0) >= min_score]
        logger.info("After min_score=%d filter: %d papers", min_score, len(papers))

    if max_papers > 0 and len(papers) > max_papers:
        papers = papers[:max_papers]
        logger.info("Limited to top %d papers", max_papers)

    return papers


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def save_outputs(
    run_dir: Path,
    rq_context: RQContext,
    results: List[ExtractionResult],
):
    """Save extraction results to run directory."""
    # Flatten all evidence items with paper metadata
    all_evidence = []
    for r in results:
        for e in r.evidence_items:
            item = e.to_dict()
            item["paper_id"] = r.paper_id
            item["paper_title"] = r.paper_title
            item["text_source"] = r.text_source
            all_evidence.append(item)

    # Per-paper summary
    paper_summaries = []
    for r in results:
        paper_summaries.append({
            "paper_id": r.paper_id,
            "paper_title": r.paper_title,
            "text_source": r.text_source,
            "text_length": r.text_length,
            "evidence_count": len(r.evidence_items),
            "error": r.error,
        })

    output = {
        "run_id": run_dir.name,
        "rq_context": rq_context.to_dict(),
        "query_mode": "rq",
        "papers_processed": len(results),
        "papers_succeeded": sum(1 for r in results if not r.error),
        "papers_failed": sum(1 for r in results if r.error),
        "total_evidence": len(all_evidence),
        "evidence_items": all_evidence,
        "paper_summaries": paper_summaries,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # evidence.json
    json_path = run_dir / "evidence.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.info("Saved evidence.json (%d items)", len(all_evidence))

    # evidence.md
    md_path = run_dir / "evidence.md"
    md_path.write_text(_render_markdown(rq_context, results, all_evidence))
    logger.info("Saved evidence.md")


def _render_markdown(
    rq_context: RQContext,
    results: List[ExtractionResult],
    all_evidence: List[Dict[str, Any]],
) -> str:
    """Render evidence as readable Markdown."""
    lines = [
        f"# Evidence Extraction Results",
        f"",
        f"## RQ: {rq_context.title}",
        f"",
        f"- Papers processed: {len(results)}",
        f"- Total evidence items: {len(all_evidence)}",
        f"- Query mode: rq",
        f"",
    ]

    # Dimension distribution
    dim_counts = Counter(e.get("dimension", "unknown") for e in all_evidence)
    lines.append("## Dimension Distribution")
    lines.append("")
    for dim, count in dim_counts.most_common():
        lines.append(f"- {dim}: {count}")
    lines.append("")

    # Per-paper evidence
    for r in results:
        lines.append(f"---")
        lines.append(f"### {r.paper_title}")
        lines.append(f"")
        lines.append(f"- Text source: {r.text_source} ({r.text_length:,} chars)")
        lines.append(f"- Evidence items: {len(r.evidence_items)}")
        if r.error:
            lines.append(f"- **Error**: {r.error}")
        lines.append(f"")

        for i, e in enumerate(r.evidence_items, 1):
            lines.append(f"**[{i}] {e.dimension}** (confidence: {e.confidence})")
            lines.append(f"")
            lines.append(f"> {e.claim_or_point}")
            lines.append(f"")
            lines.append(f"根拠: {e.evidence_text[:200]}")
            lines.append(f"")
            lines.append(f"RQ関連: {e.relevance_to_rq}")
            lines.append(f"")

    return "\n".join(lines)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.081_query_evidence_extractor",
        description="081 Query Evidence Extractor — extract RQ-focused evidence from papers",
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--run-id", type=str, help="Run ID from 079")
    input_group.add_argument("--input", type=str, help="Path to candidate_papers.json")

    p.add_argument("--max-papers", type=int, default=20,
                    help="Max papers to process (default: 20)")
    p.add_argument("--min-score", type=int, default=0,
                    help="Minimum relevance score to process (default: 0 = all included)")
    p.add_argument("--mode", type=str, default="rq",
                    choices=["rq", "hypothesis", "policy", "strategic"],
                    help="Extraction mode (default: rq)")
    p.add_argument("--dry-run", action="store_true",
                    help="Show papers to process without extracting")
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

    # Load inputs
    run_dir, rq_context, included_papers = load_inputs(
        run_id=args.run_id, input_path=args.input,
    )
    logger.info("=== 081 Query Evidence Extractor === run_dir=%s", run_dir)
    logger.info("RQ: %s", rq_context.title)

    # Filter
    papers = filter_papers(
        included_papers,
        min_score=args.min_score,
        max_papers=args.max_papers,
    )
    logger.info("Processing %d papers (mode=%s)", len(papers), args.mode)

    if args.dry_run:
        print(f"\nDRY RUN — {len(papers)} papers would be processed:")
        for p in papers:
            print(f"  [{p.get('relevance_score', '?'):3}] {p.get('title', '')[:65]}")
        return

    if not papers:
        logger.error("No papers to process")
        return

    # Extract evidence
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    results = batch_extract(
        papers, rq_context,
        mode=args.mode, llm_client=llm_client,
        get_text_fn=get_paper_text,
    )

    # Save
    save_outputs(run_dir, rq_context, results)

    # Print summary
    succeeded = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    total_evidence = sum(len(r.evidence_items) for r in results)

    dim_counts = Counter()
    conf_values = []
    for r in results:
        for e in r.evidence_items:
            dim_counts[e.dimension] += 1
            conf_values.append(e.confidence)

    print(f"\n{'=' * 60}")
    print(f"081 Query Evidence Extractor — Complete")
    print(f"{'=' * 60}")
    print(f"Run: {run_dir.name}")
    print(f"RQ: {rq_context.title}")
    print(f"Papers: {len(results)} processed, {len(succeeded)} succeeded, {len(failed)} failed")
    print(f"Total evidence: {total_evidence} items")
    if succeeded:
        avg = total_evidence / len(succeeded)
        print(f"Average per paper: {avg:.1f}")

    if dim_counts:
        print(f"\nDimension distribution:")
        for dim, count in dim_counts.most_common():
            print(f"  {dim:12s}: {count}")

    if conf_values:
        print(f"\nConfidence: mean={sum(conf_values)/len(conf_values):.2f}, "
              f"range={min(conf_values):.2f}–{max(conf_values):.2f}")

    if failed:
        print(f"\nFailed papers:")
        for r in failed:
            print(f"  - {r.paper_title[:60]}: {r.error[:80]}")

    print(f"\nOutputs: {run_dir}")
    print(f"  evidence.json")
    print(f"  evidence.md")

    logger.info("=== 081 Done ===")


if __name__ == "__main__":
    main()
