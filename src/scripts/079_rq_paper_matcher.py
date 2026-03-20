#!/usr/bin/env python
"""079 RQ Paper Matcher — Block 3 entry point.

Matches papers from LIT DB against a Research Question (RQ) and
selects candidates above a relevance threshold.

With --full-pipeline, runs the complete MVP pipeline:
  079 (match) → 081 (evidence) → 082 (synthesis)

Usage::

    # Single step
    python -m src.scripts.079_rq_paper_matcher --rq-id <notion_page_id>

    # Full pipeline
    python -m src.scripts.079_rq_paper_matcher --full-pipeline
    python -m src.scripts.079_rq_paper_matcher --rq-text "..." --full-pipeline
    python -m src.scripts.079_rq_paper_matcher --full-pipeline --min-score 65 --max-papers 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env, get_db_id
from src.notion import build_notion_client_from_env, NotionDataSourceResolver
from src.notion.rq_normalize import normalize_rqs
from src.lit_review.rq_context import RQContext
from src.lit_review.matcher import query_lit_papers, score_papers, match_papers

logger = logging.getLogger("079_rq_paper_matcher")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


# ------------------------------------------------------------------
# RQ resolution
# ------------------------------------------------------------------

def resolve_rq(
    notion_client,
    resolver,
    *,
    rq_id: str | None = None,
    rq_text: str | None = None,
) -> RQContext:
    """Resolve RQ from Notion page ID or free text."""
    if rq_id:
        rq_db_id = get_db_id("NOTION_RQ_DB_ID")
        resolved = resolver.resolve_once(name="RQ_DB", database_id=rq_db_id)
        pages = notion_client.query_data_source(
            data_source_id=resolved.data_source_id, fetch_all=True,
        )
        all_rqs = normalize_rqs(pages)
        for rq in all_rqs:
            if rq["page_id"] == rq_id:
                return RQContext.from_notion_rq(rq)
        for rq in all_rqs:
            if rq_id in rq.get("page_id", ""):
                return RQContext.from_notion_rq(rq)
        raise ValueError(f"RQ not found: {rq_id}")

    if rq_text:
        return RQContext.from_text(rq_text)

    # Default: first High-priority RQ
    rq_db_id = get_db_id("NOTION_RQ_DB_ID")
    resolved = resolver.resolve_once(name="RQ_DB", database_id=rq_db_id)
    pages = notion_client.query_data_source(
        data_source_id=resolved.data_source_id, fetch_all=True,
    )
    all_rqs = normalize_rqs(pages)
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    all_rqs.sort(key=lambda r: (priority_order.get(r.get("priority", ""), 9), r.get("title", "")))
    if not all_rqs:
        raise ValueError("No RQs found in RQ DB")
    logger.info("No RQ specified — using first High-priority RQ")
    return RQContext.from_notion_rq(all_rqs[0])


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------

def save_079_outputs(run_id: str, rq_context: RQContext, match_result) -> Path:
    run_dir = _DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rq_path = run_dir / "rq_context.json"
    rq_path.write_text(json.dumps(rq_context.to_dict(), ensure_ascii=False, indent=2))

    result_path = run_dir / "candidate_papers.json"
    result_path.write_text(json.dumps(match_result.to_dict(), ensure_ascii=False, indent=2))

    md_path = run_dir / "candidate_papers.md"
    md_path.write_text(match_result.to_markdown())

    logger.info("079 outputs saved to %s", run_dir)
    return run_dir


# ------------------------------------------------------------------
# Full pipeline: 081 Evidence Extraction
# ------------------------------------------------------------------

def run_081_evidence(
    run_dir: Path,
    rq_context: RQContext,
    included_papers: list,
    llm_client,
    *,
    min_score: int = 0,
    max_papers: int = 20,
) -> dict:
    """Run 081 evidence extraction inline."""
    from src.lit_review.extractor import batch_extract, get_paper_text

    # Filter
    papers = included_papers
    if min_score > 0:
        papers = [p for p in papers if p.get("relevance_score", 0) >= min_score]
        logger.info("081: After min_score=%d: %d papers", min_score, len(papers))
    if max_papers > 0 and len(papers) > max_papers:
        papers = papers[:max_papers]
        logger.info("081: Limited to %d papers", max_papers)

    if not papers:
        logger.error("081: No papers to process")
        return {"papers_processed": 0, "evidence_items": [], "error": "no papers"}

    # Extract
    results = batch_extract(
        papers, rq_context,
        mode="rq", llm_client=llm_client,
        get_text_fn=get_paper_text,
    )

    # Build output
    all_evidence = []
    paper_summaries = []
    for r in results:
        for e in r.evidence_items:
            item = e.to_dict()
            item["paper_id"] = r.paper_id
            item["paper_title"] = r.paper_title
            item["text_source"] = r.text_source
            all_evidence.append(item)
        paper_summaries.append({
            "paper_id": r.paper_id,
            "paper_title": r.paper_title,
            "text_source": r.text_source,
            "text_length": r.text_length,
            "evidence_count": len(r.evidence_items),
            "error": r.error,
        })

    succeeded = sum(1 for r in results if not r.error)
    failed = sum(1 for r in results if r.error)

    output = {
        "run_id": run_dir.name,
        "rq_context": rq_context.to_dict(),
        "query_mode": "rq",
        "papers_processed": len(results),
        "papers_succeeded": succeeded,
        "papers_failed": failed,
        "total_evidence": len(all_evidence),
        "evidence_items": all_evidence,
        "paper_summaries": paper_summaries,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save
    json_path = run_dir / "evidence.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.info("081: Saved evidence.json (%d items from %d papers, %d failed)",
                len(all_evidence), succeeded, failed)

    return output


# ------------------------------------------------------------------
# Full pipeline: 082 Synthesis
# ------------------------------------------------------------------

def run_082_synthesis(
    run_dir: Path,
    rq_context: RQContext,
    evidence_data: dict,
    papers: list,
    llm_client,
):
    """Run 082 synthesis inline."""
    from src.lit_review.synthesizer import synthesize_lit_review

    evidence_items = evidence_data.get("evidence_items", [])
    if not evidence_items:
        logger.error("082: No evidence items to synthesize")
        return None

    result = synthesize_lit_review(
        rq_context=rq_context,
        evidence_items=evidence_items,
        papers=papers,
        llm_client=llm_client,
        run_id=run_dir.name,
    )

    # Save
    json_path = run_dir / "lit_review.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("082: Saved lit_review.json")

    md_path = run_dir / "lit_review.md"
    md_path.write_text(result.to_markdown())
    logger.info("082: Saved lit_review.md")

    return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.079_rq_paper_matcher",
        description="079 RQ Paper Matcher — match LIT papers against an RQ, optionally run full pipeline",
    )
    rq_group = p.add_mutually_exclusive_group()
    rq_group.add_argument("--rq-id", type=str, default=None,
                          help="Notion page ID of the RQ")
    rq_group.add_argument("--rq-text", type=str, default=None,
                          help="Free-form RQ text")

    p.add_argument("--threshold", type=int, default=50,
                    help="Minimum relevance score for inclusion (default: 50)")
    p.add_argument("--full-pipeline", action="store_true",
                    help="Run full pipeline: 079 → 081 → 082")
    p.add_argument("--min-score", type=int, default=65,
                    help="Min score for 081 evidence extraction (default: 65, used with --full-pipeline)")
    p.add_argument("--max-papers", type=int, default=20,
                    help="Max papers for 081 (default: 20, used with --full-pipeline)")
    p.add_argument("--dry-run", action="store_true",
                    help="Show RQ and paper count without scoring")
    p.add_argument("--run-id", type=str, default=None,
                    help="Use existing run ID")
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

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    logger.info("=== 079 RQ Paper Matcher === run_id=%s", run_id)

    # Setup
    notion_client = build_notion_client_from_env()
    resolver = NotionDataSourceResolver(notion_client)

    rq_context = resolve_rq(
        notion_client, resolver,
        rq_id=args.rq_id, rq_text=args.rq_text,
    )
    logger.info("RQ: %s", rq_context.title)

    papers = query_lit_papers(notion_client, resolver)

    if args.dry_run:
        logger.info("DRY RUN — %d papers would be scored", len(papers))
        if args.full_pipeline:
            logger.info("Full pipeline: 079 → 081 (min_score=%d, max_papers=%d) → 082",
                        args.min_score, args.max_papers)
        return

    if not papers:
        logger.error("No papers found in LIT DB")
        return

    # --- 079: Score & Match ---
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    scores = score_papers(rq_context, papers, llm_client=llm_client)
    match_result = match_papers(rq_context, papers, scores, threshold=args.threshold)
    run_dir = save_079_outputs(run_id, rq_context, match_result)

    included = match_result.included_papers()
    included_dicts = [p.to_dict() for p in included]

    print(f"\n{'=' * 60}")
    print(f"079 RQ Paper Matcher")
    print(f"{'=' * 60}")
    print(f"Run ID: {run_id}")
    print(f"RQ: {rq_context.title}")
    print(f"Total: {match_result.total_papers} | Included (>={match_result.threshold}): {match_result.included} | Excluded: {match_result.excluded}")
    if included:
        s_list = [p.relevance_score for p in included]
        print(f"Score range: {min(s_list)}–{max(s_list)}")

    if not args.full_pipeline:
        print(f"\nOutputs: {run_dir}")
        print(f"  Run full pipeline: python -m src.scripts.079_rq_paper_matcher --run-id {run_id} --full-pipeline")
        logger.info("=== 079 Done (single step) ===")
        return

    # --- 081: Evidence Extraction ---
    print(f"\n{'=' * 60}")
    print(f"081 Evidence Extraction (min_score={args.min_score}, max_papers={args.max_papers})")
    print(f"{'=' * 60}")

    evidence_data = run_081_evidence(
        run_dir, rq_context, included_dicts, llm_client,
        min_score=args.min_score, max_papers=args.max_papers,
    )

    ev_succeeded = evidence_data.get("papers_succeeded", 0)
    ev_failed = evidence_data.get("papers_failed", 0)
    ev_total = evidence_data.get("total_evidence", 0)
    print(f"Papers: {ev_succeeded} succeeded, {ev_failed} failed")
    print(f"Evidence: {ev_total} items")

    if ev_total == 0:
        logger.error("No evidence extracted — skipping 082")
        print(f"\nOutputs: {run_dir}")
        return

    # Dimension distribution
    dim_counts = Counter(e.get("dimension", "?") for e in evidence_data.get("evidence_items", []))
    print(f"Dimensions: {dict(dim_counts.most_common())}")

    # --- 082: Synthesis ---
    print(f"\n{'=' * 60}")
    print(f"082 Lit Review Synthesis")
    print(f"{'=' * 60}")

    lit_review = run_082_synthesis(
        run_dir, rq_context, evidence_data, included_dicts, llm_client,
    )

    if lit_review:
        print(f"Theoretical streams: {len(lit_review.theoretical_streams)}")
        for s in lit_review.theoretical_streams:
            print(f"  - {s.name}")
        print(f"Established findings: {len(lit_review.established)}")
        print(f"Emerging findings: {len(lit_review.emerging)}")
        print(f"Contested points: {len(lit_review.contested)}")
        print(f"Open questions: {len(lit_review.open_questions)}")
        print(f"\nExecutive Summary:")
        print(f"  {lit_review.executive_summary[:200]}...")

    # --- Final Summary ---
    print(f"\n{'=' * 60}")
    print(f"Full Pipeline Complete")
    print(f"{'=' * 60}")
    print(f"Run ID: {run_id}")
    print(f"RQ: {rq_context.title}")
    print(f"")
    print(f"079 Matching:   {match_result.total_papers} scored → {match_result.included} included")
    print(f"081 Evidence:   {ev_succeeded} papers → {ev_total} evidence items ({ev_failed} failed)")
    if lit_review:
        print(f"082 Synthesis:  {len(lit_review.theoretical_streams)} streams, "
              f"{len(lit_review.established)} established, "
              f"{len(lit_review.emerging)} emerging, "
              f"{len(lit_review.contested)} contested, "
              f"{len(lit_review.open_questions)} gaps")
    print(f"")
    print(f"Outputs: {run_dir}")
    print(f"  rq_context.json")
    print(f"  candidate_papers.json / .md")
    print(f"  evidence.json")
    print(f"  lit_review.json / .md")

    logger.info("=== Full Pipeline Done ===")


if __name__ == "__main__":
    main()
