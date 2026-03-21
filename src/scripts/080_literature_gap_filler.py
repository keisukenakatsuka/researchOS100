#!/usr/bin/env python
"""080 Literature Gap Filler — search external sources for missing papers.

Searches Semantic Scholar and arXiv for papers related to the RQ
but not yet in LIT DB. Optionally writes new papers to LIT DB.

Usage::

    # Search and score (no LIT write)
    python -m src.scripts.080_literature_gap_filler --run-id <id>

    # With writeback to LIT DB
    python -m src.scripts.080_literature_gap_filler --run-id <id> --writeback

    # Dry run
    python -m src.scripts.080_literature_gap_filler --run-id <id> --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env, get_db_id, is_notion_writeback_enabled
from src.lit_review.rq_context import RQContext
from src.lit_review.gap_filler import fill_gaps, GapCandidate

logger = logging.getLogger("080_literature_gap_filler")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------

def load_inputs(run_id: str) -> tuple[Path, RQContext, List[Dict[str, Any]]]:
    """Load RQ context and candidate papers (for dedup)."""
    run_dir = _DATA_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    rq_context = RQContext.from_dict(json.loads((run_dir / "rq_context.json").read_text()))

    lit_papers = []
    candidates_path = run_dir / "candidate_papers.json"
    if candidates_path.exists():
        data = json.loads(candidates_path.read_text())
        lit_papers = data.get("scored_papers", [])

    return run_dir, rq_context, lit_papers


# ------------------------------------------------------------------
# LIT DB writeback
# ------------------------------------------------------------------

def _build_source_uid(c: GapCandidate) -> str:
    """Build a source UID for dedup."""
    if c.arxiv_id:
        return f"arxiv:{c.arxiv_id}"
    if c.doi:
        return f"doi:{c.doi}"
    if c.source_id:
        return f"{c.source}:{c.source_id}"
    return ""


def _check_exists_in_lit(
    notion_client,
    data_source_id: str,
    candidate: GapCandidate,
) -> bool:
    """Check if a paper already exists in LIT DB.

    Uses three dedup strategies in order:
    1. Source UID (arxiv:xxx, doi:xxx)
    2. Title exact match
    3. Title contains (for partial matches)
    """
    source_uid = _build_source_uid(candidate)

    # Strategy 1: Source UID
    if source_uid:
        try:
            pages = notion_client.query_data_source(
                data_source_id=data_source_id,
                filter={"property": "Source UID", "rich_text": {"equals": source_uid}},
                page_size=1, fetch_all=False,
            )
            if pages:
                logger.debug("Dedup hit (Source UID=%s): %s", source_uid, candidate.title[:40])
                return True
        except Exception:
            pass

    # Strategy 2: Title exact match
    try:
        pages = notion_client.query_data_source(
            data_source_id=data_source_id,
            filter={"property": "Name", "title": {"equals": candidate.title}},
            page_size=1, fetch_all=False,
        )
        if pages:
            logger.debug("Dedup hit (title exact): %s", candidate.title[:40])
            return True
    except Exception:
        pass

    return False


def preview_writeback(
    candidates: List[GapCandidate],
) -> Dict[str, Any]:
    """Preview what would be written to LIT DB (no Notion calls)."""
    to_add = [c for c in candidates if c.decision == "add"]
    return {
        "total_recommended": len(to_add),
        "papers": [
            {
                "title": c.title[:70],
                "source": c.source,
                "year": c.year,
                "score": c.relevance_score,
                "source_uid": _build_source_uid(c),
            }
            for c in to_add
        ],
    }


def write_to_lit_db(
    candidates: List[GapCandidate],
    *,
    run_id: str,
) -> Dict[str, Any]:
    """Write recommended papers to LIT DB with multi-strategy dedup.

    Dedup strategies (checked in order):
    1. Source UID (arxiv:xxx, doi:xxx, semantic_scholar:xxx)
    2. Title exact match in LIT DB
    """
    from src.notion import build_notion_client_from_env, NotionDataSourceResolver

    notion_client = build_notion_client_from_env()
    resolver = NotionDataSourceResolver(notion_client)
    lit_db_id = get_db_id("NOTION_LIT_DB_ID")
    resolved = resolver.resolve_once(name="LIT_DB", database_id=lit_db_id)

    written = 0
    skipped = 0
    errors = []

    to_add = [c for c in candidates if c.decision == "add"]
    logger.info("Writing %d recommended papers to LIT DB", len(to_add))

    for i, c in enumerate(to_add):
        # Multi-strategy dedup
        if _check_exists_in_lit(notion_client, resolved.data_source_id, c):
            skipped += 1
            continue

        source_uid = _build_source_uid(c)

        props: Dict[str, Any] = {
            "Name": {"title": [{"type": "text", "text": {"content": c.title[:2000]}}]},
            "Status": {"select": {"name": "INBOX"}},
            "Authors & Year": {"rich_text": [{"type": "text", "text": {
                "content": f"{c.authors} ({c.year})" if c.year else c.authors
            }}]},
            "Core Idea": {"rich_text": [{"type": "text", "text": {
                "content": c.abstract[:2000]
            }}]},
            "Tags": {"multi_select": [
                {"name": "block3_gap_filler"},
                {"name": c.source},
            ]},
            "Run ID": {"rich_text": [{"type": "text", "text": {"content": run_id}}]},
            "Ingested At": {"date": {"start": datetime.now(timezone.utc).isoformat()}},
            "Importance": {"number": c.relevance_score},
            "Decision Reason": {"rich_text": [{"type": "text", "text": {
                "content": f"[080 gap_filler] {c.reasoning[:1900]}"
            }}]},
        }

        if source_uid:
            props["Source UID"] = {"rich_text": [{"type": "text", "text": {"content": source_uid}}]}
        if c.url:
            props["PDF Link"] = {"url": c.url}

        try:
            notion_client.create_page(parent_db_id=lit_db_id, properties=props)
            written += 1
            logger.info("[%d/%d] Added to LIT: %s (uid=%s)", i + 1, len(to_add), c.title[:50], source_uid)
        except Exception as e:
            logger.warning("Failed to add: %s: %s", c.title[:50], e)
            errors.append(f"{c.title[:50]}: {e}")

    logger.info("LIT writeback: %d written, %d skipped (duplicate), %d errors", written, skipped, len(errors))
    return {"written": written, "skipped": skipped, "errors": errors}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.080_literature_gap_filler",
        description="080 Literature Gap Filler — search external sources for related papers",
    )
    p.add_argument("--run-id", type=str, required=True, help="Run ID from 079")
    p.add_argument("--max-results", type=int, default=50,
                    help="Max papers to search per source (default: 50)")
    p.add_argument("--threshold", type=int, default=60,
                    help="Min relevance score to recommend (default: 60)")
    p.add_argument("--sources", type=str, default="semantic_scholar,arxiv",
                    help="Comma-separated sources (default: semantic_scholar,arxiv)")
    p.add_argument("--writeback", action="store_true",
                    help="Write recommended papers to LIT DB")
    p.add_argument("--dry-run", action="store_true",
                    help="Show search queries without executing")
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

    run_dir, rq_context, lit_papers = load_inputs(args.run_id)
    logger.info("=== 080 Literature Gap Filler === run_id=%s", args.run_id)
    logger.info("RQ: %s", rq_context.title)
    logger.info("Existing LIT papers for dedup: %d", len(lit_papers))

    sources = [s.strip() for s in args.sources.split(",")]

    if args.dry_run:
        from src.llm.claude_client import build_claude_client_from_env
        llm_client = build_claude_client_from_env()
        queries = fill_gaps.__wrapped__(rq_context, llm_client=llm_client) if hasattr(fill_gaps, '__wrapped__') else None
        # Just generate and show queries
        from src.lit_review.gap_filler import generate_search_queries
        queries = generate_search_queries(rq_context, llm_client=llm_client)
        print(f"\nDRY RUN — would search with:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
        print(f"\nSources: {sources}")
        print(f"Max results: {args.max_results}")
        print(f"Threshold: {args.threshold}")
        print(f"Writeback: {args.writeback}")
        return

    # Run gap filler
    from src.llm.claude_client import build_claude_client_from_env
    llm_client = build_claude_client_from_env()

    result = fill_gaps(
        rq_context, lit_papers,
        llm_client=llm_client,
        max_results=args.max_results,
        threshold=args.threshold,
        sources=sources,
    )

    # Save outputs
    json_path = run_dir / "gap_candidates.json"
    json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    logger.info("Saved gap_candidates.json")

    md_path = run_dir / "gap_candidates.md"
    md_path.write_text(result.to_markdown())
    logger.info("Saved gap_candidates.md")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"080 Literature Gap Filler — Complete")
    print(f"{'=' * 60}")
    print(f"RQ: {rq_context.title}")
    print(f"")
    print(f"Search queries: {len(result.search_queries)}")
    for q in result.search_queries:
        print(f"  - {q}")
    print(f"")
    print(f"Total found: {result.total_found}")
    print(f"Duplicates removed: {result.duplicates_removed}")
    print(f"Scored: {result.scored}")
    print(f"Recommended (>={args.threshold}): {result.recommended}")
    print(f"")

    recommended = result.recommended_papers()
    if recommended:
        print(f"Recommended papers:")
        for c in recommended[:10]:
            print(f"  [{c.relevance_score}] {c.title[:60]} ({c.source}, {c.year or '?'})")
        if len(recommended) > 10:
            print(f"  ... and {len(recommended) - 10} more")

    # Writeback preview / execution
    if recommended:
        preview = preview_writeback(result.candidates)
        print(f"\nWriteback preview ({preview['total_recommended']} papers):")
        for p in preview["papers"]:
            uid = p["source_uid"] or "(no uid)"
            print(f"  [{p['score']}] {p['title']} ({p['source']}, {p['year'] or '?'}) {uid}")

    if args.writeback and recommended:
        if not is_notion_writeback_enabled():
            logger.error("ENABLE_NOTION_WRITEBACK is not 'true'")
            print(f"\nWriteback skipped: set ENABLE_NOTION_WRITEBACK=true in notebooks/env.txt")
        else:
            print(f"\nWriting {len(recommended)} papers to LIT DB (with dedup check)...")
            wb_result = write_to_lit_db(result.candidates, run_id=args.run_id)
            print(f"  Written: {wb_result['written']}")
            print(f"  Skipped (already in LIT): {wb_result['skipped']}")
            if wb_result['errors']:
                print(f"  Errors: {len(wb_result['errors'])}")
                for e in wb_result['errors'][:5]:
                    print(f"    - {e}")
    elif args.writeback and not recommended:
        print(f"\nNo papers to write (none above threshold)")
    elif not args.writeback and recommended:
        print(f"\nTo add these to LIT DB, re-run with --writeback")

    print(f"\nOutputs: {run_dir}")
    print(f"  gap_candidates.json")
    print(f"  gap_candidates.md")

    logger.info("=== 080 Done ===")


if __name__ == "__main__":
    main()
