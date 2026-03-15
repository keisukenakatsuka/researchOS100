# src/lit_review/writer.py
"""Block 3 Notion writeback orchestration.

Writes Block 3 outputs (Evidence, Claims, Memos, Research Run) to KML.
Uses existing repos from src/notion/ (evidence_repo, claims_repo, memos_repo, research_runs_repo).

Usage::

    from src.lit_review.writer import write_to_notion

    result = write_to_notion(run_dir, dry_run=False)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_db_id, is_notion_writeback_enabled
from src.notion import build_notion_client_from_env, NotionDataSourceResolver
from src.notion.evidence_repo import EvidenceRepo
from src.notion.claims_repo import ClaimsRepo
from src.notion.memos_repo import MemosRepo
from src.notion.research_runs_repo import ResearchRunsRepo
from src.notion.research_schema import (
    ENV_EVIDENCE_DB_ID,
    ENV_CLAIMS_DB_ID,
    ENV_MEMOS_DB_ID,
    ENV_RESEARCH_RUNS_DB_ID,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------

def preflight_check() -> Dict[str, Any]:
    """Check that writeback is enabled and all DB IDs are available."""
    result = {
        "ok": True,
        "writeback_enabled": is_notion_writeback_enabled(),
        "missing_db_ids": [],
    }

    for env_name in [ENV_EVIDENCE_DB_ID, ENV_CLAIMS_DB_ID, ENV_MEMOS_DB_ID, ENV_RESEARCH_RUNS_DB_ID]:
        try:
            get_db_id(env_name)
        except RuntimeError:
            result["missing_db_ids"].append(env_name)
            result["ok"] = False

    if not result["writeback_enabled"]:
        result["ok"] = False

    return result


# ------------------------------------------------------------------
# Evidence ID generation
# ------------------------------------------------------------------

def _generate_evidence_id(run_id: str, index: int) -> str:
    """Generate a stable Evidence ID for a run.

    NOTE: Currently uses run-local index for stability within a single run.
    This means re-ordering evidence items within the same run would change IDs.
    Future improvement: migrate to content-hash based IDs for cross-run stability.
    For now, this is treated as a run-local stable ID for upsert idempotency.
    """
    return f"{run_id}__ev_{index:04d}"


def _confidence_to_select(conf: float) -> str:
    """Convert numeric confidence to Notion select value."""
    if conf >= 0.8:
        return "high"
    elif conf >= 0.5:
        return "medium"
    else:
        return "low"


# ------------------------------------------------------------------
# Core writeback functions
# ------------------------------------------------------------------

def _write_evidence(
    evidence_repo: EvidenceRepo,
    evidence_items: List[Dict[str, Any]],
    run_id: str,
    now_iso: str,
) -> Dict[str, Any]:
    """Write evidence items to Evidence DB. Returns {page_ids, errors}."""
    page_ids: List[str] = []
    errors: List[str] = []

    for i, ev in enumerate(evidence_items):
        evidence_id = _generate_evidence_id(run_id, i)

        # Build tags: dimension + query_mode + block3
        tags = ["block3"]
        if ev.get("dimension"):
            tags.append(ev["dimension"])
        if ev.get("query_mode"):
            tags.append(ev["query_mode"])

        record = {
            "evidence_id": evidence_id,
            "statement": ev.get("claim_or_point", ""),
            "confidence": _confidence_to_select(ev.get("confidence", 0.5)),
            "confidence_reason": ev.get("relevance_to_rq", ""),
            "tags": tags,
            "extracted_at": now_iso,
        }

        try:
            page = evidence_repo.upsert_evidence(record)
            page_ids.append(page["id"])
            if (i + 1) % 20 == 0:
                logger.info("  Evidence: %d/%d written", i + 1, len(evidence_items))
        except Exception as e:
            logger.warning("  Evidence write failed [%d] %s: %s", i, evidence_id, e)
            errors.append(f"evidence:{evidence_id}:{e}")

    logger.info("Evidence: %d written, %d failed", len(page_ids), len(errors))
    return {"page_ids": page_ids, "errors": errors}


# ------------------------------------------------------------------
# Claims extraction + writing
# ------------------------------------------------------------------

def _generate_claim_id(run_id: str, category: str, index: int) -> str:
    """Generate a stable Claim ID for a run.

    NOTE: Currently uses run-local category + index for stability within a single run.
    Future improvement: migrate to canonical claim hash for cross-run dedup.
    For now, this is treated as a run-local stable ID for upsert idempotency.
    """
    return f"{run_id}__cl_{category}_{index:03d}"


def _extract_claims_from_lit_review(
    lit_review: Dict[str, Any],
    run_id: str,
    now_iso: str,
) -> List[Dict[str, Any]]:
    """Extract Claim records from lit_review.json findings."""
    claims = []
    findings = lit_review.get("empirical_findings", {})

    # Established findings → confidence=high
    for i, f in enumerate(findings.get("established", [])):
        papers_str = ", ".join(f.get("supporting_papers", [])[:5])
        reason = f.get("evidence_summary", "")
        if papers_str:
            reason += f"\n論文: {papers_str}"

        claims.append({
            "claim_id": _generate_claim_id(run_id, "established", i),
            "statement": f.get("statement", ""),
            "confidence": "high",
            "confidence_reason": reason[:2000],
            "tags": ["block3", "lit_review", "established"],
            "created_at": now_iso,
            "_category": "established",
        })

    # Emerging findings → confidence=medium
    for i, f in enumerate(findings.get("emerging", [])):
        papers_str = ", ".join(f.get("supporting_papers", [])[:5])
        reason = f.get("evidence_summary", "")
        if papers_str:
            reason += f"\n論文: {papers_str}"

        claims.append({
            "claim_id": _generate_claim_id(run_id, "emerging", i),
            "statement": f.get("statement", ""),
            "confidence": "medium",
            "confidence_reason": reason[:2000],
            "tags": ["block3", "lit_review", "emerging"],
            "created_at": now_iso,
            "_category": "emerging",
        })

    # Contested points → each position is a separate Claim, confidence=low
    for i, c in enumerate(findings.get("contested", [])):
        topic = c.get("topic", "")
        disagreement = c.get("nature_of_disagreement", "")
        for j, pos in enumerate(c.get("positions", [])):
            papers_str = ", ".join(pos.get("papers", [])[:5])
            reason = f"[論争: {topic}] {disagreement}"
            if papers_str:
                reason += f"\n論文: {papers_str}"

            claims.append({
                "claim_id": _generate_claim_id(run_id, "contested", i * 10 + j),
                "statement": pos.get("statement", ""),
                "confidence": "low",
                "confidence_reason": reason[:2000],
                "tags": ["block3", "lit_review", "contested"],
                "created_at": now_iso,
                "_category": "contested",
            })

    return claims


def _write_claims(
    claims_repo: ClaimsRepo,
    claims: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Write claims to Claims DB. Returns {page_ids, errors, by_category}."""
    page_ids: List[str] = []
    errors: List[str] = []
    by_category: Dict[str, int] = {}

    for claim in claims:
        category = claim.pop("_category", "unknown")
        claim_id = claim.get("claim_id", "")

        try:
            page = claims_repo.upsert_claim(claim)
            page_ids.append(page["id"])
            by_category[category] = by_category.get(category, 0) + 1
        except Exception as e:
            logger.warning("Claim write failed [%s] %s: %s", category, claim_id, e)
            errors.append(f"claim:{claim_id}:{e}")

    logger.info("Claims: %d written (%s), %d failed",
                len(page_ids), dict(by_category), len(errors))
    return {"page_ids": page_ids, "errors": errors, "by_category": by_category}


def _write_memo(
    memos_repo: MemosRepo,
    *,
    memo_id: str,
    title: str,
    summary: str,
    memo_type: str,
    body_md: str,
    run_page_id: Optional[str],
    evidence_page_ids: Optional[List[str]],
    now_iso: str,
) -> Dict[str, Any]:
    """Write a single memo. Returns {page_id, error}."""
    memo = {
        "memo_id": memo_id,
        "title": title,
        "summary": summary[:2000],
        "type": memo_type,
        "created_at": now_iso,
    }

    try:
        page = memos_repo.upsert_memo(
            memo,
            memo_body_md=body_md,
            run_page_id=run_page_id,
            evidence_page_ids=evidence_page_ids,
        )
        logger.info("Memo written: %s (page=%s)", memo_id, page["id"])
        return {"page_id": page["id"], "error": None}
    except Exception as e:
        logger.error("Memo write failed: %s: %s", memo_id, e)
        return {"page_id": None, "error": f"memo:{memo_id}:{e}"}


def _write_research_run(
    runs_repo: ResearchRunsRepo,
    *,
    run_id: str,
    rq_context: Dict[str, Any],
    status: str,
    started_at: str,
    completed_at: str,
    evidence_page_ids: Optional[List[str]],
    memo_page_ids: Optional[List[str]],
) -> Dict[str, Any]:
    """Write research run. Returns {page_id, error}."""
    request_text = rq_context.get("title", "")
    if rq_context.get("background"):
        request_text += f"\n背景: {rq_context['background']}"
    if rq_context.get("gap"):
        request_text += f"\nギャップ: {rq_context['gap']}"

    run = {
        "run_id": run_id,
        "request": request_text[:2000],
        "run_type": "lit_review",
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
    }

    try:
        page = runs_repo.upsert_run(
            run,
            evidence_page_ids=evidence_page_ids,
            memo_page_ids=memo_page_ids,
        )
        logger.info("Research Run written: %s (page=%s)", run_id, page["id"])
        return {"page_id": page["id"], "error": None}
    except Exception as e:
        logger.error("Research Run write failed: %s: %s", run_id, e)
        return {"page_id": None, "error": f"run:{run_id}:{e}"}


# ------------------------------------------------------------------
# Main orchestration
# ------------------------------------------------------------------

def write_to_notion(
    run_dir: Path,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Write Block 3 outputs from run_dir to KML Notion DBs.

    Writes in dependency order:
    1. Evidence → Evidence DB
    2. Memos → Memos DB
    3. Research Run → Research Runs DB (with relations)

    Parameters
    ----------
    run_dir:
        Path to run directory containing evidence.json, lit_review.md, landscape.md, rq_context.json.
    dry_run:
        If True, only report what would be written.

    Returns
    -------
    Dict with page counts, page IDs, and errors.
    """
    run_id = run_dir.name
    now_iso = datetime.now(timezone.utc).isoformat()

    # Load inputs
    rq_context = json.loads((run_dir / "rq_context.json").read_text())

    evidence_data = {}
    evidence_path = run_dir / "evidence.json"
    if evidence_path.exists():
        evidence_data = json.loads(evidence_path.read_text())
    evidence_items = evidence_data.get("evidence_items", [])

    lit_review_md = ""
    lit_review_md_path = run_dir / "lit_review.md"
    if lit_review_md_path.exists():
        lit_review_md = lit_review_md_path.read_text()

    lit_review_json = {}
    lit_review_json_path = run_dir / "lit_review.json"
    if lit_review_json_path.exists():
        lit_review_json = json.loads(lit_review_json_path.read_text())

    landscape_md = ""
    landscape_md_path = run_dir / "landscape.md"
    if landscape_md_path.exists():
        landscape_md = landscape_md_path.read_text()

    landscape_json = {}
    landscape_json_path = run_dir / "landscape.json"
    if landscape_json_path.exists():
        landscape_json = json.loads(landscape_json_path.read_text())

    # Build summary for landscape memo
    landscape_summary = ""
    if landscape_json:
        hotspots = landscape_json.get("hotspots", [])
        blindspots = landscape_json.get("blindspots", [])
        parts = []
        if hotspots:
            parts.append("Hotspots: " + "; ".join(h.get("area", "") for h in hotspots[:3]))
        if blindspots:
            parts.append("Blindspots: " + "; ".join(b.get("area", "") for b in blindspots[:3]))
        landscape_summary = ". ".join(parts)

    # Dry run report
    rq_title = rq_context.get("title", "")
    has_lit_review = bool(lit_review_md)
    has_landscape = bool(landscape_md)

    # Extract claims from lit review
    claims_to_write = []
    if lit_review_json:
        claims_to_write = _extract_claims_from_lit_review(lit_review_json, run_id, now_iso)

    if dry_run:
        n_memos = (1 if has_lit_review else 0) + (1 if has_landscape else 0)
        total_calls = 1 + len(evidence_items) + len(claims_to_write) + n_memos
        print(f"\n{'=' * 60}")
        print(f"Writeback DRY RUN — {run_id}")
        print(f"{'=' * 60}")
        print(f"RQ: {rq_title}")
        print(f"")
        print(f"Would write:")
        print(f"  Research Run: 1 record (run_id={run_id})")
        print(f"  Evidence:     {len(evidence_items)} records")
        print(f"  Claims:       {len(claims_to_write)} records")
        if claims_to_write:
            from collections import Counter
            cat_counts = Counter(c.get("_category", "?") for c in claims_to_write)
            for cat, count in sorted(cat_counts.items()):
                print(f"    {cat}: {count}")
        print(f"  Memo (Lit Review):  {'1 record' if has_lit_review else 'SKIP (no lit_review.md)'}")
        print(f"  Memo (Landscape):   {'1 record' if has_landscape else 'SKIP (no landscape.md)'}")
        print(f"")
        print(f"Total Notion API calls (estimate): {total_calls}")
        return {"dry_run": True, "claims_count": len(claims_to_write)}

    # Setup repos
    notion_client = build_notion_client_from_env()
    resolver = NotionDataSourceResolver(notion_client)

    ev_db_id = get_db_id(ENV_EVIDENCE_DB_ID)
    ev_resolved = resolver.resolve_once(name="EVIDENCE_DB", database_id=ev_db_id)
    evidence_repo = EvidenceRepo(
        client=notion_client, database_id=ev_db_id,
        data_source_id=ev_resolved.data_source_id,
    )

    cl_db_id = get_db_id(ENV_CLAIMS_DB_ID)
    cl_resolved = resolver.resolve_once(name="CLAIMS_DB", database_id=cl_db_id)
    claims_repo = ClaimsRepo(
        client=notion_client, database_id=cl_db_id,
        data_source_id=cl_resolved.data_source_id,
    )

    memos_db_id = get_db_id(ENV_MEMOS_DB_ID)
    memos_resolved = resolver.resolve_once(name="MEMOS_DB", database_id=memos_db_id)
    memos_repo = MemosRepo(
        client=notion_client, database_id=memos_db_id,
        data_source_id=memos_resolved.data_source_id,
    )

    runs_db_id = get_db_id(ENV_RESEARCH_RUNS_DB_ID)
    runs_resolved = resolver.resolve_once(name="RUNS_DB", database_id=runs_db_id)
    runs_repo = ResearchRunsRepo(
        client=notion_client, database_id=runs_db_id,
        data_source_id=runs_resolved.data_source_id,
    )

    all_errors: List[str] = []

    # Step 1: Evidence
    logger.info("Step 1/4: Writing %d evidence items", len(evidence_items))
    ev_result = _write_evidence(evidence_repo, evidence_items, run_id, now_iso)
    all_errors.extend(ev_result["errors"])

    # Step 2: Claims
    cl_result = {"page_ids": [], "errors": [], "by_category": {}}
    if claims_to_write:
        logger.info("Step 2/4: Writing %d claims", len(claims_to_write))
        cl_result = _write_claims(claims_repo, claims_to_write)
        all_errors.extend(cl_result["errors"])
    else:
        logger.info("Step 2/4: No claims to write (no lit_review.json)")

    # Step 3: Memos
    memo_page_ids: List[str] = []

    if has_lit_review:
        logger.info("Step 3a/4: Writing Lit Review memo")
        exec_summary = lit_review_json.get("executive_summary", "")
        lr_result = _write_memo(
            memos_repo,
            memo_id=f"{run_id}__lit_review",
            title=f"Lit Review: {rq_title[:60]}",
            summary=exec_summary[:2000],
            memo_type="lit_review",
            body_md=lit_review_md,
            run_page_id=None,  # Set after Research Run is created
            evidence_page_ids=ev_result["page_ids"][:100],  # Notion relation limit
            now_iso=now_iso,
        )
        if lr_result["page_id"]:
            memo_page_ids.append(lr_result["page_id"])
        if lr_result["error"]:
            all_errors.append(lr_result["error"])

    if has_landscape:
        logger.info("Step 3b/4: Writing Landscape memo")
        ls_result = _write_memo(
            memos_repo,
            memo_id=f"{run_id}__landscape",
            title=f"Landscape: {rq_title[:60]}",
            summary=landscape_summary[:2000],
            memo_type="research_landscape",
            body_md=landscape_md,
            run_page_id=None,
            evidence_page_ids=None,
            now_iso=now_iso,
        )
        if ls_result["page_id"]:
            memo_page_ids.append(ls_result["page_id"])
        if ls_result["error"]:
            all_errors.append(ls_result["error"])

    # Step 4: Research Run
    logger.info("Step 4/4: Writing Research Run")
    status = "completed" if not all_errors else "partial"
    run_result = _write_research_run(
        runs_repo,
        run_id=run_id,
        rq_context=rq_context,
        status=status,
        started_at=evidence_data.get("created_at", now_iso),
        completed_at=now_iso,
        evidence_page_ids=ev_result["page_ids"][:100],
        memo_page_ids=memo_page_ids,
    )
    # Update Research Run with claim relations (separate call to avoid overloading create)
    if run_result["page_id"] and cl_result["page_ids"]:
        try:
            notion_client.update_page(
                page_id=run_result["page_id"],
                properties={"Claims": {"relation": [
                    {"id": pid} for pid in cl_result["page_ids"][:100]
                ]}},
            )
        except Exception as e:
            logger.warning("Failed to link claims to run: %s", e)
    if run_result["error"]:
        all_errors.append(run_result["error"])

    # Step 4: Update Memos with Research Run relation
    if run_result["page_id"]:
        for memo_pid in memo_page_ids:
            try:
                notion_client.update_page(
                    page_id=memo_pid,
                    properties={"Research Run": {"relation": [{"id": run_result["page_id"]}]}},
                )
            except Exception as e:
                logger.warning("Failed to link memo %s to run: %s", memo_pid, e)

    result = {
        "run_id": run_id,
        "evidence_written": len(ev_result["page_ids"]),
        "evidence_failed": len(ev_result["errors"]),
        "claims_written": len(cl_result["page_ids"]),
        "claims_failed": len(cl_result["errors"]),
        "claims_by_category": cl_result.get("by_category", {}),
        "memos_written": len(memo_page_ids),
        "run_page_id": run_result["page_id"],
        "errors": all_errors,
        "status": "completed" if not all_errors else "partial",
    }

    logger.info(
        "Writeback complete: %d evidence, %d claims (%s), %d memos, run=%s, errors=%d",
        result["evidence_written"],
        result["claims_written"], dict(cl_result.get("by_category", {})),
        result["memos_written"],
        "OK" if run_result["page_id"] else "FAILED",
        len(all_errors),
    )

    return result
