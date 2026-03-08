"""072 Publish Deep Research Memo — service logic.

Reads plan.json, sources.json, credibility.json, claims.json,
generates an evidence-based memo, saves locally, and writes to
Notion Knowledge Memory Layer.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.deep_research import (
    DATA_DIR,
    load_step_output,
    make_memo_id,
    save_step_output,
)

logger = logging.getLogger("072_publisher")


# -- memo generation ----------------------------------------------------


def _build_summary(
    claims: List[Dict[str, Any]],
    plan: Dict[str, Any],
) -> str:
    """Generate a 2-3 sentence summary from high/medium confidence claims."""
    high_claims = [c for c in claims if c.get("confidence") == "high"]
    if not high_claims:
        high_claims = claims[:3]

    parts: List[str] = []
    for cl in high_claims[:3]:
        stmt = cl.get("statement", "")
        # Take first sentence only for summary
        first_sentence = stmt.split(";")[0].split(",")[0].strip()
        if first_sentence:
            parts.append(first_sentence)

    request = plan.get("request", "")
    if parts:
        return f"Research on \"{request}\": {'; '.join(parts)}."
    return f"Research on \"{request}\" completed with {len(claims)} claims."


def _build_memo_md(
    title: str,
    summary: str,
    claims: List[Dict[str, Any]],
    evidence_index: Dict[str, Dict[str, Any]],
    sources_index: Dict[str, Dict[str, Any]],
) -> str:
    """Generate the full memo in markdown format."""
    lines: List[str] = []

    # Title
    lines.append(f"# {title}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(summary)
    lines.append("")

    # Key Claims
    lines.append("## Key Claims")
    lines.append("")

    high_med_claims = [
        c for c in claims if c.get("confidence") in ("high", "medium")
    ]
    low_claims = [c for c in claims if c.get("confidence") == "low"]

    for i, cl in enumerate(high_med_claims, 1):
        conf = cl.get("confidence", "?")
        stmt = cl.get("statement", "")
        tags = ", ".join(cl.get("tags", []))
        lines.append(
            f"- **Claim {i}** [{conf}]: {stmt}"
        )
        if tags:
            lines.append(f"  - Tags: {tags}")
    lines.append("")

    # Supporting Evidence
    lines.append("## Supporting Evidence")
    lines.append("")

    for i, cl in enumerate(high_med_claims, 1):
        lines.append(f"### Claim {i}: {cl.get('statement', '')[:80]}")
        lines.append("")
        evidence_ids = cl.get("evidence_ids", [])
        for eid in evidence_ids:
            ev = evidence_index.get(eid, {})
            ev_stmt = ev.get("statement", eid)
            ev_conf = ev.get("confidence", "?")
            ev_source = ev.get("source_title", "unknown")
            lines.append(f"- [{ev_conf}] {ev_stmt} (source: {ev_source})")
        lines.append("")

    # Caveats / Uncertainty
    lines.append("## Caveats / Uncertainty")
    lines.append("")

    if low_claims:
        lines.append("**Low confidence claims:**")
        lines.append("")
        for cl in low_claims:
            lines.append(f"- {cl.get('statement', '')}")
            reason = cl.get("confidence_reason", "")
            if reason:
                lines.append(f"  - Reason: {reason}")
        lines.append("")

    # Check for evidence with only low confidence
    low_evidence = [
        ev for ev in evidence_index.values()
        if ev.get("confidence") == "low"
    ]
    if low_evidence:
        lines.append("**Low confidence evidence:**")
        lines.append("")
        for ev in low_evidence:
            lines.append(f"- {ev.get('statement', '')} (source: {ev.get('source_title', 'unknown')})")
        lines.append("")

    if not low_claims and not low_evidence:
        lines.append("- No significant caveats identified.")
        lines.append("")

    # Sources
    lines.append("## Sources")
    lines.append("")

    # Collect source_ids used in claims
    used_source_ids: List[str] = []
    for cl in claims:
        for sid in cl.get("source_ids", []):
            if sid not in used_source_ids:
                used_source_ids.append(sid)

    for sid in used_source_ids:
        src = sources_index.get(sid, {})
        title = src.get("title", sid)
        domain = src.get("domain", "")
        url = src.get("url", "")
        if url:
            lines.append(f"- {title} ({domain}) — {url}")
        else:
            lines.append(f"- {title} ({domain})")

    # Also list unused sources
    unused = [
        s for sid, s in sources_index.items()
        if sid not in used_source_ids
        and s.get("fetch_status") == "success"
    ]
    if unused:
        lines.append("")
        lines.append("*Additional sources (not cited in claims):*")
        for s in unused:
            lines.append(f"- {s.get('title', '')} ({s.get('domain', '')})")

    lines.append("")
    return "\n".join(lines)


def generate_memo(
    run_id: str,
    plan: Dict[str, Any],
    sources_data: Dict[str, Any],
    credibility_data: Dict[str, Any],
    claims_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate evidence-based memo from pipeline outputs.

    Returns dict with memo_id, title, summary, body (markdown),
    claim_ids, evidence_ids, source_ids, generated_at.
    """
    claims = claims_data.get("claims", [])
    evidence_list = credibility_data.get("annotated_evidence", [])
    sources_list = sources_data.get("sources", [])

    # Build indices
    evidence_index = {ev["evidence_id"]: ev for ev in evidence_list}
    sources_index = {s["source_id"]: s for s in sources_list}

    # Title
    request = plan.get("request", "")
    targets = plan.get("targets", [])
    target_str = ", ".join(targets) if targets else request[:40]
    title = f"Research Memo: {target_str}"

    # Summary
    summary = _build_summary(claims, plan)

    # Body
    body_md = _build_memo_md(title, summary, claims, evidence_index, sources_index)

    # Collect all IDs
    claim_ids = [c["claim_id"] for c in claims]
    evidence_ids = list(evidence_index.keys())
    source_ids = [s["source_id"] for s in sources_list if s.get("fetch_status") == "success"]

    memo_id = make_memo_id(run_id, 1)
    now = datetime.now().isoformat()

    return {
        "run_id": run_id,
        "memo_id": memo_id,
        "type": "research memo",
        "title": title,
        "summary": summary,
        "body": body_md,
        "claim_ids": claim_ids,
        "evidence_ids": evidence_ids,
        "source_ids": source_ids,
        "generated_at": now,
    }


# -- local save ----------------------------------------------------------


def save_memo_local(
    run_id: str,
    memo: Dict[str, Any],
) -> Dict[str, Path]:
    """Save memo.md and memo.json locally.

    Returns dict with paths: {"md": Path, "json": Path}.
    """
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save memo.md
    md_path = run_dir / "memo.md"
    md_path.write_text(memo["body"], encoding="utf-8")
    logger.info("Saved memo.md: %s", md_path)

    # Save memo.json (without body to avoid duplication)
    memo_json = {
        "run_id": memo["run_id"],
        "memo_id": memo["memo_id"],
        "type": memo["type"],
        "title": memo["title"],
        "summary": memo["summary"],
        "claim_ids": memo["claim_ids"],
        "evidence_ids": memo["evidence_ids"],
        "source_ids": memo["source_ids"],
        "generated_at": memo["generated_at"],
    }
    json_path = save_step_output(run_id, "072", memo_json)
    logger.info("Saved memo.json: %s", json_path)

    return {"md": md_path, "json": json_path}


# -- preflight check -----------------------------------------------------


def preflight_check() -> Dict[str, Any]:
    """Check all prerequisites for Notion writeback.

    Returns dict with 'ok' bool, 'writeback_enabled', 'missing_db_ids'.
    """
    import os
    from src.notion.research_schema import (
        ENV_SOURCES_DB_ID,
        ENV_EVIDENCE_DB_ID,
        ENV_CLAIMS_DB_ID,
        ENV_MEMOS_DB_ID,
        ENV_RESEARCH_RUNS_DB_ID,
    )

    writeback = os.getenv("ENABLE_NOTION_WRITEBACK", "false").strip().lower() == "true"
    token = bool(os.getenv("NOTION_TOKEN", "").strip())

    required_vars = [
        ENV_SOURCES_DB_ID,
        ENV_EVIDENCE_DB_ID,
        ENV_CLAIMS_DB_ID,
        ENV_MEMOS_DB_ID,
        ENV_RESEARCH_RUNS_DB_ID,
    ]
    missing = [v for v in required_vars if not os.getenv(v, "").strip()]

    ok = writeback and token and not missing
    return {
        "ok": ok,
        "writeback_enabled": writeback,
        "notion_token": token,
        "missing_db_ids": missing,
    }


# -- Notion writeback ----------------------------------------------------


def _build_repos(notion_client: Any) -> Dict[str, Any]:
    """Build all 5 research repos from environment variables."""
    from src.config import get_db_id
    from src.notion.sources_repo import SourcesRepo
    from src.notion.evidence_repo import EvidenceRepo
    from src.notion.claims_repo import ClaimsRepo
    from src.notion.memos_repo import MemosRepo
    from src.notion.research_runs_repo import ResearchRunsRepo
    from src.notion.research_schema import (
        ENV_SOURCES_DB_ID,
        ENV_EVIDENCE_DB_ID,
        ENV_CLAIMS_DB_ID,
        ENV_MEMOS_DB_ID,
        ENV_RESEARCH_RUNS_DB_ID,
    )

    def _resolve_data_source_id(db_id: str) -> str:
        """Resolve data_source_id from database metadata."""
        try:
            db_meta = notion_client.get_database(database_id=db_id)
            ds_list = db_meta.get("data_sources", [])
            if ds_list:
                return ds_list[0]["id"]
        except Exception:
            pass
        return db_id  # fallback

    def _make_repo(cls, env_name):
        db_id = get_db_id(env_name)
        ds_id = _resolve_data_source_id(db_id)
        logger.debug("Repo %s: db=%s, ds=%s", cls.__name__, db_id[:8], ds_id[:8])
        return cls(client=notion_client, database_id=db_id, data_source_id=ds_id)

    return {
        "sources": _make_repo(SourcesRepo, ENV_SOURCES_DB_ID),
        "evidence": _make_repo(EvidenceRepo, ENV_EVIDENCE_DB_ID),
        "claims": _make_repo(ClaimsRepo, ENV_CLAIMS_DB_ID),
        "memos": _make_repo(MemosRepo, ENV_MEMOS_DB_ID),
        "runs": _make_repo(ResearchRunsRepo, ENV_RESEARCH_RUNS_DB_ID),
    }


def publish_to_notion(
    memo: Dict[str, Any],
    plan: Dict[str, Any],
    sources_data: Dict[str, Any],
    credibility_data: Dict[str, Any],
    claims_data: Dict[str, Any],
    notion_client: Any,
) -> Dict[str, Any]:
    """Write all entities to Notion in dependency order.

    Returns mapping result with page IDs and ID mappings.
    """
    repos = _build_repos(notion_client)

    sources_list = sources_data.get("sources", [])
    evidence_list = credibility_data.get("annotated_evidence", [])
    claims = claims_data.get("claims", [])

    # Result tracking
    result: Dict[str, Any] = {
        "source_id_map": {},   # local source_id → Notion page_id
        "evidence_id_map": {}, # local evidence_id → Notion page_id
        "claim_id_map": {},    # local claim_id → Notion page_id
        "memo_page_id": None,
        "run_page_id": None,
        "errors": [],
    }

    # --------------------------------------------------
    # 1. Sources (upsert)
    # --------------------------------------------------
    logger.info("Writing %d sources to Notion...", len(sources_list))
    for src in sources_list:
        source_id = src.get("source_id", "")
        try:
            page = repos["sources"].upsert_source(src)
            result["source_id_map"][source_id] = page["id"]
            logger.debug("Source %s → %s", source_id, page["id"])
        except Exception as e:
            logger.warning("Failed to write source %s: %s", source_id, e)
            result["errors"].append(f"source:{source_id}:{e}")

    # --------------------------------------------------
    # 2. Evidence (upsert)
    # --------------------------------------------------
    logger.info("Writing %d evidence to Notion...", len(evidence_list))
    for ev in evidence_list:
        evidence_id = ev.get("evidence_id", "")
        source_page_id = result["source_id_map"].get(ev.get("source_id", ""))
        try:
            page = repos["evidence"].upsert_evidence(
                ev,
                source_page_id=source_page_id,
            )
            result["evidence_id_map"][evidence_id] = page["id"]
            logger.debug("Evidence %s → %s", evidence_id, page["id"])
        except Exception as e:
            logger.warning("Failed to write evidence %s: %s", evidence_id, e)
            result["errors"].append(f"evidence:{evidence_id}:{e}")

    # --------------------------------------------------
    # 3. Claims (upsert)
    # --------------------------------------------------
    logger.info("Writing %d claims to Notion...", len(claims))
    for cl in claims:
        claim_id = cl.get("claim_id", "")
        ev_page_ids = [
            result["evidence_id_map"][eid]
            for eid in cl.get("evidence_ids", [])
            if eid in result["evidence_id_map"]
        ]
        src_page_ids = [
            result["source_id_map"][sid]
            for sid in cl.get("source_ids", [])
            if sid in result["source_id_map"]
        ]
        try:
            page = repos["claims"].upsert_claim(
                cl,
                evidence_page_ids=ev_page_ids,
                source_page_ids=src_page_ids,
            )
            result["claim_id_map"][claim_id] = page["id"]
            logger.debug("Claim %s → %s", claim_id, page["id"])
        except Exception as e:
            logger.warning("Failed to write claim %s: %s", claim_id, e)
            result["errors"].append(f"claim:{claim_id}:{e}")

    # --------------------------------------------------
    # 4. Memo (upsert)
    # --------------------------------------------------
    logger.info("Writing memo to Notion...")
    claim_page_ids = list(result["claim_id_map"].values())
    evidence_page_ids = list(result["evidence_id_map"].values())
    source_page_ids = list(result["source_id_map"].values())

    try:
        page = repos["memos"].upsert_memo(
            memo,
            memo_body_md=memo.get("body", ""),
            claim_page_ids=claim_page_ids,
            evidence_page_ids=evidence_page_ids,
            source_page_ids=source_page_ids,
        )
        result["memo_page_id"] = page["id"]
        logger.info("Memo → %s", page["id"])
    except Exception as e:
        logger.warning("Failed to write memo: %s", e)
        result["errors"].append(f"memo:{e}")

    # --------------------------------------------------
    # 5. Research Run (upsert)
    # --------------------------------------------------
    logger.info("Writing research run to Notion...")
    now = datetime.now().isoformat()
    run_data = {
        "run_id": memo["run_id"],
        "request": plan.get("request", ""),
        "run_type": "deep_research",
        "status": "completed" if not result["errors"] else "partial",
        "started_at": plan.get("created_at") or sources_data.get("collected_at"),
        "completed_at": now,
    }

    memo_page_ids = [result["memo_page_id"]] if result["memo_page_id"] else []

    try:
        page = repos["runs"].upsert_run(
            run_data,
            source_page_ids=source_page_ids,
            evidence_page_ids=evidence_page_ids,
            claim_page_ids=claim_page_ids,
            memo_page_ids=memo_page_ids,
        )
        result["run_page_id"] = page["id"]
        logger.info("Research Run → %s", page["id"])
    except Exception as e:
        logger.warning("Failed to write research run: %s", e)
        result["errors"].append(f"run:{e}")

    # --------------------------------------------------
    # 6. Update memo with run relation (if both exist)
    # --------------------------------------------------
    if result["memo_page_id"] and result["run_page_id"]:
        try:
            from src.notion.research_schema import MEMO_PROP_RESEARCH_RUN
            from src.notion.client import normalize_uuid
            notion_client.update_page(
                page_id=result["memo_page_id"],
                properties={
                    MEMO_PROP_RESEARCH_RUN: {
                        "relation": [{"id": normalize_uuid(result["run_page_id"])}],
                    },
                },
            )
            logger.debug("Linked memo to research run")
        except Exception as e:
            logger.warning("Failed to link memo to run: %s", e)

    return result


def log_writeback_preview(
    memo: Dict[str, Any],
    sources_data: Dict[str, Any],
    credibility_data: Dict[str, Any],
    claims_data: Dict[str, Any],
) -> None:
    """Log what would be written to Notion (dry-run mode)."""
    sources = sources_data.get("sources", [])
    evidence = credibility_data.get("annotated_evidence", [])
    claims = claims_data.get("claims", [])

    logger.info("=== Notion Writeback Preview (dry-run) ===")
    logger.info("  Sources:  %d pages to upsert", len(sources))
    for s in sources:
        logger.info("    %s: %s (%s)", s.get("source_id"), s.get("title", ""), s.get("domain", ""))
    logger.info("  Evidence: %d pages to create", len(evidence))
    for ev in evidence:
        logger.info("    %s: %s [%s]", ev.get("evidence_id"), ev.get("statement", "")[:60], ev.get("confidence", ""))
    logger.info("  Claims:   %d pages to create", len(claims))
    for cl in claims:
        logger.info("    %s: %s [%s]", cl.get("claim_id"), cl.get("statement", "")[:60], cl.get("confidence", ""))
    logger.info("  Memo:     1 page to create — %s", memo.get("title", ""))
    logger.info("  Run:      1 page to create — %s", memo.get("run_id", ""))
    logger.info("=== End Preview ===")


# -- entry point ---------------------------------------------------------


def run(
    run_id: str,
    notion_client: Any = None,
    enable_writeback: bool = False,
) -> Dict[str, Any]:
    """Execute the full 072 publish pipeline.

    1. Load all pipeline outputs.
    2. Generate evidence-based memo.
    3. Save locally (memo.md + memo.json).
    4. Optionally write to Notion.

    Args:
        run_id: Research run identifier.
        notion_client: Optional NotionClient. Required if enable_writeback=True.
        enable_writeback: If True, write to Notion DBs.

    Returns:
        Dict with memo data, local paths, and optional Notion result.
    """
    # 1. Load inputs
    plan = load_step_output(run_id, "067")
    sources_data = load_step_output(run_id, "068")
    credibility_data = load_step_output(run_id, "070")
    claims_data = load_step_output(run_id, "071")

    logger.info(
        "=== 072 Publisher: run_id=%s, claims=%d, evidence=%d, sources=%d ===",
        run_id,
        len(claims_data.get("claims", [])),
        len(credibility_data.get("annotated_evidence", [])),
        len(sources_data.get("sources", [])),
    )

    # 2. Generate memo
    memo = generate_memo(run_id, plan, sources_data, credibility_data, claims_data)
    logger.info("Memo generated: %s", memo["title"])

    # 3. Save locally
    paths = save_memo_local(run_id, memo)

    # 4. Notion writeback
    notion_result = None
    if enable_writeback and notion_client is not None:
        logger.info("Notion writeback enabled — publishing...")
        try:
            notion_result = publish_to_notion(
                memo, plan, sources_data, credibility_data, claims_data, notion_client,
            )
            errors = notion_result.get("errors", [])
            if errors:
                logger.warning("Writeback completed with %d errors", len(errors))
            else:
                logger.info("Writeback completed successfully")
        except Exception as e:
            logger.error("Writeback failed: %s", e)
            notion_result = {"error": str(e)}
    else:
        # Dry-run: log preview
        log_writeback_preview(memo, sources_data, credibility_data, claims_data)

    result = {
        "run_id": run_id,
        "memo": memo,
        "paths": {k: str(v) for k, v in paths.items()},
        "writeback_enabled": enable_writeback,
        "notion_result": notion_result,
    }

    logger.info("=== 072 Publisher done ===")
    return result
