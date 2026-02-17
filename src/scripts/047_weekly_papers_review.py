#!/usr/bin/env python
# src/scripts/047_weekly_papers_review.py
"""Weekly papers review — fetch, classify via LLM, persist.

Responsibilities:

1. **Ensure a WEEKLY_DIGESTS_DB row** for ``{week_id}`` (upsert).
   This establishes the weekly context that 052 later enriches.
2. **Fetch weekly papers** from LIT DB.
3. **Use OpenAI to classify each paper**: READ | KEEP | SKIP.
4. **Write Decision + Decision Reason back to LIT DB** per paper.
5. **Persist local artifacts** (JSON + Markdown).

Theme clustering is handled by **048** (event-based), NOT here.

Usage::

    # Dry-run (default) — prints plan, no API calls
    python -m src.scripts.047_weekly_papers_review

    # Live run — full pipeline with LLM + Notion writes
    python -m src.scripts.047_weekly_papers_review --run

    # Live run without Notion writes (debug)
    python -m src.scripts.047_weekly_papers_review --run --no-write
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# ----------------------------------------------------------------
# sys.path bridge (INTERIM)
# ----------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    RunMetadata,
    WeekContext,
    get_db_id,
    get_output_dir,
    get_week_context,
    get_iso_week_context,
    load_env,
    setup_logging,
)
from src.notion import (
    NotionClient,
    NotionDataSourceResolver,
    build_notion_client_from_env,
)
from src.notion.papers_schema import (
    PAPERS_REQUIRED_PROPERTIES,
)
from src.notion.properties import page_to_record
from src.notion.weekly_digests_repo import WeeklyDigestsRepo
from src.llm.openai_client import (
    build_openai_client_from_env,
)

logger = logging.getLogger("047_weekly_papers_review")

SCRIPT_NAME = "047_weekly_papers_review"
JST = ZoneInfo("Asia/Tokyo")

# Maximum papers per LLM batch (to stay within context limits)
LLM_BATCH_SIZE = 10


# ================================================================
# Core pipeline steps
# ================================================================

def resolve_papers_data_source(
    client: NotionClient,
    db_id: str,
) -> str:
    resolver = NotionDataSourceResolver(client)
    resolved = resolver.resolve_once(name="PAPERS_DB", database_id=db_id)
    logger.info(
        "Resolved data_source_id=%s for PAPERS_DB (database_id=%s)",
        resolved.data_source_id, resolved.database_id,
    )
    return resolved.data_source_id


def fetch_recent_papers(
    client: NotionClient,
    data_source_id: str,
    *,
    since: datetime,
    date_property: str = "Ingested At",
) -> List[dict]:
    filt = {
        "property": date_property,
        "date": {"on_or_after": since.isoformat()},
    }
    sorts = [{"property": date_property, "direction": "descending"}]
    logger.info("Fetching papers where '%s' >= %s …", date_property, since.strftime("%Y-%m-%d"))
    pages = client.query_data_source(
        data_source_id=data_source_id, filter=filt, sorts=sorts, fetch_all=True,
    )
    logger.info("Fetched %d papers.", len(pages))
    return pages


def normalize_papers(pages: List[dict], property_names: List[str]) -> List[Dict[str, Any]]:
    records = [page_to_record(p, property_names) for p in pages]
    logger.info("Normalised %d paper records.", len(records))
    return records


# ================================================================
# LLM paper classification
# ================================================================

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a research intelligence analyst specializing in startup ecosystems, \
venture capital dynamics, and entrepreneurship policy.

For each paper below, classify it into one of three categories:
- READ: Strongly aligned with high-priority research questions. Should be read this week.
- KEEP: Relevant but not urgent. Archive for reference.
- SKIP: Weak alignment. Low value for the current research program.

Criteria:
- READ: directly addresses RQ gaps, provides novel data on VC/startup dynamics, \
  or offers new policy frameworks.
- KEEP: covers adjacent topics or useful background.
- SKIP: tangential, duplicates known findings, or unrelated domains.

Return a JSON object with key "classifications" containing an array of objects, \
one per paper, each with:
  - paper_id: string (the page_id provided)
  - decision: string (READ | KEEP | SKIP)
  - reason: string (one sentence explaining the classification)
"""


_SKIP_FALLBACK = {"decision": "SKIP", "reason": "LLM did not return classification; auto-filled as SKIP."}

_MAX_BATCH_RETRIES = 1  # retry once on mismatch before falling back


def _build_batch_payload(batch):
    """Build the JSON payload list for a batch of paper records."""
    return [
        {
            "paper_id": r.get("notion_page_id", ""),
            "title": r.get("Name", ""),
            "core_idea": (r.get("Core Idea") or "")[:500],
            "findings": (r.get("Findings") or "")[:500],
            "source": r.get("Source", ""),
            "tags": r.get("Tags", ""),
        }
        for r in batch
    ]


def _call_llm_for_batch(llm, papers_payload, rq_context):
    """Send one batch to OpenAI and return the raw classifications list."""
    user_prompt = (
        f"Classify these {len(papers_payload)} papers. "
        f"You MUST return exactly {len(papers_payload)} classifications, "
        f"one per paper_id provided.{rq_context}\n\n"
        + json.dumps(papers_payload, indent=2, ensure_ascii=False)
    )
    resp = llm.call_json(system=CLASSIFICATION_SYSTEM_PROMPT, user=user_prompt)
    return resp.parsed.get("classifications", [])


def _reconcile_batch(classifications, expected_ids):
    """Match LLM classifications to expected paper_ids.

    Returns a dict ``{paper_id: {"decision": ..., "reason": ...}}``
    covering **every** id in *expected_ids*.  Missing ids are filled
    with ``_SKIP_FALLBACK``.
    """
    returned: Dict[str, Dict[str, str]] = {}
    for c in classifications:
        pid = c.get("paper_id", "")
        if pid and pid in expected_ids:
            returned[pid] = {
                "decision": c.get("decision", "SKIP"),
                "reason": c.get("reason", ""),
            }

    # Fill any missing ids with deterministic SKIP fallback
    missing_ids = expected_ids - returned.keys()
    for pid in sorted(missing_ids):
        logger.warning("Paper %s missing from LLM response — auto-filling SKIP", pid)
        returned[pid] = dict(_SKIP_FALLBACK)

    return returned


def classify_papers_llm(llm, records, *, rq_titles=None):
    """Classify papers via OpenAI in batches.

    **Hard guarantee:** returns exactly one classification per paper in
    *records*.  If the LLM omits a paper, we retry once and then
    deterministically fill the gap with SKIP.

    Returns dict: paper_page_id -> {"decision": ..., "reason": ...}.

    Raises
    ------
    RuntimeError
        If, after retry + fallback, the count still doesn't match (should
        never happen, but guards against logic bugs).
    """
    rq_context = ""
    if rq_titles:
        rq_list = "\n".join(f"  - {t}" for t in rq_titles[:20])
        rq_context = f"\n\nCurrent high-priority Research Questions:\n{rq_list}\n"

    results: Dict[str, Dict[str, str]] = {}

    for batch_start in range(0, len(records), LLM_BATCH_SIZE):
        batch = records[batch_start:batch_start + LLM_BATCH_SIZE]
        papers_payload = _build_batch_payload(batch)
        expected_ids = {p["paper_id"] for p in papers_payload}

        # --- First attempt ---
        classifications = _call_llm_for_batch(llm, papers_payload, rq_context)
        returned_ids = {c.get("paper_id", "") for c in classifications}
        missing_ids = expected_ids - returned_ids

        # --- Retry once if mismatch ---
        if missing_ids and _MAX_BATCH_RETRIES > 0:
            logger.warning(
                "LLM batch mismatch (attempt 1): sent %d, got %d — "
                "retrying for %d missing papers",
                len(batch), len(returned_ids & expected_ids), len(missing_ids),
            )
            retry_classifications = _call_llm_for_batch(
                llm, papers_payload, rq_context,
            )
            # Merge: prefer retry results for missing ids, keep originals for rest
            retry_by_id = {
                c.get("paper_id", ""): c for c in retry_classifications
            }
            for pid in missing_ids:
                if pid in retry_by_id:
                    classifications.append(retry_by_id[pid])
            # Recalculate
            returned_ids = {c.get("paper_id", "") for c in classifications}
            still_missing = expected_ids - returned_ids
            if still_missing:
                logger.warning(
                    "Still missing %d papers after retry — will SKIP-fill",
                    len(still_missing),
                )

        # --- Reconcile: match + fill ---
        batch_results = _reconcile_batch(classifications, expected_ids)

        # --- Hard assert ---
        if len(batch_results) != len(batch):
            raise RuntimeError(
                f"classify_papers_llm: FATAL — batch expected {len(batch)} "
                f"classifications but reconciled {len(batch_results)}.  "
                f"expected_ids={sorted(expected_ids)}, "
                f"got_ids={sorted(batch_results.keys())}"
            )

        results.update(batch_results)

        logger.info(
            "LLM classified batch %d-%d (%d/%d from LLM, %d SKIP-filled)",
            batch_start + 1,
            min(batch_start + LLM_BATCH_SIZE, len(records)),
            len(batch_results) - sum(
                1 for v in batch_results.values()
                if v.get("reason") == _SKIP_FALLBACK["reason"]
            ),
            len(batch),
            sum(
                1 for v in batch_results.values()
                if v.get("reason") == _SKIP_FALLBACK["reason"]
            ),
        )

    return results


# ================================================================
# LIT DB write-back (Decision + Decision Reason)
# ================================================================

def write_decisions_to_lit_db(client, classifications, records):
    """Update each paper in LIT DB with Decision + Decision Reason. Returns count."""
    count = 0
    for r in records:
        page_id = r.get("notion_page_id", "")
        if not page_id or page_id not in classifications:
            continue
        cls = classifications[page_id]
        try:
            client.update_page(
                page_id=page_id,
                properties={
                    "Decision": {"select": {"name": cls["decision"]}},
                    "Decision Reason": {"rich_text": [{"type": "text", "text": {"content": cls["reason"][:2000]}}]},
                },
            )
            count += 1
        except Exception as e:
            logger.warning("Failed to update Decision for paper %s: %s", page_id, e)
    logger.info("Updated Decision on %d / %d papers in LIT DB", count, len(records))
    return count


# ================================================================
# Output writers
# ================================================================

def write_papers_json(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False, default=str) + "\n")
    logger.info("Wrote %d papers to %s", len(records), path)


def write_summary_md(records, classifications, wk, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    decision_counts = {"READ": 0, "KEEP": 0, "SKIP": 0}
    for cls in classifications.values():
        d = cls.get("decision", "SKIP")
        decision_counts[d] = decision_counts.get(d, 0) + 1

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Weekly Papers Review — {wk.week_id}\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write("## Classification Summary\n\n")
        f.write(f"- **Total papers:** {len(records)}\n")
        for d in ["READ", "KEEP", "SKIP"]:
            f.write(f"- **{d}:** {decision_counts.get(d, 0)}\n")
        f.write("\n")

        for decision in ["READ", "KEEP", "SKIP"]:
            matching = [
                r for r in records
                if classifications.get(r.get("notion_page_id", ""), {}).get("decision") == decision
            ]
            if not matching:
                continue
            f.write(f"## {decision} ({len(matching)} papers)\n\n")
            for r in matching:
                name = r.get("Name", "(untitled)")
                reason = classifications.get(r.get("notion_page_id", ""), {}).get("reason", "")
                source = r.get("Source", "")
                f.write(f"- **{name}**")
                if source:
                    f.write(f" ({source})")
                if reason:
                    f.write(f" — {reason}")
                f.write("\n")
            f.write("\n")

        f.write(f"---\n\n*Generated by {SCRIPT_NAME}*\n")
    logger.info("Wrote summary to %s", path)


# ================================================================
# CLI
# ================================================================

def build_parser():
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Weekly papers review: LLM classification + Notion write-back.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", default=False, help="Execute live pipeline.")
    mode.add_argument("--dry-run", action="store_true", default=True, help="Print plan (default).")
    p.add_argument("--days", type=int, default=7, help="Lookback window (default: 7).")
    p.add_argument("--date-property", default="Ingested At", help="Date property (default: 'Ingested At').")
    p.add_argument("--output-base", default="outputs", help="Output base dir (default: outputs/).")
    p.add_argument("--write", action="store_true", default=False, help="Persist to Notion (default: off).")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    return p


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    is_live = args.run
    write_enabled = args.write
    if args.run:
        args.dry_run = False

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    load_env()

    wk = get_week_context()
    wk_iso = get_iso_week_context()
    db_id = get_db_id("NOTION_LIT_DB_ID")

    date_to_utc = wk.now_utc
    date_from_utc = date_to_utc - timedelta(days=args.days)
    date_from_iso = date_from_utc.isoformat(timespec="seconds")
    date_to_iso = date_to_utc.isoformat(timespec="seconds")

    out_dir = get_output_dir(SCRIPT_NAME, wk.week_id, base=args.output_base, create=is_live)

    logger.info("=== %s ===", SCRIPT_NAME)
    logger.info("Week: %s  |  Lookback: %d days", wk.week_id, args.days)
    logger.info("Date window (UTC): %s → %s", date_from_iso, date_to_iso)
    logger.info("Output: %s", out_dir)
    logger.info("Mode: %s  |  Write: %s", "LIVE" if is_live else "DRY-RUN", write_enabled)

    result: Dict[str, Any] = {
        "ok": False, "week_id": wk.week_id, "output_dir": str(out_dir),
        "summary": {}, "errors": [],
    }

    if not is_live:
        logger.info("[DRY-RUN] Would fetch papers from NOTION_LIT_DB_ID=%s", db_id)
        logger.info("[DRY-RUN] Would classify via OpenAI + write Decision to LIT DB")
        logger.info("[DRY-RUN] Would upsert digest row in WEEKLY_DIGESTS_DB")
        logger.info("[DRY-RUN] Pass --run to execute.")
        result["ok"] = True
        return result

    # ---- Live pipeline ----
    import uuid as _uuid
    now_jst = datetime.now(JST)
    run_id = str(_uuid.uuid4())[:8]

    client = build_notion_client_from_env()
    llm = build_openai_client_from_env(
        cache_dir=Path(args.output_base) / "weekly" / wk.week_id / ".cache" / "llm",
    )

    # 1) Ensure WEEKLY_DIGESTS_DB row
    digest_page_id = None
    if write_enabled:
        try:
            digests_db_id = get_db_id("NOTION_WEEKLY_DIGESTS_DB_ID")
            digests_resolver = NotionDataSourceResolver(client)
            digests_resolved = digests_resolver.resolve_once(name="WEEKLY_DIGESTS_DB", database_id=digests_db_id)
            digests_repo = WeeklyDigestsRepo(
                client=client, database_id=digests_resolved.database_id,
                data_source_id=digests_resolved.data_source_id,
            )
            digests_repo.validate_schema()
            key, props = digests_repo.build_digest_properties(
                week_id=wk.week_id,
                week_start=wk_iso.start_date if hasattr(wk_iso, "start_date") else "",
                week_end=wk_iso.end_date if hasattr(wk_iso, "end_date") else "",
                run_id=run_id, now_jst=now_jst,
            )
            page = digests_repo.upsert_row(key=key, properties=props)
            digest_page_id = page.get("id")
            logger.info("Ensured WEEKLY_DIGESTS_DB row for %s (page_id=%s)", wk.week_id, digest_page_id)
        except Exception as e:
            logger.warning("Failed to ensure WEEKLY_DIGESTS_DB row: %s", e)
            result["errors"].append(f"Digest row creation failed: {e}")

    # 2) Fetch papers
    ds_id = resolve_papers_data_source(client, db_id)
    raw_pages = fetch_recent_papers(client, ds_id, since=date_from_utc, date_property=args.date_property)
    property_names = list(PAPERS_REQUIRED_PROPERTIES.keys())
    records = normalize_papers(raw_pages, property_names)

    # 3) LLM classification
    classifications: Dict[str, Dict[str, str]] = {}
    if records:
        classifications = classify_papers_llm(llm, records)
        dc = {"READ": 0, "KEEP": 0, "SKIP": 0}
        for cls in classifications.values():
            dc[cls.get("decision", "SKIP")] = dc.get(cls.get("decision", "SKIP"), 0) + 1
        logger.info("OpenAI: classified %d papers (READ:%d, KEEP:%d, SKIP:%d)",
                     len(classifications), dc["READ"], dc["KEEP"], dc["SKIP"])

    # 4) Write Decision back to LIT DB
    decisions_written = 0
    if write_enabled and classifications:
        decisions_written = write_decisions_to_lit_db(client, classifications, records)

    # 5) Write local outputs
    write_papers_json(records, out_dir / "papers.json")
    write_summary_md(records, classifications, wk, out_dir / "summary.md")
    (out_dir / "classifications.json").write_text(
        json.dumps(classifications, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    # 6) Run metadata
    paper_page_ids = [r.get("notion_page_id", "") for r in records if r.get("notion_page_id")]

    meta = RunMetadata.build(
        notebook=SCRIPT_NAME, week_id=wk.week_id, date_from=date_from_iso, date_to=date_to_iso,
        counts={
            "papers_fetched": len(raw_pages), "papers_normalised": len(records),
            "papers_read": sum(1 for c in classifications.values() if c.get("decision") == "READ"),
            "papers_keep": sum(1 for c in classifications.values() if c.get("decision") == "KEEP"),
            "papers_skip": sum(1 for c in classifications.values() if c.get("decision") == "SKIP"),
        },
        extra={
            "date_property": args.date_property, "data_source_id": ds_id,
            "write_enabled": write_enabled, "decisions_written": decisions_written,
            "digest_page_id": digest_page_id,
        },
    )
    meta.save(out_dir / "run_metadata.json")

    result["ok"] = True
    result["summary"] = {
        "papers_fetched": len(records),
        "classifications": {d: sum(1 for c in classifications.values() if c.get("decision") == d) for d in ["READ", "KEEP", "SKIP"]},
        "decisions_written": decisions_written,
        "digest_page_id": digest_page_id,
        "paper_page_ids": paper_page_ids,
    }
    logger.info("=== Done: %d papers classified → %s ===", len(records), out_dir)
    return result


if __name__ == "__main__":
    r = main()
    raise SystemExit(0 if r["ok"] else 1)
