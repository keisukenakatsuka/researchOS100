# src/question/writer.py
"""105 RQ Writeback — service logic (optional).

Writes promoted RQ candidates to Notion RQ DB.
Only runs when ENABLE_NOTION_WRITEBACK=true.

Also provides `export_promoted_for_next_run()` which converts
promoted candidates into rq_context.json format for 079 input.
This works WITHOUT Notion and is the primary way to close the cycle.

Usage::

    from src.question.writer import writeback_promoted, export_promoted_for_next_run

    # Option A: Notion writeback (optional)
    result = writeback_promoted(portfolio_path)

    # Option B: Export for next run (always works)
    contexts = export_promoted_for_next_run(portfolio_path, candidates_path)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class WritebackEntry:
    candidate_id: str
    title: str
    notion_page_id: Optional[str] = None
    status: str = "pending"   # pending | written | skipped | error
    error: str = ""


@dataclass
class WritebackResult:
    status: str = "failed"
    entries: List[WritebackEntry] = field(default_factory=list)
    promoted_contexts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "entries": [asdict(e) for e in self.entries],
            "promoted_contexts": self.promoted_contexts,
            "error": self.error,
        }


# ------------------------------------------------------------------
# Export for next run (no Notion, always works)
# ------------------------------------------------------------------

def export_promoted_for_next_run(
    portfolio_path: Path,
    candidates_path: Path,
) -> List[Dict[str, Any]]:
    """Convert promoted candidates to rq_context.json format.

    Returns a list of RQContext-compatible dicts, each ready to be
    saved as rq_context.json for a new 079 run.
    """
    portfolio = json.loads(portfolio_path.read_text())
    candidates_data = json.loads(candidates_path.read_text())

    # Build candidate lookup
    cand_map: Dict[str, Dict] = {}
    for c in candidates_data.get("candidates", []):
        cand_map[c.get("candidate_id", "")] = c

    parent_run_id = candidates_data.get("parent_run_id", "")

    contexts: List[Dict[str, Any]] = []
    for entry in portfolio.get("portfolio", []):
        if entry.get("recommendation") != "promote":
            continue

        cid = entry.get("candidate_id", "")
        cand = cand_map.get(cid, {})

        context = {
            "rq_id": None,  # will be assigned by Notion or user
            "title": cand.get("question", cand.get("title", "")),
            "background": cand.get("background", ""),
            "gap": cand.get("gap", ""),
            "approach": cand.get("approach", ""),
            "keywords": cand.get("keywords", []),
            # Block 1 metadata
            "_block1_metadata": {
                "candidate_id": cid,
                "parent_run_id": parent_run_id,
                "parent_rq_title": candidates_data.get("parent_rq_title", ""),
                "source_type": cand.get("source_type", ""),
                "derived_from": cand.get("derived_from", ""),
                "composite_score": entry.get("composite_score", 0),
                "portfolio_role": entry.get("portfolio_role", ""),
                "rationale": cand.get("rationale", ""),
            },
        }
        contexts.append(context)

    return contexts


# ------------------------------------------------------------------
# Notion writeback (optional)
# ------------------------------------------------------------------

def writeback_promoted(
    portfolio_path: Path,
    candidates_path: Path,
) -> WritebackResult:
    """Write promoted RQs to Notion RQ DB.

    Only executes if ENABLE_NOTION_WRITEBACK=true.
    Returns result with entries and exported contexts regardless.
    """
    result = WritebackResult()

    try:
        # Always export contexts (works without Notion)
        contexts = export_promoted_for_next_run(portfolio_path, candidates_path)
        result.promoted_contexts = contexts

        if not contexts:
            result.status = "generated"
            result.error = "No promoted candidates to write"
            return result

        # Check writeback flag
        writeback_enabled = os.environ.get("ENABLE_NOTION_WRITEBACK", "").lower() == "true"

        if not writeback_enabled:
            logger.info("ENABLE_NOTION_WRITEBACK is not true — skipping Notion write")
            for ctx in contexts:
                meta = ctx.get("_block1_metadata", {})
                result.entries.append(WritebackEntry(
                    candidate_id=meta.get("candidate_id", ""),
                    title=ctx.get("title", ""),
                    status="skipped",
                    error="ENABLE_NOTION_WRITEBACK not set",
                ))
            result.status = "generated"
            return result

        # Notion write
        from src.notion.client import NotionClient
        db_id = os.environ.get("NOTION_RQ_DB_ID", "")
        if not db_id:
            result.error = "NOTION_RQ_DB_ID not set"
            for ctx in contexts:
                meta = ctx.get("_block1_metadata", {})
                result.entries.append(WritebackEntry(
                    candidate_id=meta.get("candidate_id", ""),
                    title=ctx.get("title", ""),
                    status="error",
                    error="NOTION_RQ_DB_ID not set",
                ))
            result.status = "generated"
            return result

        client = NotionClient()

        for ctx in contexts:
            meta = ctx.get("_block1_metadata", {})
            cid = meta.get("candidate_id", "")
            title = ctx.get("title", "")

            entry = WritebackEntry(candidate_id=cid, title=title)

            try:
                properties = {
                    "Name": {"title": [{"text": {"content": title[:2000]}}]},
                    "Status": {"select": {"name": "Candidate"}},
                    "Priority": {"select": {"name": "Medium"}},
                    "Rationale / Background": {
                        "rich_text": [{"text": {"content": ctx.get("background", "")[:2000]}}]
                    },
                    "Proposed Approach": {
                        "rich_text": [{"text": {"content": ctx.get("approach", "")[:2000]}}]
                    },
                    "Gap Identified": {
                        "rich_text": [{"text": {"content": ctx.get("gap", "")[:2000]}}]
                    },
                    "Tags": {
                        "multi_select": [
                            {"name": "block1-generated"},
                            {"name": meta.get("portfolio_role", "")},
                        ]
                    },
                }

                resp = client.create_page(parent_db_id=db_id, properties=properties)
                entry.notion_page_id = resp.get("id", "")
                entry.status = "written"
                logger.info("Written to Notion: %s → %s", cid, entry.notion_page_id)

            except Exception as e:
                entry.status = "error"
                entry.error = str(e)
                logger.error("Notion write failed for %s: %s", cid, e)

            result.entries.append(entry)

        result.status = "generated"

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.error("105: %s", e)

    return result
