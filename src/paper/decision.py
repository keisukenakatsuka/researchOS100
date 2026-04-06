"""Decision log management — append-only decisions.jsonl."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.paper.models import (
    DECISION_TYPES,
    Decision,
    RejectedAlternative,
)
from src.paper.registry import _paper_dir, get_paper

logger = logging.getLogger(__name__)


def _decisions_path(paper_id: str) -> Path:
    return _paper_dir(paper_id) / "decisions.jsonl"


def _next_id(paper_id: str) -> str:
    """Auto-increment decision ID: d001, d002, ..."""
    decisions = get_decisions(paper_id)
    if not decisions:
        return "d001"
    max_num = max(int(d.id[1:]) for d in decisions)
    return f"d{max_num + 1:03d}"


def add_decision(
    paper_id: str,
    decision: str,
    reason: str,
    *,
    decision_type: str = "other",
    stage: str = "",
    rejected_alternatives: Optional[List[Dict[str, str]]] = None,
    refs: Optional[List[str]] = None,
    effective_at: Optional[str] = None,
    source: str = "cli",
) -> Decision:
    """Add a decision to the paper's log.

    Only ``decision`` and ``reason`` are required. Everything else is optional
    to keep the barrier to writing decisions as low as possible.

    Args:
        rejected_alternatives: List of {"option": "...", "rejection_reason": "..."}.
        effective_at: For migration. If set, source should be "migration".
    """
    if decision_type not in DECISION_TYPES:
        raise ValueError(
            f"Invalid decision_type '{decision_type}'. Must be one of {sorted(DECISION_TYPES)}"
        )

    paper = get_paper(paper_id)  # validates paper exists

    # Auto-fill stage from paper if not provided
    if not stage:
        stage = paper.current_stage

    rejected = [
        RejectedAlternative(**ra) for ra in (rejected_alternatives or [])
    ]

    record = Decision(
        id=_next_id(paper_id),
        decision=decision,
        reason=reason,
        stage=stage,
        decision_type=decision_type,
        rejected_alternatives=rejected,
        refs=refs or [],
        effective_at=effective_at or "",
        source=source,
    )

    path = _decisions_path(paper_id)
    with path.open("a") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    logger.info("Decision %s added to paper '%s'", record.id, paper_id)
    return record


def get_decisions(paper_id: str) -> List[Decision]:
    """Read all decisions from decisions.jsonl."""
    path = _decisions_path(paper_id)
    if not path.exists():
        return []

    text = path.read_text().strip()
    if not text:
        return []

    results = []
    for line in text.splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        # Reconstruct RejectedAlternative objects
        ra_list = data.pop("rejected_alternatives", [])
        d = Decision(
            **data,
            rejected_alternatives=[RejectedAlternative(**ra) for ra in ra_list],
        )
        results.append(d)
    return results
