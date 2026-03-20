# src/lit_review/review.py
"""Block 5: Hypothesis Review (089b).

Applies human review decisions to the selector output.
If no human review file exists, auto-accepts the selector result.

Usage::

    from src.lit_review.review import apply_review

    decision = apply_review(run_dir)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class ReviewDecision:
    run_id: str
    created_at: str = ""
    review_source: str = "auto_accept"  # "auto_accept" | "human_review"
    primary: Dict[str, Any] = field(default_factory=dict)
    secondary: Optional[Dict[str, Any]] = None
    has_secondary: bool = False
    rationale: str = ""
    notes_for_downstream: str = ""
    non_selected: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "review_source": self.review_source,
            "primary": self.primary,
            "secondary": self.secondary,
            "has_secondary": self.has_secondary,
            "rationale": self.rationale,
            "notes_for_downstream": self.notes_for_downstream,
            "non_selected": self.non_selected,
        }


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def _validate_review_input(review: Dict, candidate_ids: set) -> List[str]:
    """Validate human review input. Returns list of errors."""
    errors = []

    primary_id = review.get("final_primary_id")
    if not primary_id:
        errors.append("final_primary_id is required")
    elif primary_id not in candidate_ids:
        errors.append(f"final_primary_id '{primary_id}' not found in candidates")

    secondary_id = review.get("final_secondary_id")
    if secondary_id and secondary_id not in candidate_ids:
        errors.append(f"final_secondary_id '{secondary_id}' not found in candidates")

    if secondary_id and secondary_id == primary_id:
        errors.append("final_secondary_id cannot be the same as final_primary_id")

    for override in review.get("overrides", []):
        oid = override.get("hypothesis_id")
        if oid and oid not in candidate_ids:
            errors.append(f"override hypothesis_id '{oid}' not found in candidates")

    return errors


# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------

def apply_review(run_dir: Path) -> ReviewDecision:
    """Apply human review if present, otherwise auto-accept selector output."""
    run_id = run_dir.name
    now_iso = datetime.now(timezone.utc).isoformat()

    # Load selector output
    sel_path = run_dir / "hypothesis_selection.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"hypothesis_selection.json not found in {run_dir}")

    selection = json.loads(sel_path.read_text())
    candidates = selection.get("candidates", [])
    candidate_by_id = {c.get("hypothesis_id", ""): c for c in candidates}
    candidate_ids = set(candidate_by_id.keys())

    # Check for human review
    review_path = run_dir / "hypothesis_review.json"
    if review_path.exists():
        logger.info("Human review file found: %s", review_path)
        try:
            raw = review_path.read_text().strip()
            if not raw:
                raise ValueError("hypothesis_review.json is empty")
            review = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"hypothesis_review.json is not valid JSON: {e}") from e
        return _apply_human_review(run_id, now_iso, selection, candidate_by_id, candidate_ids, review)
    else:
        logger.info("No human review file found — auto-accepting selector output")
        return _auto_accept(run_id, now_iso, selection, candidate_by_id)


def _auto_accept(
    run_id: str,
    now_iso: str,
    selection: Dict,
    candidate_by_id: Dict[str, Dict],
) -> ReviewDecision:
    """Auto-accept selector result."""
    primary_id = selection.get("primary_id", "")
    secondary_id = selection.get("secondary_id")

    primary = candidate_by_id.get(primary_id, {})
    secondary = candidate_by_id.get(secondary_id, {}) if secondary_id else None

    non_selected = [
        {
            "hypothesis_id": c.get("hypothesis_id", ""),
            "status": c.get("status", "deferred"),
            "reason": c.get("selection_reason", ""),
        }
        for c in selection.get("candidates", [])
        if c.get("hypothesis_id") not in (primary_id, secondary_id)
    ]

    return ReviewDecision(
        run_id=run_id,
        created_at=now_iso,
        review_source="auto_accept",
        primary={
            "hypothesis_id": primary.get("hypothesis_id", ""),
            "hypothesis_statement": primary.get("hypothesis_statement", ""),
            "selection_reason": primary.get("selection_reason", ""),
        },
        secondary={
            "hypothesis_id": secondary.get("hypothesis_id", ""),
            "hypothesis_statement": secondary.get("hypothesis_statement", ""),
            "selection_reason": secondary.get("selection_reason", ""),
        } if secondary else None,
        has_secondary=secondary is not None,
        rationale=selection.get("selection_rationale", ""),
        notes_for_downstream="",
        non_selected=non_selected,
    )


def _apply_human_review(
    run_id: str,
    now_iso: str,
    selection: Dict,
    candidate_by_id: Dict[str, Dict],
    candidate_ids: set,
    review: Dict,
) -> ReviewDecision:
    """Apply human review decisions."""
    errors = _validate_review_input(review, candidate_ids)
    if errors:
        raise ValueError(f"Invalid hypothesis_review.json: {'; '.join(errors)}")

    primary_id = review["final_primary_id"]
    secondary_id = review.get("final_secondary_id")
    drop_secondary = review.get("drop_secondary", False)

    if drop_secondary:
        secondary_id = None

    primary = candidate_by_id.get(primary_id, {})
    secondary = candidate_by_id.get(secondary_id, {}) if secondary_id else None

    # Determine non-selected statuses (apply overrides)
    override_map = {}
    for o in review.get("overrides", []):
        override_map[o.get("hypothesis_id", "")] = o

    non_selected = []
    for c in selection.get("candidates", []):
        cid = c.get("hypothesis_id", "")
        if cid in (primary_id, secondary_id):
            continue
        override = override_map.get(cid, {})
        non_selected.append({
            "hypothesis_id": cid,
            "status": override.get("new_status", c.get("status", "deferred")),
            "reason": override.get("reason", c.get("selection_reason", "")),
        })

    return ReviewDecision(
        run_id=run_id,
        created_at=now_iso,
        review_source="human_review",
        primary={
            "hypothesis_id": primary.get("hypothesis_id", primary_id),
            "hypothesis_statement": primary.get("hypothesis_statement", ""),
            "selection_reason": primary.get("selection_reason", ""),
        },
        secondary={
            "hypothesis_id": secondary.get("hypothesis_id", secondary_id),
            "hypothesis_statement": secondary.get("hypothesis_statement", ""),
            "selection_reason": secondary.get("selection_reason", ""),
        } if secondary else None,
        has_secondary=secondary is not None,
        rationale=review.get("rationale", ""),
        notes_for_downstream=review.get("notes_for_downstream", ""),
        non_selected=non_selected,
    )
