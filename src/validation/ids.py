# src/validation/ids.py
"""Stable ID generation for Validation & Grounding entities.

All IDs are content-hash based (SHA-256, 12 hex chars) so they are:
- Deterministic: same input → same ID
- Cross-run stable: re-running 081 doesn't change evidence IDs
- Prefix-typed: ev__, ds__ for entity type identification

See .steering/20260318-lit_validation_and_grounding/design.md Section 1.5.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


def evidence_id(claim_or_point: str, paper_title: str) -> str:
    """Generate stable evidence ID from content hash.

    Same claim_or_point + paper_title → same ID, regardless of array position.
    """
    normalized = f"{claim_or_point.strip().lower()}|{paper_title.strip().lower()}"
    content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:12]
    return f"ev__{content_hash}"


def dataset_id(name: str) -> str:
    """Generate stable dataset ID from normalized name."""
    normalized = name.strip().lower()
    content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:12]
    return f"ds__{content_hash}"


def resolve_paper_id(
    evidence_item: Dict[str, Any],
    candidate_papers: List[Dict[str, Any]],
) -> Optional[str]:
    """Resolve Notion page_id for an evidence item.

    Resolution order:
    1. source_paper_id if non-empty
    2. Reverse lookup by paper_title in candidate_papers
    3. None if unresolvable
    """
    # 1. Direct field
    source_id = evidence_item.get("source_paper_id", "")
    if source_id:
        return source_id

    # Also check paper_id field (some evidence formats have it directly)
    paper_id_field = evidence_item.get("paper_id", "")
    if paper_id_field:
        return paper_id_field

    # 2. Reverse lookup by title
    title = evidence_item.get("paper_title", "").strip().lower()
    if not title:
        return None

    for paper in candidate_papers:
        candidate_title = paper.get("title", "").strip().lower()
        if candidate_title == title:
            return paper.get("paper_id", None)

    return None
