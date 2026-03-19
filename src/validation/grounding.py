# src/validation/grounding.py
"""Paper text retrieval for literature validation.

Reuses 081 extractor's existing text retrieval logic (get_paper_text).
See design.md Section 2.2, Step 1.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIT_DATA_DIR = PROJECT_ROOT / "data" / "lit_review"


def retrieve_paper_text(
    paper_title: str,
    candidate_papers: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Get best available text for a paper by title.

    Reuses 081 extractor's get_paper_text() with candidate_papers metadata.

    Returns (text, source) where source is "pdf", "metadata", or "title_only".
    """
    from src.lit_review.extractor import get_paper_text

    # Find paper metadata in candidate_papers by title match
    paper_meta = _find_paper_by_title(paper_title, candidate_papers)
    if paper_meta is None:
        logger.warning("Paper not found in candidates: '%s'", paper_title[:60])
        return f"Title: {paper_title}", "title_only"

    try:
        text, source = get_paper_text(paper_meta)
        return text, source
    except Exception as e:
        logger.warning("Text retrieval failed for '%s': %s", paper_title[:60], e)
        return f"Title: {paper_title}", "title_only"


def _find_paper_by_title(
    title: str,
    candidate_papers: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find a paper in candidate_papers by title (case-insensitive)."""
    title_lower = title.strip().lower()
    for paper in candidate_papers:
        if paper.get("title", "").strip().lower() == title_lower:
            return paper
    return None


def load_candidate_papers(run_dir: Path) -> List[Dict[str, Any]]:
    """Load candidate_papers.json and extract the scored_papers list."""
    path = run_dir / "candidate_papers.json"
    if not path.exists():
        logger.warning("candidate_papers.json not found at %s", path)
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("scored_papers", [])
    except Exception as e:
        logger.error("Failed to load candidate_papers.json: %s", e)
        return []


def group_evidence_by_paper(
    evidence_items: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group evidence items by paper_title for batch processing."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in evidence_items:
        title = item.get("paper_title", "unknown")
        groups.setdefault(title, []).append(item)
    return groups
