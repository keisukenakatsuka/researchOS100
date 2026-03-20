# src/lit_review/exporter.py
"""100 Export Bundle — service logic.

Assembles all generated drafts into an exportable research bundle:

  research_output_bundle/
  ├── paper_draft.md          # Combined paper (all sections)
  ├── paper_outline.json      # Outline (094)
  ├── review_report.md        # Review (099)
  ├── review_report.json      # Review data (099)
  └── export_bundle.json      # Bundle metadata

Usage::

    from src.lit_review.exporter import export_bundle

    result = export_bundle(run_dir)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.lit_review.drafters.base import BaseDrafter

logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "1.0.0"

# Sections in paper order
_SECTIONS = [
    ("introduction", "draft_introduction.md"),
    ("literature_review", "draft_literature_review.md"),
    ("hypotheses", "draft_hypotheses.md"),
    ("methods", "draft_methods.md"),
]

# Additional files to include in bundle
_BUNDLE_FILES = [
    "paper_outline.json",
    "paper_outline.md",
    "review_report.json",
    "review_report.md",
]


# ------------------------------------------------------------------
# Result type
# ------------------------------------------------------------------

@dataclass
class ExportResult:
    status: str = "failed"
    bundle_dir: str = ""
    paper_word_count: int = 0
    sections_included: List[str] = field(default_factory=list)
    sections_missing: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------
# Citation counting
# ------------------------------------------------------------------

def _count_citations(text: str) -> int:
    p1 = re.findall(
        r"[(\uff08][A-Z][a-zé]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-zé]+))?[.,]?\s*\d{4}[)\uff09]",
        text,
    )
    p2 = re.findall(
        r"[A-Z][a-zé]+(?:\s+et\s+al\.?)?\s*[(\uff08]\d{4}[)\uff09]",
        text,
    )
    return len(set(p1 + p2))


# ------------------------------------------------------------------
# Paper assembly
# ------------------------------------------------------------------

def _assemble_paper(run_dir: Path) -> tuple[str, List[str], List[str]]:
    """Combine draft sections into a single paper_draft.md.

    Returns (paper_text, included_sections, missing_sections).
    """
    parts: List[str] = []
    included: List[str] = []
    missing: List[str] = []

    for section_id, fname in _SECTIONS:
        p = run_dir / fname
        if p.exists():
            text = p.read_text().strip()
            if text:
                parts.append(text)
                included.append(section_id)
            else:
                missing.append(section_id)
                parts.append(f"# {section_id.replace('_', ' ').title()}\n\n*(Section not yet generated)*\n")
        else:
            missing.append(section_id)
            parts.append(f"# {section_id.replace('_', ' ').title()}\n\n*(Section not yet generated)*\n")

    # Table of contents
    toc_lines = ["# Table of Contents\n"]
    for section_id, _ in _SECTIONS:
        title = section_id.replace("_", " ").title()
        status = "" if section_id in included else " *(missing)*"
        toc_lines.append(f"- {title}{status}")
    toc_lines.append("")

    paper = "\n".join(toc_lines) + "\n---\n\n" + "\n\n---\n\n".join(parts) + "\n"
    return paper, included, missing


def _build_metadata(
    run_dir: Path,
    paper_text: str,
    included: List[str],
    missing: List[str],
) -> Dict[str, Any]:
    """Build export_bundle.json metadata."""
    # RQ
    rq_title = ""
    rq_path = run_dir / "rq_context.json"
    if rq_path.exists():
        rq_title = json.loads(rq_path.read_text()).get("title", "")

    # Word counts per section
    section_words: Dict[str, int] = {}
    for section_id, fname in _SECTIONS:
        p = run_dir / fname
        if p.exists():
            section_words[section_id] = BaseDrafter._count_words(p.read_text())
        else:
            section_words[section_id] = 0

    total_words = BaseDrafter._count_words(paper_text)
    citation_count = _count_citations(paper_text)

    # Quality score from review
    quality_score = 0.0
    review_path = run_dir / "review_report.json"
    if review_path.exists():
        review = json.loads(review_path.read_text())
        quality_score = review.get("overall_quality_score", 0.0)

    # Focused hypotheses info
    focused_info: Dict[str, Any] = {}
    focused_path = run_dir / "focused_hypotheses.json"
    if focused_path.exists():
        focused = json.loads(focused_path.read_text())
        primary = focused.get("primary", {})
        secondary = focused.get("secondary", {})
        focused_info = {
            "has_focused": True,
            "primary_id": primary.get("hypothesis_id", ""),
            "primary_statement": primary.get("hypothesis_statement", "")[:100],
            "has_secondary": focused.get("has_secondary", False),
            "secondary_id": secondary.get("hypothesis_id", "") if secondary else "",
            "review_source": focused.get("review_source", ""),
        }

    return {
        "run_id": run_dir.name,
        "rq_title": rq_title,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": _PIPELINE_VERSION,
        "sections_included": included,
        "sections_missing": missing,
        "section_word_counts": section_words,
        "total_word_count": total_words,
        "citation_count": citation_count,
        "quality_score": quality_score,
        "focused_hypotheses": focused_info,
        "files": [
            "paper_draft.md",
            "paper_outline.json",
            "paper_outline.md",
            "review_report.json",
            "review_report.md",
            "export_bundle.json",
        ],
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def export_bundle(run_dir: Path) -> ExportResult:
    """Assemble and export research bundle. Entry point for 100 script."""
    result = ExportResult()

    try:
        # 1. Assemble paper
        paper_text, included, missing = _assemble_paper(run_dir)

        if not included:
            result.error = "No draft sections found"
            return result

        result.sections_included = included
        result.sections_missing = missing

        # 2. Build metadata
        metadata = _build_metadata(run_dir, paper_text, included, missing)

        # 3. Write outputs to run_dir (flat — same dir as other artifacts)
        # paper_draft.md
        paper_path = run_dir / "paper_draft.md"
        paper_path.write_text(paper_text)
        logger.info("Saved paper_draft.md (%d words)", metadata["total_word_count"])

        # export_bundle.json
        bundle_path = run_dir / "export_bundle.json"
        bundle_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        logger.info("Saved export_bundle.json")

        # Copy supporting files (they already exist in run_dir, so just verify)
        for fname in _BUNDLE_FILES:
            p = run_dir / fname
            if not p.exists():
                logger.warning("Bundle file not found (skipped): %s", fname)

        result.status = "generated"
        result.bundle_dir = str(run_dir)
        result.paper_word_count = metadata["total_word_count"]
        result.metadata = metadata

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.error("100: unexpected error: %s", e)

    return result
