# src/lit_review/dimensions.py
"""Research dimension extraction (082/083).

Extracts structured research dimensions from papers:
- theoretical_lens — theoretical frameworks and concepts
- method — research methodologies and analysis techniques
- dataset — data sources (categorized + specific)
- context — geographic, temporal, institutional setting
- research_focus — primary research themes

Design decisions from T0.4 spike:
- dataset: include both category (e.g., "VC investment data") and specific name
- context: restrict to geographic/temporal/institutional; exclude research design
- Synonyms are normalized in 083 landscape mapper, not here

Usage::

    from src.lit_review.dimensions import extract_research_dimensions

    dims = extract_research_dimensions(paper, llm_client=client)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Dimension categories
DIMENSION_CATEGORIES = [
    "theoretical_lens",
    "method",
    "dataset",
    "context",
    "research_focus",
]


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class PaperDimensions:
    """Research dimensions extracted from a single paper."""

    paper_id: str
    paper_title: str
    theoretical_lens: List[str] = field(default_factory=list)
    method: List[str] = field(default_factory=list)
    dataset: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    research_focus: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PaperDimensions:
        return cls(
            paper_id=data.get("paper_id", ""),
            paper_title=data.get("paper_title", data.get("title", "")),
            theoretical_lens=list(data.get("theoretical_lens", [])),
            method=list(data.get("method", [])),
            dataset=list(data.get("dataset", [])),
            context=list(data.get("context", [])),
            research_focus=list(data.get("research_focus", [])),
        )


@dataclass
class AggregatedDimensions:
    """Aggregated dimensions across multiple papers.

    Each category maps dimension values to their frequency count.
    """

    theoretical_lens: Dict[str, int] = field(default_factory=dict)
    method: Dict[str, int] = field(default_factory=dict)
    dataset: Dict[str, int] = field(default_factory=dict)
    context: Dict[str, int] = field(default_factory=dict)
    research_focus: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_research_dimensions(
    paper: Dict[str, Any],
    *,
    llm_client: Any = None,
    paper_text: str = "",
) -> PaperDimensions:
    """Extract research dimensions from a single paper.

    Parameters
    ----------
    paper:
        Paper metadata dict with at least ``page_id`` and ``title``.
    llm_client:
        Claude client instance.
    paper_text:
        Full text or metadata text of the paper.

    Returns
    -------
    PaperDimensions with extracted dimension lists.
    """
    # TODO: T1.3 — implement using validated T0.4 prompt
    raise NotImplementedError(
        "extract_research_dimensions: full implementation pending T1.3. "
        "See src/lit_review/spikes/spike_rq_dimensions.py for validated prototype."
    )


def extract_dimensions_batch(
    papers: List[Dict[str, Any]],
    *,
    rq_context: Any = None,
    llm_client: Any = None,
    batch_size: int = 5,
) -> List[PaperDimensions]:
    """Extract dimensions from multiple papers in batches.

    Parameters
    ----------
    papers:
        List of paper metadata dicts.
    rq_context:
        Optional RQContext for RQ-aware dimension extraction.
    llm_client:
        Claude client instance.
    batch_size:
        Papers per LLM call (default 5, validated in T0.4).

    Returns
    -------
    List of PaperDimensions.
    """
    # TODO: T1.3 — implement
    raise NotImplementedError(
        "extract_dimensions_batch: full implementation pending T1.3."
    )


def aggregate_dimensions(
    paper_dims: List[PaperDimensions],
) -> AggregatedDimensions:
    """Aggregate dimension frequencies across all papers.

    Normalizes values to lowercase for counting.
    """
    agg = AggregatedDimensions()
    for pd in paper_dims:
        for cat in DIMENSION_CATEGORIES:
            items = getattr(pd, cat, [])
            freq = getattr(agg, cat)
            for item in items:
                normalized = str(item).strip().lower()
                freq[normalized] = freq.get(normalized, 0) + 1
    return agg
