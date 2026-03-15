# src/lit_review/rq_context.py
"""RQContext — shared query context for Block 3 pipeline.

RQContext is the canonical internal representation of a Research Question
used across all Block 3 scripts (079–083). It can be constructed from:

1. A Notion RQ page (via ``from_notion_rq``)
2. Free-form text (via ``from_text``)

Usage::

    from src.lit_review.rq_context import RQContext

    # From a normalized Notion RQ dict (output of rq_normalize.normalize_rq)
    ctx = RQContext.from_notion_rq(rq_dict)

    # From free text
    ctx = RQContext.from_text("How do co-investment networks affect ...")

    # Serialize / deserialize for intermediate JSON
    data = ctx.to_dict()
    ctx2 = RQContext.from_dict(data)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RQContext:
    """Structured representation of a Research Question.

    Shared across matcher (079), extractor (081), synthesizer (082),
    and landscape mapper (083).
    """

    rq_id: Optional[str]
    """Notion page ID. None when constructed from free text."""

    title: str
    """RQ title / question text."""

    background: str = ""
    """Rationale / background motivating the RQ."""

    gap: str = ""
    """Identified research gap."""

    approach: str = ""
    """Proposed approach (from Notion RQ DB)."""

    keywords: List[str] = field(default_factory=list)
    """Auto-extracted keywords for pre-filtering."""

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_notion_rq(cls, rq: Dict[str, Any]) -> RQContext:
        """Create from a normalized Notion RQ dict.

        Expected keys match the output of
        ``src.notion.rq_normalize.normalize_rq()``:
        page_id, title, rationale, approach, gap, keywords.
        """
        return cls(
            rq_id=rq.get("page_id") or None,
            title=rq.get("title", ""),
            background=rq.get("rationale", ""),
            gap=rq.get("gap", ""),
            approach=rq.get("approach", ""),
            keywords=list(rq.get("keywords", [])),
        )

    @classmethod
    def from_text(cls, text: str) -> RQContext:
        """Create from free-form text.

        Keywords are extracted via simple tokenization.
        """
        keywords = _extract_keywords(text)
        return cls(
            rq_id=None,
            title=text.strip(),
            keywords=keywords,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RQContext:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            rq_id=data.get("rq_id"),
            title=data.get("title", ""),
            background=data.get("background", ""),
            gap=data.get("gap", ""),
            approach=data.get("approach", ""),
            keywords=list(data.get("keywords", [])),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    # ------------------------------------------------------------------
    # LLM prompt helpers
    # ------------------------------------------------------------------

    def to_prompt_text(self) -> str:
        """Format for inclusion in LLM prompts."""
        parts = [f"Research Question: {self.title}"]
        if self.background:
            parts.append(f"背景: {self.background}")
        if self.approach:
            parts.append(f"アプローチ: {self.approach}")
        if self.gap:
            parts.append(f"ギャップ: {self.gap}")
        if self.keywords:
            parts.append(f"キーワード: {', '.join(self.keywords)}")
        return "\n".join(parts)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "this", "that", "with", "from", "into", "over", "will", "have",
    "has", "been", "were", "their", "about", "after", "before",
    "also", "than", "then", "them", "they", "what", "when", "where",
    "which", "while", "said", "says", "the", "and", "for", "are",
    "not", "but", "was", "can", "all", "may", "its", "does", "how",
})


def _extract_keywords(text: str, *, max_k: int = 15) -> List[str]:
    """Simple stopword-filtered keyword extraction.

    Mirrors ``src.notion.rq_normalize._extract_keywords``.
    """
    tokens = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    seen: List[str] = []
    for t in tokens:
        if t in _STOP_WORDS or t in seen:
            continue
        seen.append(t)
        if len(seen) >= max_k:
            break
    return seen
