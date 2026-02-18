# src/notion/truncation.py
"""Shared Notion rich-text truncation helper.

Notion rich_text properties have a hard 2 000-character limit.  This module
provides a single ``notion_truncate`` function that:

- caps text at a *safe* inner limit (default 1 900 chars) so that the
  ``TRUNCATION_SUFFIX`` always fits within the Notion limit,
- records which fields were truncated (caller aggregates into run_metadata).

Usage::

    from src.notion.truncation import notion_truncate, TruncationTracker

    tracker = TruncationTracker()
    value = notion_truncate("very long ...", field_name="Evidence Detail",
                            tracker=tracker)
    # After all fields:
    metadata["truncated_fields"] = tracker.report()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Notion hard limit for a single rich_text block is 2 000 chars.
# We leave headroom for the suffix so the final payload never exceeds 2 000.
NOTION_RT_HARD_LIMIT: int = 2000
TRUNCATION_SUFFIX: str = " ...(truncated)"
# Inner limit = hard limit minus suffix length (guarantees the final string fits).
SAFE_LIMIT: int = NOTION_RT_HARD_LIMIT - len(TRUNCATION_SUFFIX)  # 1 985


def notion_truncate(
    text: Optional[str],
    *,
    field_name: str = "",
    limit: int = SAFE_LIMIT,
    tracker: Optional["TruncationTracker"] = None,
) -> str:
    """Truncate *text* to fit a Notion rich_text property.

    Parameters
    ----------
    text:
        The raw string (``None`` / empty → ``""``).
    field_name:
        Logical property name; recorded in *tracker* when truncation occurs.
    limit:
        Maximum characters *before* the suffix.  Defaults to ``SAFE_LIMIT``
        (1 985 chars), which guarantees the full string including suffix fits
        within Notion's 2 000-char hard limit.
    tracker:
        Optional :class:`TruncationTracker`; if provided, truncated fields
        are recorded for later inclusion in ``run_metadata.json``.

    Returns
    -------
    str
        The (possibly truncated) string, always ≤ ``limit + len(TRUNCATION_SUFFIX)``
        characters.
    """
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    truncated = text[:limit].rstrip() + TRUNCATION_SUFFIX
    if tracker is not None and field_name:
        tracker.record(field_name, original_len=len(text), kept_len=limit)
    return truncated


@dataclass
class TruncationTracker:
    """Collects truncation events for inclusion in ``run_metadata.json``."""

    _events: List[Dict[str, object]] = field(default_factory=list)

    def record(self, field_name: str, *, original_len: int, kept_len: int) -> None:
        self._events.append({
            "field": field_name,
            "original_chars": original_len,
            "kept_chars": kept_len,
        })

    def report(self) -> List[Dict[str, object]]:
        """Return a de-duplicated, sorted list of truncation events."""
        # Aggregate: one entry per field_name with max original_len
        by_field: Dict[str, Dict[str, object]] = {}
        for ev in self._events:
            fname = str(ev["field"])
            if fname not in by_field:
                by_field[fname] = dict(ev)
            else:
                existing = by_field[fname]
                if ev["original_chars"] > existing["original_chars"]:  # type: ignore[operator]
                    by_field[fname] = dict(ev)
        return sorted(by_field.values(), key=lambda e: str(e["field"]))

    @property
    def had_truncations(self) -> bool:
        return len(self._events) > 0
