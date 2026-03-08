# src/deep_research/constants.py
"""Shared constants for the Deep Research pipeline (067-072).

Centralizes confidence levels, valid tags, source types,
and tag aliases so that all modules use the same definitions.
"""

from __future__ import annotations

from typing import Dict

# -- Confidence levels -------------------------------------------------------

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

CONFIDENCE_ORDER: Dict[str, int] = {
    CONFIDENCE_HIGH: 3,
    CONFIDENCE_MEDIUM: 2,
    CONFIDENCE_LOW: 1,
}

# -- Valid tags --------------------------------------------------------------

VALID_TAGS = [
    "strategy",
    "funding",
    "product",
    "partnership",
    "governance",
    "research",
    "market",
    "legal",
    "personnel",
]

VALID_TAGS_SET = frozenset(VALID_TAGS)

# -- Tag aliases (for cluster merging) --------------------------------------

TAG_ALIASES: Dict[str, str] = {
    "legal": "governance",
    "personnel": "governance",
}

# -- Source types ------------------------------------------------------------

VALID_SOURCE_TYPES = [
    "article",
    "news",
    "paper",
    "report",
    "webpage",
    "dataset",
    "unknown",
]

# source_type → base confidence mapping
SOURCE_TYPE_CONFIDENCE: Dict[str, str] = {
    "paper": CONFIDENCE_HIGH,
    "report": CONFIDENCE_HIGH,
    "news": CONFIDENCE_MEDIUM,
    "article": CONFIDENCE_MEDIUM,
    "dataset": CONFIDENCE_MEDIUM,
    "webpage": CONFIDENCE_LOW,
    "unknown": CONFIDENCE_LOW,
}
