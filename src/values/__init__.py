# src/values/__init__.py
"""Values Foundation — directional alignment framework.

This package provides:
- Value domain schema (frozen dataclasses) with versioning support
- Alignment entry schema (Log layer — two-dimensional evaluation)
- Refinement policy (prevents value drift toward KPI/moralizing language)
- LLM refinement layer (OpenAI via router, structured suggestions)
- Notion DB formatting for ROS_Values_Codex and ROS_Alignment_Log
- Voice interface with Japanese default (VoiceConfig)

Architecture:
  Codex  = identity definitions (stable, principled, no numeric scores)
  Log    = lived evaluation (dynamic, importance 1–5 + alignment 1–5)

Value Evaluation Scale:
  importance_score (1–5) — how important is this domain?
  alignment_score  (1–5) — how consistently am I living this value?
  gap_score        (computed) — importance − alignment
  significant_gap  (computed) — True when gap >= 2

Voice Language:
  Default language for voice I/O is Japanese (ja).
  Configurable via VoiceConfig(language="en") or --lang CLI flag.

Designed for quarterly review and iterative refinement.
"""

from src.values.schema import (  # noqa: F401
    AlignmentEntry,
    DEFAULT_POLICY,
    DOMAIN_IDS,
    PolicyViolation,
    RefinementPolicy,
    RefinementResult,
    RefinementSuggestion,
    ValueBehavior,
    ValueDomain,
    ValueRecord,
    VALID_SOURCES,
    alignment_entry_from_dict,
    value_record_from_dict,
)
from src.values.generator import (  # noqa: F401
    generate_all_domains,
    generate_value_record,
)
from src.values.voice import (  # noqa: F401
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    VoiceConfig,
    ReflectionLog,
    ReflectionQuestion,
    ReflectionResponse,
)
