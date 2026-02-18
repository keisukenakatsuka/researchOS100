# src/values/schema.py
"""Data schema for the Values Foundation system.

All domain data is represented as frozen dataclasses for immutability.
The schema is JSON-serializable via ``ValueRecord.to_dict()``.

Value Domains (Codex Layer — definitions only)
-----------------------------------------------
12 life domains, each with:
- identity-oriented definition
- behavioral translation (how value becomes visible)
- concrete example behaviors (executable today)
- misalignment description (what off-track looks like)
- reflection questions (for voice + avatar sessions)
- micro habits (smallest viable daily actions)

NOTE: Numeric embodiment scores do NOT belong here.
The Codex defines *what* and *how*, not *how well*.
All numeric evaluation lives in AlignmentEntry (Log layer).

Designed for storage in ROS_Values_Codex (Notion) and quarterly review.
Supports revision tracking within quarters.

Source Types
------------
- Manual  : hand-authored seed data or user edits
- LLM     : generated entirely by LLM refinement
- Hybrid  : seed data enhanced by LLM suggestions, user-approved
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple


# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------

DOMAIN_IDS: Tuple[str, ...] = (
    "family",
    "marriage_romantic",
    "parenting",
    "friendships_interpersonal",
    "career_work",
    "personal_growth",
    "leisure",
    "spirituality",
    "community_social",
    "health",
    "environment",
    "creative_arts",
)

SourceType = Literal["Manual", "LLM", "Hybrid"]

VALID_SOURCES: Tuple[str, ...] = ("Manual", "LLM", "Hybrid")


# ----------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------

@dataclass(frozen=True)
class ValueBehavior:
    """A single concrete behavior that expresses a value domain."""
    description: str
    frequency_hint: str = ""  # e.g. "daily", "weekly", "as-needed"


@dataclass(frozen=True)
class ValueDomain:
    """One of 12 life value domains with behavioral translation.

    This is the canonical in-memory representation.  The Notion schema
    in ``values_schema.py`` maps these fields to DB properties.
    """

    domain_id: str
    domain_label: str
    value_definition: str
    behavioral_translation: str
    example_behaviors: Tuple[ValueBehavior, ...]
    misalignment_description: str = ""
    reflection_questions: Tuple[str, ...] = ()
    micro_habits: Tuple[str, ...] = ()
    source: str = "Manual"              # Manual / LLM / Hybrid
    version: int = 1
    revision: int = 0
    change_notes: str = ""

    # NOTE: No embodiment_scale here.  Numeric evaluation belongs
    # exclusively in AlignmentEntry (the Log layer).

    def __post_init__(self) -> None:
        if self.domain_id not in DOMAIN_IDS:
            raise ValueError(
                f"Unknown domain_id: {self.domain_id!r}. "
                f"Expected one of: {DOMAIN_IDS}"
            )
        if len(self.example_behaviors) < 1:
            raise ValueError(
                f"Domain {self.domain_id!r} must have at least 1 example behavior"
            )
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source: {self.source!r}. "
                f"Expected one of: {VALID_SOURCES}"
            )


@dataclass(frozen=True)
class ValueRecord:
    """Complete values snapshot — all 12 domains for one review cycle.

    This is the top-level container that gets serialized to JSON and
    written to the Notion ROS_Values_Codex.
    """

    version: str                         # schema version ("1.0", "2.0", ...)
    review_quarter: str                  # e.g. "2026-Q1"
    domains: Tuple[ValueDomain, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        ids = [d.domain_id for d in self.domains]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate domain_id found in domains")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict representation."""
        return {
            "version": self.version,
            "review_quarter": self.review_quarter,
            "notes": self.notes,
            "domains": [_domain_to_dict(d) for d in self.domains],
        }

    def get_domain(self, domain_id: str) -> Optional[ValueDomain]:
        """Lookup a single domain by its stable ID."""
        for d in self.domains:
            if d.domain_id == domain_id:
                return d
        return None


# ----------------------------------------------------------------
# Alignment Log entry (in-memory representation)
# ----------------------------------------------------------------

def _validate_score_1_5(value: int, name: str) -> None:
    """Validate that a score is an integer in [1, 5]."""
    if not isinstance(value, int):
        raise TypeError(
            f"{name} must be int, got {type(value).__name__}"
        )
    if value < 1 or value > 5:
        raise ValueError(
            f"{name} must be 1–5 (integer), got {value}. "
            f"Every alignment entry requires a concrete self-assessment."
        )


@dataclass(frozen=True)
class AlignmentEntry:
    """One row in the ROS_Alignment_Log.

    Captures a daily, weekly, or quarterly reflection against a
    specific value domain using a two-dimensional evaluation scale:

    Value Evaluation Scale
    ----------------------
    - **importance_score** (1–5): How important is this domain to me?
    - **alignment_score** (1–5): How consistently am I living this value?
    - **gap_score** (computed): importance_score − alignment_score
    - **significant_gap** (computed): True if gap_score >= 2

    All scores are strictly integer 1–5.  0 is NOT valid.
    The gap indicates where energy and attention are most needed.
    A gap >= 2 signals meaningful misalignment requiring action.
    """

    date_iso: str                        # "2026-02-17"
    review_type: str                     # "Daily" / "Weekly" / "Quarterly"
    domain_id: str                       # links to ValueDomain

    # Two-dimensional evaluation scale (both required, both 1–5)
    importance_score: int                # 1–5: how important is this domain?
    alignment_score: int                 # 1–5: how consistently am I living it?

    reflection_text: str = ""
    misalignment_notes: str = ""
    next_adjustment: str = ""
    transcript: str = ""                 # voice transcription
    audio_url: str = ""                  # link to audio file
    ai_summary: str = ""                 # LLM-generated summary
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.review_type not in ("Daily", "Weekly", "Quarterly"):
            raise ValueError(
                f"Invalid review_type: {self.review_type!r}. "
                f"Expected 'Daily', 'Weekly', or 'Quarterly'."
            )
        _validate_score_1_5(self.importance_score, "importance_score")
        _validate_score_1_5(self.alignment_score, "alignment_score")

    @property
    def gap_score(self) -> int:
        """Importance − Alignment.  Positive = under-living the value."""
        return self.importance_score - self.alignment_score

    @property
    def significant_gap(self) -> bool:
        """True when gap >= 2, indicating meaningful misalignment."""
        return self.gap_score >= 2


# ----------------------------------------------------------------
# Refinement policy (prevents value drift)
# ----------------------------------------------------------------

@dataclass(frozen=True)
class RefinementPolicy:
    """Constraints that govern what an LLM is allowed to suggest.

    Each flag, when True, activates a corresponding validation rule
    applied AFTER the LLM returns suggestions and BEFORE suggestions
    are accepted.  Violations are flagged — never silently dropped.

    Purpose: prevent value drift toward KPI language, moral preaching,
    over-specific tactical wording, or loss of identity framing.
    """

    preserve_identity_orientation: bool = True
    """value_definition must start with 'I am someone who...' or similar."""

    forbid_goal_language: bool = True
    """Reject suggestions containing goal/KPI language
    (e.g. 'achieve', 'maximize', 'hit target', 'KPI', 'OKR')."""

    forbid_moralizing_language: bool = True
    """Reject suggestions with moral-preaching tone
    (e.g. 'you should', 'one must', 'it is wrong to', 'always be')."""

    preserve_length_bounds: bool = True
    """Definitions must stay within ±50% of the original length.
    Prevents bloat or over-compression."""

    preserve_principle_based_tone: bool = True
    """Reject suggestions that drift from principle-based ('I am...')
    to prescription-based ('Do X every day')."""

    # Configurable thresholds
    length_tolerance: float = 0.5
    """Fraction tolerance for length bounds (0.5 = ±50%)."""


# Frozen sets of forbidden patterns (compiled once, used by policy validator)
_GOAL_LANGUAGE_PATTERNS: Tuple[str, ...] = (
    "achieve", "maximize", "minimize", "hit target", "kpi", "okr",
    "measurable outcome", "metric", "target score", "goal of",
    "optimize for", "benchmark", "deliverable", "performance indicator",
)

_MORALIZING_PATTERNS: Tuple[str, ...] = (
    "you should", "one must", "it is wrong to", "always be",
    "you need to", "you have to", "you ought to", "never fail to",
    "it is important that you", "make sure you always",
    "don't ever", "you are obligated",
)

_PRESCRIPTION_PATTERNS: Tuple[str, ...] = (
    "do x every", "every single day", "you must do",
    "complete this daily", "always execute", "mandatory action",
    "required daily", "non-negotiable task",
)


@dataclass(frozen=True)
class PolicyViolation:
    """A single policy violation detected in a refinement suggestion."""
    domain_id: str
    field_name: str
    rule_name: str
    detail: str
    suggestion_text: str = ""


DEFAULT_POLICY = RefinementPolicy()


# ----------------------------------------------------------------
# Refinement diff (for LLM suggestions)
# ----------------------------------------------------------------

@dataclass(frozen=True)
class RefinementSuggestion:
    """A single field-level suggestion from the LLM refinement layer.

    Does NOT auto-apply.  The caller must decide whether to accept
    each suggestion (or the entire set) via the --apply flag.
    """
    domain_id: str
    field_name: str                      # which ValueDomain field
    original_value: str
    suggested_value: str
    rationale: str = ""


@dataclass(frozen=True)
class RefinementResult:
    """Complete LLM refinement output for one domain.

    Contains both accepted suggestions and policy violations.
    Suggestions that violate policy are moved to ``violations``
    and excluded from ``suggestions``.
    """
    domain_id: str
    suggestions: Tuple[RefinementSuggestion, ...]
    violations: Tuple[PolicyViolation, ...] = ()
    model_used: str = ""
    token_usage: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # default mutable field workaround for frozen dataclass
        if self.token_usage is None:
            object.__setattr__(self, "token_usage", {})

    @property
    def has_changes(self) -> bool:
        return len(self.suggestions) > 0

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0


# ----------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------

def _domain_to_dict(domain: ValueDomain) -> Dict[str, Any]:
    return {
        "domain_id": domain.domain_id,
        "domain_label": domain.domain_label,
        "value_definition": domain.value_definition,
        "behavioral_translation": domain.behavioral_translation,
        "example_behaviors": [
            {
                "description": b.description,
                "frequency_hint": b.frequency_hint,
            }
            for b in domain.example_behaviors
        ],
        "misalignment_description": domain.misalignment_description,
        "reflection_questions": list(domain.reflection_questions),
        "micro_habits": list(domain.micro_habits),
        "source": domain.source,
        "version": domain.version,
        "revision": domain.revision,
        "change_notes": domain.change_notes,
    }


def _alignment_entry_to_dict(entry: AlignmentEntry) -> Dict[str, Any]:
    return {
        "date_iso": entry.date_iso,
        "review_type": entry.review_type,
        "domain_id": entry.domain_id,
        "importance_score": entry.importance_score,
        "alignment_score": entry.alignment_score,
        "gap_score": entry.gap_score,
        "significant_gap": entry.significant_gap,
        "reflection_text": entry.reflection_text,
        "misalignment_notes": entry.misalignment_notes,
        "next_adjustment": entry.next_adjustment,
        "transcript": entry.transcript,
        "audio_url": entry.audio_url,
        "ai_summary": entry.ai_summary,
        "tags": list(entry.tags),
    }


def _refinement_suggestion_to_dict(s: RefinementSuggestion) -> Dict[str, Any]:
    return {
        "domain_id": s.domain_id,
        "field_name": s.field_name,
        "original_value": s.original_value,
        "suggested_value": s.suggested_value,
        "rationale": s.rationale,
    }


def _policy_violation_to_dict(v: PolicyViolation) -> Dict[str, Any]:
    return {
        "domain_id": v.domain_id,
        "field_name": v.field_name,
        "rule_name": v.rule_name,
        "detail": v.detail,
        "suggestion_text": v.suggestion_text,
    }


def _refinement_result_to_dict(r: RefinementResult) -> Dict[str, Any]:
    return {
        "domain_id": r.domain_id,
        "model_used": r.model_used,
        "token_usage": dict(r.token_usage),
        "has_changes": r.has_changes,
        "has_violations": r.has_violations,
        "suggestions": [_refinement_suggestion_to_dict(s) for s in r.suggestions],
        "violations": [_policy_violation_to_dict(v) for v in r.violations],
    }


def value_record_from_dict(data: Dict[str, Any]) -> ValueRecord:
    """Reconstruct a ValueRecord from a JSON-parsed dict."""
    domains = []
    for d in data["domains"]:
        behaviors = tuple(
            ValueBehavior(
                description=b["description"],
                frequency_hint=b.get("frequency_hint", ""),
            )
            for b in d["example_behaviors"]
        )
        domains.append(ValueDomain(
            domain_id=d["domain_id"],
            domain_label=d["domain_label"],
            value_definition=d["value_definition"],
            behavioral_translation=d["behavioral_translation"],
            example_behaviors=behaviors,
            misalignment_description=d.get("misalignment_description", ""),
            reflection_questions=tuple(d.get("reflection_questions", ())),
            micro_habits=tuple(d.get("micro_habits", ())),
            source=d.get("source", "Manual"),
            version=d.get("version", 1),
            revision=d.get("revision", 0),
            change_notes=d.get("change_notes", ""),
        ))
    return ValueRecord(
        version=data["version"],
        review_quarter=data["review_quarter"],
        domains=tuple(domains),
        notes=data.get("notes", ""),
    )


def alignment_entry_from_dict(data: Dict[str, Any]) -> AlignmentEntry:
    """Reconstruct an AlignmentEntry from a JSON-parsed dict.

    Note: gap_score and significant_gap are computed properties and
    are NOT read from the dict — they are derived from the two scores.
    """
    return AlignmentEntry(
        date_iso=data["date_iso"],
        review_type=data["review_type"],
        domain_id=data["domain_id"],
        importance_score=data["importance_score"],
        alignment_score=data["alignment_score"],
        reflection_text=data.get("reflection_text", ""),
        misalignment_notes=data.get("misalignment_notes", ""),
        next_adjustment=data.get("next_adjustment", ""),
        transcript=data.get("transcript", ""),
        audio_url=data.get("audio_url", ""),
        ai_summary=data.get("ai_summary", ""),
        tags=tuple(data.get("tags", ())),
    )
