# src/values/refine.py
"""LLM refinement layer for value domains with policy enforcement.

Accepts a ValueDomain, calls the LLM via the router (OpenAI), validates
suggestions against a RefinementPolicy, and returns a RefinementResult
with clean suggestions and flagged violations.

Key principles:
- NEVER overwrites automatically — returns suggestions only.
- The caller decides whether to apply via --apply flag.
- Policy violations are flagged, never silently dropped.
- All LLM calls go through src/llm/router.py (routed to OpenAI).
- No direct API calls in this module.

Usage::

    from src.values.refine import refine_domain, apply_suggestions
    from src.llm.router import build_router_from_env

    router = build_router_from_env()
    result = refine_domain(domain, router=router)

    if result.has_violations:
        for v in result.violations:
            print(f"  REJECTED [{v.rule_name}]: {v.detail}")

    if result.has_changes:
        for s in result.suggestions:
            print(f"  {s.field_name}: {s.rationale}")

        # Apply only if --apply flag is set
        if apply_flag:
            refined = apply_suggestions(domain, result)
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from src.llm.router import LLMRouter
from src.values.schema import (
    DEFAULT_POLICY,
    PolicyViolation,
    RefinementPolicy,
    RefinementResult,
    RefinementSuggestion,
    ValueBehavior,
    ValueDomain,
    _GOAL_LANGUAGE_PATTERNS,
    _MORALIZING_PATTERNS,
    _PRESCRIPTION_PATTERNS,
    _refinement_result_to_dict,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# System prompt for refinement (policy-aware)
# ----------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are a values alignment coach specializing in behavioral psychology \
and personal development frameworks. Your role is to refine value domain \
definitions to be more actionable, identity-oriented, and measurable.

Output JSON with this exact structure:
{
  "suggestions": [
    {
      "field_name": "<field to change>",
      "original_value": "<current value>",
      "suggested_value": "<improved value>",
      "rationale": "<why this is better>"
    }
  ]
}

Valid field_name values:
- "value_definition"
- "behavioral_translation"
- "misalignment_description"
- "example_behavior_1", "example_behavior_2", "example_behavior_3"
- "reflection_question_1", "reflection_question_2", "reflection_question_3"
- "micro_habit_1", "micro_habit_2", "micro_habit_3"

If no improvements are needed, return: {"suggestions": []}
"""

_POLICY_RULES_TEMPLATE = """
MANDATORY CONSTRAINTS — every suggestion MUST obey these rules:
{rules}
Suggestions that violate any constraint will be automatically rejected.
"""


def _build_system_prompt(policy: RefinementPolicy) -> str:
    """Build the full system prompt with policy rules injected."""
    rules: list[str] = []

    if policy.preserve_identity_orientation:
        rules.append(
            "- value_definition MUST use identity-oriented framing "
            "('I am someone who...').  Never rewrite it as a goal, "
            "instruction, or aspiration."
        )
    if policy.forbid_goal_language:
        rules.append(
            "- NEVER use goal/KPI language: avoid 'achieve', 'maximize', "
            "'minimize', 'hit target', 'KPI', 'OKR', 'metric', "
            "'benchmark', 'deliverable', 'performance indicator'."
        )
    if policy.forbid_moralizing_language:
        rules.append(
            "- NEVER use moralizing language: avoid 'you should', "
            "'one must', 'it is wrong to', 'always be', 'you need to', "
            "'you have to', 'you ought to', 'never fail to'."
        )
    if policy.preserve_length_bounds:
        pct = int(policy.length_tolerance * 100)
        rules.append(
            f"- Keep all suggested text within ±{pct}% of the original "
            f"length.  Do not bloat or over-compress."
        )
    if policy.preserve_principle_based_tone:
        rules.append(
            "- Maintain principle-based tone throughout.  Definitions "
            "describe identity ('I am...'), not prescriptions "
            "('Do X every day').  Behaviors and habits may be specific, "
            "but definitions and translations must stay principled."
        )

    # Always include baseline rules
    rules.extend([
        "- Preserve the user's voice and intent — enhance, don't replace.",
        "- Only suggest changes where genuine improvement is possible.",
        "- If a field is already strong, do NOT suggest a change for it.",
    ])

    constraints_block = _POLICY_RULES_TEMPLATE.format(
        rules="\n".join(rules)
    )

    return _SYSTEM_PROMPT_BASE + constraints_block


# ----------------------------------------------------------------
# Policy validation (post-LLM, pre-accept)
# ----------------------------------------------------------------

def _check_pattern_match(text: str, patterns: tuple[str, ...]) -> Optional[str]:
    """Return the first matching pattern found in text (case-insensitive)."""
    text_lower = text.lower()
    for pat in patterns:
        if pat in text_lower:
            return pat
    return None


def validate_suggestion(
    suggestion: RefinementSuggestion,
    original_domain: ValueDomain,
    policy: RefinementPolicy,
) -> list[PolicyViolation]:
    """Validate a single suggestion against the policy.

    Returns a list of violations (empty if clean).
    """
    violations: list[PolicyViolation] = []
    sv = suggestion.suggested_value
    fn = suggestion.field_name

    # --- preserve_identity_orientation ---
    if policy.preserve_identity_orientation and fn == "value_definition":
        # Must start with identity framing
        if not re.match(r"^I am someone who\b", sv, re.IGNORECASE):
            violations.append(PolicyViolation(
                domain_id=suggestion.domain_id,
                field_name=fn,
                rule_name="preserve_identity_orientation",
                detail=(
                    "value_definition must start with 'I am someone who...'. "
                    f"Got: '{sv[:60]}...'"
                ),
                suggestion_text=sv,
            ))

    # --- forbid_goal_language ---
    if policy.forbid_goal_language:
        match = _check_pattern_match(sv, _GOAL_LANGUAGE_PATTERNS)
        if match:
            violations.append(PolicyViolation(
                domain_id=suggestion.domain_id,
                field_name=fn,
                rule_name="forbid_goal_language",
                detail=f"Contains goal/KPI language: '{match}'",
                suggestion_text=sv,
            ))

    # --- forbid_moralizing_language ---
    if policy.forbid_moralizing_language:
        match = _check_pattern_match(sv, _MORALIZING_PATTERNS)
        if match:
            violations.append(PolicyViolation(
                domain_id=suggestion.domain_id,
                field_name=fn,
                rule_name="forbid_moralizing_language",
                detail=f"Contains moralizing language: '{match}'",
                suggestion_text=sv,
            ))

    # --- preserve_length_bounds ---
    if policy.preserve_length_bounds:
        original_text = _resolve_original_text(original_domain, fn)
        if original_text and len(original_text) > 0:
            ratio = len(sv) / len(original_text)
            lo = 1.0 - policy.length_tolerance
            hi = 1.0 + policy.length_tolerance
            if ratio < lo or ratio > hi:
                violations.append(PolicyViolation(
                    domain_id=suggestion.domain_id,
                    field_name=fn,
                    rule_name="preserve_length_bounds",
                    detail=(
                        f"Length ratio {ratio:.2f} outside [{lo:.2f}, {hi:.2f}]. "
                        f"Original: {len(original_text)} chars, "
                        f"suggested: {len(sv)} chars."
                    ),
                    suggestion_text=sv,
                ))

    # --- preserve_principle_based_tone ---
    if policy.preserve_principle_based_tone:
        # Only enforce on definition and translation fields
        if fn in ("value_definition", "behavioral_translation"):
            match = _check_pattern_match(sv, _PRESCRIPTION_PATTERNS)
            if match:
                violations.append(PolicyViolation(
                    domain_id=suggestion.domain_id,
                    field_name=fn,
                    rule_name="preserve_principle_based_tone",
                    detail=f"Contains prescription language: '{match}'",
                    suggestion_text=sv,
                ))

    return violations


def _resolve_original_text(domain: ValueDomain, field_name: str) -> str:
    """Resolve the original text for a field from the domain."""
    if field_name == "value_definition":
        return domain.value_definition
    elif field_name == "behavioral_translation":
        return domain.behavioral_translation
    elif field_name == "misalignment_description":
        return domain.misalignment_description
    elif field_name.startswith("example_behavior_"):
        idx = int(field_name.split("_")[-1]) - 1
        if 0 <= idx < len(domain.example_behaviors):
            return domain.example_behaviors[idx].description
    elif field_name.startswith("reflection_question_"):
        idx = int(field_name.split("_")[-1]) - 1
        if 0 <= idx < len(domain.reflection_questions):
            return domain.reflection_questions[idx]
    elif field_name.startswith("micro_habit_"):
        idx = int(field_name.split("_")[-1]) - 1
        if 0 <= idx < len(domain.micro_habits):
            return domain.micro_habits[idx]
    return ""


# ----------------------------------------------------------------
# Refinement logic
# ----------------------------------------------------------------

def _build_refinement_prompt(domain: ValueDomain) -> str:
    """Build the user prompt for refining a single domain."""
    behaviors = "\n".join(
        f"  {i+1}. {b.description} [{b.frequency_hint}]"
        for i, b in enumerate(domain.example_behaviors)
    )
    questions = "\n".join(
        f"  {i+1}. {q}"
        for i, q in enumerate(domain.reflection_questions)
    )
    habits = "\n".join(
        f"  {i+1}. {h}"
        for i, h in enumerate(domain.micro_habits)
    )

    return f"""\
Domain: {domain.domain_label}
Domain ID: {domain.domain_id}

Value Definition:
  {domain.value_definition}

Behavioral Translation:
  {domain.behavioral_translation}

Example Behaviors:
{behaviors}

Misalignment Description:
  {domain.misalignment_description}

Reflection Questions:
{questions}

Micro Habits:
{habits}

Please analyze this value domain and suggest specific improvements. \
Focus on making definitions more precise, behaviors more executable, \
and questions more probing. Only suggest changes where real improvement \
is possible.
"""


def _parse_suggestions(
    raw: dict,
    domain_id: str,
) -> List[RefinementSuggestion]:
    """Parse LLM JSON output into typed RefinementSuggestion objects."""
    suggestions = []
    for item in raw.get("suggestions", []):
        field_name = item.get("field_name", "")
        if not field_name:
            logger.warning("Skipping suggestion with empty field_name")
            continue

        suggestions.append(RefinementSuggestion(
            domain_id=domain_id,
            field_name=field_name,
            original_value=item.get("original_value", ""),
            suggested_value=item.get("suggested_value", ""),
            rationale=item.get("rationale", ""),
        ))

    return suggestions


def refine_domain(
    domain: ValueDomain,
    *,
    router: LLMRouter,
    policy: RefinementPolicy = DEFAULT_POLICY,
    use_cache: bool = True,
) -> RefinementResult:
    """Call the LLM to generate refinement suggestions, then validate against policy.

    Parameters
    ----------
    domain : ValueDomain
        The domain to refine.
    router : LLMRouter
        Configured LLM router (routes to OpenAI for refinement).
    policy : RefinementPolicy
        Constraints for suggestion validation.
    use_cache : bool
        Whether to use disk caching for the LLM call.

    Returns
    -------
    RefinementResult
        Typed result with accepted suggestions and policy violations.
        Suggestions that violate policy are moved to ``violations``
        and excluded from ``suggestions``.
    """
    logger.info("Refining domain: %s", domain.domain_label)

    system_prompt = _build_system_prompt(policy)
    user_prompt = _build_refinement_prompt(domain)

    result = router.call_refinement(
        system=system_prompt,
        user=user_prompt,
        use_cache=use_cache,
    )

    raw_suggestions = _parse_suggestions(result.parsed, domain.domain_id)

    # --- Policy validation ---
    accepted: list[RefinementSuggestion] = []
    all_violations: list[PolicyViolation] = []

    for s in raw_suggestions:
        violations = validate_suggestion(s, domain, policy)
        if violations:
            all_violations.extend(violations)
            logger.warning(
                "Domain %s: suggestion for %r rejected — %d violation(s): %s",
                domain.domain_id, s.field_name, len(violations),
                [v.rule_name for v in violations],
            )
        else:
            accepted.append(s)

    logger.info(
        "Domain %s: %d accepted, %d rejected (from %d raw suggestions)",
        domain.domain_id, len(accepted), len(all_violations), len(raw_suggestions),
    )

    return RefinementResult(
        domain_id=domain.domain_id,
        suggestions=tuple(accepted),
        violations=tuple(all_violations),
        model_used=router.get_route("refinement").model,
        token_usage={
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        },
    )


def refine_all_domains(
    domains: tuple[ValueDomain, ...],
    *,
    router: LLMRouter,
    policy: RefinementPolicy = DEFAULT_POLICY,
    use_cache: bool = True,
) -> list[RefinementResult]:
    """Refine all domains in sequence with policy enforcement.

    Returns a list of RefinementResult, one per domain.
    """
    results = []
    for domain in domains:
        result = refine_domain(
            domain, router=router, policy=policy, use_cache=use_cache,
        )
        results.append(result)
    return results


# ----------------------------------------------------------------
# Apply suggestions (only when --apply is set)
# ----------------------------------------------------------------

def apply_suggestions(
    domain: ValueDomain,
    refinement: RefinementResult,
) -> ValueDomain:
    """Apply accepted suggestions to a ValueDomain, returning a new instance.

    Increments the revision counter and sets source to "Hybrid".
    NEVER called automatically — only when the user passes --apply.

    Only applies suggestions that PASSED policy validation
    (i.e. those in ``refinement.suggestions``, NOT in ``violations``).

    Parameters
    ----------
    domain : ValueDomain
        The original domain.
    refinement : RefinementResult
        Validated result with accepted suggestions only.

    Returns
    -------
    ValueDomain
        A new ValueDomain with suggestions applied and revision incremented.
    """
    if not refinement.has_changes:
        return domain

    # Start with current field values
    value_definition = domain.value_definition
    behavioral_translation = domain.behavioral_translation
    misalignment_description = domain.misalignment_description
    behaviors = list(domain.example_behaviors)
    questions = list(domain.reflection_questions)
    habits = list(domain.micro_habits)
    change_parts: list[str] = []

    for s in refinement.suggestions:
        fn = s.field_name

        if fn == "value_definition":
            value_definition = s.suggested_value
            change_parts.append(f"value_definition: {s.rationale}")

        elif fn == "behavioral_translation":
            behavioral_translation = s.suggested_value
            change_parts.append(f"behavioral_translation: {s.rationale}")

        elif fn == "misalignment_description":
            misalignment_description = s.suggested_value
            change_parts.append(f"misalignment_description: {s.rationale}")

        elif fn.startswith("example_behavior_"):
            idx = int(fn.split("_")[-1]) - 1
            if 0 <= idx < len(behaviors):
                behaviors[idx] = ValueBehavior(
                    description=s.suggested_value,
                    frequency_hint=behaviors[idx].frequency_hint,
                )
                change_parts.append(f"{fn}: {s.rationale}")

        elif fn.startswith("reflection_question_"):
            idx = int(fn.split("_")[-1]) - 1
            if 0 <= idx < len(questions):
                questions[idx] = s.suggested_value
                change_parts.append(f"{fn}: {s.rationale}")

        elif fn.startswith("micro_habit_"):
            idx = int(fn.split("_")[-1]) - 1
            if 0 <= idx < len(habits):
                habits[idx] = s.suggested_value
                change_parts.append(f"{fn}: {s.rationale}")

        else:
            logger.warning("Unknown field_name in suggestion: %r", fn)

    change_notes = "; ".join(change_parts)

    return ValueDomain(
        domain_id=domain.domain_id,
        domain_label=domain.domain_label,
        value_definition=value_definition,
        behavioral_translation=behavioral_translation,
        example_behaviors=tuple(behaviors),
        misalignment_description=misalignment_description,
        reflection_questions=tuple(questions),
        micro_habits=tuple(habits),
        source="Hybrid",
        version=domain.version,
        revision=domain.revision + 1,
        change_notes=change_notes,
    )


# ----------------------------------------------------------------
# Diff formatting (for --diff flag)
# ----------------------------------------------------------------

def format_diff(results: list[RefinementResult]) -> str:
    """Format refinement results as a human-readable diff string.

    Includes both accepted suggestions and policy violations.
    Used by the CLI when --diff is passed.
    """
    lines: list[str] = []
    total_suggestions = sum(len(r.suggestions) for r in results)
    total_violations = sum(len(r.violations) for r in results)

    lines.append(
        f"=== Refinement Diff "
        f"({total_suggestions} accepted, {total_violations} rejected) "
        f"across {len(results)} domains ==="
    )
    lines.append("")

    for result in results:
        if not result.has_changes and not result.has_violations:
            lines.append(f"[{result.domain_id}] No changes suggested.")
            lines.append("")
            continue

        if result.suggestions:
            lines.append(
                f"[{result.domain_id}] {len(result.suggestions)} accepted suggestion(s):"
            )
            for s in result.suggestions:
                lines.append(f"  Field: {s.field_name}")
                lines.append(
                    f"  - Original: "
                    f"{s.original_value[:120]}{'...' if len(s.original_value) > 120 else ''}"
                )
                lines.append(
                    f"  + Suggested: "
                    f"{s.suggested_value[:120]}{'...' if len(s.suggested_value) > 120 else ''}"
                )
                lines.append(f"  Rationale: {s.rationale}")
                lines.append("")

        if result.violations:
            lines.append(
                f"[{result.domain_id}] {len(result.violations)} REJECTED by policy:"
            )
            for v in result.violations:
                lines.append(f"  Field: {v.field_name}")
                lines.append(f"  Rule: {v.rule_name}")
                lines.append(f"  Detail: {v.detail}")
                lines.append("")

    return "\n".join(lines)


def refinement_results_to_json(results: list[RefinementResult]) -> list[dict]:
    """Convert refinement results to JSON-serializable dicts."""
    return [_refinement_result_to_dict(r) for r in results]
