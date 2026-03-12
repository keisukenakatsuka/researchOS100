"""067 Deep Research Planner — service logic.

Transforms a free-form research request into a structured ResearchPlan
that drives all downstream pipeline steps (068-072).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.models.research_plan import ResearchPlan
from src.deep_research import generate_run_id
from src.deep_research.frameworks import get_framework

logger = logging.getLogger("067_planner")

# LLM model to use for request analysis
_MODEL = "claude-sonnet-4-20250514"


# ── Topic / subtype taxonomy ───────────────────────────────────────

VALID_TOPICS = [
    "company",
    "person",
    "market",
    "technology",
    "policy",
    "product",
]

VALID_SUBTYPES: Dict[str, List[str]] = {
    "company":    ["startup", "enterprise", "lp", "vc", "academic", "general"],
    "person":     ["executive", "academic_person", "investor_person", "public_figure", "general"],
    "market":     ["general"],
    "technology": ["general"],
    "policy":     ["general"],
    "product":    ["general"],
}

# ── Legacy intent types (kept for backward compatibility) ──────────

VALID_INTENTS = [
    "company_research",
    "person_research",
    "interview_prep",
    "tech_review",
    "policy_analysis",
    "issue_analysis",
    "general_research",
]

# intent → (topic, subtype) mapping for legacy fallback
_INTENT_TO_TOPIC: Dict[str, tuple] = {
    "company_research": ("company", "general"),
    "person_research":  ("person", "general"),
    "interview_prep":   ("person", "general"),
    "tech_review":      ("technology", "general"),
    "policy_analysis":  ("policy", "general"),
    "issue_analysis":   ("general", "general"),
    "general_research": ("general", "general"),
}

# topic → intent mapping for generating backward-compatible intent
_TOPIC_TO_INTENT: Dict[str, str] = {
    "company":    "company_research",
    "person":     "person_research",
    "market":     "general_research",
    "technology": "tech_review",
    "policy":     "policy_analysis",
    "product":    "general_research",
    "general":    "general_research",
}

# ── Query type categories ──────────────────────────────────────────

QUERY_TYPES = ["factual", "news", "background", "opinion"]

# ── LLM prompt ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a research planning assistant. Given a free-form research request,
produce a structured research plan as a JSON object.

Analyze the request and output the following fields:

- topic: one of [company, person, market, technology, policy, product]
  - "company": any entity with legal personality or organizational substance \
(startup, enterprise, VC fund, LP/pension fund, university, research institute)
  - "person": a specific individual (executive, researcher, investor, public figure)
  - "market": a market, industry, or sector (market size, trends, players)
  - "technology": a technology, methodology, or technical approach
  - "policy": a policy, regulation, or institutional framework
  - "product": a specific product or service (features, competitors, reviews)
- subtype: refine the topic further
  - For company: one of [startup, enterprise, lp, vc, academic, general]
    - "startup": early-stage, high-growth, pre-IPO (Series A/B/C, Y Combinator, etc.)
    - "enterprise": large/public company, established business
    - "lp": institutional investor (pension fund, insurance, SWF, endowment)
    - "vc": venture capital or private equity fund
    - "academic": university, research institute, think tank
    - "general": if unsure or does not fit above
  - For person: one of [executive, academic_person, investor_person, public_figure, general]
    - "executive": corporate executive, C-suite, board member
    - "academic_person": professor, researcher, PhD
    - "investor_person": individual GP, angel investor, fund manager
    - "public_figure": politician, bureaucrat, journalist
    - "general": if unsure or does not fit above
  - For other topics: use "general"
- targets: list of entities to research (companies, people, technologies, etc.)
- key_questions: 3-7 specific questions the research should answer
- search_queries: 5-10 search queries, each as an object with:
    - query: the search string
    - type: one of [factual, news, background, opinion]
- constraints: object with the following keys:
    - language: "ja" or "en" (detect from request)
    - depth: "shallow", "medium", or "deep" (infer from request complexity)
    - region: "global", "jp", "us", or other ISO 3166-1 alpha-2 code

The request may be in Japanese or English. Detect the language and set
constraints.language accordingly. Generate search_queries in the same
language as the request, but also include English queries if the topic
is international.

Do NOT include "intent" or "deliverables" fields — they will be determined \
automatically.

Return ONLY a single JSON object. No markdown fences, no explanation."""


# ── Core functions ──────────────────────────────────────────────────

def analyze_request(request: str, llm_client: Any) -> Dict[str, Any]:
    """Use LLM to parse a free-form request into structured fields.

    Uses messages_create directly (prompt-based JSON) since not all
    models support output_config structured output.

    Args:
        request: Free-form research request text.
        llm_client: A ClaudeClient instance (from src.llm.claude_client).

    Returns:
        Dict with intent, targets, key_questions, search_queries,
        constraints.
    """
    body = {
        "model": _MODEL,
        "max_tokens": 2048,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": request}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.warning("LLM call failed (%s), falling back to rule-based analysis", e)
        return _fallback_analyze(request)

    text = _extract_text(resp)
    if not text:
        logger.warning("LLM returned empty text, falling back to rule-based analysis")
        return _fallback_analyze(request)

    try:
        analysis = json.loads(text)
    except json.JSONDecodeError:
        # Try stripping markdown fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        try:
            analysis = json.loads(cleaned.strip())
        except json.JSONDecodeError as e2:
            logger.warning("LLM JSON parse failed (%s), falling back", e2)
            return _fallback_analyze(request)

    logger.info(
        "LLM analysis: topic=%s, subtype=%s, targets=%s, queries=%d",
        analysis.get("topic"),
        analysis.get("subtype"),
        analysis.get("targets"),
        len(analysis.get("search_queries", [])),
    )
    return analysis


def _extract_text(resp: Dict[str, Any]) -> str:
    """Extract text from a Claude messages API response."""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def _fallback_analyze(request: str) -> Dict[str, Any]:
    """Rule-based fallback when LLM is unavailable."""
    is_ja = any("\u3000" <= c <= "\u9fff" for c in request)
    lang = "ja" if is_ja else "en"

    return {
        "topic": "general",
        "subtype": "general",
        "targets": [request[:60]],
        "key_questions": [
            "この調査対象の概要は？" if is_ja else "What is the overview?",
            "最新の動向は？" if is_ja else "What are the latest developments?",
            "主要な論点は？" if is_ja else "What are the key issues?",
        ],
        "search_queries": [
            {"query": request, "type": "background"},
            {"query": f"{request} {'最新' if is_ja else 'latest'}", "type": "news"},
            {"query": f"{request} {'概要' if is_ja else 'overview'}", "type": "factual"},
        ],
        "constraints": {"language": lang, "depth": "medium", "region": "global"},
    }


def _normalize_queries(raw_queries: list) -> tuple[list[str], list[dict]]:
    """Normalize search_queries from LLM into flat list and typed list.

    Handles both formats:
    - list[str]: legacy flat format → type defaults to "background"
    - list[dict]: typed format with {query, type}

    Returns:
        (flat_queries, typed_queries) where flat_queries is list[str]
        and typed_queries is list[{query, type}].
    """
    flat: list[str] = []
    typed: list[dict] = []

    for item in raw_queries:
        if isinstance(item, str):
            flat.append(item)
            typed.append({"query": item, "type": "background"})
        elif isinstance(item, dict):
            q = item.get("query", "")
            t = item.get("type", "background")
            if t not in QUERY_TYPES:
                t = "background"
            flat.append(q)
            typed.append({"query": q, "type": t})

    return flat, typed


def _normalize_constraints(raw: dict) -> dict:
    """Ensure constraints has all required keys with defaults."""
    return {
        "language": raw.get("language", "ja"),
        "depth": raw.get("depth", "medium"),
        "region": raw.get("region", "global"),
    }


def recall_knowledge(
    request: str,
    *,
    notion_client: Any = None,
) -> Dict[str, List[str]]:
    """Search existing Knowledge Memory Layer for related entities.

    Delegates to src.deep_research.recall for the actual search.

    Args:
        request: The original research request text.
        notion_client: Optional NotionClient for Notion queries.

    Returns:
        Dict with recalled_evidence_ids and recalled_claim_ids.
    """
    from src.deep_research.recall import recall_knowledge as _recall

    result = _recall(request, notion_client=notion_client)
    return {
        "recalled_evidence_ids": result.get("evidence_ids", []),
        "recalled_claim_ids": result.get("claim_ids", []),
    }


def _resolve_topic_subtype(analysis: Dict[str, Any]) -> tuple:
    """Resolve topic and subtype from LLM analysis with fallback.

    Handles three cases:
    1. LLM returned topic + subtype (new prompt) → validate and use
    2. LLM returned intent only (old prompt or cached) → derive from intent
    3. Neither → default to general/general

    Returns:
        (topic, subtype) tuple.
    """
    topic = analysis.get("topic", "")
    subtype = analysis.get("subtype", "")

    # Case 1: topic present — validate
    if topic:
        if topic not in VALID_TOPICS:
            logger.warning("Unknown topic '%s', defaulting to general", topic)
            topic = "general"
        valid_subs = VALID_SUBTYPES.get(topic, ["general"])
        if subtype and subtype not in valid_subs:
            logger.warning(
                "Unknown subtype '%s' for topic '%s', defaulting to general",
                subtype, topic,
            )
            subtype = "general"
        if not subtype:
            subtype = "general"
        return topic, subtype

    # Case 2: legacy intent present — derive topic/subtype
    intent = analysis.get("intent", "")
    if intent and intent in _INTENT_TO_TOPIC:
        topic, subtype = _INTENT_TO_TOPIC[intent]
        logger.info("Derived topic=%s, subtype=%s from legacy intent=%s", topic, subtype, intent)
        return topic, subtype

    # Case 3: nothing — general fallback
    return "general", "general"


def build_plan(
    run_id: str,
    request: str,
    analysis: Dict[str, Any],
    recalled: Dict[str, List[str]],
) -> ResearchPlan:
    """Assemble a ResearchPlan from analysis results.

    Args:
        run_id: The generated run identifier.
        request: Original request text.
        analysis: Structured fields from LLM analysis.
        recalled: Knowledge Recall results.

    Returns:
        A fully populated ResearchPlan.
    """
    # Resolve topic/subtype (canonical internal model)
    topic, subtype = _resolve_topic_subtype(analysis)
    framework_id = f"{topic}.{subtype}"

    # Derive intent for backward compatibility
    intent = _TOPIC_TO_INTENT.get(topic, "general_research")

    # Determine deliverables from framework
    fw = get_framework(topic, subtype)
    deliverables = fw.deliverables if fw.deliverables else ["evidence_memo"]

    # Normalize search_queries (typed → flat)
    flat_queries, typed_queries = _normalize_queries(
        analysis.get("search_queries", [])
    )
    logger.info(
        "Query types: %s",
        {t: sum(1 for q in typed_queries if q["type"] == t) for t in QUERY_TYPES if any(q["type"] == t for q in typed_queries)},
    )

    # Normalize constraints
    constraints = _normalize_constraints(analysis.get("constraints", {}))

    return ResearchPlan(
        run_id=run_id,
        request=request,
        intent=intent,
        created_at=datetime.now(),
        targets=analysis.get("targets", []),
        key_questions=analysis.get("key_questions", []),
        search_queries=flat_queries,
        deliverables=deliverables,
        recalled_evidence_ids=recalled.get("recalled_evidence_ids", []),
        recalled_claim_ids=recalled.get("recalled_claim_ids", []),
        constraints=constraints,
        topic=topic,
        subtype=subtype,
        framework_id=framework_id,
    )


def run(
    request: str,
    llm_client: Any,
    run_id: Optional[str] = None,
    notion_client: Any = None,
) -> ResearchPlan:
    """Execute the planner pipeline.

    This is the main entry point called by the CLI script.

    Args:
        request: Free-form research request text.
        llm_client: A ClaudeClient instance.
        run_id: Optional pre-generated run_id (for re-runs).
        notion_client: Optional NotionClient for Knowledge Recall.

    Returns:
        A ResearchPlan ready for downstream pipeline steps.
    """
    if run_id is None:
        run_id = generate_run_id()

    logger.info("Planning research run %s for request: %s", run_id, request[:80])

    analysis = analyze_request(request, llm_client)
    recalled = recall_knowledge(request, notion_client=notion_client)
    plan = build_plan(run_id, request, analysis, recalled)

    logger.info(
        "Plan generated: topic=%s, subtype=%s, framework=%s, intent=%s, "
        "targets=%s, queries=%d, deliverables=%s",
        plan.topic,
        plan.subtype,
        plan.framework_id,
        plan.intent,
        plan.targets,
        len(plan.search_queries),
        plan.deliverables,
    )
    return plan
