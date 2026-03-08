"""069 Extract Structured Evidence — service logic.

Reads sources.json, extracts fine-grained Evidence items from each
successfully-fetched Source using LLM (with rule-based fallback),
and produces evidence.json.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.deep_research import load_step_output, make_evidence_id
from src.deep_research.constants import VALID_TAGS, VALID_TAGS_SET

logger = logging.getLogger("069_extractor")

# -- constants -----------------------------------------------------------

_MODEL = "claude-sonnet-4-20250514"

# Maximum characters of fetched_text sent to LLM per source.
MAX_SOURCE_TEXT_CHARS = 8000

# Maximum evidence items extracted per source (LLM or fallback).
MAX_EVIDENCE_PER_SOURCE = 10

# -- LLM prompt ----------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an evidence extraction assistant.
Given the title, URL, and body text of a web source, extract concrete,
fine-grained evidence items.

Each evidence item must be:
- ONE fact, ONE data point, or ONE quotation (not a paragraph).
- Verifiable: it should reference a specific event, number, date, or quote.
- Self-contained: understandable without reading the full source.

For each evidence item output a JSON object with:
- statement: a concise factual sentence (1-2 sentences max)
- tags: list of 1-3 tags from: strategy, funding, product, partnership,
  governance, research, market, legal, personnel

Return a JSON array of evidence objects.
Extract at most 10 items.  Prefer quality over quantity.

Return ONLY a JSON array.  No markdown fences, no explanation."""

# -- core functions ------------------------------------------------------


def _filter_sources(sources_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return sources eligible for extraction."""
    eligible = []
    for src in sources_data.get("sources", []):
        if src.get("fetch_status") == "success" and src.get("fetched_text"):
            eligible.append(src)
    logger.info(
        "Eligible sources: %d / %d",
        len(eligible),
        len(sources_data.get("sources", [])),
    )
    return eligible


def extract_from_source_llm(
    source: Dict[str, Any],
    llm_client: Any,
) -> List[Dict[str, Any]]:
    """Use LLM to extract evidence from a single source."""
    text = source.get("fetched_text", "")[:MAX_SOURCE_TEXT_CHARS]
    title = source.get("title", "")
    url = source.get("url", "")

    user_msg = (
        f"Title: {title}\n"
        f"URL: {url}\n"
        f"---\n"
        f"{text}"
    )

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.warning("LLM extraction failed for %s: %s", source.get("source_id"), e)
        return []

    # Extract text from response
    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    if not resp_text:
        return []

    # Parse JSON array
    try:
        items = json.loads(resp_text)
    except json.JSONDecodeError:
        # Strip markdown fences
        cleaned = resp_text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        try:
            items = json.loads(cleaned.strip())
        except json.JSONDecodeError as e2:
            logger.warning("LLM JSON parse failed for %s: %s", source.get("source_id"), e2)
            return []

    if not isinstance(items, list):
        items = [items]

    return items[:MAX_EVIDENCE_PER_SOURCE]


def extract_from_source_fallback(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rule-based fallback: split text into sentence-level evidence."""
    text = source.get("fetched_text", "") or ""
    if not text:
        return []

    # Split into sentences
    sentences = re.split(r'(?<=[.。!！?？])\s+', text)

    items: List[Dict[str, Any]] = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 500:
            continue
        # Prefer sentences with factual signals
        if not _has_factual_signal(sent):
            continue
        items.append({
            "statement": sent,
            "tags": _infer_tags(sent),
        })
        if len(items) >= MAX_EVIDENCE_PER_SOURCE:
            break

    return items


def _has_factual_signal(sent: str) -> bool:
    """Check if a sentence likely contains a factual claim."""
    # Numbers, dates, percentages, currency, proper nouns (capitalized words)
    patterns = [
        r'\d',                       # contains a digit
        r'\$|€|¥|£',                 # currency
        r'%|percent',                # percentage
        r'billion|million|thousand', # magnitude
        r'(?:19|20)\d{2}',             # year (1900-2099)
        r'announce|launch|release|invest|acquire|partner|fund',
    ]
    text_lower = sent.lower()
    return any(re.search(p, text_lower) for p in patterns)


def _infer_tags(sent: str) -> List[str]:
    """Simple keyword-based tag inference."""
    text_lower = sent.lower()
    tags: List[str] = []
    tag_keywords = {
        "funding": ["invest", "fund", "raise", "valuation", "billion", "million", "$"],
        "product": ["launch", "release", "product", "feature", "api", "model", "gpt", "chatgpt"],
        "partnership": ["partner", "collaboration", "deal", "agreement", "microsoft", "google"],
        "strategy": ["strategy", "plan", "vision", "roadmap", "pivot", "expansion"],
        "governance": ["board", "governance", "policy", "regulation", "safety", "alignment"],
        "research": ["research", "paper", "study", "breakthrough", "benchmark"],
        "market": ["market", "revenue", "growth", "user", "subscriber", "share"],
        "legal": ["lawsuit", "legal", "copyright", "patent", "litigation"],
        "personnel": ["hire", "ceo", "cto", "appoint", "resign", "fired", "joined"],
    }
    for tag, keywords in tag_keywords.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    return tags[:3] if tags else ["strategy"]


def extract_evidence(
    sources_data: Dict[str, Any],
    run_id: str,
    llm_client: Any = None,
) -> List[Dict[str, Any]]:
    """Extract evidence from all eligible sources.

    Uses LLM if *llm_client* is provided, otherwise falls back to
    rule-based sentence extraction.

    Returns list of evidence dicts (not yet wrapped in output envelope).
    """
    eligible = _filter_sources(sources_data)
    all_evidence: List[Dict[str, Any]] = []
    seq = 0

    now = datetime.now().isoformat()

    for src in eligible:
        source_id = src.get("source_id", "")
        source_title = src.get("title", "")

        # Extract
        if llm_client is not None:
            items = extract_from_source_llm(src, llm_client)
            if not items:
                logger.info("LLM returned 0 items for %s, trying fallback", source_id)
                items = extract_from_source_fallback(src)
        else:
            items = extract_from_source_fallback(src)

        # Build evidence records
        for item in items:
            seq += 1
            statement = item.get("statement", "").strip()
            if not statement:
                continue

            tags = item.get("tags", [])
            tags = [t for t in tags if t in VALID_TAGS_SET][:3]
            if not tags:
                tags = ["strategy"]

            all_evidence.append({
                "evidence_id": make_evidence_id(run_id, seq),
                "statement": statement,
                "source_id": source_id,
                "source_title": source_title,
                "confidence": None,
                "confidence_reason": None,
                "extracted_at": now,
                "tags": tags,
            })

        logger.info(
            "Source %s: %d evidence items extracted", source_id, len(items),
        )

    return all_evidence


def build_output(
    run_id: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the evidence.json envelope."""
    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "total_evidence": len(evidence),
        "evidence": evidence,
    }


# -- main entry point ----------------------------------------------------

def run(
    run_id: str,
    llm_client: Any = None,
) -> Dict[str, Any]:
    """Execute the full 069 extractor pipeline.

    1. Load sources.json.
    2. Filter eligible sources (fetch_status=success, fetched_text!=null).
    3. Extract evidence (LLM or fallback).
    4. Build and return evidence.json dict.

    Args:
        run_id: Research run identifier.
        llm_client: Optional ClaudeClient. If None, uses rule-based fallback.

    Returns:
        Dict ready to be saved as ``evidence.json``.
    """
    sources_data = load_step_output(run_id, "068")

    logger.info(
        "=== 069 Extractor: run_id=%s, total_sources=%d ===",
        run_id,
        len(sources_data.get("sources", [])),
    )

    evidence = extract_evidence(sources_data, run_id, llm_client)
    result = build_output(run_id, evidence)

    logger.info(
        "Extractor done: %d evidence items from %d sources",
        result["total_evidence"],
        len(_filter_sources(sources_data)),
    )
    return result
