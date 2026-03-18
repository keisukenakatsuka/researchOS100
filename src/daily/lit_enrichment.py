# src/daily/lit_enrichment.py
"""109 LIT Enrichment — service logic.

Enriches LIT DB papers that have PDF but lack content fields
(Core Idea, Methods, Datasets, Findings, Notes, Fundings).

Uses Claude Sonnet to extract structured fields from PDF text,
caches results locally, and writes back to Notion.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("109_lit_enrichment")

# -- constants ---------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LIT_INBOX_DIR = _PROJECT_ROOT / "data" / "downloads" / "lit_inbox"
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "lit_enrichment"
_LLM_MODEL = "claude-sonnet-4-20250514"
_PDF_TEXT_MAX_CHARS = 120_000
_PDF_TEXT_MAX_PAGES = 20
_NOTION_DELAY_SEC = 0.5
_LLM_MAX_RETRIES = 3

_ENRICH_SYSTEM = """\
You are a research paper analyst who creates structured database entries.
Given paper text, extract and return a JSON object with the following fields.

Output language rules:
- Core Idea: Japanese. Summarize the paper's core contribution and novelty in 2-3 sentences.
- Methods: Japanese. Describe the research methodology concisely.
- Datasets: Japanese for descriptions, keep dataset names in English. List datasets used. \
If none mentioned, return "データセット記載なし".
- Findings: Japanese. Summarize the key findings in 2-4 sentences.
- Notes: Japanese. Any additional noteworthy points (limitations, future work, etc.). \
If nothing notable, return empty string.
- Fundings: Japanese for descriptions, keep institution/grant names in English. \
List funding sources. If none mentioned, return "資金情報なし".

Proper nouns (paper titles, author names, institution names, dataset names, method names, \
grant numbers) must remain in their original language (usually English).

Return JSON ONLY with keys: core_idea, methods, datasets, findings, notes, fundings
No markdown fences. No extra text."""

_FIELD_DEFAULTS = {
    "core_idea": "不明",
    "methods": "不明",
    "datasets": "データセット記載なし",
    "findings": "不明",
    "notes": "",
    "fundings": "資金情報なし",
}


# -- helpers -----------------------------------------------------------------


def is_japanese(text: str) -> bool:
    """Check if text contains Japanese characters (hiragana/katakana/kanji)."""
    if not text:
        return False
    return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))


def _normalize_paper(page: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw Notion LIT page into a flat record."""
    from src.notion.properties import extract_property_value as ev

    page_id = page.get("id", "")
    name = ev(page, "Name") or ""
    if not name:
        return None

    return {
        "page_id": page_id,
        "name": name,
        "source_uid": ev(page, "Source UID") or "",
        "pdf_link": ev(page, "PDF Link") or "",
        "pdf_status": ev(page, "PDF Status") or "",
        "status": ev(page, "Status") or "",
        "decision": ev(page, "Decision") or "",
        "core_idea": ev(page, "Core Idea") or "",
        "findings": ev(page, "Findings") or "",
        "methods": ev(page, "Methods") or "",
        "datasets": ev(page, "Datasets") or "",
        "notes": ev(page, "Notes") or "",
        "authors_year": ev(page, "Authors & Year") or "",
    }


# -- Notion query ------------------------------------------------------------


def query_enrichment_candidates(
    notion_client: Any,
    data_source_id: str,
    *,
    backfill: bool = False,
    normalize_lang: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch LIT DB papers eligible for enrichment."""
    if normalize_lang:
        # Get all READ/KEEP + HAS_PDF — filter by language in Python
        filt = {
            "and": [
                {"property": "Status", "select": {"equals": "INBOX"}},
                {"property": "PDF Status", "select": {"equals": "HAS_PDF"}},
                {
                    "or": [
                        {"property": "Decision", "select": {"equals": "READ"}},
                        {"property": "Decision", "select": {"equals": "KEEP"}},
                    ]
                },
            ]
        }
    elif backfill:
        filt = {
            "and": [
                {"property": "Status", "select": {"equals": "INBOX"}},
                {"property": "PDF Status", "select": {"equals": "HAS_PDF"}},
                {
                    "or": [
                        {"property": "Decision", "select": {"equals": "READ"}},
                        {"property": "Decision", "select": {"equals": "KEEP"}},
                    ]
                },
                {"property": "Core Idea", "rich_text": {"is_empty": True}},
            ]
        }
    else:
        # Daily flow: any Decision set + HAS_PDF + Core Idea empty
        filt = {
            "and": [
                {"property": "Status", "select": {"equals": "INBOX"}},
                {"property": "Decision", "select": {"is_not_empty": True}},
                {"property": "PDF Status", "select": {"equals": "HAS_PDF"}},
                {"property": "Core Idea", "rich_text": {"is_empty": True}},
            ]
        }

    pages = notion_client.query_data_source(
        data_source_id=data_source_id,
        filter=filt,
        fetch_all=True,
    )
    logger.info("Fetched %d pages from LIT DB", len(pages))

    records = []
    for page in pages:
        rec = _normalize_paper(page)
        if rec is None:
            continue
        if normalize_lang:
            # Skip if Core Idea is already Japanese or empty
            core_idea = rec["core_idea"]
            if not core_idea or is_japanese(core_idea):
                continue
        records.append(rec)

    logger.info("Papers eligible for enrichment: %d", len(records))
    return records


# -- PDF path resolution -----------------------------------------------------


def resolve_pdf_path(paper: Dict[str, Any]) -> Optional[Path]:
    """Locate the downloaded PDF for a paper.

    Uses the same safe_name convention as 074 (src/daily/lit_inbox.py).
    """
    source_uid = paper.get("source_uid", "")
    page_id = paper.get("page_id", "")
    safe_name = re.sub(r"[^\w\-.]", "_", source_uid or page_id)
    pdf_path = _LIT_INBOX_DIR / f"{safe_name}.pdf"
    if pdf_path.exists():
        return pdf_path
    return None


# -- cache -------------------------------------------------------------------


def _load_cache(page_id: str) -> Optional[Dict[str, Any]]:
    """Load cached enrichment result for a page."""
    cache_path = _CACHE_DIR / f"{page_id}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "fields" in data:
            return data
    except Exception as e:
        logger.warning("Cache load failed for %s: %s", page_id, e)
    return None


def _save_cache(page_id: str, data: Dict[str, Any]) -> None:
    """Save enrichment result to cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{page_id}.json"
    cache_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# -- LLM enrichment ----------------------------------------------------------


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, tolerating markdown fences."""
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        m2 = re.search(r"\{.*\}", text, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(0))
            except json.JSONDecodeError:
                pass
    return None


def enrich_paper_fields(
    text: str,
    llm_client: Any,
    *,
    title: str = "",
    authors_year: str = "",
    model: str = _LLM_MODEL,
) -> Dict[str, str]:
    """Call Claude to generate enrichment fields from paper text.

    Returns dict with keys: core_idea, methods, datasets, findings, notes, fundings.
    Falls back to defaults on failure.
    """
    if not text.strip():
        logger.warning("Empty paper text — returning defaults")
        return dict(_FIELD_DEFAULTS)

    user_content = f"""\
Paper text (truncated to {_PDF_TEXT_MAX_CHARS} chars):
{text[:_PDF_TEXT_MAX_CHARS]}

Metadata hints:
- title: {title}
- authors_year: {authors_year}"""

    body = {
        "model": model,
        "max_tokens": 2048,
        "system": _ENRICH_SYSTEM,
        "messages": [{"role": "user", "content": user_content}],
    }

    for attempt in range(_LLM_MAX_RETRIES):
        try:
            resp = llm_client.messages_create(body=body)
            resp_text = ""
            for block in resp.get("content", []):
                if block.get("type") == "text":
                    resp_text = block.get("text", "")
                    break

            parsed = _parse_json_response(resp_text)
            if parsed and isinstance(parsed, dict):
                # Apply defaults for missing keys
                result = {}
                for key, default in _FIELD_DEFAULTS.items():
                    val = parsed.get(key, "")
                    result[key] = str(val).strip() if val else default
                return result

            logger.warning("LLM response not parseable (attempt %d)", attempt + 1)
        except Exception as e:
            logger.warning("LLM call failed (attempt %d): %s", attempt + 1, e)

        if attempt < _LLM_MAX_RETRIES - 1:
            time.sleep(1.0)

    logger.error("LLM enrichment failed after %d retries — returning defaults", _LLM_MAX_RETRIES)
    return dict(_FIELD_DEFAULTS)


# -- Notion update ------------------------------------------------------------


def update_paper_enrichment(
    notion_client: Any,
    page_id: str,
    fields: Dict[str, str],
) -> bool:
    """Update LIT DB page with enrichment fields.

    Strategy: try all fields (including Fundings) in one call.
    If it fails (e.g. Fundings property doesn't exist), retry without Fundings.
    """

    def _build_properties(include_fundings: bool) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        field_map = {
            "core_idea": "Core Idea",
            "methods": "Methods",
            "datasets": "Datasets",
            "findings": "Findings",
            "notes": "Notes",
        }
        if include_fundings:
            field_map["fundings"] = "Fundings"

        for key, prop_name in field_map.items():
            value = fields.get(key, "")
            if value:
                properties[prop_name] = {
                    "rich_text": [
                        {"type": "text", "text": {"content": value[:2000]}}
                    ]
                }
        return properties

    # First attempt: all fields including Fundings
    try:
        props = _build_properties(include_fundings=True)
        if props:
            notion_client.update_page(page_id=page_id, properties=props)
        return True
    except Exception as e:
        logger.warning("Update with Fundings failed, retrying without: %s", e)

    # Retry without Fundings
    try:
        props = _build_properties(include_fundings=False)
        if props:
            notion_client.update_page(page_id=page_id, properties=props)
        return True
    except Exception as e:
        logger.error("Update without Fundings also failed: %s", e)
        return False


# -- single paper processing -------------------------------------------------


def process_single_paper(
    paper: Dict[str, Any],
    *,
    llm_client: Any,
    notion_client: Any,
    enable_writeback: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Process a single paper: cache check → PDF → LLM → cache → Notion.

    Returns a result dict with keys:
      paper_name, page_id, status, fields, error
    """
    page_id = paper["page_id"]
    result: Dict[str, Any] = {
        "paper_name": paper["name"],
        "page_id": page_id,
        "status": "skipped",
        "fields": None,
        "error": None,
    }

    # Check cache
    cached = _load_cache(page_id)
    if cached and not force:
        if cached.get("notion_written"):
            logger.info("[%s] Already enriched (cached + written) — skip", paper["name"][:60])
            result["status"] = "cached_skip"
            result["fields"] = cached.get("fields")
            return result
        else:
            # Cache exists but Notion write pending
            logger.info("[%s] Cache hit, Notion write pending", paper["name"][:60])
            fields = cached.get("fields", {})
            result["fields"] = fields

            if enable_writeback:
                written = update_paper_enrichment(notion_client, page_id, fields)
                if written:
                    cached["notion_written"] = True
                    _save_cache(page_id, cached)
                    result["status"] = "written_from_cache"
                else:
                    result["status"] = "write_failed"
                    result["error"] = "Notion write failed (from cache)"
            else:
                result["status"] = "cached_no_writeback"

            return result

    # Resolve PDF path
    pdf_path = resolve_pdf_path(paper)
    if pdf_path is None:
        logger.warning("[%s] PDF not found — skip", paper["name"][:60])
        result["status"] = "no_pdf"
        result["error"] = "PDF file not found in lit_inbox"
        return result

    # Extract text
    try:
        from src.pdf.metadata import extract_pdf_text_for_llm
        text = extract_pdf_text_for_llm(
            pdf_path, max_pages=_PDF_TEXT_MAX_PAGES, max_chars=_PDF_TEXT_MAX_CHARS
        )
    except Exception as e:
        logger.warning("[%s] Text extraction failed: %s", paper["name"][:60], e)
        result["status"] = "extraction_failed"
        result["error"] = str(e)
        return result

    if not text.strip():
        logger.warning("[%s] Empty text extracted — skip", paper["name"][:60])
        result["status"] = "empty_text"
        result["error"] = "PDF text extraction returned empty"
        return result

    # LLM enrichment
    fields = enrich_paper_fields(
        text,
        llm_client,
        title=paper["name"],
        authors_year=paper.get("authors_year", ""),
    )
    result["fields"] = fields

    # Save to cache
    cache_data = {
        "page_id": page_id,
        "paper_name": paper["name"],
        "enriched_at": datetime.now(timezone.utc).isoformat(),
        "llm_model": _LLM_MODEL,
        "fields": fields,
        "notion_written": False,
    }

    if enable_writeback:
        written = update_paper_enrichment(notion_client, page_id, fields)
        if written:
            cache_data["notion_written"] = True
            result["status"] = "enriched"
            logger.info("[%s] Enriched + written to Notion", paper["name"][:60])
        else:
            result["status"] = "write_failed"
            result["error"] = "Notion write failed"
            logger.warning("[%s] Enriched but Notion write failed", paper["name"][:60])
    else:
        result["status"] = "enriched_no_writeback"
        logger.info("[%s] Enriched (writeback off — cached only)", paper["name"][:60])

    _save_cache(page_id, cache_data)
    return result


# -- batch processing ---------------------------------------------------------


def run_lit_enrichment(
    *,
    notion_client: Any,
    data_source_id: str,
    llm_client: Any,
    backfill: bool = False,
    normalize_lang: bool = False,
    force: bool = False,
    dry_run: bool = False,
    limit: int = 0,
    enable_writeback: bool = False,
) -> Dict[str, Any]:
    """Run the full LIT enrichment pipeline.

    Returns a summary dict.
    """
    summary: Dict[str, Any] = {
        "total": 0,
        "processed": 0,
        "enriched": 0,
        "written": 0,
        "skipped": 0,
        "errors": 0,
        "results": [],
        "dry_run": dry_run,
    }

    papers = query_enrichment_candidates(
        notion_client, data_source_id,
        backfill=backfill,
        normalize_lang=normalize_lang,
    )
    if limit > 0:
        papers = papers[:limit]
    summary["total"] = len(papers)

    if not papers:
        logger.info("No papers eligible for enrichment")
        return summary

    if dry_run:
        logger.info("=== DRY-RUN: %d papers eligible ===", len(papers))
        for p in papers:
            pdf_path = resolve_pdf_path(p)
            cached = _load_cache(p["page_id"])
            logger.info(
                "  [%s] source_uid=%s pdf=%s cached=%s core_idea=%s",
                p["name"][:50],
                p.get("source_uid", "")[:30],
                "yes" if pdf_path else "no",
                "yes" if cached else "no",
                (p.get("core_idea") or "")[:30] or "(empty)",
            )
            summary["results"].append({
                "paper_name": p["name"],
                "page_id": p["page_id"],
                "has_pdf": pdf_path is not None,
                "has_cache": cached is not None,
                "dry_run": True,
            })
        return summary

    # Process each paper
    for i, paper in enumerate(papers, 1):
        logger.info("--- Processing %d/%d ---", i, len(papers))

        result = process_single_paper(
            paper,
            llm_client=llm_client,
            notion_client=notion_client,
            enable_writeback=enable_writeback,
            force=force,
        )

        summary["processed"] += 1
        status = result["status"]

        if status in ("enriched", "written_from_cache"):
            summary["enriched"] += 1
            summary["written"] += 1
        elif status == "enriched_no_writeback":
            summary["enriched"] += 1
        elif status in ("cached_skip", "cached_no_writeback"):
            summary["skipped"] += 1
        elif status in ("no_pdf", "extraction_failed", "empty_text", "write_failed"):
            summary["errors"] += 1

        summary["results"].append(result)

        # Rate limit
        time.sleep(_NOTION_DELAY_SEC)

    return summary
