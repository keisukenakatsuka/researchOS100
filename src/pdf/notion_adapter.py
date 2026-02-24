# src/pdf/notion_adapter.py
"""Notion record creation adapter for the PDF inbox pipeline.

Ported from notebook 031 cells 05, 06, 08.

This module provides:
- check_for_duplicate(): capability-aware deduplication
- create_notion_paper_record(): create Papers DB entry with rich fields
"""
from __future__ import annotations

import inspect
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.pdf.metadata import (
    extract_pdf_text_for_llm,
    make_dedup_key,
    normalize_title,
    openai_text_json,
    parse_json_object_loose,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# ID helpers
# ----------------------------------------------------------------


def make_source_uid(metadata: Dict[str, Any], pdf_path: Path) -> str:
    doi = (metadata.get("doi") or "").strip()
    arxiv = (metadata.get("arxiv_id") or "").strip()
    if doi:
        return f"doi:{doi.lower()}"
    if arxiv:
        return f"arxiv:{arxiv.lower()}"
    return f"file:{pdf_path.name.lower()}"


def make_paper_dedup_key(source_uid: str) -> str:
    return make_dedup_key(f"paper|{source_uid}")


def to_authors_year(metadata: Dict[str, Any]) -> str:
    authors = metadata.get("authors") or []
    authors = [str(a).strip() for a in authors if str(a).strip()]
    authors_str = ", ".join(authors)

    year = metadata.get("year")
    year_str = ""
    if isinstance(year, int):
        year_str = str(year)
    elif isinstance(year, str) and re.match(r"^\d{4}$", year.strip()):
        year_str = year.strip()

    if authors_str and year_str:
        return f"{authors_str} ({year_str})"
    return authors_str or ""


# ----------------------------------------------------------------
# Dedup match extraction
# ----------------------------------------------------------------


def _extract_match_fields(match: Any) -> Dict[str, Any]:
    if match is None:
        return {"page_id": None, "title": None}
    if isinstance(match, list) and match:
        match = match[0]
    if isinstance(match, dict):
        page_id = match.get("id") or match.get("page_id")
        title = match.get("title") or match.get("name")
        props = match.get("properties")
        if title is None and isinstance(props, dict):
            for k in ["Title", "title", "Name", "Paper", "Paper Title"]:
                v = props.get(k)
                if isinstance(v, dict):
                    tarr = v.get("title") or v.get("rich_text")
                    if isinstance(tarr, list) and tarr:
                        title = tarr[0].get("plain_text")
                        break
        return {"page_id": page_id, "title": title}
    return {"page_id": None, "title": None}


# ----------------------------------------------------------------
# Deduplication checker (Cell 06)
# ----------------------------------------------------------------


def check_for_duplicate(
    metadata: Dict[str, Any],
    *,
    find_duplicate_paper: Optional[Callable] = None,
    find_duplicate_by_doi: Optional[Callable] = None,
    find_duplicate_by_arxiv_id: Optional[Callable] = None,
    find_duplicate_by_title: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Check if a paper already exists in Notion.

    Parameters
    ----------
    metadata : dict
        Must contain at least 'title'; optionally 'doi', 'arxiv_id'.
    find_duplicate_paper : callable, optional
        Generic dedup function from 029 (preferred).
    find_duplicate_by_doi, find_duplicate_by_arxiv_id, find_duplicate_by_title : callable, optional
        Specific dedup functions.

    Returns
    -------
    dict
        Keys: is_duplicate, match_type, page_id, title, error, checked
    """
    result: Dict[str, Any] = {
        "is_duplicate": False,
        "match_type": None,
        "page_id": None,
        "title": None,
        "error": None,
        "checked": [],
    }

    title_raw = (metadata.get("title") or "").strip()
    doi = (metadata.get("doi") or "").strip()
    arxiv_id = (metadata.get("arxiv_id") or "").strip()
    title_norm = normalize_title(title_raw)

    # --- 0) Generic dedup ---
    if find_duplicate_paper and title_raw:
        try:
            source_uid = None
            if doi:
                source_uid = f"doi:{doi.lower()}"
            elif arxiv_id:
                source_uid = f"arxiv:{arxiv_id}"

            dedup_key = make_dedup_key(source_uid or title_norm or title_raw)
            result["checked"].append("generic:find_duplicate_paper")

            match = find_duplicate_paper(
                name=title_raw,
                dedup_key=dedup_key,
                source_uid=source_uid,
                pdf_link=None,
            )

            if isinstance(match, tuple) and len(match) >= 2:
                is_dup = bool(match[0])
                existing_id = match[1]
                reason = match[2] if len(match) >= 3 else None
                if is_dup and existing_id:
                    result.update({
                        "is_duplicate": True,
                        "match_type": f"generic:{reason or 'dedup_key'}",
                        "page_id": existing_id,
                    })
                    return result
            elif match:
                fields = _extract_match_fields(match)
                result.update({"is_duplicate": True, "match_type": "generic", **fields})
                return result
        except Exception as e:
            msg = f"generic dedup failed: {e}"
            logger.warning(msg)
            result["error"] = msg

    # --- 1) DOI ---
    if doi and find_duplicate_by_doi:
        result["checked"].append("doi")
        try:
            match = find_duplicate_by_doi(doi)
            if match:
                fields = _extract_match_fields(match)
                result.update({"is_duplicate": True, "match_type": "doi", **fields})
                return result
        except Exception as e:
            msg = f"DOI dedup failed: {e}"
            logger.warning(msg)
            result["error"] = (result["error"] + "; " if result["error"] else "") + msg

    # --- 2) arXiv ---
    if arxiv_id and find_duplicate_by_arxiv_id:
        result["checked"].append("arxiv")
        try:
            match = find_duplicate_by_arxiv_id(arxiv_id)
            if match:
                fields = _extract_match_fields(match)
                result.update({"is_duplicate": True, "match_type": "arxiv", **fields})
                return result
        except Exception as e:
            msg = f"arXiv dedup failed: {e}"
            logger.warning(msg)
            result["error"] = (result["error"] + "; " if result["error"] else "") + msg

    # --- 3) Title ---
    if title_norm and find_duplicate_by_title:
        result["checked"].append("title")
        try:
            match = find_duplicate_by_title(title_norm)
            if match:
                fields = _extract_match_fields(match)
                result.update({"is_duplicate": True, "match_type": "title", **fields})
                return result
        except Exception as e:
            msg = f"title dedup failed: {e}"
            logger.warning(msg)
            result["error"] = (result["error"] + "; " if result["error"] else "") + msg

    logger.info("No duplicate found. checked=%s", result["checked"])
    return result


# ----------------------------------------------------------------
# Notion fields generation (Cell 08)
# ----------------------------------------------------------------

_PDF_RUNS_DIRNAME = "pdf_runs"
_PDF_TEXT_MAX_CHARS = 120_000
_PDF_TEXT_MAX_PAGES = 20


def _get_pdf_run_dir(artifacts_path: Path, dedup_key: str) -> Path:
    d = artifacts_path / _PDF_RUNS_DIRNAME / dedup_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_or_generate_notion_fields(
    pdf_path: Path,
    metadata: Dict[str, Any],
    pdf_run_dir: Path,
    openai_client: Any,
    openai_model: str,
    openai_temperature: float,
) -> Dict[str, Any]:
    """Generate or load cached notion_fields.json for a paper."""
    notion_fields_path = pdf_run_dir / "notion_fields.json"

    if notion_fields_path.exists():
        try:
            nf = json.loads(notion_fields_path.read_text(encoding="utf-8"))
            if isinstance(nf, dict) and nf:
                return nf
        except Exception as e:
            logger.warning("notion_fields.json exists but failed to load; regenerating. err=%s", e)

    # Minimal fallback if no OpenAI client
    if openai_client is None:
        nf = {
            "name": (metadata.get("title") or pdf_path.stem).strip() or "Untitled Paper",
            "authors_year": to_authors_year(metadata),
            "source": "",
            "type": "",
            "core_idea": "",
            "datasets": "",
            "methods": "",
            "findings": "",
            "notes": "openai_client is not initialized; metadata-only record.",
            "tags": [],
        }
        notion_fields_path.write_text(json.dumps(nf, ensure_ascii=False, indent=2), encoding="utf-8")
        return nf

    paper_text = extract_pdf_text_for_llm(
        pdf_path, max_pages=_PDF_TEXT_MAX_PAGES, max_chars=_PDF_TEXT_MAX_CHARS
    )

    if not paper_text.strip():
        nf = {
            "name": (metadata.get("title") or pdf_path.stem).strip() or "Untitled Paper",
            "authors_year": to_authors_year(metadata),
            "source": "",
            "type": "",
            "core_idea": "",
            "datasets": "",
            "methods": "",
            "findings": "",
            "notes": "PDF text extraction empty; metadata-only record.",
            "tags": [],
        }
        notion_fields_path.write_text(json.dumps(nf, ensure_ascii=False, indent=2), encoding="utf-8")
        return nf

    system = "You are a precise research assistant who writes concise database-ready summaries. Return JSON only."
    user = f"""
Create a Literature Database entry from the paper text below.

Output language rules:
- Name (title): English (single line)
- Authors & Year: English (format: "Last, First (Year)" if possible; else best effort)
- Source: English (journal / venue) (unknown => empty string)
- Type: English (short noun phrase, unknown => empty string)
- Tags: English (JSON array of short tags, 2\u20136 items, Title Case; unknown => empty array)

- Core Idea: Japanese (unknown => "\u4e0d\u660e")
- Datasets: Japanese (unknown => "\u4e0d\u660e")
- Methods: Japanese (unknown => "\u4e0d\u660e")
- Findings: Japanese (unknown => "\u4e0d\u660e")
- Notes: Japanese (short; unknown => "")

Return JSON ONLY with keys:
name, authors_year, source, type, core_idea, datasets, methods, findings, notes, tags

Metadata hints:
- title_hint: {metadata.get("title") or ""}
- doi_hint: {(metadata.get("doi") or "").strip()}
- arxiv_hint: {(metadata.get("arxiv_id") or "").strip()}

Paper text (truncated):
{paper_text[:_PDF_TEXT_MAX_CHARS]}
""".strip()

    out = openai_text_json(openai_client, openai_model, system, user, temperature=openai_temperature)
    nf = parse_json_object_loose(out)

    if not isinstance(nf, dict):
        nf = {}

    nf.setdefault("name", (metadata.get("title") or pdf_path.stem).strip() or "Untitled Paper")
    nf.setdefault("authors_year", to_authors_year(metadata))
    nf.setdefault("source", "")
    nf.setdefault("type", "")
    nf.setdefault("core_idea", "\u4e0d\u660e")
    nf.setdefault("datasets", "\u4e0d\u660e")
    nf.setdefault("methods", "\u4e0d\u660e")
    nf.setdefault("findings", "\u4e0d\u660e")
    nf.setdefault("notes", "")
    nf.setdefault("tags", [])

    if not isinstance(nf.get("tags"), list):
        nf["tags"] = []
    nf["tags"] = [str(t).strip() for t in nf["tags"] if str(t).strip()]

    notion_fields_path.write_text(json.dumps(nf, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved notion_fields.json: %s", notion_fields_path)
    return nf


# ----------------------------------------------------------------
# Notion record creation
# ----------------------------------------------------------------


def _filter_kwargs_for_callable(fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by fn signature. If **kwargs exists, pass all."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        allowed = set(params.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _extract_page_id(notion_result: Any) -> Optional[str]:
    if notion_result is None:
        return None
    if isinstance(notion_result, dict):
        return notion_result.get("page_id") or notion_result.get("id")
    return None


def create_notion_paper_record(
    metadata: Dict[str, Any],
    pdf_path: Path,
    *,
    slide_path: Optional[Path] = None,
    artifacts_path: Path,
    run_id: str,
    create_paper_inbox_fn: Callable,
    openai_client: Any = None,
    openai_model: str = "gpt-4.1-mini",
    openai_temperature: float = 0.2,
) -> Dict[str, Any]:
    """Create a Notion Papers record via create_paper_inbox wrapper.

    Returns
    -------
    dict
        Keys: success, page_id, used_wrapper, payload_preview, error, dedup_key, source_uid
    """
    result: Dict[str, Any] = {
        "success": False,
        "page_id": None,
        "used_wrapper": "create_paper_inbox",
        "payload_preview": None,
        "error": None,
        "dedup_key": None,
        "source_uid": None,
    }

    title = (metadata.get("title") or "").strip() or (pdf_path.stem.strip() or "Untitled Paper")
    source_uid = make_source_uid(metadata, pdf_path)
    dedup_key = make_paper_dedup_key(source_uid)

    result["source_uid"] = source_uid
    result["dedup_key"] = dedup_key

    pdf_run_dir = _get_pdf_run_dir(artifacts_path, dedup_key)

    notion_fields: Dict[str, Any] = {}
    try:
        notion_fields = _get_or_generate_notion_fields(
            pdf_path=pdf_path,
            metadata=metadata,
            pdf_run_dir=pdf_run_dir,
            openai_client=openai_client,
            openai_model=openai_model,
            openai_temperature=openai_temperature,
        )
    except Exception as e:
        logger.warning("notion_fields generation failed; continuing. err=%s", e)

    name = (notion_fields.get("name") or "").strip() or title
    authors_year = (notion_fields.get("authors_year") or "").strip() or to_authors_year(metadata)
    tags = notion_fields.get("tags") if isinstance(notion_fields.get("tags"), list) else None
    if tags is not None:
        tags = [str(t).strip() for t in tags if str(t).strip()] or None

    pdf_status = "LOCAL" if pdf_path.exists() else "NONE"

    extra = {
        "pipeline": "031_pdf_inbox_processor",
        "ingested_at": datetime.today().date().isoformat(),
        "local_pdf_path": str(pdf_path),
        "local_slide_path": str(slide_path) if (slide_path and slide_path.exists()) else None,
        "pdf_run_dir": str(pdf_run_dir),
        "doi": (metadata.get("doi") or "").strip() or None,
        "arxiv_id": (metadata.get("arxiv_id") or "").strip() or None,
        "raw_title": title,
        "authors_list": metadata.get("authors") or [],
        "notion_fields": notion_fields or None,
    }
    extra = {k: v for k, v in extra.items() if v is not None}

    kwargs = dict(
        name=name,
        authors_year=authors_year,
        pdf_link=None,
        tags=tags,
        status="INBOX",
        pdf_status=pdf_status,
        dedup_key=dedup_key,
        source_uid=source_uid,
        run_id=run_id,
        slide1_url=None,
        extra=extra,
    )

    kwargs = _filter_kwargs_for_callable(create_paper_inbox_fn, kwargs)

    result["payload_preview"] = {
        "name": (kwargs.get("name") or "")[:90],
        "authors_year": (kwargs.get("authors_year") or "")[:90],
        "status": kwargs.get("status"),
        "dedup_key": kwargs.get("dedup_key"),
        "tags": kwargs.get("tags"),
    }

    logger.info("Creating Notion paper (INBOX): %.80s...", name)

    try:
        notion_result = create_paper_inbox_fn(**kwargs)
        page_id = _extract_page_id(notion_result)

        if page_id:
            result["success"] = True
            result["page_id"] = page_id
            logger.info("Notion record created: page_id=%s", page_id)
        else:
            msg = f"create_paper_inbox returned no page id (type={type(notion_result)})"
            logger.warning(msg)
            result["error"] = msg
    except Exception as e:
        msg = f"Notion record creation failed: {e}"
        logger.error(msg)
        result["error"] = msg

    return result
