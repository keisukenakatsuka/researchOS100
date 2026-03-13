# src/daily/lit_inbox.py
"""074 LIT Inbox Processor — service logic.

Queries LIT DB for INBOX papers without a Decision, attempts PDF download,
extracts text via src.pdf.metadata, runs LLM relevance judgment
(READ / KEEP / SKIP), and writes results back to Notion.

Does NOT call 031's process_single_pdf or run_batch.
Reuses src.pdf.metadata individual functions only.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("074_lit_inbox")

# -- constants ---------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LIT_INBOX_DIR = _PROJECT_ROOT / "data" / "downloads" / "lit_inbox"
_LLM_MODEL = "claude-sonnet-4-20250514"
_ARXIV_DELAY_SEC = 3.0
_NOTION_DELAY_SEC = 0.5
_PDF_DOWNLOAD_TIMEOUT_SEC = 60

_JUDGE_SYSTEM = """\
You are a research paper relevance classifier.

Given the paper text below, classify it as one of:
- READ: Directly relevant to innovation policy, VC/LP investment strategy, \
startup ecosystems, or network analysis of these domains. \
Contains actionable insights or novel findings.
- KEEP: Indirectly relevant. Methodology or findings may be useful as \
reference. Related to economics, technology policy, entrepreneurship, \
or organizational theory.
- SKIP: Not relevant to the above domains.

Return ONLY a JSON object: {"decision": "READ|KEEP|SKIP", "reason": "1-2 sentence explanation"}
No markdown fences. No extra text."""


# -- PDF URL resolution ------------------------------------------------------


def resolve_pdf_url(paper: Dict[str, Any]) -> Optional[str]:
    """Resolve a PDF download URL from paper metadata.

    Priority:
    1. Source UID with arxiv: prefix  → arXiv PDF link
    2. PDF Link field ending with .pdf → direct link
    3. DOI via Unpaywall API          → best-effort OA PDF
    4. None (unresolvable)
    """
    source_uid = (paper.get("source_uid") or "").strip()
    if source_uid.lower().startswith("arxiv:"):
        arxiv_id = source_uid.split(":", 1)[1].strip()
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    pdf_link = (paper.get("pdf_link") or "").strip()
    if pdf_link and (pdf_link.lower().endswith(".pdf") or "arxiv.org/pdf" in pdf_link.lower()):
        return pdf_link
    # arXiv abstract URL → convert to PDF URL
    if pdf_link:
        m = re.match(r"https?://arxiv\.org/abs/(\S+)", pdf_link, re.IGNORECASE)
        if m:
            return f"https://arxiv.org/pdf/{m.group(1)}.pdf"

    # DOI → Unpaywall (best-effort, returns None on failure)
    doi = (paper.get("doi") or "").strip()
    if doi:
        return _try_unpaywall_pdf(doi)

    return None


def _try_unpaywall_pdf(doi: str) -> Optional[str]:
    """Try Unpaywall API to find an OA PDF for a DOI."""
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email=researchos@example.com"
        req = urllib.request.Request(url, headers={"User-Agent": "researchOS/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if pdf_url and isinstance(pdf_url, str):
            return pdf_url
    except Exception as e:
        logger.debug("Unpaywall lookup failed for DOI %s: %s", doi, e)
    return None


# -- PDF download -------------------------------------------------------------


def download_pdf(
    url: str,
    dest_path: Path,
    *,
    timeout: int = _PDF_DOWNLOAD_TIMEOUT_SEC,
) -> bool:
    """Download a PDF from URL to dest_path. Returns True on success."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "researchOS/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
        if len(content) < 1024:
            logger.warning("Downloaded file too small (%d bytes): %s", len(content), url)
            return False
        dest_path.write_bytes(content)
        logger.info("Downloaded PDF (%d KB): %s", len(content) // 1024, dest_path.name)
        return True
    except Exception as e:
        logger.warning("PDF download failed for %s: %s", url, e)
        return False


# -- LIT DB query -------------------------------------------------------------


def _normalize_paper(page: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw Notion LIT page into a flat record for processing."""
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
        "tags": ev(page, "Tags") or "",
    }


def query_inbox_papers(
    notion_client: Any,
    data_source_id: str,
) -> List[Dict[str, Any]]:
    """Fetch LIT DB papers with Status=INBOX and no Decision set."""
    filt = {
        "and": [
            {"property": "Status", "select": {"equals": "INBOX"}},
        ]
    }
    pages = notion_client.query_data_source(
        data_source_id=data_source_id,
        filter=filt,
        fetch_all=True,
    )
    logger.info("Fetched %d INBOX papers from LIT DB", len(pages))

    records = []
    for page in pages:
        rec = _normalize_paper(page)
        if rec is None:
            continue
        # Skip papers that already have a Decision
        if rec["decision"]:
            continue
        records.append(rec)

    logger.info("Papers eligible for processing (INBOX, no Decision): %d", len(records))
    return records


# -- LLM relevance judgment ---------------------------------------------------


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON response, tolerating markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def judge_relevance(
    text: str,
    llm_client: Any,
    *,
    paper_title: str = "",
    model: str = _LLM_MODEL,
) -> Dict[str, str]:
    """Use LLM to classify paper as READ / KEEP / SKIP.

    Returns dict with 'decision' and 'reason' keys.
    Falls back to SKIP on failure.
    """
    if not text and not paper_title:
        return {"decision": "SKIP", "reason": "No text available for judgment"}

    user_content = text[:100_000]  # cap for safety
    if not text:
        user_content = f"Title: {paper_title}\n(No full text available — abstract-only judgment)"

    body = {
        "model": model,
        "max_tokens": 256,
        "system": _JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": user_content}],
    }

    for attempt in range(3):
        try:
            resp = llm_client.messages_create(body=body)
            resp_text = ""
            for block in resp.get("content", []):
                if block.get("type") == "text":
                    resp_text = block.get("text", "")
                    break

            parsed = _parse_json_response(resp_text)
            if parsed and "decision" in parsed:
                decision = parsed["decision"].upper().strip()
                if decision in ("READ", "KEEP", "SKIP"):
                    reason = str(parsed.get("reason", ""))[:2000]
                    if not text:
                        reason = f"[abstract-only] {reason}"
                    return {"decision": decision, "reason": reason}
        except Exception as e:
            logger.warning("LLM judgment attempt %d failed: %s", attempt + 1, e)

    return {"decision": "SKIP", "reason": "LLM judgment failed after retries"}


# -- Notion update ------------------------------------------------------------


def update_paper_decision(
    notion_client: Any,
    page_id: str,
    *,
    decision: str,
    reason: str,
    pdf_status: str,
) -> bool:
    """Update a LIT DB page with Decision, Decision Reason, and PDF Status."""
    properties: Dict[str, Any] = {
        "Decision": {"select": {"name": decision}},
        "Decision Reason": {
            "rich_text": [{"type": "text", "text": {"content": reason[:2000]}}]
        },
    }
    if pdf_status:
        properties["PDF Status"] = {"select": {"name": pdf_status}}

    try:
        notion_client.update_page(page_id=page_id, properties=properties)
        return True
    except Exception as e:
        logger.error("Failed to update paper %s: %s", page_id, e)
        return False


# -- Single paper processing --------------------------------------------------


def process_inbox_paper(
    paper: Dict[str, Any],
    *,
    llm_client: Any,
    inbox_dir: Path = _LIT_INBOX_DIR,
    openai_client: Any = None,
    no_slides: bool = True,
) -> Dict[str, Any]:
    """Process a single INBOX paper: resolve URL → download → extract → judge.

    Returns a result dict with keys:
      paper_name, pdf_url, pdf_downloaded, decision, reason, pdf_status, error
    """
    result: Dict[str, Any] = {
        "paper_name": paper["name"],
        "page_id": paper["page_id"],
        "pdf_url": None,
        "pdf_downloaded": False,
        "decision": None,
        "reason": None,
        "pdf_status": paper.get("pdf_status") or "",
        "error": None,
    }

    # Step 1: Resolve PDF URL
    pdf_url = resolve_pdf_url(paper)
    result["pdf_url"] = pdf_url

    if not pdf_url:
        # No PDF URL — judge from title + abstract only
        result["pdf_status"] = "NO_PDF"
        text_for_llm = ""
        title_for_fallback = paper["name"]
        logger.info("[%s] No PDF URL found — abstract-only judgment", paper["name"][:60])
    else:
        # Step 2: Download PDF
        safe_name = re.sub(r"[^\w\-.]", "_", paper.get("source_uid") or paper["page_id"])
        pdf_path = inbox_dir / f"{safe_name}.pdf"

        if pdf_path.exists():
            logger.info("[%s] PDF already exists: %s", paper["name"][:60], pdf_path.name)
            result["pdf_downloaded"] = True
        else:
            # arXiv rate limit
            if "arxiv.org" in pdf_url:
                time.sleep(_ARXIV_DELAY_SEC)
            result["pdf_downloaded"] = download_pdf(pdf_url, pdf_path)

        if result["pdf_downloaded"]:
            result["pdf_status"] = "HAS_PDF"
            # Step 3: Extract text for LLM
            try:
                from src.pdf.metadata import extract_pdf_text_for_llm
                text_for_llm = extract_pdf_text_for_llm(pdf_path)
            except Exception as e:
                logger.warning("[%s] Text extraction failed: %s", paper["name"][:60], e)
                text_for_llm = ""
            title_for_fallback = paper["name"]
        else:
            result["pdf_status"] = "NO_PDF"
            text_for_llm = ""
            title_for_fallback = paper["name"]

    # Step 4: LLM relevance judgment
    judgment = judge_relevance(
        text_for_llm,
        llm_client,
        paper_title=title_for_fallback,
    )
    result["decision"] = judgment["decision"]
    result["reason"] = judgment["reason"]

    logger.info(
        "[%s] → %s (pdf=%s)",
        paper["name"][:60],
        result["decision"],
        result["pdf_status"],
    )
    return result


# -- Batch processing ---------------------------------------------------------


def run_lit_inbox(
    *,
    notion_client: Any,
    data_source_id: str,
    llm_client: Any,
    dry_run: bool = False,
    limit: int = 0,
    no_slides: bool = True,
    inbox_dir: Path = _LIT_INBOX_DIR,
) -> Dict[str, Any]:
    """Run the full LIT inbox processing pipeline.

    Returns a summary dict with counts and per-paper results.
    """
    summary: Dict[str, Any] = {
        "total": 0,
        "processed": 0,
        "read": 0,
        "keep": 0,
        "skip": 0,
        "pdf_downloaded": 0,
        "no_pdf": 0,
        "errors": 0,
        "results": [],
        "dry_run": dry_run,
    }

    papers = query_inbox_papers(notion_client, data_source_id)
    if limit > 0:
        papers = papers[:limit]
    summary["total"] = len(papers)

    if not papers:
        logger.info("No INBOX papers to process")
        return summary

    if dry_run:
        logger.info("=== DRY-RUN: %d papers eligible ===", len(papers))
        for p in papers:
            pdf_url = resolve_pdf_url(p)
            logger.info(
                "  [%s] source_uid=%s pdf_url=%s",
                p["name"][:60],
                p.get("source_uid", "")[:30],
                pdf_url or "(none)",
            )
            summary["results"].append({
                "paper_name": p["name"],
                "page_id": p["page_id"],
                "pdf_url": pdf_url,
                "dry_run": True,
            })
        return summary

    # Process each paper
    inbox_dir.mkdir(parents=True, exist_ok=True)

    for i, paper in enumerate(papers, 1):
        logger.info("--- Processing %d/%d ---", i, len(papers))

        result = process_inbox_paper(
            paper,
            llm_client=llm_client,
            inbox_dir=inbox_dir,
            no_slides=no_slides,
        )

        # Write back to Notion
        updated = update_paper_decision(
            notion_client,
            paper["page_id"],
            decision=result["decision"],
            reason=result["reason"],
            pdf_status=result["pdf_status"],
        )
        if not updated:
            result["error"] = "Notion update failed"
            summary["errors"] += 1

        # Track counts
        summary["processed"] += 1
        d = result["decision"]
        if d == "READ":
            summary["read"] += 1
        elif d == "KEEP":
            summary["keep"] += 1
        elif d == "SKIP":
            summary["skip"] += 1

        if result["pdf_downloaded"]:
            summary["pdf_downloaded"] += 1
        if result["pdf_status"] == "NO_PDF":
            summary["no_pdf"] += 1

        summary["results"].append(result)

        # Notion API rate limit
        time.sleep(_NOTION_DELAY_SEC)

    return summary
