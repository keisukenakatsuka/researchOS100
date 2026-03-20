# src/validation/grounding.py
"""Paper text retrieval for literature validation.

Enhances 081 extractor's get_paper_text() with:
1. Google Drive PDF download + local caching
2. Fixed source_uid path resolution (double-extension bug)
3. Fallback: PDF → metadata → title_only

See design.md Section 2.2, Step 1.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIT_DATA_DIR = PROJECT_ROOT / "data" / "lit_review"
LIT_INBOX_DIR = PROJECT_ROOT / "data" / "downloads" / "lit_inbox"
_VALIDATION_PDF_CACHE = PROJECT_ROOT / "data" / "cache" / "validation_pdfs"


def retrieve_paper_text(
    paper_title: str,
    candidate_papers: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Get best available text for a paper by title.

    Resolution order:
    1. Local PDF via source_uid (with double-extension fix)
    2. Google Drive PDF download (cached locally)
    3. LIT DB metadata (core_idea, findings, methods, notes)
    4. Title only

    Returns (text, source) where source is "pdf", "metadata", or "title_only".
    """
    paper_meta = _find_paper_by_title(paper_title, candidate_papers)
    if paper_meta is None:
        logger.warning("Paper not found in candidates: '%s'", paper_title[:60])
        return f"Title: {paper_title}", "title_only"

    # Try 1: Local PDF via source_uid (enhanced path resolution)
    text = _try_local_pdf(paper_meta)
    if text:
        return text, "pdf"

    # Try 2: Google Drive download
    text = _try_gdrive_pdf(paper_meta)
    if text:
        return text, "pdf"

    # Try 3: Metadata fallback (reuse 081 logic)
    text = _try_metadata(paper_meta)
    if text:
        return text, "metadata"

    return f"Title: {paper_meta.get('title', paper_title)}", "title_only"


# ------------------------------------------------------------------
# PDF resolution strategies
# ------------------------------------------------------------------

def _try_local_pdf(paper: Dict[str, Any]) -> Optional[str]:
    """Try to find and extract text from a local PDF.

    Fixes the double-extension bug in 081: file:xxx.pdf → xxx.pdf.pdf
    """
    source_uid = paper.get("source_uid", "")
    if not source_uid:
        return None

    # Strategy 1: Original 081 path (safe_name + .pdf)
    safe_name = re.sub(r"[^\w\-.]", "_", source_uid)
    candidates = [
        LIT_INBOX_DIR / f"{safe_name}.pdf",
    ]

    # Strategy 2: Fix double-extension for file:xxx.pdf UIDs
    if safe_name.endswith(".pdf"):
        candidates.append(LIT_INBOX_DIR / safe_name)  # already has .pdf

    # Strategy 3: Strip prefix (file: → raw filename)
    if source_uid.startswith("file:"):
        raw_name = source_uid[5:]  # remove "file:" prefix
        candidates.append(LIT_INBOX_DIR / raw_name)

    for pdf_path in candidates:
        if pdf_path.exists():
            text = _extract_pdf_text(pdf_path)
            if text:
                logger.info("PDF found via local path: %s", pdf_path.name)
                return text

    return None


def _try_gdrive_pdf(paper: Dict[str, Any]) -> Optional[str]:
    """Download PDF from Google Drive link and cache locally."""
    pdf_link = paper.get("pdf_link", "")
    if not pdf_link or "drive.google.com/file/d/" not in pdf_link:
        return None

    # Extract Google Drive file ID
    m = re.search(r"/file/d/([^/]+)", pdf_link)
    if not m:
        return None
    file_id = m.group(1)

    # Check cache first
    _VALIDATION_PDF_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = _VALIDATION_PDF_CACHE / f"gdrive_{file_id}.pdf"
    if cache_path.exists():
        text = _extract_pdf_text(cache_path)
        if text:
            logger.debug("PDF cache hit: %s", cache_path.name)
            return text

    # Download from Google Drive (handle virus scan confirmation for large files)
    try:
        import requests
        session = requests.Session()

        # First attempt: direct download
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        resp = session.get(download_url, timeout=30, allow_redirects=True)

        # Handle large-file confirmation page
        if resp.status_code == 200 and b"%PDF-" not in resp.content[:10]:
            # Try confirm token extraction
            confirm_token = _extract_gdrive_confirm_token(resp)
            if confirm_token:
                confirmed_url = f"{download_url}&confirm={confirm_token}"
                resp = session.get(confirmed_url, timeout=60, allow_redirects=True)

        if resp.status_code == 200 and len(resp.content) > 1000:
            if resp.content[:5] == b"%PDF-":
                cache_path.write_bytes(resp.content)
                logger.info("Downloaded PDF from Google Drive: %s (%d bytes)", file_id[:12], len(resp.content))
                text = _extract_pdf_text(cache_path)
                if text:
                    return text
            else:
                logger.debug("Google Drive response is not a PDF (file_id=%s)", file_id[:12])
        else:
            logger.debug("Google Drive download failed: status=%d, size=%d", resp.status_code, len(resp.content))
    except Exception as e:
        logger.debug("Google Drive download error for %s: %s", file_id[:12], e)

    return None


def _try_metadata(paper: Dict[str, Any]) -> Optional[str]:
    """Extract text from LIT DB metadata fields."""
    parts = []
    if paper.get("core_idea"):
        parts.append(f"Core Idea: {paper['core_idea']}")
    if paper.get("findings"):
        parts.append(f"Findings: {paper['findings']}")
    if paper.get("methods"):
        parts.append(f"Methods: {paper['methods']}")
    if paper.get("notes"):
        parts.append(f"Notes: {paper['notes']}")

    if parts:
        return "\n\n".join(parts)
    return None


def _extract_gdrive_confirm_token(resp) -> Optional[str]:
    """Extract confirmation token from Google Drive virus scan page."""
    import re
    # Look for confirm token in response text
    for pattern in [
        r'confirm=([0-9A-Za-z_-]+)&',
        r'name="confirm" value="([^"]+)"',
        r'/uc\?export=download&amp;confirm=([^&]+)&',
    ]:
        m = re.search(pattern, resp.text)
        if m:
            return m.group(1)
    # Also check cookies
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _extract_pdf_text(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF file. Returns None on failure or insufficient text."""
    try:
        from src.pdf.metadata import extract_pdf_text_for_llm
        text = extract_pdf_text_for_llm(pdf_path)
        if text and len(text) > 200:
            return text
    except Exception as e:
        logger.warning("PDF text extraction failed for %s: %s", pdf_path.name, e)
    return None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_paper_by_title(
    title: str,
    candidate_papers: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find a paper in candidate_papers by title (case-insensitive)."""
    title_lower = title.strip().lower()
    for paper in candidate_papers:
        if paper.get("title", "").strip().lower() == title_lower:
            return paper
    return None


def load_candidate_papers(run_dir: Path) -> List[Dict[str, Any]]:
    """Load candidate_papers.json and extract the scored_papers list."""
    path = run_dir / "candidate_papers.json"
    if not path.exists():
        logger.warning("candidate_papers.json not found at %s", path)
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("scored_papers", [])
    except Exception as e:
        logger.error("Failed to load candidate_papers.json: %s", e)
        return []


def group_evidence_by_paper(
    evidence_items: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group evidence items by paper_title for batch processing."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in evidence_items:
        title = item.get("paper_title", "unknown")
        groups.setdefault(title, []).append(item)
    return groups
