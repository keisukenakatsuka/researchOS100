# src/pdf/metadata.py
"""PDF metadata extraction and text normalization utilities.

Ported from notebook 031 cells 03 (text normalization) and 04 (metadata extraction).

Provides:
- normalize_text / normalize_title / make_dedup_key  — dedup helpers
- extract_pdf_metadata — local extraction + optional OpenAI repair
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Text normalization (Cell 03)
# ----------------------------------------------------------------


def normalize_text(text: Optional[str]) -> str:
    """Normalize text for dedup comparison (lowercase, collapse whitespace, strip punctuation)."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    normalized = normalized.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[.,;:!?()\[\]{}\"\'\\/]", "", normalized)
    return normalized


def normalize_title(title: Optional[str]) -> str:
    """Normalize paper title — strips common academic prefixes."""
    normalized = normalize_text(title)
    if not normalized:
        return ""
    prefixes = ["towards ", "toward ", "the ", "an ", "a ", "on "]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized.strip()


def make_dedup_key(*parts: Optional[str]) -> str:
    """Generate stable dedup key (SHA-256 truncated to 16 chars) from 1+ parts."""
    cleaned = []
    for p in parts:
        s = normalize_text(p) if p is not None else ""
        if s:
            cleaned.append(s)
    if not cleaned:
        return ""
    blob = "|".join(cleaned)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------
# PDF text extraction helpers (Cell 04)
# ----------------------------------------------------------------

try:
    import fitz  # PyMuPDF

    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False

try:
    import PyPDF2  # noqa: N811

    _HAS_PYPDF2 = True
except Exception:
    _HAS_PYPDF2 = False


def extract_first_pages_text(pdf_path: Path, pages: int = 2) -> str:
    """Extract text from the first *pages* of a PDF (PyMuPDF preferred, PyPDF2 fallback)."""
    if _HAS_FITZ:
        try:
            doc = fitz.open(pdf_path)
            chunks = []
            for i in range(min(pages, len(doc))):
                txt = doc[i].get_text("text") or ""
                if txt.strip():
                    chunks.append(txt)
            doc.close()
            return "\n\n".join(chunks).strip()
        except Exception as e:
            logger.warning("fitz text extraction failed; falling back. err=%s", e)

    if _HAS_PYPDF2:
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                chunks = []
                for i in range(min(pages, len(reader.pages))):
                    try:
                        txt = reader.pages[i].extract_text() or ""
                        if txt.strip():
                            chunks.append(txt)
                    except Exception:
                        pass
                return "\n\n".join(chunks).strip()
        except Exception:
            pass

    return ""


def extract_pdf_text_for_llm(
    pdf_path: Path, max_pages: int = 20, max_chars: int = 120_000
) -> str:
    """Best-effort text extraction for LLM consumption (no OCR)."""
    if _HAS_FITZ:
        try:
            doc = fitz.open(pdf_path)
            chunks: List[str] = []
            for i in range(min(max_pages, len(doc))):
                t = (doc[i].get_text("text") or "").strip()
                if t:
                    chunks.append(t)
                if sum(len(x) for x in chunks) >= max_chars:
                    break
            doc.close()
            return "\n\n".join(chunks).strip()[:max_chars]
        except Exception:
            pass

    if _HAS_PYPDF2:
        try:
            chunks = []
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages[:max_pages]:
                    t = (page.extract_text() or "").strip()
                    if t:
                        chunks.append(t)
                    if sum(len(x) for x in chunks) >= max_chars:
                        break
            return "\n\n".join(chunks).strip()[:max_chars]
        except Exception as e:
            logger.warning("PDF text extraction failed for LLM: %s", e)

    return ""


# ----------------------------------------------------------------
# Suspicious title detection
# ----------------------------------------------------------------

_BAD_TITLE_SUBSTRINGS = [
    "authors listed",
    "acknowledg",
    "table of contents",
    "contents",
    "microsoft word",
    "working paper",
    "preprint",
    "draft",
    "copyright",
    "all rights reserved",
    "editorial",
    "supplementary",
]


def _is_suspicious_title(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return True
    tl = t.lower()
    for s in _BAD_TITLE_SUBSTRINGS:
        if s in tl:
            return True
    if re.fullmatch(r"[0-9a-z\.\-_]{10,}\.pdf", tl):
        return True
    if re.search(r"\b10\.\d{4,}/", tl):
        return True
    if len(t) < 8 or len(t) > 200:
        return True
    sym_ratio = sum(1 for ch in t if not (ch.isalnum() or ch.isspace())) / max(len(t), 1)
    if sym_ratio > 0.25:
        return True
    if tl in {"my title", "untitled", "title"}:
        return True
    return False


# ----------------------------------------------------------------
# Local heuristic: title / authors from first N pages
# ----------------------------------------------------------------

_STOPLINE_RE = re.compile(
    r"(abstract|keywords|jel|introduction|1\.?\s+introduction|contents|table of contents|references)\b",
    flags=re.IGNORECASE,
)


def _clean_line(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _looks_like_name_token(tok: str) -> bool:
    if "@" in tok or len(tok) < 2:
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z\.\-']+$", tok))


def _extract_title_from_text(first_pages_text: str) -> Optional[str]:
    text = (first_pages_text or "").strip()
    if not text:
        return None
    lines = [_clean_line(ln) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln][:200]

    candidates = []
    for ln in lines:
        if _STOPLINE_RE.search(ln):
            break
        ll = ln.lower()
        if any(bad in ll for bad in _BAD_TITLE_SUBSTRINGS):
            continue
        if re.search(r"\bdoi\b|arxiv", ll):
            continue
        if len(ln) < 8 or len(ln) > 220:
            continue
        alpha_ratio = sum(ch.isalpha() for ch in ln) / max(len(ln), 1)
        if alpha_ratio < 0.55:
            continue
        candidates.append(ln)
        if len(candidates) >= 15:
            break

    if not candidates:
        return None

    def score(line: str) -> float:
        alpha = sum(ch.isalpha() for ch in line)
        return alpha + min(len(line), 120) * 0.2

    best = sorted(candidates, key=score, reverse=True)[0].strip()[:200]
    return best or None


def _extract_authors_from_text(first_pages_text: str) -> List[str]:
    text = (first_pages_text or "").strip()
    if not text:
        return []
    lines = [_clean_line(ln) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln][:200]

    author_line = None
    for ln in lines[:60]:
        if _STOPLINE_RE.search(ln):
            break
        ll = ln.lower()
        if ("," in ln) or (" and " in ll) or (" & " in ln):
            toks = re.split(r"[\s,;&]+", ln)
            nameish = sum(1 for t in toks if _looks_like_name_token(t))
            if nameish >= 3:
                author_line = ln
                break

    if not author_line:
        return []

    author_line = re.sub(r"\S+@\S+", "", author_line)
    author_line = re.sub(r"\s+", " ", author_line).strip()
    parts = re.split(r"\s*(?:,|;|\band\b|&)\s*", author_line, flags=re.IGNORECASE)
    parts = [_clean_line(p) for p in parts if _clean_line(p)]

    authors: List[str] = []
    for p in parts:
        if len(p) > 60:
            continue
        if re.search(
            r"\b(university|department|school|institute|center|centre|lab)\b",
            p,
            flags=re.IGNORECASE,
        ):
            continue
        toks = p.split()
        if sum(1 for t in toks if _looks_like_name_token(t)) >= 2:
            authors.append(p)

    seen: set[str] = set()
    out: List[str] = []
    for a in authors:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out[:10]


# ----------------------------------------------------------------
# OCR fallback
# ----------------------------------------------------------------


def _ocr_first_pages_text(pdf_path: Path, pages: int = 2, lang: str = "eng") -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        return ""
    try:
        images = convert_from_path(str(pdf_path), first_page=1, last_page=min(pages, 6))
        chunks = []
        for img in images:
            txt = pytesseract.image_to_string(img, lang=lang) or ""
            if txt.strip():
                chunks.append(txt)
        return "\n\n".join(chunks).strip()
    except Exception:
        return ""


# ----------------------------------------------------------------
# OpenAI JSON helper
# ----------------------------------------------------------------


def openai_text_json(
    openai_client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
) -> str:
    """Call OpenAI and return raw text response (Responses API preferred)."""
    if hasattr(openai_client, "responses"):
        r = openai_client.responses.create(
            model=model,
            temperature=temperature,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        out = (getattr(r, "output_text", None) or "").strip()
        if out:
            return out
        try:
            chunks = []
            for item in getattr(r, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if isinstance(c, dict):
                        if c.get("type") == "output_text" and "text" in c:
                            chunks.append(c["text"])
                    else:
                        if getattr(c, "type", None) == "output_text" and getattr(c, "text", None):
                            chunks.append(c.text)
            return "\n".join(chunks).strip()
        except Exception:
            return str(r)

    if hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
        r = openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (r.choices[0].message.content or "").strip()

    raise AttributeError("openai_client does not support responses or chat.completions.")


def parse_json_object_loose(text: str) -> dict:
    """Parse JSON from text, tolerating markdown fences / surrounding text."""
    if not text or not text.strip():
        raise ValueError("Empty response text (cannot parse JSON).")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in response text.")
        return json.loads(m.group(0))


# ----------------------------------------------------------------
# OpenAI metadata repair
# ----------------------------------------------------------------

_LOCAL_PAGES_FOR_INFERENCE = 2
_LLM_MAX_CHARS = 18_000
_LLM_PAGES_FOR_TEXT = 2


def _repair_metadata_with_openai(
    openai_client: Any,
    openai_model: str,
    openai_temperature: float,
    pdf_path: Path,
    filename: str,
    extracted_text: str,
    current: Dict[str, Any],
) -> Dict[str, Any]:
    txt = (extracted_text or "").strip()
    if len(txt) > _LLM_MAX_CHARS:
        txt = txt[:_LLM_MAX_CHARS]

    system = "You are a precise academic metadata extractor. Output JSON only."
    user = f"""
Return JSON ONLY with keys:
- title: string (clean paper title)
- authors: array of strings (best-effort; empty allowed)
- year: integer or null (publication year if confidently present)

Rules:
- Infer from the most likely title/authors block near the beginning.
- Avoid acknowledgements, author contribution notes, 'Microsoft Word - ...', DOI strings as titles.
- Authors should be personal names (not affiliations).

Context:
- Filename: {filename}
- Current extracted title: {current.get('title') or ""}
- Current extracted authors: {current.get('authors') or []}
- Current DOI: {current.get('doi') or ""}
- Current arXiv: {current.get('arxiv_id') or ""}

First pages text:
{txt}
""".strip()

    raw = openai_text_json(
        openai_client=openai_client,
        model=openai_model,
        system=system,
        user=user,
        temperature=openai_temperature,
    )
    obj = parse_json_object_loose(raw)

    patch: Dict[str, Any] = {}
    title = (obj.get("title") or "").strip()
    if title:
        patch["title"] = title[:200]

    authors = obj.get("authors")
    if isinstance(authors, list):
        patch["authors"] = [str(a).strip() for a in authors if str(a).strip()]

    year = obj.get("year", None)
    if isinstance(year, int) and 1800 <= year <= (datetime.now().year + 1):
        patch["year"] = year
    else:
        patch["year"] = None

    return patch


# ----------------------------------------------------------------
# Main extractor
# ----------------------------------------------------------------


def extract_pdf_metadata(
    pdf_path: Path,
    *,
    openai_client: Any = None,
    openai_model: str = "gpt-4.1-mini",
    openai_temperature: float = 0.2,
    llm_repair_enabled: bool = True,
) -> Dict[str, Any]:
    """Extract metadata from a PDF (local heuristics + optional OpenAI repair).

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.
    openai_client : optional
        Initialized OpenAI client for LLM metadata repair.
    openai_model : str
        Model name for OpenAI calls.
    openai_temperature : float
        Temperature for OpenAI calls.
    llm_repair_enabled : bool
        Whether to attempt LLM-based metadata repair when local heuristics
        produce suspicious results.

    Returns
    -------
    dict
        Keys: doi, arxiv_id, title, authors, year, authors_year,
              raw_text_sample, extraction_errors, llm_used, llm_error
    """
    result: Dict[str, Any] = {
        "doi": None,
        "arxiv_id": None,
        "title": None,
        "authors": [],
        "year": None,
        "authors_year": "",
        "raw_text_sample": "",
        "extraction_errors": [],
        "llm_used": False,
        "llm_error": None,
    }

    # ---------- Local extraction (PyPDF2 metadata + page 1) ----------
    if _HAS_PYPDF2:
        try:
            import PyPDF2 as _PyPDF2

            with open(pdf_path, "rb") as f:
                reader = _PyPDF2.PdfReader(f)
                metadata = reader.metadata
                if metadata:
                    if metadata.get("/Title"):
                        result["title"] = str(metadata["/Title"]).strip()
                    if metadata.get("/Author"):
                        author_str = str(metadata["/Author"]).strip()
                        authors = re.split(r"[;,]|\band\b", author_str, flags=re.IGNORECASE)
                        result["authors"] = [a.strip() for a in authors if a.strip()]
                    if metadata.get("/Subject"):
                        subject = str(metadata["/Subject"])
                        doi_match = re.search(r"10\.\d{4,}/[-._;()/:A-Z0-9]+", subject, re.IGNORECASE)
                        if doi_match:
                            result["doi"] = doi_match.group(0)

                if len(reader.pages) > 0:
                    try:
                        text1 = reader.pages[0].extract_text() or ""
                        if text1:
                            result["raw_text_sample"] = text1[:500]
                            if not result["doi"]:
                                doi_match = re.search(
                                    r"10\.\d{4,}/[-._;()/:A-Z0-9]+", text1, re.IGNORECASE
                                )
                                if doi_match:
                                    result["doi"] = doi_match.group(0)
                            arxiv_match = re.search(
                                r"arXiv:(\d{4}\.\d{4,5})(v\d+)?", text1, re.IGNORECASE
                            )
                            if arxiv_match:
                                result["arxiv_id"] = arxiv_match.group(1)
                    except Exception as e:
                        msg = f"Text extraction failed (page1): {e}"
                        result["extraction_errors"].append(msg)
                        logger.warning(msg)
        except Exception as e:
            msg = f"PDF read failed: {e}"
            result["extraction_errors"].append(msg)
            logger.error("%s for %s", msg, pdf_path.name)

    # ---------- Local inference from first 2 pages ----------
    try:
        first_pages_text = extract_first_pages_text(pdf_path, pages=_LOCAL_PAGES_FOR_INFERENCE)
        if _is_suspicious_title(result.get("title")):
            t2 = _extract_title_from_text(first_pages_text)
            if t2:
                result["title"] = t2
        if not isinstance(result.get("authors"), list) or len(result["authors"]) == 0:
            a2 = _extract_authors_from_text(first_pages_text)
            if a2:
                result["authors"] = a2
    except Exception as e:
        logger.warning("Local 2-page inference failed; continuing. err=%s", e)

    # ---------- Filename fallback for arXiv ----------
    if not result["arxiv_id"]:
        arxiv_match = re.search(r"(\d{4}\.\d{4,5})", pdf_path.stem)
        if arxiv_match:
            result["arxiv_id"] = arxiv_match.group(1)

    # ---------- Filename fallback for title ----------
    if not result["title"]:
        filename = pdf_path.stem
        cleaned = re.sub(r"\d{4}\.\d{4,5}(v\d+)?", "", filename)
        cleaned = re.sub(r"[_-]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        result["title"] = (cleaned[:200] if cleaned else filename[:200])

    # ---------- Conditional OpenAI repair ----------
    try:

        def _likely_bad_authors(authors: list) -> bool:
            return not authors or len(authors) == 1

        def _likely_bad_title(title: str) -> bool:
            t = (title or "").strip()
            if _is_suspicious_title(t):
                return True
            letters = [c for c in t if c.isalpha()]
            if letters:
                lower_ratio = sum(1 for c in letters if c.islower()) / len(letters)
                if lower_ratio > 0.85:
                    return True
            return False

        suspicious = _likely_bad_title(result["title"]) or _likely_bad_authors(result["authors"])

        if llm_repair_enabled and suspicious and openai_client is not None:
            pages_text = extract_first_pages_text(pdf_path, pages=_LLM_PAGES_FOR_TEXT)

            if not pages_text or len(pages_text.strip()) < 300:
                ocr_text = _ocr_first_pages_text(
                    pdf_path, pages=min(2, _LLM_PAGES_FOR_TEXT), lang="eng"
                )
                if ocr_text and len(ocr_text.strip()) >= 200:
                    logger.info("Using OCR text for OpenAI metadata repair.")
                    pages_text = ocr_text

            if pages_text.strip():
                patch = _repair_metadata_with_openai(
                    openai_client=openai_client,
                    openai_model=openai_model,
                    openai_temperature=openai_temperature,
                    pdf_path=pdf_path,
                    filename=pdf_path.name,
                    extracted_text=pages_text,
                    current=result,
                )
                if patch.get("title") and not _is_suspicious_title(patch["title"]):
                    result["title"] = patch["title"]
                if isinstance(patch.get("authors"), list) and patch["authors"]:
                    result["authors"] = patch["authors"]
                result["year"] = patch.get("year", None)
                result["llm_used"] = True

    except Exception as e:
        result["llm_error"] = str(e)
        logger.warning("OpenAI repair failed; continuing with local metadata. err=%s", e)

    # ---------- authors_year ----------
    if result.get("authors"):
        first = result["authors"][0]
        if result.get("year"):
            result["authors_year"] = f"{first} et al., {result['year']}"
        else:
            result["authors_year"] = f"{first}"
    else:
        result["authors_year"] = ""

    logger.info("Metadata: %s | DOI=%s arXiv=%s title=%.50s authors=%d",
                pdf_path.name,
                result["doi"] or "(none)",
                result["arxiv_id"] or "(none)",
                result["title"] or "",
                len(result["authors"]))

    return result
