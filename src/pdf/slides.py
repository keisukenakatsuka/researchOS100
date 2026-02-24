# src/pdf/slides.py
"""Slide IR generation + Gemini rendering for one-page paper summaries.

Ported from notebook 031 cell 07.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pdf.metadata import (
    extract_first_pages_text,
    make_dedup_key,
    openai_text_json,
    parse_json_object_loose,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------

GEMINI_IMAGE_MODEL: str = "gemini-3-pro-image-preview"

OPENAI_MAX_CHARS = 45_000
OPENAI_PAGES_FOR_TEXT = 6

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def safe_slug(s: str, max_len: int = 90) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else "paper"


def _wrap(text: str, max_chars: int = 70) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "\u2026"


def _englishish_title(title: str) -> str:
    t = re.sub(r"\s+", " ", (title or "").strip())
    return t if t else "Untitled Paper"


def _englishish_authors(metadata: Dict[str, Any]) -> str:
    ay = (metadata.get("authors_year") or "").strip()
    if ay:
        return ay
    authors = metadata.get("authors") or []
    if isinstance(authors, list) and authors:
        if len(authors) == 1:
            return str(authors[0]).strip()
        if len(authors) == 2:
            return f"{authors[0]} & {authors[1]}"
        return f"{authors[0]} et al."
    return "Unknown authors"


def _best_source_uid(metadata: Dict[str, Any], pdf_path: Path) -> str:
    doi = (metadata.get("doi") or "").strip()
    arxiv = (metadata.get("arxiv_id") or "").strip()
    if doi:
        return f"doi:{doi.lower()}"
    if arxiv:
        return f"arxiv:{arxiv}"
    return f"file:{pdf_path.stem}"


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------------------------------------------------
# Text extraction for slides (with OCR fallback)
# ----------------------------------------------------------------


def _extract_pages_text(
    pdf_path: Path, metadata: Optional[Dict[str, Any]] = None, pages: int = 6
) -> str:
    metadata = metadata or {}

    try:
        import fitz

        doc = fitz.open(pdf_path)
        chunks = []
        for i in range(min(pages, len(doc))):
            txt = doc[i].get_text("text") or ""
            if txt.strip():
                chunks.append(txt)
        doc.close()
        joined = "\n\n".join(chunks).strip()
        if len(joined) >= 300:
            return joined
    except Exception:
        pass

    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path), first_page=1, last_page=min(pages, 6))
        ocr_chunks = []
        for img in images:
            ocr_txt = pytesseract.image_to_string(img, lang="eng")
            if ocr_txt.strip():
                ocr_chunks.append(ocr_txt)
        ocr_joined = "\n\n".join(ocr_chunks).strip()
        if ocr_joined:
            return ocr_joined
    except Exception:
        pass

    return (metadata.get("raw_text_sample") or "").strip()


# ----------------------------------------------------------------
# OpenAI JP structured summary
# ----------------------------------------------------------------


def _generate_jp_structured_summary(
    pdf_path: Path,
    metadata: Dict[str, Any],
    openai_client: Any,
    openai_model: str,
    openai_temperature: float,
) -> Dict[str, str]:
    txt = _extract_pages_text(pdf_path, metadata, pages=OPENAI_PAGES_FOR_TEXT)
    txt = (txt or "").strip()
    if not txt:
        return {}
    if len(txt) > OPENAI_MAX_CHARS:
        txt = txt[:OPENAI_MAX_CHARS]

    system = "You are a careful research assistant. Output JSON only. If unknown, write '\u4e0d\u660e'."
    user = f"""
\u4ee5\u4e0b\u306e\u8ad6\u6587\u30c6\u30ad\u30b9\u30c8\uff08\u5192\u982d\u30da\u30fc\u30b8\u4e2d\u5fc3\uff09\u304b\u3089\u3001\u30b9\u30e9\u30a4\u30c91\u679a\u7528\u306e\u8981\u7d04\u3092\u4f5c\u3063\u3066\u304f\u3060\u3055\u3044\u3002
**\u5fc5\u305aJSON\u306e\u307f**\u3067\u8fd4\u3057\u3001\u63a8\u6e2c\u3067\u65ad\u5b9a\u3057\u306a\u3044\u3067\u304f\u3060\u3055\u3044\u3002\u4e0d\u660e\u306a\u5834\u5408\u306f\u300c\u4e0d\u660e\u300d\u3002

\u51fa\u529b\u30ad\u30fc\uff08\u3059\u3079\u3066\u65e5\u672c\u8a9e\uff09:
- rq: \u7814\u7a76\u30af\u30a8\u30b9\u30c1\u30e7\u30f3\uff081\u6587\uff09
- prior: \u65e2\u5b58\u7814\u7a76\u306f\u4f55\u304b\uff081-2\u6587\uff09
- novelty: \u65b0\u898f\u6027\u30fb\u8ca2\u732e\uff081-2\u6587\uff09
- data_method: \u30c7\u30fc\u30bf\u3068\u5206\u6790\u624b\u6cd5\uff081-2\u6587\uff09
- results: \u7d50\u679c\u30fb\u4f55\u304c\u8a00\u3048\u305f\u304b\uff081-2\u6587\uff09
- takeaway: \u4e00\u756a\u306e\u793a\u5506\uff08\u8d85\u77ed\u30441\u6587\uff09

\u53c2\u8003\u30e1\u30bf\u60c5\u5831:
- title: {metadata.get("title") or ""}
- doi: {metadata.get("doi") or ""}
- arxiv_id: {metadata.get("arxiv_id") or ""}

\u8ad6\u6587\u30c6\u30ad\u30b9\u30c8:
{txt}
""".strip()

    raw = openai_text_json(openai_client, openai_model, system, user, temperature=openai_temperature)
    obj = parse_json_object_loose(raw)

    def _s(k: str) -> str:
        v = obj.get(k)
        return "" if v is None else str(v).strip()

    return {
        "rq": _s("rq"),
        "prior": _s("prior"),
        "novelty": _s("novelty"),
        "data_method": _s("data_method"),
        "results": _s("results"),
        "takeaway": _s("takeaway"),
    }


# ----------------------------------------------------------------
# IR Builder
# ----------------------------------------------------------------


def build_slide_spec_ir(
    pdf_path: Path,
    metadata: Dict[str, Any],
    *,
    openai_client: Any = None,
    openai_model: str = "gpt-4.1-mini",
    openai_temperature: float = 0.2,
) -> Dict[str, Any]:
    """Build a slide spec intermediate representation (IR) for Gemini rendering."""
    title_en = _englishish_title(metadata.get("title") or pdf_path.stem)
    authors_en = _englishish_authors(metadata)

    jp: Dict[str, str] = {}
    jp_error = None
    if openai_client is not None:
        try:
            jp = _generate_jp_structured_summary(
                pdf_path, metadata, openai_client, openai_model, openai_temperature
            )
        except Exception as e:
            jp_error = str(e)
            logger.warning("JP summary via OpenAI failed; continuing with placeholders. err=%s", e)

    def _jp_or(key: str, placeholder: str) -> str:
        v = (jp.get(key) or "").strip()
        return v if v else placeholder

    ir = {
        "version": "ir_v2_jp_body_en_header",
        "source": {
            "pdf_name": pdf_path.name,
            "source_uid": _best_source_uid(metadata, pdf_path),
            "doi": metadata.get("doi"),
            "arxiv_id": metadata.get("arxiv_id"),
        },
        "slide": {
            "title_en": _wrap(title_en, 140),
            "authors_en": _wrap(authors_en, 80),
            "blocks_jp": [
                {"icon": "\U0001f3af", "heading_jp": "\u7814\u7a76\u30af\u30a8\u30b9\u30c1\u30e7\u30f3", "text_jp": _jp_or("rq", "\u4e0d\u660e\uff08\u30c6\u30ad\u30b9\u30c8\u62bd\u51fa\u304c\u4e0d\u8db3\u3057\u3066\u3044\u308b\u53ef\u80fd\u6027\uff09")},
                {"icon": "\U0001f4da", "heading_jp": "\u65e2\u5b58\u7814\u7a76", "text_jp": _jp_or("prior", "\u4e0d\u660e")},
                {"icon": "\u2728", "heading_jp": "\u65b0\u898f\u6027\u30fb\u8ca2\u732e", "text_jp": _jp_or("novelty", "\u4e0d\u660e")},
                {"icon": "\U0001f9ea", "heading_jp": "\u30c7\u30fc\u30bf\u30fb\u5206\u6790\u624b\u6cd5", "text_jp": _jp_or("data_method", "\u4e0d\u660e")},
                {"icon": "\U0001f4c8", "heading_jp": "\u7d50\u679c\u30fb\u793a\u5506", "text_jp": _jp_or("results", "\u4e0d\u660e")},
            ],
            "takeaway_jp": _jp_or("takeaway", "\u4e0d\u660e"),
            "badges": [b for b in [
                ("DOI" if metadata.get("doi") else None),
                ("arXiv" if metadata.get("arxiv_id") else None),
            ] if b],
            "style": {
                "theme": "clean_academic",
                "layout": "header + 2col cards + takeaway bar",
                "density": "high_but_readable",
                "color_policy": "neutral",
                "aspect": "16:9",
                "language_policy": "Header EN / Body JP",
            },
            "runtime_notes": {
                "openai_used_for_jp": bool(jp),
                "openai_error": jp_error,
                "openai_model": openai_model if bool(jp) else None,
            },
        },
    }
    return ir


# ----------------------------------------------------------------
# Gemini renderer
# ----------------------------------------------------------------


def render_slide_with_gemini(
    gemini_client: Any, ir: Dict[str, Any]
) -> Tuple[bytes, str]:
    """Render a slide image using Gemini. Returns (image_bytes, prompt)."""
    ir_json = json.dumps(ir, ensure_ascii=False, indent=2)
    title_fixed = ir["slide"]["title_en"]
    authors_fixed = ir["slide"]["authors_en"]

    prompt = f"""
You are a professional academic slide designer.

Create exactly ONE 16:9 slide image based on the Slide Spec (IR) below.

CRITICAL FIXED TEXT (MUST COPY VERBATIM, DO NOT REPHRASE):
- Title (English): {title_fixed}
- Authors (English): {authors_fixed}

Hard requirements:
- Header (top): MUST show the exact Title and Authors above (verbatim).
- Body: Japanese only.
- Use a clean academic theme. White background, subtle dividers, plenty of padding.
- Layout: Top header bar (title + authors), 2-column grid of 5 small cards, bottom takeaway strip.
- Each card MUST include: small icon, Japanese heading, Japanese text (2-3 lines max).
- No long paragraphs. No bullet symbols.
- Do NOT invent details. If text says '\u4e0d\u660e', keep it as-is.
- Output should be presentation-ready (sharp text).

Slide Spec (IR):
{ir_json}
""".strip()

    resp = gemini_client.models.generate_content(model=GEMINI_IMAGE_MODEL, contents=prompt)

    image_bytes = None

    try:
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    image_bytes = inline.data
                    break
            if image_bytes is not None:
                break
    except Exception:
        pass

    if image_bytes is None:
        try:
            for part in getattr(resp, "parts", []) or []:
                inline = getattr(part, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    image_bytes = inline.data
                    break
        except Exception:
            pass

    if image_bytes is None:
        raise RuntimeError("Gemini returned no inline image bytes (unexpected response shape).")

    return image_bytes, prompt


# ----------------------------------------------------------------
# Public entry: generate slide artifact
# ----------------------------------------------------------------


def generate_slide_artifact(
    pdf_path: Path,
    metadata: Dict[str, Any],
    *,
    slides_path: Path,
    gemini_client: Any,
    openai_client: Any = None,
    openai_model: str = "gpt-4.1-mini",
    openai_temperature: float = 0.2,
) -> Optional[Path]:
    """Generate a one-slide summary PNG. Returns the path or None on failure."""
    slides_path.mkdir(parents=True, exist_ok=True)
    spec_dir = slides_path / "_spec"
    spec_dir.mkdir(parents=True, exist_ok=True)

    title_for_name = metadata.get("title") or pdf_path.stem
    uid = _best_source_uid(metadata, pdf_path)
    dk = make_dedup_key(uid) or make_dedup_key(title_for_name) or safe_slug(pdf_path.stem, 32)
    base = f"{safe_slug(title_for_name, 60)}__{dk}__{_now_tag()}"

    slide_path = slides_path / f"{base}.png"
    ir_path = spec_dir / f"{base}__ir.json"
    prompt_path = spec_dir / f"{base}__prompt.txt"

    try:
        ir = build_slide_spec_ir(
            pdf_path,
            metadata,
            openai_client=openai_client,
            openai_model=openai_model,
            openai_temperature=openai_temperature,
        )
        ir_path.write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")

        last_err = None
        for attempt in range(1, 4):
            try:
                img_bytes, prompt_text = render_slide_with_gemini(gemini_client, ir)
                slide_path.write_bytes(img_bytes)
                prompt_path.write_text(prompt_text, encoding="utf-8")
                logger.info("Slide artifact saved (Gemini): %s", slide_path.name)
                return slide_path
            except Exception as e:
                last_err = e
                logger.warning("Gemini render attempt %d/3 failed: %s", attempt, e)
                time.sleep(1.5 * attempt)

        raise RuntimeError(f"Gemini render failed after retries: {last_err}")

    except Exception as e:
        logger.error("Slide generation failed for %s: %s", pdf_path.name, e)
        return None
