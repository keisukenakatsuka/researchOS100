# src/lit_review/extractor.py
"""Query-focused Evidence extraction (081).

Extracts structured Evidence from papers relative to a query context.
The ``mode`` parameter selects the extraction strategy:

- ``"rq"``         — Research Question focused (Phase 1)
- ``"hypothesis"`` — Hypothesis testing focused (future)
- ``"policy"``     — Policy analysis focused (future)
- ``"strategic"``  — Strategic analysis focused (future)

Usage::

    from src.lit_review.extractor import extract_evidence, batch_extract
    from src.lit_review.rq_context import RQContext

    ctx = RQContext.from_text("How do co-investment networks ...")
    items = extract_evidence(paper, ctx, mode="rq", llm_client=client)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.lit_review.rq_context import RQContext

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_LIT_INBOX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "downloads" / "lit_inbox"

# Supported modes. Only "rq" is implemented in Phase 1.
SUPPORTED_MODES = {"rq", "hypothesis", "policy", "strategic"}

# Evidence dimensions (from T0.2 spike results)
EVIDENCE_DIMENSIONS = [
    "mechanism",    # causal pathway
    "outcome",      # measured results / effects
    "condition",    # boundary conditions / moderators
    "method",       # research methodology
    "dataset",      # data sources used (optional)
    "limitation",   # acknowledged limitations
    "implication",  # implications (optional)
]


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Evidence:
    """A single piece of extracted evidence."""

    claim_or_point: str
    """What the paper claims or demonstrates (Japanese)."""

    evidence_text: str
    """Supporting text from the paper — specific data, quotes, numbers."""

    relevance_to_rq: str
    """How this evidence relates to the RQ (Japanese, 1-2 sentences)."""

    dimension: str
    """Classification: mechanism | outcome | condition | method | dataset | limitation | implication."""

    confidence: float
    """0.0–1.0. Based on: 0.8+ empirical, 0.5–0.7 suggestive, 0.2–0.4 theoretical."""

    query_mode: str = "rq"
    """The extraction mode used."""

    source_paper_id: str = ""
    """Notion page_id of the source paper."""

    source_section: str = ""
    """Section reference in the paper (when available from PDF)."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Evidence:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExtractionResult:
    """Result of extracting evidence from a single paper."""

    paper_id: str
    paper_title: str
    query_mode: str
    text_source: str  # "pdf" | "metadata" | "title_only"
    text_length: int
    evidence_items: List[Evidence] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_items"] = [e.to_dict() for e in self.evidence_items]
        return d


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_evidence(
    paper: Dict[str, Any],
    rq_context: RQContext,
    *,
    mode: str = "rq",
    llm_client: Any = None,
    paper_text: str = "",
    text_source: str = "metadata",
) -> ExtractionResult:
    """Extract query-focused evidence from a single paper.

    Parameters
    ----------
    paper:
        Paper metadata dict with at least ``page_id`` and ``title``.
    rq_context:
        The query context to extract evidence against.
    mode:
        Extraction strategy. Phase 1 supports ``"rq"`` only.
    llm_client:
        Claude client instance (from ``build_claude_client_from_env``).
    paper_text:
        Full text or metadata text of the paper.
    text_source:
        How ``paper_text`` was obtained: "pdf", "metadata", "title_only".

    Returns
    -------
    ExtractionResult with extracted Evidence items.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unknown mode: {mode!r}. Supported: {SUPPORTED_MODES}")

    if mode != "rq":
        raise NotImplementedError(
            f"Mode {mode!r} is not yet implemented. Phase 1 supports 'rq' only."
        )

    if llm_client is None:
        raise ValueError("llm_client is required")

    return _extract_rq_evidence(paper, rq_context, llm_client, paper_text, text_source)


def batch_extract(
    papers: List[Dict[str, Any]],
    rq_context: RQContext,
    *,
    mode: str = "rq",
    llm_client: Any = None,
    get_text_fn=None,
) -> List[ExtractionResult]:
    """Extract evidence from multiple papers sequentially.

    Parameters
    ----------
    papers:
        List of paper metadata dicts.
    rq_context:
        The query context.
    mode:
        Extraction strategy.
    llm_client:
        Claude client instance.
    get_text_fn:
        Callable (paper) -> (text, source) to retrieve paper text.
        Defaults to metadata-based extraction.

    Returns
    -------
    List of ExtractionResult, one per paper. Errors are captured per-paper.
    """
    results = []
    for i, paper in enumerate(papers):
        paper_text, text_source = ("", "title_only")
        if get_text_fn:
            try:
                paper_text, text_source = get_text_fn(paper)
            except Exception as e:
                logger.warning("Text retrieval failed for '%s': %s", paper.get("title", "?")[:50], e)

        logger.info(
            "[%d/%d] Extracting from '%s' (source=%s, len=%d)",
            i + 1, len(papers), paper.get("title", "?")[:50], text_source, len(paper_text),
        )

        try:
            result = extract_evidence(
                paper, rq_context,
                mode=mode, llm_client=llm_client,
                paper_text=paper_text, text_source=text_source,
            )
        except Exception as e:
            logger.error("Extraction failed for '%s': %s", paper.get("title", "?")[:50], e)
            result = ExtractionResult(
                paper_id=paper.get("page_id", ""),
                paper_title=paper.get("title", ""),
                query_mode=mode,
                text_source=text_source,
                text_length=len(paper_text),
                error=str(e),
            )
        results.append(result)

    return results


# ------------------------------------------------------------------
# Text retrieval
# ------------------------------------------------------------------

def get_paper_text(paper: Dict[str, Any]) -> tuple[str, str]:
    """Get best available text for a paper.

    Tries PDF first, falls back to LIT DB metadata fields.

    Returns (text, source) where source is "pdf", "metadata", or "title_only".
    """
    # Try PDF
    source_uid = paper.get("source_uid", "")
    if source_uid:
        safe_name = re.sub(r"[^\w\-.]", "_", source_uid)
        pdf_path = _LIT_INBOX_DIR / f"{safe_name}.pdf"
        if pdf_path.exists():
            try:
                from src.pdf.metadata import extract_pdf_text_for_llm
                text = extract_pdf_text_for_llm(pdf_path)
                if text and len(text) > 200:
                    return text, "pdf"
            except Exception as e:
                logger.warning("PDF extraction failed for %s: %s", safe_name, e)

    # Fallback to metadata
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
        return "\n\n".join(parts), "metadata"

    return f"Title: {paper.get('title', '')}", "title_only"


# ------------------------------------------------------------------
# LLM prompts
# ------------------------------------------------------------------

_RQ_EXTRACTION_SYSTEM = """\
あなたは学術文献のレビュー専門家です。
Research Question (RQ) の観点から、論文の内容を分析し、RQ に関連する Evidence を構造化して抽出してください。

重要な指示:
- 論文の一般的な要約ではなく、RQ に直接関係する知見のみを抽出してください
- 各 Evidence は具体的で、他の論文と比較可能な粒度にしてください
- RQ と無関係な記述は含めないでください
- 各 Evidence に dimension（分類）を付与してください
- evidence_text には可能な限り具体的な数値・データ・引用を含めてください

dimension の分類:
- mechanism: 因果メカニズム、作用経路（「〜を通じて〜が生じる」）
- outcome: 測定された結果、効果、パフォーマンス指標
- condition: 効果が成立する条件、モデレーター、境界条件
- method: 使用された研究手法、分析手法
- limitation: 著者が認めた限界、未検証事項
- dataset: 使用されたデータセット、サンプル（該当する場合）
- implication: 政策・実務・理論への示唆（該当する場合）

confidence は 0.0–1.0 で以下の基準:
- 0.8–1.0: 実証的に裏付けられた強い Evidence
- 0.5–0.7: 示唆的だが追加検証が必要
- 0.2–0.4: 理論的推論や限定的なデータに基づく"""


def _build_rq_extraction_prompt(rq_context: RQContext, paper: Dict[str, Any], text: str) -> str:
    return (
        f"## Research Question\n"
        f"{rq_context.to_prompt_text()}\n\n"
        f"## 論文\n"
        f"タイトル: {paper.get('title', '')}\n"
        f"Tags: {paper.get('tags', '')}\n\n"
        f"## 論文の内容\n"
        f"{text[:80_000]}\n\n"
        f"## 指示\n"
        f"上記の論文から、RQ に関連する Evidence を抽出してください。\n"
        f"各 Evidence について、以下の JSON 形式で出力してください。\n"
        f"一般的な要約ではなく、RQ の視点から見た具体的な知見を抽出してください。\n"
        f"1論文あたり 3〜8 件程度の Evidence を目安にしてください。\n\n"
        f"出力形式:\n"
        f'{{"evidence_items": [\n'
        f'  {{\n'
        f'    "claim_or_point": "この論文が示している主張や知見（日本語で簡潔に）",\n'
        f'    "evidence_text": "論文中の根拠となる具体的な記述や数値（可能な限り原文に近い形で）",\n'
        f'    "relevance_to_rq": "この Evidence が RQ にどう関係するか（日本語で1-2文）",\n'
        f'    "dimension": "mechanism | outcome | condition | method | dataset | limitation | implication のいずれか",\n'
        f'    "confidence": 0.0\n'
        f'  }}\n'
        f']}}'
    )


def _parse_json_response(text: str) -> Optional[Any]:
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------
# Mode implementations
# ------------------------------------------------------------------

def _extract_rq_evidence(
    paper: Dict[str, Any],
    rq_context: RQContext,
    llm_client: Any,
    paper_text: str,
    text_source: str,
) -> ExtractionResult:
    """mode="rq" implementation. Validated in T0.2 spike."""
    paper_id = paper.get("page_id", "")
    paper_title = paper.get("title", "")

    # Build prompt
    user_msg = _build_rq_extraction_prompt(rq_context, paper, paper_text)

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _RQ_EXTRACTION_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    # Call LLM
    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("LLM call failed for '%s': %s", paper_title[:50], e)
        return ExtractionResult(
            paper_id=paper_id, paper_title=paper_title,
            query_mode="rq", text_source=text_source,
            text_length=len(paper_text), error=f"LLM call failed: {e}",
        )

    # Parse response
    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    parsed = _parse_json_response(resp_text)
    if not parsed or "evidence_items" not in parsed:
        logger.error("JSON parse failed for '%s'", paper_title[:50])
        return ExtractionResult(
            paper_id=paper_id, paper_title=paper_title,
            query_mode="rq", text_source=text_source,
            text_length=len(paper_text),
            error=f"JSON parse failed. Raw: {resp_text[:200]}",
        )

    # Convert to Evidence dataclass instances
    items = []
    for raw_item in parsed["evidence_items"]:
        try:
            items.append(Evidence(
                claim_or_point=raw_item.get("claim_or_point", ""),
                evidence_text=raw_item.get("evidence_text", ""),
                relevance_to_rq=raw_item.get("relevance_to_rq", ""),
                dimension=raw_item.get("dimension", "unknown"),
                confidence=float(raw_item.get("confidence", 0.5)),
                query_mode="rq",
                source_paper_id=paper_id,
                source_section=raw_item.get("source_section", ""),
            ))
        except Exception as e:
            logger.warning("Failed to parse evidence item: %s", e)

    usage = resp.get("usage", {})
    logger.info(
        "Extracted %d evidence items from '%s' (in=%d, out=%d tokens)",
        len(items), paper_title[:50],
        usage.get("input_tokens", 0), usage.get("output_tokens", 0),
    )

    return ExtractionResult(
        paper_id=paper_id, paper_title=paper_title,
        query_mode="rq", text_source=text_source,
        text_length=len(paper_text), evidence_items=items,
    )
