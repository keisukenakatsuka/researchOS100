# src/lit_review/outline.py
"""094 Paper Outline Generator — service logic.

Generates a structured paper outline (paper_outline.json + paper_outline.md)
from Block 2–5 artifacts.  The JSON outline serves as the shared reference
for all section drafters (095–098).

Usage::

    from src.lit_review.outline import generate_outline

    result = generate_outline(run_dir, llm_client=client)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 8192


# ------------------------------------------------------------------
# Input files
# ------------------------------------------------------------------

# Required inputs (must exist and be non-empty)
_REQUIRED = ["rq_context.json", "lit_review.json", "hypotheses.json"]

# Optional inputs (used if available, gracefully omitted otherwise)
_OPTIONAL = [
    "research_plan.md",
    "landscape.json",
    "assumptions.json",
    "hypothesis_portfolio.json",
    "focused_hypotheses.json",
    "validation_designs.json",
    "data_requirements.json",
    "method_selection.json",
]


# ------------------------------------------------------------------
# Result type
# ------------------------------------------------------------------

@dataclass
class OutlineResult:
    """Return value of generate_outline()."""
    status: str = "failed"        # generated | failed
    outline: Dict[str, Any] = field(default_factory=dict)
    word_count: int = 0           # of the markdown rendering
    error: Optional[str] = None
    retryable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------
# Context builder
# ------------------------------------------------------------------

def _load_inputs(run_dir: Path) -> Dict[str, Any]:
    """Load all available upstream artifacts."""
    inputs: Dict[str, Any] = {}
    for fname in _REQUIRED + _OPTIONAL:
        p = run_dir / fname
        if not p.exists():
            if fname in _REQUIRED:
                raise FileNotFoundError(f"Required input missing: {fname}")
            continue
        if fname.endswith(".json"):
            data = json.loads(p.read_text())
            if fname in _REQUIRED and (not isinstance(data, dict) or not data):
                raise ValueError(f"Required input empty or invalid: {fname}")
            inputs[fname] = data
        else:
            text = p.read_text()
            if fname in _REQUIRED and not text.strip():
                raise ValueError(f"Required input empty: {fname}")
            inputs[fname] = text
    return inputs


def _build_context(inputs: Dict[str, Any]) -> str:
    """Build full context string for outline generation.

    Unlike 093's compact context, this uses full data from upstream
    artifacts to give the LLM maximum information for outline planning.
    """
    rq = inputs.get("rq_context.json", {})
    lr = inputs.get("lit_review.json", {})
    hyp = inputs.get("hypotheses.json", {})
    asmp = inputs.get("assumptions.json", {})
    port = inputs.get("hypothesis_portfolio.json", {})
    vd = inputs.get("validation_designs.json", {})
    dr = inputs.get("data_requirements.json", {})
    ms = inputs.get("method_selection.json", {})
    plan = inputs.get("research_plan.md", "")

    parts: List[str] = []

    # RQ context (full)
    parts.append(f"## Research Question\n{rq.get('title', '')}")
    if rq.get("background"):
        parts.append(f"Background: {rq['background']}")
    if rq.get("gap"):
        parts.append(f"Gap: {rq['gap']}")

    # Research plan (if exists from 093)
    if plan:
        parts.append(f"\n## Research Plan\n{plan[:3000]}")

    # Literature review (full summary + streams)
    if lr.get("executive_summary"):
        parts.append(f"\n## Literature Review Summary\n{lr['executive_summary']}")

    for stream in lr.get("theoretical_streams", []):
        name = stream.get("name", "")
        desc = stream.get("description", "")
        parts.append(f"\n### Theoretical Stream: {name}\n{desc[:500]}")

    # Empirical findings summary
    findings = lr.get("empirical_findings", {})
    for category in ["established", "emerging", "contested"]:
        items = findings.get(category, [])
        if items:
            parts.append(f"\n### {category.title()} Findings ({len(items)})")
            for f in items[:5]:
                parts.append(f"- {f.get('finding', f.get('description', ''))[:200]}")

    # Open questions
    oqs = lr.get("open_questions", [])
    if oqs:
        parts.append(f"\n## Open Questions ({len(oqs)})")
        for q in oqs:
            parts.append(f"- {q.get('description', q.get('question', ''))[:200]}")

    # Hypotheses — use focused (089c) when available
    from src.lit_review.focus import is_focused
    focused = inputs.get("focused_hypotheses.json", {})
    if is_focused(focused):
        parts.append(f"\n## Focused Hypotheses (from convergence layer)")
        parts.append(f"NOTE: This paper should be structured around H1 (primary) and optionally H2 (secondary).")
        primary = focused["primary"]
        parts.append(f"\nH1 (PRIMARY) [{primary.get('strategy', '')}]: {primary.get('hypothesis_statement', '')}")
        parts.append(f"Rationale: {primary.get('rationale', '')[:300]}")
        secondary = focused.get("secondary")
        if focused.get("has_secondary") and secondary:
            parts.append(f"\nH2 (SECONDARY) [{secondary.get('strategy', '')}]: {secondary.get('hypothesis_statement', '')}")
            parts.append(f"Rationale: {secondary.get('rationale', '')[:300]}")
        notes = focused.get("notes_for_downstream", "")
        if notes:
            parts.append(f"\nNotes: {notes}")
    else:
        hypotheses = hyp.get("hypotheses", [])
        parts.append(f"\n## Hypotheses ({len(hypotheses)})")
        for i, h in enumerate(hypotheses):
            stmt = h.get("hypothesis_statement", "")
            strat = h.get("strategy", "")
            rationale = h.get("rationale", "")[:300]
            parts.append(f"\nH{i+1} [{strat}]: {stmt}\nRationale: {rationale}")

        # Portfolio priority (if available)
        if port:
            scored = port.get("scored_hypotheses", [])
            high = [s for s in scored if s.get("recommendation") == "high_priority"]
            if high:
                ids = [s.get("hypothesis_id", "") for s in high]
                parts.append(f"\n## High Priority Hypotheses: {', '.join(ids)}")

    # Assumptions summary
    if asmp:
        total = asmp.get("total_assumptions", 0)
        parts.append(f"\n## Assumptions: {total} total across {asmp.get('hypotheses_analyzed', 0)} hypotheses")

    # Validation designs (all)
    if vd:
        designs = vd.get("validation_designs", [])
        parts.append(f"\n## Validation Designs ({len(designs)})")
        for d in designs:
            dtype = d.get("design_type", "")
            ident = d.get("identification_strategy", "")
            parts.append(f"- {dtype} / {ident}")

    # Method selections (all)
    if ms:
        sels = ms.get("method_selections", [])
        parts.append(f"\n## Method Selections ({len(sels)})")
        for s in sels:
            parts.append(f"- Primary: {s.get('primary_method', '')} | Secondary: {s.get('secondary_method', '')}")

    # Data requirements summary
    if dr:
        plans = dr.get("data_plans", [])
        total_vars = sum(len(p.get("variables", [])) for p in plans)
        parts.append(f"\n## Data Requirements: {total_vars} variables across {len(plans)} designs")

    return "\n".join(parts)


# ------------------------------------------------------------------
# LLM call
# ------------------------------------------------------------------

def _parse_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response (handles ```json blocks)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _generate_outline_json(
    context: str,
    rq_title: str,
    llm_client: Any,
) -> Optional[Dict[str, Any]]:
    """Call LLM to generate structured outline JSON."""
    system = (
        "あなたは学術論文の構成設計の専門家です。\n"
        "研究成果をもとに、学術論文の詳細なアウトラインを設計してください。\n"
        "出力は JSON のみ。説明文は不要です。"
    )

    user = (
        f"{context}\n\n"
        f"## 指示\n"
        f"上記の研究成果をもとに、学術論文の詳細アウトラインを以下の JSON 形式で出力してください。\n\n"
        f"```json\n"
        f'{{\n'
        f'  "rq_title": "{rq_title[:100]}",\n'
        f'  "total_target_words": 10000,\n'
        f'  "sections": [\n'
        f'    {{\n'
        f'      "section_id": "introduction",\n'
        f'      "title": "Introduction",\n'
        f'      "target_words": 2000,\n'
        f'      "argument_flow": [\n'
        f'        "Step 1: ...",\n'
        f'        "Step 2: ..."\n'
        f'      ],\n'
        f'      "key_references": ["Author1 (2020)", "Author2 (2021)"],\n'
        f'      "connects_from": null,\n'
        f'      "connects_to": "literature_review"\n'
        f'    }}\n'
        f'  ]\n'
        f'}}\n'
        f"```\n\n"
        f"## セクション構成 (必須)\n"
        f"以下の 4 セクションを必ず含めてください:\n"
        f"1. introduction — motivation → gap → contribution → paper structure\n"
        f"2. literature_review — theoretical streams → empirical findings → research gap\n"
        f"3. hypotheses — H1 を中心に理論的根拠と正式な仮説文を展開。H2 がある場合は補助仮説として扱う\n"
        f"4. methods — research design → identification → data → variables → estimation\n\n"
        f"## 要件\n"
        f"- 各セクションの target_words の合計は 8,000–12,000 words\n"
        f"- argument_flow は各セクションの論理展開を 3–6 ステップで記述\n"
        f"- key_references は各セクションで引用すべき主要先行研究 (著者名 + 年)\n"
        f"- connects_from / connects_to でセクション間の接続を明示\n"
        f"- JSON のみ出力。説明文は不要。"
    )

    body = {
        "model": _MODEL,
        "max_tokens": _MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None

    text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info(
        "LLM: in=%d, out=%d tokens",
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )

    return _parse_json_from_response(text)


def _outline_to_markdown(outline: Dict[str, Any]) -> str:
    """Render paper_outline.json as human-readable Markdown."""
    parts: List[str] = []
    rq = outline.get("rq_title", "")
    total = outline.get("total_target_words", 0)
    parts.append(f"# Paper Outline\n")
    parts.append(f"**RQ**: {rq}\n")
    parts.append(f"**Total target**: {total:,} words\n")

    for section in outline.get("sections", []):
        sid = section.get("section_id", "")
        title = section.get("title", sid)
        target = section.get("target_words", 0)
        parts.append(f"\n## {title} ({target:,} words)")

        flow = section.get("argument_flow", [])
        if flow:
            parts.append("\n**Argument flow**:")
            for i, step in enumerate(flow, 1):
                parts.append(f"  {i}. {step}")

        refs = section.get("key_references", [])
        if refs:
            parts.append(f"\n**Key references**: {', '.join(refs)}")

        conn_from = section.get("connects_from")
        conn_to = section.get("connects_to")
        if conn_from or conn_to:
            parts.append(f"\n**Flow**: {conn_from or '(start)'} → {sid} → {conn_to or '(end)'}")

    return "\n".join(parts) + "\n"


def _validate_outline(outline: Dict[str, Any]) -> List[str]:
    """Return warnings if outline is structurally incomplete."""
    warnings: List[str] = []
    sections = outline.get("sections", [])
    if not sections:
        warnings.append("No sections in outline")
        return warnings

    required_ids = {"introduction", "literature_review", "hypotheses", "methods"}
    found_ids = {s.get("section_id") for s in sections}
    missing = required_ids - found_ids
    if missing:
        warnings.append(f"Missing required sections: {missing}")

    total = sum(s.get("target_words", 0) for s in sections)
    if total < 8000:
        warnings.append(f"Total target words {total} below minimum 8,000")
    if total > 15000:
        warnings.append(f"Total target words {total} above maximum 15,000")

    for s in sections:
        if not s.get("argument_flow"):
            warnings.append(f"Section '{s.get('section_id')}' missing argument_flow")

    return warnings


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_outline(
    run_dir: Path,
    *,
    llm_client: Any,
) -> OutlineResult:
    """Generate paper_outline.json + paper_outline.md.

    Single entry point for 094_paper_outline_generator script.
    """
    result = OutlineResult()

    try:
        # 1. Load inputs
        inputs = _load_inputs(run_dir)
        rq_title = inputs.get("rq_context.json", {}).get("title", "")

        # 2. Build context
        context = _build_context(inputs)
        logger.info("Context built: %d chars", len(context))

        # 3. Generate outline via LLM
        outline = _generate_outline_json(context, rq_title, llm_client)

        if not outline:
            result.error = "LLM failed to produce valid JSON outline"
            result.retryable = True
            return result

        # 4. Validate
        warnings = _validate_outline(outline)
        if warnings:
            for w in warnings:
                logger.warning("Outline: %s", w)

        # 5. Save JSON
        json_path = run_dir / "paper_outline.json"
        json_path.write_text(json.dumps(outline, ensure_ascii=False, indent=2))
        logger.info("Saved paper_outline.json")

        # 6. Save Markdown
        md_text = _outline_to_markdown(outline)
        md_path = run_dir / "paper_outline.md"
        md_path.write_text(md_text)
        logger.info("Saved paper_outline.md (%d words)", len(md_text.split()))

        result.status = "generated"
        result.outline = outline
        result.word_count = len(md_text.split())
        result.metadata = {
            "model": _MODEL,
            "sections": len(outline.get("sections", [])),
            "total_target_words": outline.get("total_target_words", 0),
            "warnings": warnings,
        }

    except (FileNotFoundError, ValueError) as e:
        result.error = str(e)
        result.retryable = False
        logger.error("094: %s", e)

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.retryable = True
        logger.error("094: unexpected error: %s", e)

    return result
