# src/lit_review/research_output.py
"""Block 6: Research Plan Generator (093).

Generates research_plan.md from Block 2–5 pipeline outputs.

As of v1 (094–100 split), 093 is responsible ONLY for research plan
generation.  Outline generation has moved to 094 (src/lit_review/outline.py),
and section drafts to 095–098 (src/lit_review/drafters/).

Usage::

    from src.lit_review.research_output import generate_research_plan, collect_all_inputs

    inputs = collect_all_inputs(run_dir)
    plan_text = generate_research_plan(inputs, llm_client=client)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Input collection
# ------------------------------------------------------------------

INPUT_FILES = [
    "rq_context.json",
    "lit_review.json",
    "landscape.json",
    "hypotheses.json",
    "assumptions.json",
    "hypothesis_portfolio.json",
    "validation_designs.json",
    "data_requirements.json",
    "method_selection.json",
]


def collect_all_inputs(run_dir: Path) -> Dict[str, Any]:
    """Load all Block 2–5 outputs and build summarized context."""
    inputs: Dict[str, Any] = {"raw": {}, "available": {}}

    for fname in INPUT_FILES:
        path = run_dir / fname
        if path.exists():
            inputs["raw"][fname] = json.loads(path.read_text())
            inputs["available"][fname] = True
        else:
            inputs["available"][fname] = False

    # Extract key fields for LLM context
    rq = inputs["raw"].get("rq_context.json", {})
    lr = inputs["raw"].get("lit_review.json", {})
    hyp = inputs["raw"].get("hypotheses.json", {})
    port = inputs["raw"].get("hypothesis_portfolio.json", {})
    vd = inputs["raw"].get("validation_designs.json", {})
    dr = inputs["raw"].get("data_requirements.json", {})
    ms = inputs["raw"].get("method_selection.json", {})

    inputs["summary"] = {
        "rq_title": rq.get("title", ""),
        "rq_background": rq.get("background", ""),
        "rq_gap": rq.get("gap", ""),
        "executive_summary": lr.get("executive_summary", ""),
        "theoretical_streams": [s.get("name", "") for s in lr.get("theoretical_streams", [])],
        "established_count": len(lr.get("empirical_findings", {}).get("established", [])),
        "emerging_count": len(lr.get("empirical_findings", {}).get("emerging", [])),
        "contested_count": len(lr.get("empirical_findings", {}).get("contested", [])),
        "open_questions": [q.get("description", "") for q in lr.get("open_questions", [])],
        "hypotheses": [
            {"statement": h.get("hypothesis_statement", ""), "strategy": h.get("strategy", ""),
             "testability": h.get("testability", "")}
            for h in hyp.get("hypotheses", [])
        ],
        "high_priority_hypotheses": [
            s.get("hypothesis_id", "")
            for s in port.get("scored_hypotheses", [])
            if s.get("recommendation") == "high_priority"
        ],
        "validation_designs_count": len(vd.get("validation_designs", [])),
        "primary_methods": [
            s.get("primary_method", "")
            for s in ms.get("method_selections", [])
        ],
        "total_variables": sum(
            len(p.get("variables", []))
            for p in dr.get("data_plans", [])
        ),
    }

    missing = [f for f, avail in inputs["available"].items() if not avail]
    if missing:
        logger.warning("Missing inputs: %s", missing)

    return inputs


def _build_compact_context(inputs: Dict[str, Any]) -> str:
    """Build a compact text representation for LLM prompts."""
    s = inputs["summary"]
    vd_raw = inputs["raw"].get("validation_designs.json", {})
    ms_raw = inputs["raw"].get("method_selection.json", {})

    parts = [
        f"## RQ\n{s['rq_title']}\n背景: {s['rq_background'][:200]}\nギャップ: {s['rq_gap'][:200]}",
        f"\n## Lit Review Summary\n{s['executive_summary'][:500]}",
        f"\n## Theoretical Streams\n{', '.join(s['theoretical_streams'])}",
        f"\n## Findings: {s['established_count']} established, {s['emerging_count']} emerging, {s['contested_count']} contested",
        f"\n## Open Questions\n" + "\n".join(f"- {q[:100]}" for q in s["open_questions"][:4]),
    ]

    # Hypotheses (compact)
    parts.append(f"\n## Hypotheses ({len(s['hypotheses'])})")
    for i, h in enumerate(s["hypotheses"]):
        parts.append(f"H{i+1} [{h['strategy']}]: {h['statement'][:80]}")

    # Validation designs (compact)
    for d in vd_raw.get("validation_designs", [])[:5]:
        parts.append(f"\n- Design: {d.get('design_type', '')} / {d.get('identification_strategy', '')}")

    # Methods (compact)
    for sel in ms_raw.get("method_selections", [])[:5]:
        parts.append(f"- Primary: {sel.get('primary_method', '')} / Secondary: {sel.get('secondary_method', '')}")

    parts.append(f"\n## Data: {s['total_variables']} variables across {s['validation_designs_count']} designs")

    return "\n".join(parts)


# ------------------------------------------------------------------
# LLM helper
# ------------------------------------------------------------------

def _llm_generate(llm_client: Any, system: str, user: str, max_tokens: int = 8192) -> str:
    """Call LLM and return raw text."""
    body = {
        "model": _MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return ""

    text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("LLM: in=%d, out=%d tokens", usage.get("input_tokens", 0), usage.get("output_tokens", 0))
    return text


# ------------------------------------------------------------------
# Research plan generator (093 core responsibility)
# ------------------------------------------------------------------

def generate_research_plan(inputs: Dict[str, Any], *, llm_client: Any) -> str:
    """Generate research_plan.md from Block 2–5 inputs.

    This is the sole responsibility of 093 after the v1 split.
    """
    context = _build_compact_context(inputs)
    system = (
        "あなたは研究計画書の執筆専門家です。\n"
        "Block 2–5 の研究成果をもとに、一貫性のある研究計画書を日本語で作成してください。\n"
        "各セクションは具体的に記述し、仮説・方法・データ間の整合性を維持してください。"
    )
    user = (
        f"{context}\n\n"
        f"## 指示\n"
        f"上記の研究成果を統合し、以下の構成で研究計画書を Markdown で作成してください:\n"
        f"1. Research Question\n2. Literature Review Summary\n3. Research Gap\n"
        f"4. Hypotheses\n5. Research Design\n6. Data\n7. Methodology\n"
        f"8. Assumptions & Limitations\n9. Timeline & Next Steps"
    )
    return _llm_generate(llm_client, system, user)
