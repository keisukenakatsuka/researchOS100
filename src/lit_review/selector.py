# src/lit_review/selector.py
"""Block 5: Hypothesis Selector (089a).

Evaluates candidate hypotheses on 6 axes and selects H1 (primary)
and optionally H2 (secondary) for the focused paper.

Non-selected hypotheses are preserved as deferred or rejected.

Usage::

    from src.lit_review.selector import select_hypotheses

    result = select_hypotheses(run_dir, llm_client=client)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"

SELECTION_AXES = [
    "testability", "data_availability", "novelty",
    "impact", "coherence_with_rq", "specificity",
]

STATUS_PRIMARY = "selected_primary"
STATUS_SECONDARY = "selected_secondary"
STATUS_DEFERRED = "deferred"
STATUS_REJECTED = "rejected"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class HypothesisCandidate:
    hypothesis_id: str
    hypothesis_statement: str
    rank: int = 0
    score_breakdown: Dict[str, int] = field(default_factory=dict)
    composite_score: float = 0.0
    selection_reason: str = ""
    status: str = STATUS_DEFERRED
    portfolio_recommendation: str = ""
    portfolio_composite: float = 0.0
    overall_vulnerability: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionResult:
    run_id: str
    rq_title: str = ""
    selection_method: str = "llm_ranked"
    candidates: List[HypothesisCandidate] = field(default_factory=list)
    primary_id: str = ""
    secondary_id: Optional[str] = None
    selection_rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rq_title": self.rq_title,
            "created_at": self.metadata.get("created_at", ""),
            "selection_method": self.selection_method,
            "candidates": [c.to_dict() for c in self.candidates],
            "primary_id": self.primary_id,
            "secondary_id": self.secondary_id,
            "selection_rationale": self.selection_rationale,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# Input collection
# ------------------------------------------------------------------

def collect_inputs(run_dir: Path) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {
        "hypotheses": [],
        "portfolio": [],
        "hypothesis_assumptions": [],
        "rq_title": "",
        "lit_review": {},
    }

    rq_path = run_dir / "rq_context.json"
    if rq_path.exists():
        inputs["rq_title"] = json.loads(rq_path.read_text()).get("title", "")

    hyp_path = run_dir / "hypotheses.json"
    if hyp_path.exists():
        inputs["hypotheses"] = json.loads(hyp_path.read_text()).get("hypotheses", [])

    port_path = run_dir / "hypothesis_portfolio.json"
    if port_path.exists():
        inputs["portfolio"] = json.loads(port_path.read_text()).get("scored_hypotheses", [])

    asmp_path = run_dir / "assumptions.json"
    if asmp_path.exists():
        inputs["hypothesis_assumptions"] = json.loads(asmp_path.read_text()).get("hypothesis_assumptions", [])

    lr_path = run_dir / "lit_review.json"
    if lr_path.exists():
        inputs["lit_review"] = json.loads(lr_path.read_text())

    logger.info("Selector inputs: %d hypotheses, %d portfolio scores, %d assumption sets",
                len(inputs["hypotheses"]), len(inputs["portfolio"]),
                len(inputs["hypothesis_assumptions"]))
    return inputs


# ------------------------------------------------------------------
# LLM selection
# ------------------------------------------------------------------

def _parse_json_response(text: str) -> Optional[Any]:
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


_SELECTOR_SYSTEM = """\
あなたは研究方法論の専門家です。
候補仮説群から、学術論文の中核仮説として最適な H1（primary）と、
必要に応じて H2（secondary）を選定してください。

論文は Max 2 仮説、理想 1 仮説で構成します。
H2 は H1 を補完する場合のみ選定してください。
冗長な場合や H1 だけで十分な場合は H2 を選ばないでください。

各仮説を以下の 6 軸で 1-5 評価してください:
- testability: 経験的に検証可能か
- data_availability: 必要データが入手可能か
- novelty: 既存研究に対する新規貢献があるか
- impact: 研究上または実務上の重要性があるか
- coherence_with_rq: 元の RQ と整合しているか
- specificity: 曖昧すぎず、具体的な研究設計を導けるか"""


def _build_selector_prompt(inputs: Dict[str, Any]) -> str:
    port_by_id = {p.get("hypothesis_id", ""): p for p in inputs["portfolio"]}
    asmp_by_id = {a.get("hypothesis_id", ""): a for a in inputs["hypothesis_assumptions"]}

    hyp_lines = []
    for i, h in enumerate(inputs["hypotheses"]):
        hyp_id = h.get("hypothesis_id", "")
        stmt = h.get("hypothesis_statement", "")
        strategy = h.get("strategy", "")
        test = h.get("suggested_test", "")
        port = port_by_id.get(hyp_id, {})
        asmp = asmp_by_id.get(hyp_id, {})

        hyp_lines.append(
            f"[H{i}] id={hyp_id}\n"
            f"  statement: {stmt}\n"
            f"  strategy: {strategy}\n"
            f"  suggested_test: {test}\n"
            f"  portfolio_recommendation: {port.get('recommendation', 'N/A')}\n"
            f"  portfolio_composite: {port.get('composite_score', 'N/A')}\n"
            f"  overall_vulnerability: {asmp.get('overall_vulnerability', 'N/A')}"
        )

    # Context from lit review
    lr = inputs.get("lit_review", {})
    open_qs = lr.get("open_questions", [])
    open_qs_str = "; ".join(q if isinstance(q, str) else q.get("question", "") for q in open_qs[:5])

    return (
        f"## RQ: {inputs.get('rq_title', '')}\n\n"
        f"## Open Questions (from lit review): {open_qs_str or '(none)'}\n\n"
        f"## Candidate Hypotheses ({len(inputs['hypotheses'])} total)\n\n"
        + "\n\n".join(hyp_lines) + "\n\n"
        f"## Instructions\n"
        f"1. Score each hypothesis on 6 axes (1-5)\n"
        f"2. Rank all hypotheses\n"
        f"3. Select H1 (primary) — required\n"
        f"4. Decide if H2 (secondary) adds genuine complementary value\n"
        f"5. Classify remaining as 'deferred' (promising for future) or 'rejected' (weak)\n\n"
        f"Output JSON:\n"
        f'{{"candidates": [\n'
        f'  {{"hypothesis_index": 0, "testability": 4, "data_availability": 3, "novelty": 5, '
        f'"impact": 4, "coherence_with_rq": 5, "specificity": 4, '
        f'"selection_reason": "...", "status": "selected_primary"}}\n'
        f'], "selection_rationale": "..."}}'
    )


def evaluate_hypotheses(
    inputs: Dict[str, Any],
    *,
    llm_client: Any,
    max_secondary_score: float = 3.0,
    force_single: bool = False,
) -> Optional[Dict]:
    """Call LLM to evaluate and rank hypotheses."""
    user_msg = _build_selector_prompt(inputs)

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _SELECTOR_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Selector LLM call failed: %s", e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Selector LLM: in=%d, out=%d tokens",
                usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    return _parse_json_response(resp_text)


# ------------------------------------------------------------------
# Selection assembly
# ------------------------------------------------------------------

def select_hypotheses(
    run_dir: Path,
    *,
    llm_client: Any,
    max_secondary_score: float = 3.0,
    force_single: bool = False,
) -> SelectionResult:
    """Main entry: evaluate hypotheses and select H1/H2."""
    run_id = run_dir.name
    now_iso = datetime.now(timezone.utc).isoformat()

    inputs = collect_inputs(run_dir)
    if not inputs["hypotheses"]:
        logger.error("No hypotheses found")
        return SelectionResult(run_id=run_id, metadata={"created_at": now_iso, "error": "no_hypotheses"})

    # Build lookup tables
    port_by_id = {p.get("hypothesis_id", ""): p for p in inputs["portfolio"]}
    asmp_by_id = {a.get("hypothesis_id", ""): a for a in inputs["hypothesis_assumptions"]}

    # LLM evaluation
    llm_result = evaluate_hypotheses(
        inputs, llm_client=llm_client,
        max_secondary_score=max_secondary_score, force_single=force_single,
    )

    if not llm_result:
        logger.warning("LLM returned no result; using portfolio fallback")
        return _fallback_from_portfolio(inputs, run_id, now_iso)

    # Parse LLM candidates
    candidates = []
    llm_candidates = llm_result.get("candidates", [])
    selection_rationale = llm_result.get("selection_rationale", "")

    for lc in llm_candidates:
        idx = lc.get("hypothesis_index", -1)
        if idx < 0 or idx >= len(inputs["hypotheses"]):
            continue

        hyp = inputs["hypotheses"][idx]
        hyp_id = hyp.get("hypothesis_id", "")
        port = port_by_id.get(hyp_id, {})
        asmp = asmp_by_id.get(hyp_id, {})

        scores = {axis: lc.get(axis, 3) for axis in SELECTION_AXES}
        composite = sum(scores.values()) / len(scores)

        candidates.append(HypothesisCandidate(
            hypothesis_id=hyp_id,
            hypothesis_statement=hyp.get("hypothesis_statement", ""),
            score_breakdown=scores,
            composite_score=round(composite, 2),
            selection_reason=lc.get("selection_reason", ""),
            status=lc.get("status", STATUS_DEFERRED),
            portfolio_recommendation=port.get("recommendation", ""),
            portfolio_composite=port.get("composite_score", 0.0),
            overall_vulnerability=asmp.get("overall_vulnerability", ""),
        ))

    # Ensure hypotheses not in LLM output are included as deferred
    seen_ids = {c.hypothesis_id for c in candidates}
    for hyp in inputs["hypotheses"]:
        hyp_id = hyp.get("hypothesis_id", "")
        if hyp_id not in seen_ids:
            port = port_by_id.get(hyp_id, {})
            asmp = asmp_by_id.get(hyp_id, {})
            candidates.append(HypothesisCandidate(
                hypothesis_id=hyp_id,
                hypothesis_statement=hyp.get("hypothesis_statement", ""),
                status=STATUS_DEFERRED,
                selection_reason="Not evaluated by selector",
                portfolio_recommendation=port.get("recommendation", ""),
                portfolio_composite=port.get("composite_score", 0.0),
                overall_vulnerability=asmp.get("overall_vulnerability", ""),
            ))

    # Sort by composite descending
    candidates.sort(key=lambda c: c.composite_score, reverse=True)

    # Assign ranks
    for i, c in enumerate(candidates, 1):
        c.rank = i

    # Enforce constraints: exactly 1 primary, at most 1 secondary
    primary_id = _enforce_primary(candidates)
    secondary_id = _enforce_secondary(candidates, max_secondary_score, force_single)

    return SelectionResult(
        run_id=run_id,
        rq_title=inputs.get("rq_title", ""),
        candidates=candidates,
        primary_id=primary_id,
        secondary_id=secondary_id,
        selection_rationale=selection_rationale,
        metadata={
            "created_at": now_iso,
            "model": _MODEL,
            "total_candidates": len(inputs["hypotheses"]),
            "selected_count": 1 + (1 if secondary_id else 0),
            "max_secondary_score": max_secondary_score,
            "force_single": force_single,
        },
    )


def _enforce_primary(candidates: List[HypothesisCandidate]) -> str:
    """Ensure exactly one candidate is selected_primary."""
    primaries = [c for c in candidates if c.status == STATUS_PRIMARY]
    if len(primaries) == 1:
        return primaries[0].hypothesis_id

    # No primary from LLM → pick highest composite
    if not primaries:
        if candidates:
            candidates[0].status = STATUS_PRIMARY
            return candidates[0].hypothesis_id
        return ""

    # Multiple primaries → keep only the highest-ranked
    primaries.sort(key=lambda c: c.composite_score, reverse=True)
    for p in primaries[1:]:
        p.status = STATUS_SECONDARY
    return primaries[0].hypothesis_id


def _enforce_secondary(
    candidates: List[HypothesisCandidate],
    max_secondary_score: float,
    force_single: bool,
) -> Optional[str]:
    """Ensure at most one candidate is selected_secondary."""
    if force_single:
        for c in candidates:
            if c.status == STATUS_SECONDARY:
                c.status = STATUS_DEFERRED
        return None

    secondaries = [c for c in candidates if c.status == STATUS_SECONDARY]
    if not secondaries:
        return None

    # Keep only the best secondary
    secondaries.sort(key=lambda c: c.composite_score, reverse=True)
    best = secondaries[0]

    # Check threshold
    if best.composite_score < max_secondary_score:
        best.status = STATUS_DEFERRED
        for s in secondaries[1:]:
            s.status = STATUS_DEFERRED
        return None

    for s in secondaries[1:]:
        s.status = STATUS_DEFERRED

    return best.hypothesis_id


def _fallback_from_portfolio(
    inputs: Dict[str, Any],
    run_id: str,
    now_iso: str,
) -> SelectionResult:
    """Fallback: use portfolio scores when LLM fails."""
    port = sorted(inputs["portfolio"], key=lambda p: p.get("composite_score", 0), reverse=True)
    asmp_by_id = {a.get("hypothesis_id", ""): a for a in inputs["hypothesis_assumptions"]}
    hyp_by_id = {h.get("hypothesis_id", ""): h for h in inputs["hypotheses"]}

    candidates = []
    for i, p in enumerate(port):
        hyp_id = p.get("hypothesis_id", "")
        hyp = hyp_by_id.get(hyp_id, {})
        asmp = asmp_by_id.get(hyp_id, {})

        status = STATUS_DEFERRED
        if i == 0:
            status = STATUS_PRIMARY
        elif i == 1 and p.get("recommendation") in ("high_priority", "promising"):
            status = STATUS_SECONDARY

        candidates.append(HypothesisCandidate(
            hypothesis_id=hyp_id,
            hypothesis_statement=hyp.get("hypothesis_statement", p.get("statement", "")),
            rank=i + 1,
            composite_score=p.get("composite_score", 0.0),
            selection_reason="Portfolio fallback (LLM unavailable)",
            status=status,
            portfolio_recommendation=p.get("recommendation", ""),
            portfolio_composite=p.get("composite_score", 0.0),
            overall_vulnerability=asmp.get("overall_vulnerability", ""),
        ))

    primary_id = candidates[0].hypothesis_id if candidates else ""
    secondary_id = None
    secondaries = [c for c in candidates if c.status == STATUS_SECONDARY]
    if secondaries:
        secondary_id = secondaries[0].hypothesis_id

    return SelectionResult(
        run_id=run_id,
        rq_title=inputs.get("rq_title", ""),
        selection_method="portfolio_fallback",
        candidates=candidates,
        primary_id=primary_id,
        secondary_id=secondary_id,
        selection_rationale="LLM evaluation failed; fell back to portfolio composite scores",
        metadata={
            "created_at": now_iso,
            "model": "fallback",
            "total_candidates": len(inputs["hypotheses"]),
            "selected_count": 1 + (1 if secondary_id else 0),
        },
    )


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: SelectionResult) -> str:
    lines = [
        f"# Hypothesis Selection",
        f"",
        f"## RQ: {result.rq_title}",
        f"",
        f"**Method**: {result.selection_method}",
        f"**Selected**: {result.metadata.get('selected_count', 0)} of {result.metadata.get('total_candidates', 0)}",
        f"",
    ]

    # Primary
    primary = next((c for c in result.candidates if c.status == STATUS_PRIMARY), None)
    if primary:
        lines.extend([
            f"## H1 (Primary): {primary.hypothesis_statement[:80]}",
            f"",
            f"- **ID**: `{primary.hypothesis_id}`",
            f"- **Composite**: {primary.composite_score}",
            f"- **Scores**: {_format_scores(primary.score_breakdown)}",
            f"- **Reason**: {primary.selection_reason}",
            f"- **Portfolio**: {primary.portfolio_recommendation} (composite={primary.portfolio_composite})",
            f"",
        ])

    # Secondary
    secondary = next((c for c in result.candidates if c.status == STATUS_SECONDARY), None)
    if secondary:
        lines.extend([
            f"## H2 (Secondary): {secondary.hypothesis_statement[:80]}",
            f"",
            f"- **ID**: `{secondary.hypothesis_id}`",
            f"- **Composite**: {secondary.composite_score}",
            f"- **Scores**: {_format_scores(secondary.score_breakdown)}",
            f"- **Reason**: {secondary.selection_reason}",
            f"- **Portfolio**: {secondary.portfolio_recommendation} (composite={secondary.portfolio_composite})",
            f"",
        ])
    else:
        lines.extend([f"## H2 (Secondary): Not selected", f""])

    # Rationale
    lines.extend([
        f"## Selection Rationale",
        f"",
        f"{result.selection_rationale}",
        f"",
    ])

    # All candidates table
    lines.extend([
        f"## All Candidates (ranked)",
        f"",
        f"| Rank | Status | Comp | Test | Data | Nov | Imp | CoRQ | Spec | Hypothesis |",
        f"|------|--------|------|------|------|-----|-----|------|------|------------|",
    ])
    for c in result.candidates:
        s = c.score_breakdown
        lines.append(
            f"| {c.rank} | {c.status[:8]} | {c.composite_score} | "
            f"{s.get('testability', '-')} | {s.get('data_availability', '-')} | "
            f"{s.get('novelty', '-')} | {s.get('impact', '-')} | "
            f"{s.get('coherence_with_rq', '-')} | {s.get('specificity', '-')} | "
            f"{c.hypothesis_statement[:45]} |"
        )
    lines.append(f"")

    # Human review template
    lines.extend([
        f"## Human Review Template",
        f"",
        f"To override this selection, create `hypothesis_review.json` in the run directory:",
        f"",
        f"```json",
        f"{{",
        f'  "reviewed_at": "{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")}",',
        f'  "reviewer": "human",',
        f'  "final_primary_id": "{result.primary_id}",',
        f'  "final_secondary_id": {json.dumps(result.secondary_id)},',
        f'  "drop_secondary": false,',
        f'  "rationale": "",',
        f'  "notes_for_downstream": "",',
        f'  "overrides": []',
        f"}}",
        f"```",
        f"",
    ])

    return "\n".join(lines)


def _format_scores(scores: Dict[str, int]) -> str:
    if not scores:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in scores.items())
