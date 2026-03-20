# src/lit_review/focus.py
"""Block 5: Focused Hypotheses (089c).

Assembles the canonical focused_hypotheses.json that all downstream
scripts (090-100) use as their primary hypothesis input.

Enriches H1/H2 from review decision with full hypothesis data,
portfolio scores, selector scores, and assumptions.

Usage::

    from src.lit_review.focus import build_focused

    result = build_focused(run_dir)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared mode detection — used by all downstream scripts
# ------------------------------------------------------------------

def load_focused(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load focused_hypotheses.json if it exists and has a valid primary.

    Returns the parsed dict if focused mode is active, None otherwise.
    All downstream scripts should use this single function to determine
    whether to use focused (H1/H2) or legacy (all hypotheses) mode.
    """
    path = run_dir / "focused_hypotheses.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("focused_hypotheses.json exists but is unreadable; falling back to legacy mode")
        return None
    if not isinstance(data, dict) or not data.get("primary"):
        return None
    return data


def is_focused(focused: Optional[Dict[str, Any]]) -> bool:
    """Check if focused mode is active. Use with load_focused() result."""
    return focused is not None and bool(focused.get("primary"))


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class FocusedResult:
    run_id: str
    rq_title: str = ""
    created_at: str = ""
    review_source: str = ""
    primary: Optional[Dict[str, Any]] = None
    secondary: Optional[Dict[str, Any]] = None
    has_secondary: bool = False
    notes_for_downstream: str = ""
    non_selected_summary: Dict[str, Any] = field(default_factory=dict)
    all_candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "rq_title": self.rq_title,
            "created_at": self.created_at,
            "review_source": self.review_source,
            "primary": self.primary,
            "secondary": self.secondary,
            "has_secondary": self.has_secondary,
            "notes_for_downstream": self.notes_for_downstream,
            "non_selected_summary": self.non_selected_summary,
            "all_candidates": self.all_candidates,
        }

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# Core logic
# ------------------------------------------------------------------

def build_focused(run_dir: Path) -> FocusedResult:
    """Assemble focused hypotheses from review decision + source data."""
    run_id = run_dir.name
    now_iso = datetime.now(timezone.utc).isoformat()

    # Load review decision
    decision_path = run_dir / "hypothesis_review_decision.json"
    if not decision_path.exists():
        raise FileNotFoundError(f"hypothesis_review_decision.json not found in {run_dir}")
    decision = json.loads(decision_path.read_text())

    # Load source data for enrichment
    hypotheses = _load_json_list(run_dir / "hypotheses.json", "hypotheses")
    portfolio = _load_json_list(run_dir / "hypothesis_portfolio.json", "scored_hypotheses")
    assumptions = _load_json_list(run_dir / "assumptions.json", "hypothesis_assumptions")
    selection = _load_json(run_dir / "hypothesis_selection.json")

    # Build lookup tables
    hyp_by_id = {h.get("hypothesis_id", ""): h for h in hypotheses}
    port_by_id = {p.get("hypothesis_id", ""): p for p in portfolio}
    asmp_by_id = {a.get("hypothesis_id", ""): a for a in assumptions}
    sel_by_id = {c.get("hypothesis_id", ""): c for c in selection.get("candidates", [])}

    rq_path = run_dir / "rq_context.json"
    rq_title = ""
    if rq_path.exists():
        rq_title = json.loads(rq_path.read_text()).get("title", "")

    # Enrich primary
    primary_id = decision.get("primary", {}).get("hypothesis_id", "")
    primary = _enrich_hypothesis(primary_id, decision.get("primary", {}),
                                  hyp_by_id, port_by_id, asmp_by_id, sel_by_id)

    # Enrich secondary
    secondary = None
    has_secondary = decision.get("has_secondary", False)
    if has_secondary and decision.get("secondary"):
        secondary_id = decision["secondary"].get("hypothesis_id", "")
        secondary = _enrich_hypothesis(secondary_id, decision["secondary"],
                                        hyp_by_id, port_by_id, asmp_by_id, sel_by_id)

    # Non-selected summary
    non_selected = decision.get("non_selected", [])
    deferred = [n for n in non_selected if n.get("status") == "deferred"]
    rejected = [n for n in non_selected if n.get("status") == "rejected"]

    non_selected_summary = {
        "deferred_count": len(deferred),
        "rejected_count": len(rejected),
        "deferred_ids": [n.get("hypothesis_id", "") for n in deferred],
        "rejected_ids": [n.get("hypothesis_id", "") for n in rejected],
    }

    # All candidates (lightweight, for reference)
    all_candidates = []
    for c in selection.get("candidates", []):
        cid = c.get("hypothesis_id", "")
        status = c.get("status", "deferred")
        # Override status for primary/secondary
        if cid == primary_id:
            status = "selected_primary"
        elif has_secondary and secondary and cid == secondary.get("hypothesis_id"):
            status = "selected_secondary"
        elif any(n.get("hypothesis_id") == cid for n in non_selected):
            ns = next(n for n in non_selected if n.get("hypothesis_id") == cid)
            status = ns.get("status", status)

        all_candidates.append({
            "hypothesis_id": cid,
            "hypothesis_statement": c.get("hypothesis_statement", ""),
            "status": status,
            "composite_score": c.get("composite_score", 0.0),
        })

    return FocusedResult(
        run_id=run_id,
        rq_title=rq_title,
        created_at=now_iso,
        review_source=decision.get("review_source", ""),
        primary=primary,
        secondary=secondary,
        has_secondary=has_secondary and secondary is not None,
        notes_for_downstream=decision.get("notes_for_downstream", ""),
        non_selected_summary=non_selected_summary,
        all_candidates=all_candidates,
    )


def _enrich_hypothesis(
    hyp_id: str,
    decision_entry: Dict[str, Any],
    hyp_by_id: Dict, port_by_id: Dict, asmp_by_id: Dict, sel_by_id: Dict,
) -> Dict[str, Any]:
    """Enrich a hypothesis with full data from all sources."""
    hyp = hyp_by_id.get(hyp_id, {})
    port = port_by_id.get(hyp_id, {})
    asmp = asmp_by_id.get(hyp_id, {})
    sel = sel_by_id.get(hyp_id, {})

    return {
        # Core hypothesis fields
        "hypothesis_id": hyp_id,
        "hypothesis_statement": hyp.get("hypothesis_statement", decision_entry.get("hypothesis_statement", "")),
        "rationale": hyp.get("rationale", ""),
        "strategy": hyp.get("strategy", ""),
        "testability": hyp.get("testability", ""),
        "suggested_test": hyp.get("suggested_test", ""),
        "source_claim_ids": hyp.get("source_claim_ids", []),
        "source_gaps": hyp.get("source_gaps", []),
        "novelty_rationale": hyp.get("novelty_rationale", ""),
        "tags": hyp.get("tags", []),
        # Portfolio scores
        "portfolio_scores": port.get("scores", {}),
        "portfolio_recommendation": port.get("recommendation", ""),
        "portfolio_composite": port.get("composite_score", 0.0),
        # Selector scores
        "selector_scores": sel.get("score_breakdown", {}),
        "selector_composite": sel.get("composite_score", 0.0),
        # Assumptions
        "assumptions": _extract_assumptions(asmp),
        "overall_vulnerability": asmp.get("overall_vulnerability", port.get("overall_vulnerability", "")),
        # Selection reason
        "selection_reason": decision_entry.get("selection_reason", sel.get("selection_reason", "")),
    }


def _extract_assumptions(asmp: Dict) -> List[Dict[str, str]]:
    """Extract compact assumption summaries."""
    result = []
    for a in asmp.get("assumptions", []):
        result.append({
            "assumption_id": a.get("assumption_id", ""),
            "statement": a.get("statement", ""),
            "category": a.get("category", ""),
            "vulnerability": a.get("vulnerability", ""),
        })
    return result


def _load_json(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _load_json_list(path: Path, key: str) -> List:
    if path.exists():
        return json.loads(path.read_text()).get(key, [])
    return []


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: FocusedResult) -> str:
    lines = [
        f"# Focused Hypotheses",
        f"",
        f"## RQ: {result.rq_title}",
        f"",
        f"**Review source**: {result.review_source}",
        f"**Has secondary**: {result.has_secondary}",
        f"",
    ]

    if result.primary:
        lines.extend(_render_hypothesis("H1 (Primary)", result.primary))

    if result.has_secondary and result.secondary:
        lines.extend(_render_hypothesis("H2 (Secondary)", result.secondary))
    else:
        lines.extend([f"## H2 (Secondary): Not selected", f""])

    if result.notes_for_downstream:
        lines.extend([
            f"## Notes for Downstream",
            f"",
            f"{result.notes_for_downstream}",
            f"",
        ])

    # Non-selected summary
    ns = result.non_selected_summary
    lines.extend([
        f"## Non-Selected Hypotheses",
        f"",
        f"- Deferred: {ns.get('deferred_count', 0)}",
        f"- Rejected: {ns.get('rejected_count', 0)}",
        f"",
    ])

    for c in result.all_candidates:
        if c.get("status") not in ("selected_primary", "selected_secondary"):
            lines.append(f"- [{c.get('status')}] `{c.get('hypothesis_id', '')}` — {c.get('hypothesis_statement', '')[:60]}")
    lines.append(f"")

    return "\n".join(lines)


def _render_hypothesis(title: str, h: Dict) -> List[str]:
    lines = [
        f"## {title}: {h.get('hypothesis_statement', '')[:80]}",
        f"",
        f"- **ID**: `{h.get('hypothesis_id', '')}`",
        f"- **Strategy**: {h.get('strategy', '')}",
        f"- **Testability**: {h.get('testability', '')}",
        f"- **Vulnerability**: {h.get('overall_vulnerability', '')}",
        f"- **Suggested test**: {h.get('suggested_test', '')}",
        f"- **Selection reason**: {h.get('selection_reason', '')}",
        f"",
        f"### Portfolio Scores",
        f"- Recommendation: {h.get('portfolio_recommendation', '')} (composite={h.get('portfolio_composite', '')})",
    ]
    for k, v in h.get("portfolio_scores", {}).items():
        lines.append(f"- {k}: {v}")

    lines.extend([f"", f"### Selector Scores"])
    for k, v in h.get("selector_scores", {}).items():
        lines.append(f"- {k}: {v}")

    if h.get("assumptions"):
        lines.extend([f"", f"### Key Assumptions"])
        for a in h["assumptions"]:
            lines.append(f"- [{a.get('vulnerability', '')}] ({a.get('category', '')}) {a.get('statement', '')[:60]}")

    lines.append(f"")
    return lines
