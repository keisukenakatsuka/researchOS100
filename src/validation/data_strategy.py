# src/validation/data_strategy.py
"""Data Strategy Planner (113 core logic).

Integrates Dataset Registry, Data Requirements, and Hypothesis Portfolio
to build a phased data acquisition roadmap.

See design.md Section 5.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class CriticalGap:
    data_need: str = ""
    required_by_hypotheses: List[str] = field(default_factory=list)
    criticality: str = ""  # must-have | nice-to-have | optional
    best_source_id: str = ""
    best_source_name: str = ""
    availability: str = ""
    workaround: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CriticalGap:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RoadmapDataset:
    dataset_id: str = ""
    name: str = ""
    action: str = ""
    effort: str = ""  # low | medium | high
    value: str = ""
    cost_justification: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d["cost_justification"]:
            del d["cost_justification"]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoadmapDataset:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RoadmapPhase:
    description: str = ""
    timeline: str = ""
    datasets: List[RoadmapDataset] = field(default_factory=list)
    estimated_cost: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "timeline": self.timeline,
            "datasets": [d.to_dict() for d in self.datasets],
            "estimated_cost": self.estimated_cost,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoadmapPhase:
        datasets = [RoadmapDataset.from_dict(d) for d in data.pop("datasets", [])]
        phase = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        phase.datasets = datasets
        return phase


@dataclass
class DataStrategy:
    run_id: str = ""
    created_at: str = ""
    gap_analysis: Dict[str, Any] = field(default_factory=dict)
    acquisition_roadmap: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    # Internal typed fields (used during construction)
    _critical_gaps: List[CriticalGap] = field(default_factory=list, repr=False)
    _covered: List[str] = field(default_factory=list, repr=False)
    _phases: Dict[str, RoadmapPhase] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        # Build gap_analysis from typed fields if available
        if self._critical_gaps:
            total_needs = len(self._critical_gaps) + len(self._covered)
            self.gap_analysis = {
                "critical_gaps": [g.to_dict() for g in self._critical_gaps],
                "covered": self._covered,
                "coverage_rate": round(len(self._covered) / total_needs, 2) if total_needs else 0.0,
            }

        if self._phases:
            self.acquisition_roadmap = {k: v.to_dict() for k, v in self._phases.items()}

        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "gap_analysis": self.gap_analysis,
            "acquisition_roadmap": self.acquisition_roadmap,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        d = self.to_dict()
        ga = d["gap_analysis"]
        roadmap = d["acquisition_roadmap"]
        recs = d["recommendations"]

        lines = [
            f"# Data Strategy",
            f"Run: {self.run_id} | Created: {self.created_at[:10]}",
            f"",
        ]

        # Gap analysis
        gaps = ga.get("critical_gaps", [])
        covered = ga.get("covered", [])
        coverage = ga.get("coverage_rate", 0)
        lines.append(f"## Gap Analysis")
        lines.append(f"")
        lines.append(f"Coverage rate: {coverage:.0%} ({len(covered)} covered, {len(gaps)} gaps)")
        lines.append(f"")

        if gaps:
            lines.append(f"### Critical Gaps ({len(gaps)})")
            lines.append(f"")
            lines.append(f"| Data Need | Criticality | Best Source | Availability | Workaround |")
            lines.append(f"|-----------|-------------|------------|-------------|-----------|")
            for g in gaps:
                src = g.get("best_source_name", g.get("best_source_id", "-"))
                hyps = ", ".join(f"`{h}`" for h in g.get("required_by_hypotheses", []))
                lines.append(
                    f"| {g['data_need'][:50]} | {g['criticality']} | {src[:25]} | "
                    f"{g['availability']} | {g['workaround'][:40]} |"
                )
            lines.append(f"")

            # Detailed gaps
            for g in gaps:
                lines.append(f"#### {g['data_need']}")
                lines.append(f"- **Criticality**: {g['criticality']}")
                lines.append(f"- **Required by**: {', '.join(f'`{h}`' for h in g.get('required_by_hypotheses', []))}")
                lines.append(f"- **Best source**: {g.get('best_source_name', '')} (`{g.get('best_source_id', '')}`)")
                lines.append(f"- **Availability**: {g['availability']}")
                lines.append(f"- **Workaround**: {g['workaround']}")
                lines.append(f"")

        if covered:
            lines.append(f"### Covered Data ({len(covered)})")
            lines.append(f"")
            for c in covered:
                lines.append(f"- {c}")
            lines.append(f"")

        # Acquisition roadmap
        lines.append(f"## Acquisition Roadmap")
        lines.append(f"")

        phase_labels = {
            "phase_1_immediate": ("Phase 1: Immediate", "free/open"),
            "phase_2_short_term": ("Phase 2: Short-term", "restricted/application"),
            "phase_3_medium_term": ("Phase 3: Medium-term", "commercial/paid"),
            "phase_4_long_term": ("Phase 4: Long-term", "custom collection"),
        }

        for phase_key, (label, category) in phase_labels.items():
            phase = roadmap.get(phase_key, {})
            if not phase:
                continue
            datasets = phase.get("datasets", [])
            lines.append(f"### {label}")
            lines.append(f"- **Timeline**: {phase.get('timeline', '-')}")
            lines.append(f"- **Estimated cost**: {phase.get('estimated_cost', '-')}")
            lines.append(f"- **Description**: {phase.get('description', '')}")
            lines.append(f"")
            if datasets:
                lines.append(f"| Dataset | Action | Effort | Value |")
                lines.append(f"|---------|--------|--------|-------|")
                for ds in datasets:
                    name = ds.get("name", ds.get("dataset_id", "-"))
                    lines.append(f"| {name[:40]} | {ds['action'][:40]} | {ds['effort']} | {ds['value'][:40]} |")
                lines.append(f"")

        # Recommendations
        if recs:
            lines.append(f"## Recommendations")
            lines.append(f"")
            for i, rec in enumerate(recs, 1):
                lines.append(f"{i}. {rec}")
            lines.append(f"")

        return "\n".join(lines)


# ------------------------------------------------------------------
# Strategy generation (single LLM call)
# ------------------------------------------------------------------

_STRATEGY_SYSTEM = """\
あなたはデータ戦略の専門家です。
研究プロジェクトのデータ取得戦略を策定してください。

以下の3つの情報源を統合して分析してください:
1. Dataset Registry: 利用可能なデータセットの一覧と入手可能性
2. Data Requirements: 仮説検証に必要な変数とデータソース
3. Hypothesis Portfolio: 仮説の優先度

出力は必ず JSON 形式で返してください。"""


def plan_strategy(
    registry: Dict[str, Any],
    data_requirements: Dict[str, Any],
    portfolio: Dict[str, Any],
    llm_client: Any,
) -> Dict[str, Any]:
    """Generate data strategy via single LLM call.

    Returns raw parsed JSON matching the DataStrategy schema.
    """

    # Build context summaries
    registry_summary = _summarize_registry(registry)
    reqs_summary = _summarize_requirements(data_requirements)
    portfolio_summary = _summarize_portfolio(portfolio)

    user_msg = (
        f"## Dataset Registry\n{registry_summary}\n\n"
        f"## Data Requirements (091)\n{reqs_summary}\n\n"
        f"## Hypothesis Portfolio (089)\n{portfolio_summary}\n\n"
        f"## Instructions\n"
        f"上記の情報を統合し、データ取得戦略を以下の JSON 形式で出力してください:\n\n"
        f'{{\n'
        f'  "gap_analysis": {{\n'
        f'    "critical_gaps": [\n'
        f'      {{\n'
        f'        "data_need": "必要だが未入手のデータ（日本語）",\n'
        f'        "required_by_hypotheses": ["hyp__xxx"],\n'
        f'        "criticality": "must-have | nice-to-have | optional",\n'
        f'        "best_source_id": "ds__xxx",\n'
        f'        "best_source_name": "データソース名",\n'
        f'        "availability": "open | restricted | commercial | unavailable",\n'
        f'        "workaround": "代替手段（日本語）"\n'
        f'      }}\n'
        f'    ],\n'
        f'    "covered": ["入手済み or 容易に入手可能なデータカテゴリ"],\n'
        f'    "coverage_rate": 0.6\n'
        f'  }},\n'
        f'  "acquisition_roadmap": {{\n'
        f'    "phase_1_immediate": {{\n'
        f'      "description": "無料・公開データの即時取得",\n'
        f'      "timeline": "1-2 weeks",\n'
        f'      "datasets": [\n'
        f'        {{\n'
        f'          "dataset_id": "ds__xxx",\n'
        f'          "name": "データセット名",\n'
        f'          "action": "具体的な取得アクション",\n'
        f'          "effort": "low | medium | high",\n'
        f'          "value": "このデータの研究上の価値"\n'
        f'        }}\n'
        f'      ],\n'
        f'      "estimated_cost": "$0"\n'
        f'    }},\n'
        f'    "phase_2_short_term": {{\n'
        f'      "description": "制限付きデータの申請・取得",\n'
        f'      "timeline": "1-3 months",\n'
        f'      "datasets": [...],\n'
        f'      "estimated_cost": "$0-500"\n'
        f'    }},\n'
        f'    "phase_3_medium_term": {{\n'
        f'      "description": "有料データの取得",\n'
        f'      "timeline": "3-6 months",\n'
        f'      "datasets": [\n'
        f'        {{\n'
        f'          "dataset_id": "ds__xxx",\n'
        f'          "name": "データセット名",\n'
        f'          "action": "取得アクション",\n'
        f'          "effort": "high",\n'
        f'          "value": "研究上の価値",\n'
        f'          "cost_justification": "コスト正当化の理由"\n'
        f'        }}\n'
        f'      ],\n'
        f'      "estimated_cost": "$5,000-20,000"\n'
        f'    }},\n'
        f'    "phase_4_long_term": {{\n'
        f'      "description": "独自データ収集",\n'
        f'      "timeline": "6+ months",\n'
        f'      "datasets": [...],\n'
        f'      "estimated_cost": "TBD"\n'
        f'    }}\n'
        f'  }},\n'
        f'  "recommendations": [\n'
        f'    "具体的な推奨アクション（日本語）"\n'
        f'  ]\n'
        f'}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _STRATEGY_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Strategy LLM call failed: %s", e)
        return {}

    resp_text = _extract_text(resp)
    parsed = _parse_json(resp_text)
    if not parsed:
        logger.error("Strategy JSON parse failed. Raw: %s", resp_text[:300])
        return {}

    usage = resp.get("usage", {})
    logger.info(
        "Strategy generated (in=%d, out=%d tokens)",
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )

    return parsed


# ------------------------------------------------------------------
# Context summarizers (keep token count manageable)
# ------------------------------------------------------------------

def _summarize_registry(registry: Dict[str, Any]) -> str:
    """Build concise registry summary for prompt."""
    summary = registry.get("summary", {})
    lines = [
        f"Total: {summary.get('total_datasets', 0)} datasets",
        f"Open: {summary.get('open', 0)}, Restricted: {summary.get('restricted', 0)}, "
        f"Commercial: {summary.get('commercial', 0)}, Unavailable: {summary.get('unavailable', 0)}",
        "",
        "Key datasets:",
    ]
    for ds in registry.get("datasets", [])[:30]:  # limit
        status = ds.get("availability_status", "?")
        cost = ds.get("cost_tier", "?")
        hyps = ds.get("used_by_hypotheses", [])
        hyps_str = f" [used by: {', '.join(hyps)}]" if hyps else ""
        lines.append(f"- {ds.get('name', '')} ({status}, cost={cost}) [{ds.get('dataset_id', '')}]{hyps_str}")

    return "\n".join(lines)


def _summarize_requirements(data_reqs: Dict[str, Any]) -> str:
    """Build concise data requirements summary."""
    lines = []
    for plan in data_reqs.get("data_plans", []):
        hyp_id = plan.get("hypothesis_id", "")
        hyp_stmt = plan.get("hypothesis_statement", "")[:80]
        feasibility = plan.get("overall_feasibility", "?")
        gaps = plan.get("critical_data_gaps", [])
        lines.append(f"Hypothesis {hyp_id} (feasibility={feasibility}): {hyp_stmt}")
        if gaps:
            for g in gaps:
                lines.append(f"  GAP: {g}")
        for var in plan.get("variables", [])[:3]:  # top 3 vars per plan
            ps = var.get("primary_source", {})
            lines.append(
                f"  Var: {var.get('name', '')} ({var.get('variable_type', '')}) "
                f"→ {ps.get('name', '')} ({ps.get('acquisition_difficulty', '')})"
            )
    return "\n".join(lines)


def _summarize_portfolio(portfolio: Dict[str, Any]) -> str:
    """Build concise portfolio summary."""
    lines = []
    for h in portfolio.get("scored_hypotheses", []):
        lines.append(
            f"- {h.get('hypothesis_id', '')} [{h.get('recommendation', '')}] "
            f"(score={h.get('composite_score', 0)}): {h.get('statement', '')[:60]}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_text(resp: Dict[str, Any]) -> str:
    for block in resp.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
