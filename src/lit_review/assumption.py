# src/lit_review/assumption.py
"""Block 5: Assumption Analyzer (088).

Analyzes hypotheses to surface implicit assumptions across three categories:
  - Theoretical: theory model applicability conditions
  - Identification: causal inference strategy prerequisites
  - Data: data availability, quality, and representativeness

Usage::

    from src.lit_review.assumption import analyze_assumptions_pipeline

    result = analyze_assumptions_pipeline(run_dir, llm_client=client)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Assumption:
    assumption_id: str
    hypothesis_id: str
    statement: str
    category: str       # theoretical | identification | data
    testability: str    # testable | partially_testable | untestable
    test_approach: str
    vulnerability: str  # critical | significant | minor
    rationale: str
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_claims_db_record(self, now_iso: str) -> Dict[str, Any]:
        reason_parts = [
            f"[assumption] category={self.category}, testability={self.testability}, vulnerability={self.vulnerability}",
            f"hypothesis: {self.hypothesis_id}",
            f"Rationale: {self.rationale[:400]}",
        ]
        if self.test_approach:
            reason_parts.append(f"test_approach: {self.test_approach[:400]}")

        return {
            "claim_id": self.assumption_id,
            "statement": self.statement,
            "confidence": _vulnerability_to_confidence(self.vulnerability),
            "confidence_reason": "\n".join(reason_parts)[:2000],
            "tags": self.tags,
            "created_at": now_iso,
        }


@dataclass
class HypothesisAssumptions:
    """Assumptions for a single hypothesis."""
    hypothesis_id: str
    hypothesis_statement: str
    assumptions: List[Assumption] = field(default_factory=list)
    overall_vulnerability: str = ""  # high | medium | low
    weakest_assumption: str = ""


@dataclass
class AssumptionResult:
    run_id: str
    rq_title: str = ""
    hypotheses_analyzed: int = 0
    total_assumptions: int = 0
    hypothesis_assumptions: List[HypothesisAssumptions] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        by_cat = Counter()
        by_test = Counter()
        by_vuln = Counter()
        for ha in self.hypothesis_assumptions:
            for a in ha.assumptions:
                by_cat[a.category] += 1
                by_test[a.testability] += 1
                by_vuln[a.vulnerability] += 1

        return {
            "run_id": self.run_id,
            "rq_title": self.rq_title,
            "hypotheses_analyzed": self.hypotheses_analyzed,
            "total_assumptions": self.total_assumptions,
            "hypothesis_assumptions": [
                {
                    "hypothesis_id": ha.hypothesis_id,
                    "hypothesis_statement": ha.hypothesis_statement,
                    "assumptions": [a.to_dict() for a in ha.assumptions],
                    "overall_vulnerability": ha.overall_vulnerability,
                    "weakest_assumption": ha.weakest_assumption,
                }
                for ha in self.hypothesis_assumptions
            ],
            "summary": {
                "by_category": dict(by_cat),
                "by_testability": dict(by_test),
                "by_vulnerability": dict(by_vuln),
            },
            "metadata": self.metadata,
        }

    def all_assumptions(self) -> List[Assumption]:
        return [a for ha in self.hypothesis_assumptions for a in ha.assumptions]

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def assumption_id(statement: str) -> str:
    """Content-hash based ID for idempotent upsert."""
    normalized = statement.strip().lower()
    return f"asmp__{hashlib.sha256(normalized.encode()).hexdigest()[:12]}"


def _vulnerability_to_confidence(vulnerability: str) -> str:
    """Map vulnerability to Claims DB confidence (inverted)."""
    return {"critical": "low", "significant": "medium", "minor": "high"}.get(vulnerability, "medium")


def _determine_overall_vulnerability(assumptions: List[Assumption]) -> str:
    vulns = [a.vulnerability for a in assumptions]
    if "critical" in vulns:
        return "high"
    if vulns.count("significant") >= 2:
        return "medium"
    return "low"


def _build_tags(category: str, testability: str, vulnerability: str) -> List[str]:
    return [
        "assumption",
        "block5",
        category,
        f"testability_{testability}",
        f"vulnerability_{vulnerability}",
    ]


# ------------------------------------------------------------------
# Input collection
# ------------------------------------------------------------------

def collect_inputs(run_dir: Path) -> Dict[str, Any]:
    """Load hypotheses and context from run directory."""
    inputs: Dict[str, Any] = {"hypotheses": [], "theoretical_streams": [], "methods": [], "rq_title": ""}

    rq_path = run_dir / "rq_context.json"
    if rq_path.exists():
        inputs["rq_title"] = json.loads(rq_path.read_text()).get("title", "")

    hyp_path = run_dir / "hypotheses.json"
    if hyp_path.exists():
        hyp_data = json.loads(hyp_path.read_text())
        inputs["hypotheses"] = hyp_data.get("hypotheses", [])

    lr_path = run_dir / "lit_review.json"
    if lr_path.exists():
        lr = json.loads(lr_path.read_text())
        inputs["theoretical_streams"] = lr.get("theoretical_streams", [])

    ls_path = run_dir / "landscape.json"
    if ls_path.exists():
        ls = json.loads(ls_path.read_text())
        dims = ls.get("methodological_landscape", {})
        for cat_items in dims.values():
            if isinstance(cat_items, list):
                for m in cat_items:
                    if isinstance(m, dict):
                        inputs["methods"].append(m.get("name", ""))
                    elif isinstance(m, str):
                        inputs["methods"].append(m)

    logger.info("Collected: %d hypotheses, %d streams, %d methods",
                len(inputs["hypotheses"]), len(inputs["theoretical_streams"]), len(inputs["methods"]))
    return inputs


# ------------------------------------------------------------------
# LLM assumption generation
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


_SYSTEM = """\
あなたは研究方法論の専門家です。
研究仮説が暗黙に仮定している前提条件を明示化してください。

3 つのカテゴリで前提条件を分析してください:
1. Theoretical: 仮説が依拠する理論モデルの適用条件
2. Identification: 因果推論の識別戦略に必要な統計的前提（例: parallel trends, exclusion restriction, SUTVA）
3. Data: データの入手可能性・品質・代表性に関する前提

各前提条件について:
- 具体的に記述してください（「データが必要」のような自明な前提は避ける）
- なぜこの前提が必要かを説明してください
- 検証可能性を評価してください (testable / partially_testable / untestable)
- 脆弱性を評価してください (critical / significant / minor)
- 検証可能な場合は検証方法を示してください

各仮説について 3–5 件の前提条件を抽出してください。"""


def generate_assumptions(
    inputs: Dict[str, Any],
    *,
    llm_client: Any,
) -> Optional[Dict]:
    """Generate assumptions for all hypotheses in 1 LLM call."""
    hyp_lines = []
    for i, h in enumerate(inputs["hypotheses"]):
        stmt = h.get("hypothesis_statement", "")
        test = h.get("suggested_test", "")
        hyp_lines.append(f"[H{i}] {stmt}\n  suggested_test: {test}")

    streams = [s.get("name", "") for s in inputs.get("theoretical_streams", [])]
    methods = inputs.get("methods", [])

    user_msg = (
        f"## RQ: {inputs.get('rq_title', '')}\n\n"
        f"## 理論的文脈\n" + (", ".join(streams) if streams else "(なし)") + "\n\n"
        f"## 方法論的文脈\n" + (", ".join(methods[:10]) if methods else "(なし)") + "\n\n"
        f"## 仮説 ({len(inputs['hypotheses'])} 件)\n\n" + "\n\n".join(hyp_lines) + "\n\n"
        f"## 指示\n"
        f"各仮説について、3–5件の前提条件を抽出してください。\n"
        f"前提条件は theoretical / identification / data の 3 カテゴリに分類してください。\n\n"
        f"出力形式 (JSON):\n"
        f'{{"hypothesis_assumptions": [\n'
        f'  {{\n'
        f'    "hypothesis_index": 0,\n'
        f'    "assumptions": [\n'
        f'      {{\n'
        f'        "statement": "前提条件の記述",\n'
        f'        "category": "theoretical | identification | data",\n'
        f'        "testability": "testable | partially_testable | untestable",\n'
        f'        "test_approach": "検証方法",\n'
        f'        "vulnerability": "critical | significant | minor",\n'
        f'        "rationale": "なぜこの前提が必要か"\n'
        f'      }}\n'
        f'    ],\n'
        f'    "overall_vulnerability": "high | medium | low",\n'
        f'    "weakest_assumption": "最も脆弱な前提の statement"\n'
        f'  }}\n'
        f']}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 8192,
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Assumption LLM call failed: %s", e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("Assumption LLM: in=%d, out=%d tokens", usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    return _parse_json_response(resp_text)


# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------

def build_assumptions(
    raw_result: Dict,
    hypotheses: List[Dict],
) -> List[HypothesisAssumptions]:
    """Convert raw LLM output to structured HypothesisAssumptions."""
    results = []

    for raw_ha in raw_result.get("hypothesis_assumptions", []):
        h_idx = raw_ha.get("hypothesis_index", 0)
        if h_idx >= len(hypotheses):
            continue

        hyp = hypotheses[h_idx]
        hyp_id = hyp.get("hypothesis_id", f"hyp_{h_idx}")
        hyp_stmt = hyp.get("hypothesis_statement", "")

        assumptions = []
        for raw_a in raw_ha.get("assumptions", []):
            stmt = raw_a.get("statement", "")
            if not stmt:
                continue
            cat = raw_a.get("category", "theoretical")
            testability = raw_a.get("testability", "partially_testable")
            vulnerability = raw_a.get("vulnerability", "significant")

            assumptions.append(Assumption(
                assumption_id=assumption_id(stmt),
                hypothesis_id=hyp_id,
                statement=stmt,
                category=cat,
                testability=testability,
                test_approach=raw_a.get("test_approach", ""),
                vulnerability=vulnerability,
                rationale=raw_a.get("rationale", ""),
                tags=_build_tags(cat, testability, vulnerability),
            ))

        overall = raw_ha.get("overall_vulnerability", "")
        if not overall:
            overall = _determine_overall_vulnerability(assumptions)

        results.append(HypothesisAssumptions(
            hypothesis_id=hyp_id,
            hypothesis_statement=hyp_stmt,
            assumptions=assumptions,
            overall_vulnerability=overall,
            weakest_assumption=raw_ha.get("weakest_assumption", ""),
        ))

    logger.info("Built assumptions for %d hypotheses (%d total)",
                len(results), sum(len(ha.assumptions) for ha in results))
    return results


# ------------------------------------------------------------------
# Claims DB writeback
# ------------------------------------------------------------------

def write_assumptions(
    assumptions: List[Assumption],
    *,
    claims_repo: Any,
    now_iso: str,
) -> Dict[str, Any]:
    """Write critical + significant assumptions to Claims DB."""
    to_write = [a for a in assumptions if a.vulnerability in ("critical", "significant")]
    page_ids = []
    errors = []

    for a in to_write:
        record = a.to_claims_db_record(now_iso)
        try:
            page = claims_repo.upsert_claim(record)
            page_ids.append(page["id"])
        except Exception as e:
            logger.warning("Assumption write failed %s: %s", a.assumption_id, e)
            errors.append(f"{a.assumption_id}: {e}")

    logger.info("Assumptions written: %d (of %d critical+significant), %d failed",
                len(page_ids), len(to_write), len(errors))
    return {"page_ids": page_ids, "errors": errors, "skipped_minor": len(assumptions) - len(to_write)}


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def analyze_assumptions_pipeline(
    run_dir: Path,
    *,
    llm_client: Any,
) -> AssumptionResult:
    """Full assumption analysis pipeline."""
    run_id = run_dir.name
    now_iso = datetime.now(timezone.utc).isoformat()

    inputs = collect_inputs(run_dir)

    if not inputs["hypotheses"]:
        logger.error("No hypotheses found")
        return AssumptionResult(run_id=run_id, rq_title=inputs.get("rq_title", ""))

    raw = generate_assumptions(inputs, llm_client=llm_client)
    if not raw:
        logger.error("Assumption generation failed")
        return AssumptionResult(
            run_id=run_id, rq_title=inputs.get("rq_title", ""),
            metadata={"error": "generation_failed"},
        )

    hypothesis_assumptions = build_assumptions(raw, inputs["hypotheses"])
    total = sum(len(ha.assumptions) for ha in hypothesis_assumptions)

    return AssumptionResult(
        run_id=run_id,
        rq_title=inputs.get("rq_title", ""),
        hypotheses_analyzed=len(hypothesis_assumptions),
        total_assumptions=total,
        hypothesis_assumptions=hypothesis_assumptions,
        metadata={"created_at": now_iso, "model": _MODEL},
    )


# ------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------

def _render_markdown(result: AssumptionResult) -> str:
    all_a = result.all_assumptions()
    by_cat = Counter(a.category for a in all_a)
    by_test = Counter(a.testability for a in all_a)
    by_vuln = Counter(a.vulnerability for a in all_a)

    lines = [
        f"# Assumption Analysis Results",
        f"",
        f"## RQ: {result.rq_title}",
        f"",
        f"## Summary",
        f"",
        f"- Hypotheses analyzed: {result.hypotheses_analyzed}",
        f"- Total assumptions: {result.total_assumptions}",
        f"",
        f"### By Category",
        f"- theoretical: {by_cat.get('theoretical', 0)}",
        f"- identification: {by_cat.get('identification', 0)}",
        f"- data: {by_cat.get('data', 0)}",
        f"",
        f"### By Testability",
        f"- testable: {by_test.get('testable', 0)}",
        f"- partially_testable: {by_test.get('partially_testable', 0)}",
        f"- untestable: {by_test.get('untestable', 0)}",
        f"",
        f"### By Vulnerability",
        f"- critical: {by_vuln.get('critical', 0)}",
        f"- significant: {by_vuln.get('significant', 0)}",
        f"- minor: {by_vuln.get('minor', 0)}",
        f"",
    ]

    # Vulnerability map
    lines.extend([f"## Vulnerability Map", f""])
    for ha in result.hypothesis_assumptions:
        vuln_icon = {"high": "!!!", "medium": "!!", "low": "!"}.get(ha.overall_vulnerability, "?")
        lines.append(f"### [{vuln_icon} {ha.overall_vulnerability}] {ha.hypothesis_statement[:70]}")
        lines.append(f"")
        if ha.weakest_assumption:
            lines.append(f"**Weakest**: {ha.weakest_assumption}")
            lines.append(f"")
        for a in ha.assumptions:
            lines.append(f"- [{a.vulnerability}] **{a.category}** ({a.testability}): {a.statement}")
            if a.test_approach:
                lines.append(f"  - Test: {a.test_approach[:100]}")
        lines.append(f"")

    return "\n".join(lines)
