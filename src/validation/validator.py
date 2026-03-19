# src/validation/validator.py
"""Evidence-Paper alignment validation (110 core logic).

Pass 1: Aligns each Evidence item against the source paper text.
Pass 2: Self-consistency check (only for uncertain/contradiction).

See design.md Section 2.2-2.4.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_BATCH_SIZE = 5  # evidence items per LLM call


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Pass1Result:
    grounding_quote: str = ""
    alignment_score: float = 0.0
    section_ref: str = ""
    reasoning: str = ""
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Pass1Result:
        return cls(
            grounding_quote=data.get("grounding_quote", ""),
            alignment_score=float(data.get("alignment_score", 0.0)),
            section_ref=data.get("section_ref", ""),
            reasoning=data.get("reasoning", ""),
            issues=data.get("issues") or [],
        )


@dataclass
class Pass2Result:
    verdict: str = ""  # "agree" | "disagree"
    revised_alignment_score: Optional[float] = None
    reasoning: str = ""
    additional_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Pass2Result:
        score = data.get("revised_alignment_score")
        return cls(
            verdict=data.get("verdict", ""),
            revised_alignment_score=float(score) if score is not None else None,
            reasoning=data.get("reasoning", ""),
            additional_issues=data.get("additional_issues") or [],
        )


@dataclass
class ValidationItem:
    evidence_id: str
    claim_or_point: str
    paper_id: Optional[str]
    paper_title: str
    text_source_used: str
    pass1: Pass1Result
    pass2: Optional[Pass2Result] = None
    final_status: str = ""  # verified | uncertain | unverifiable | contradiction
    final_alignment_score: float = 0.0
    needs_human_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "evidence_id": self.evidence_id,
            "claim_or_point": self.claim_or_point,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "text_source_used": self.text_source_used,
            "pass1": self.pass1.to_dict(),
            "pass2": self.pass2.to_dict() if self.pass2 else None,
            "final_status": self.final_status,
            "final_alignment_score": self.final_alignment_score,
            "needs_human_review": self.needs_human_review,
        }
        return d


@dataclass
class ValidationResult:
    run_id: str
    source_evidence_file: str
    validated_at: str
    model: str
    summary: Dict[str, Any] = field(default_factory=dict)
    validations: List[ValidationItem] = field(default_factory=list)

    def compute_summary(self) -> None:
        total = len(self.validations)
        verified = sum(1 for v in self.validations if v.final_status == "verified")
        uncertain = sum(1 for v in self.validations if v.final_status == "uncertain")
        unverifiable = sum(1 for v in self.validations if v.final_status == "unverifiable")
        contradiction = sum(1 for v in self.validations if v.final_status == "contradiction")
        self.summary = {
            "total": total,
            "verified": verified,
            "uncertain": uncertain,
            "unverifiable": unverifiable,
            "contradiction": contradiction,
            "coverage": round((verified + uncertain + contradiction) / total, 3) if total else 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        self.compute_summary()
        return {
            "run_id": self.run_id,
            "source_evidence_file": self.source_evidence_file,
            "validated_at": self.validated_at,
            "model": self.model,
            "summary": self.summary,
            "validations": [v.to_dict() for v in self.validations],
        }

    def to_markdown(self) -> str:
        self.compute_summary()
        s = self.summary
        total = s["total"]
        lines = [
            f"# Literature Validation Report",
            f"Run: {self.run_id} | Validated: {self.validated_at[:10]}",
            f"",
            f"## Summary",
            f"| Status | Count | % |",
            f"|--------|-------|---|",
        ]

        def _pct(n: int) -> str:
            return f"{n / total * 100:.1f}" if total else "0.0"

        lines.append(f"| Verified | {s['verified']} | {_pct(s['verified'])}% |")
        lines.append(f"| Uncertain | {s['uncertain']} | {_pct(s['uncertain'])}% |")
        lines.append(f"| Contradiction | {s['contradiction']} | {_pct(s['contradiction'])}% |")
        lines.append(f"| Unverifiable | {s['unverifiable']} | {_pct(s['unverifiable'])}% |")
        lines.append(f"| **Total** | **{total}** | |")
        lines.append(f"")
        lines.append(f"Coverage (verified + uncertain + contradiction): {s['coverage']:.1%}")
        lines.append(f"")

        # Contradictions
        contradictions = [v for v in self.validations if v.final_status == "contradiction"]
        if contradictions:
            lines.append(f"## Contradictions ({len(contradictions)} items)")
            lines.append(f"")
            for v in contradictions:
                lines.append(f"### {v.evidence_id}: {v.claim_or_point[:80]}...")
                lines.append(f"- **Paper**: {v.paper_title}")
                if v.pass1.issues:
                    lines.append(f"- **Issues**: {'; '.join(v.pass1.issues)}")
                if v.pass1.grounding_quote:
                    lines.append(f"- **Grounding Quote**: {v.pass1.grounding_quote[:200]}")
                lines.append(f"- **Alignment Score**: {v.final_alignment_score:.2f}")
                lines.append(f"- **Action Required**: Evidence statement review recommended")
                lines.append(f"")

        # Uncertain
        uncertains = [v for v in self.validations if v.final_status == "uncertain"]
        if uncertains:
            lines.append(f"## Uncertain ({len(uncertains)} items)")
            lines.append(f"")
            for v in uncertains:
                lines.append(f"### {v.evidence_id}: {v.claim_or_point[:80]}...")
                lines.append(f"- **Paper**: {v.paper_title}")
                lines.append(f"- **Alignment Score**: {v.final_alignment_score:.2f}")
                if v.pass1.issues:
                    lines.append(f"- **Issues**: {'; '.join(v.pass1.issues)}")
                lines.append(f"")

        # Verified summary (compact)
        verified = [v for v in self.validations if v.final_status == "verified"]
        if verified:
            lines.append(f"## Verified ({len(verified)} items)")
            lines.append(f"")
            lines.append(f"| Evidence ID | Paper | Score |")
            lines.append(f"|-------------|-------|-------|")
            for v in verified:
                lines.append(f"| {v.evidence_id} | {v.paper_title[:50]} | {v.final_alignment_score:.2f} |")
            lines.append(f"")

        return "\n".join(lines)


# ------------------------------------------------------------------
# Pass 1: Evidence-Paper Alignment
# ------------------------------------------------------------------

_PASS1_SYSTEM = """\
あなたは研究論文の正確性を検証する査読者です。
Evidence（論文から抽出された知見の要約）が元論文の内容と整合しているかを厳密に検証してください。

重要な指示:
- Evidence の claim_or_point と evidence_text が元論文テキストの内容と一致するかを確認
- 一致する場合は元論文の該当箇所を直接引用（grounding_quote）
- 解釈の正確性を評価（過大解釈、ニュアンスの欠落、因果関係の誤読に注意）
- alignment_score は厳密に評価してください

alignment_score の基準:
- 1.0: 完全に一致（直接引用レベル）
- 0.7-0.9: 実質的に一致（解釈に問題なし）
- 0.4-0.6: 部分的に一致（一部解釈にずれあり）
- 0.1-0.3: 大きくずれている
- 0.0: 矛盾している

出力は必ず JSON 形式で返してください。"""


def _build_pass1_prompt(evidence_items: List[Dict[str, Any]], paper_text: str) -> str:
    items_text = ""
    for i, ev in enumerate(evidence_items):
        items_text += (
            f"\n### Evidence {i + 1}\n"
            f"- claim_or_point: {ev.get('claim_or_point', '')}\n"
            f"- evidence_text: {ev.get('evidence_text', '')}\n"
        )

    return (
        f"## Evidence Items\n{items_text}\n\n"
        f"## Source Paper Text\n{paper_text[:60_000]}\n\n"
        f"## Instructions\n"
        f"上記の各 Evidence について、元論文テキストとの整合性を検証してください。\n"
        f"以下の JSON 形式で出力してください:\n\n"
        f'{{"results": [\n'
        f'  {{\n'
        f'    "evidence_index": 0,\n'
        f'    "grounding_quote": "元論文の該当箇所の直接引用",\n'
        f'    "alignment_score": 0.85,\n'
        f'    "section_ref": "Section X.X / Table Y",\n'
        f'    "reasoning": "判定根拠",\n'
        f'    "issues": ["問題点があれば記載"]\n'
        f'  }}\n'
        f']}}'
    )


def align_evidence_batch(
    evidence_items: List[Dict[str, Any]],
    paper_text: str,
    llm_client: Any,
) -> List[Pass1Result]:
    """Run Pass 1 alignment for a batch of evidence items against paper text.

    Returns a list of Pass1Result, one per evidence item.
    On LLM failure, returns default (empty) Pass1Result for each item.
    """
    user_msg = _build_pass1_prompt(evidence_items, paper_text)
    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _PASS1_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Pass 1 LLM call failed: %s", e)
        return [Pass1Result() for _ in evidence_items]

    resp_text = _extract_text(resp)
    parsed = _parse_json(resp_text)
    if not parsed or "results" not in parsed:
        logger.error("Pass 1 JSON parse failed. Raw: %s", resp_text[:300])
        return [Pass1Result() for _ in evidence_items]

    results = []
    raw_results = parsed["results"]
    for i in range(len(evidence_items)):
        if i < len(raw_results):
            results.append(Pass1Result.from_dict(raw_results[i]))
        else:
            logger.warning("Pass 1: missing result for evidence index %d", i)
            results.append(Pass1Result())

    usage = resp.get("usage", {})
    logger.info(
        "Pass 1: %d evidence items validated (in=%d, out=%d tokens)",
        len(results),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )

    return results


# ------------------------------------------------------------------
# Pass 2: Self-Consistency Check
# ------------------------------------------------------------------

_PASS2_SYSTEM = """\
あなたは別の独立した査読者です。
初回検証の結果を再検証し、判定の妥当性を評価してください。
初回検証に同意する場合は verdict: "agree"、同意しない場合は verdict: "disagree" と修正スコアを返してください。

出力は必ず JSON 形式で返してください。"""


def _build_pass2_prompt(
    evidence: Dict[str, Any],
    paper_text: str,
    pass1: Pass1Result,
) -> str:
    return (
        f"## Evidence\n"
        f"- claim_or_point: {evidence.get('claim_or_point', '')}\n"
        f"- evidence_text: {evidence.get('evidence_text', '')}\n\n"
        f"## Source Paper Text\n{paper_text[:60_000]}\n\n"
        f"## Initial Validation Result\n"
        f"- grounding_quote: {pass1.grounding_quote}\n"
        f"- alignment_score: {pass1.alignment_score}\n"
        f"- reasoning: {pass1.reasoning}\n"
        f"- issues: {pass1.issues}\n\n"
        f"## Instructions\n"
        f"初回検証の判定に同意しますか？以下の JSON 形式で出力してください:\n\n"
        f'{{\n'
        f'  "verdict": "agree or disagree",\n'
        f'  "revised_alignment_score": null,\n'
        f'  "reasoning": "判定根拠",\n'
        f'  "additional_issues": []\n'
        f'}}'
    )


def check_consistency(
    evidence: Dict[str, Any],
    paper_text: str,
    pass1: Pass1Result,
    llm_client: Any,
) -> Pass2Result:
    """Run Pass 2 self-consistency check for a single evidence item."""
    user_msg = _build_pass2_prompt(evidence, paper_text, pass1)
    body = {
        "model": _MODEL,
        "max_tokens": 1024,
        "system": _PASS2_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Pass 2 LLM call failed: %s", e)
        return Pass2Result(verdict="error", reasoning=str(e))

    resp_text = _extract_text(resp)
    parsed = _parse_json(resp_text)
    if not parsed:
        logger.error("Pass 2 JSON parse failed. Raw: %s", resp_text[:300])
        return Pass2Result(verdict="error", reasoning="JSON parse failed")

    usage = resp.get("usage", {})
    logger.debug(
        "Pass 2 done (in=%d, out=%d tokens)",
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )

    return Pass2Result.from_dict(parsed)


# ------------------------------------------------------------------
# Status assignment
# ------------------------------------------------------------------

def assign_validation_status(
    pass1: Pass1Result,
    pass2: Optional[Pass2Result],
    text_source: str,
) -> tuple[str, float, bool]:
    """Determine final_status, final_alignment_score, and needs_human_review.

    Returns (status, score, needs_review).
    """
    # Unverifiable: no paper text available
    if text_source == "title_only":
        return "unverifiable", 0.0, False

    score = pass1.alignment_score

    # If Pass 2 was run and disagreed, use revised score
    if pass2 and pass2.verdict == "disagree" and pass2.revised_alignment_score is not None:
        score = pass2.revised_alignment_score

    # Determine status based on score
    if score < 0.4:
        status = "contradiction"
    elif score < 0.7:
        status = "uncertain"
    else:
        status = "verified"

    # Pass2 disagreement downgrades to uncertain
    if pass2 and pass2.verdict == "disagree" and status == "verified":
        status = "uncertain"

    needs_review = status in ("contradiction", "uncertain")

    return status, round(score, 3), needs_review


def needs_pass2(pass1: Pass1Result, text_source: str) -> bool:
    """Determine if Pass 2 is needed (MVP optimization).

    Only run Pass 2 for uncertain or contradiction results.
    """
    if text_source == "title_only":
        return False

    score = pass1.alignment_score
    return score < 0.7  # uncertain or contradiction range


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
