# src/lit_review/reviewer.py
"""099 Research Output Review — service logic.

Cross-section review of all generated drafts.  Checks:
  1. Section-level quality (word count, citations, structure)
  2. Cross-section consistency (RQ, hypotheses, methods alignment)
  3. Logical flow integrity (RQ → Lit → Hypotheses → Methods)
  4. L2 Working Draft readiness assessment

Outputs review_report.json + review_report.md.

Usage::

    from src.lit_review.reviewer import run_review

    result = run_review(run_dir, llm_client=client)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.lit_review.drafters.base import BaseDrafter

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 8192

# Sections to review
_DRAFT_FILES = {
    "introduction": "draft_introduction.md",
    "literature_review": "draft_literature_review.md",
    "hypotheses": "draft_hypotheses.md",
    "methods": "draft_methods.md",
}

# L2 Working Draft criteria (from requirements.md)
_L2_CRITERIA = {
    "coverage": "Introduction, Literature Review, Hypotheses, Methods の 4 セクションが生成される",
    "word_count": "合計 10,000+ words",
    "depth": "各セクションが outline の target word count の 80% 以上",
    "citations": "先行研究への言及が各セクションに含まれる",
    "internal_consistency": "仮説番号・変数名・手法名がセクション間で一致",
    "completeness": "全仮説が hypotheses draft に含まれる。全 design が methods draft に含まれる",
}


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------

@dataclass
class SectionReview:
    """Review of a single section."""
    section_id: str
    available: bool = False
    word_count: int = 0
    target_words: int = 0
    word_ratio: float = 0.0
    meets_target: bool = False
    has_citations: bool = False
    citation_count: int = 0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class CrossSectionCheck:
    """Result of a cross-section consistency check."""
    check_name: str
    passed: bool = False
    detail: str = ""


@dataclass
class L2Assessment:
    """L2 Working Draft readiness assessment."""
    criterion: str
    passed: bool = False
    detail: str = ""


@dataclass
class ReviewResult:
    """Complete review output."""
    status: str = "failed"
    rq_title: str = ""
    sections: List[SectionReview] = field(default_factory=list)
    cross_section_checks: List[CrossSectionCheck] = field(default_factory=list)
    l2_assessment: List[L2Assessment] = field(default_factory=list)
    l2_passed: bool = False
    overall_quality_score: float = 0.0  # 0-10
    llm_review: str = ""  # LLM-generated narrative review
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "script_id": "099_research_output_review",
            "status": self.status,
            "rq_title": self.rq_title,
            "sections": [asdict(s) for s in self.sections],
            "cross_section_checks": [asdict(c) for c in self.cross_section_checks],
            "l2_assessment": [asdict(a) for a in self.l2_assessment],
            "l2_passed": self.l2_passed,
            "overall_quality_score": self.overall_quality_score,
            "error": self.error,
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------
# Section-level analysis (no LLM)
# ------------------------------------------------------------------

def _count_citations(text: str) -> int:
    """Count unique academic citations in text."""
    p1 = re.findall(
        r"[(\uff08][A-Z][a-zé]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-zé]+))?[.,]?\s*\d{4}[)\uff09]",
        text,
    )
    p2 = re.findall(
        r"[A-Z][a-zé]+(?:\s+et\s+al\.?)?\s*[(\uff08]\d{4}[)\uff09]",
        text,
    )
    return len(set(p1 + p2))


def _analyze_section(
    section_id: str,
    text: str,
    outline_spec: Dict[str, Any],
) -> SectionReview:
    """Analyze a single section without LLM."""
    word_count = BaseDrafter._count_words(text)
    target = outline_spec.get("target_words", 0)
    ratio = word_count / target if target > 0 else 0.0
    meets = 0.8 <= ratio <= 1.2 if target > 0 else True
    cite_count = _count_citations(text)

    return SectionReview(
        section_id=section_id,
        available=True,
        word_count=word_count,
        target_words=target,
        word_ratio=round(ratio, 3),
        meets_target=meets,
        has_citations=cite_count > 0,
        citation_count=cite_count,
    )


# ------------------------------------------------------------------
# Cross-section consistency checks (no LLM)
# ------------------------------------------------------------------

def _check_rq_consistency(drafts: Dict[str, str], rq_title: str) -> CrossSectionCheck:
    """Check that RQ is referenced consistently across sections."""
    # Extract key terms from RQ.
    # For Japanese text (no spaces), extract meaningful substrings.
    rq_lower = rq_title.lower()
    space_terms = [t for t in rq_lower.split() if len(t) > 3]

    if len(space_terms) <= 2:
        # Japanese: extract key noun phrases via character n-grams
        # Use meaningful substrings likely to appear in the text
        key_terms = []
        for pattern in [r"[a-zA-Z]{3,}", r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,4}"]:
            key_terms.extend(re.findall(pattern, rq_title))
        key_terms = list(dict.fromkeys(key_terms))[:8]  # dedupe, limit
    else:
        key_terms = space_terms[:5]

    sections_with_rq = []
    for sid, text in drafts.items():
        text_check = text.lower() if all(ord(c) < 128 for c in "".join(key_terms)) else text
        matches = sum(1 for t in key_terms if t.lower() in text_check.lower())
        if matches >= 2:
            sections_with_rq.append(sid)

    passed = len(sections_with_rq) >= 3
    return CrossSectionCheck(
        check_name="RQ consistency across sections",
        passed=passed,
        detail=f"RQ key terms ({len(key_terms)}): {key_terms[:5]}... Found in {len(sections_with_rq)}/4 sections: {sections_with_rq}",
    )


def _check_hypothesis_methods_alignment(drafts: Dict[str, str]) -> CrossSectionCheck:
    """Check that hypotheses in 096 are referenced in methods 097."""
    hyp_text = drafts.get("hypotheses", "")
    methods_text = drafts.get("methods", "")

    if not hyp_text or not methods_text:
        return CrossSectionCheck(
            check_name="Hypothesis-Methods alignment",
            passed=False,
            detail="Missing hypotheses or methods draft",
        )

    hyp_markers = sorted(set(re.findall(r"\bH\d+\b", hyp_text)))
    methods_markers = sorted(set(re.findall(r"\bH\d+\b", methods_text)))

    covered = set(hyp_markers) & set(methods_markers)
    missing = set(hyp_markers) - set(methods_markers)

    passed = len(missing) <= 1  # allow 1 missing
    return CrossSectionCheck(
        check_name="Hypothesis-Methods alignment",
        passed=passed,
        detail=f"Hypotheses {hyp_markers}, Methods references {methods_markers}, Missing in methods: {sorted(missing) if missing else 'none'}",
    )


def _check_gap_hypothesis_flow(drafts: Dict[str, str]) -> CrossSectionCheck:
    """Check that lit review gaps connect to hypotheses."""
    lit_text = drafts.get("literature_review", "")
    hyp_text = drafts.get("hypotheses", "")

    if not lit_text or not hyp_text:
        return CrossSectionCheck(
            check_name="Gap → Hypothesis flow",
            passed=False,
            detail="Missing literature review or hypotheses draft",
        )

    gap_markers = ["ギャップ", "gap", "未解明", "限界", "blindspot"]
    lit_has_gaps = sum(1 for m in gap_markers if m in lit_text.lower()) >= 2
    hyp_has_theory = any(m in hyp_text.lower() for m in ["理論", "theory", "先行研究", "既存研究"])

    passed = lit_has_gaps and hyp_has_theory
    return CrossSectionCheck(
        check_name="Gap → Hypothesis flow",
        passed=passed,
        detail=f"Lit review gaps: {lit_has_gaps}, Hypotheses theoretical grounding: {hyp_has_theory}",
    )


def _check_logical_flow(drafts: Dict[str, str]) -> CrossSectionCheck:
    """Check end-of-section bridges."""
    bridges = {}
    pairs = [
        ("introduction", "literature_review", ["文献", "レビュー", "review", "literature", "先行研究"]),
        ("literature_review", "hypotheses", ["仮説", "hypothes", "本研究", "次章"]),
        ("hypotheses", "methods", ["method", "手法", "検証", "実証", "分析"]),
    ]

    for from_sec, to_sec, markers in pairs:
        text = drafts.get(from_sec, "")
        if not text:
            bridges[f"{from_sec}→{to_sec}"] = False
            continue
        last_500 = text[-500:].lower()
        bridges[f"{from_sec}→{to_sec}"] = any(m in last_500 for m in markers)

    passed = all(bridges.values())
    detail_parts = [f"{k}: {'OK' if v else 'MISSING'}" for k, v in bridges.items()]
    return CrossSectionCheck(
        check_name="Section-to-section bridges",
        passed=passed,
        detail="; ".join(detail_parts),
    )


# ------------------------------------------------------------------
# L2 Assessment (no LLM)
# ------------------------------------------------------------------

def _assess_l2(
    sections: List[SectionReview],
    drafts: Dict[str, str],
    cross_checks: List[CrossSectionCheck],
) -> List[L2Assessment]:
    """Assess L2 Working Draft readiness."""
    assessments: List[L2Assessment] = []

    # Coverage
    available = [s for s in sections if s.available]
    required = {"introduction", "hypotheses", "methods"}
    available_ids = {s.section_id for s in available}
    missing_required = required - available_ids
    assessments.append(L2Assessment(
        criterion="coverage",
        passed=len(missing_required) == 0,
        detail=f"{len(available)}/4 sections available. Missing required: {sorted(missing_required) if missing_required else 'none'}",
    ))

    # Word count
    total_words = sum(s.word_count for s in available)
    assessments.append(L2Assessment(
        criterion="word_count",
        passed=total_words >= 10000,
        detail=f"Total: {total_words} words (target: 10,000+)",
    ))

    # Depth
    below_target = [s for s in available if s.target_words > 0 and s.word_ratio < 0.8]
    assessments.append(L2Assessment(
        criterion="depth",
        passed=len(below_target) == 0,
        detail=f"Sections below 80% target: {[s.section_id for s in below_target] if below_target else 'none'}",
    ))

    # Citations
    no_citations = [s for s in available if not s.has_citations]
    assessments.append(L2Assessment(
        criterion="citations",
        passed=len(no_citations) == 0,
        detail=f"Sections without citations: {[s.section_id for s in no_citations] if no_citations else 'none'}",
    ))

    # Internal consistency (from cross-checks)
    hyp_methods = next((c for c in cross_checks if c.check_name == "Hypothesis-Methods alignment"), None)
    assessments.append(L2Assessment(
        criterion="internal_consistency",
        passed=hyp_methods.passed if hyp_methods else False,
        detail=hyp_methods.detail if hyp_methods else "Check not run",
    ))

    # Completeness
    hyp_text = drafts.get("hypotheses", "")
    h_markers = sorted(set(re.findall(r"\bH\d+\b", hyp_text)))
    assessments.append(L2Assessment(
        criterion="completeness",
        passed=len(h_markers) >= 9,
        detail=f"Hypothesis markers in draft: {len(h_markers)} ({h_markers})",
    ))

    return assessments


# ------------------------------------------------------------------
# LLM narrative review
# ------------------------------------------------------------------

def _generate_llm_review(
    drafts: Dict[str, str],
    rq_title: str,
    sections: List[SectionReview],
    cross_checks: List[CrossSectionCheck],
    l2_results: List[L2Assessment],
    llm_client: Any,
) -> str:
    """Generate narrative review via LLM."""
    # Build compact summary of drafts for LLM
    draft_summaries: List[str] = []
    for sid, text in drafts.items():
        word_count = BaseDrafter._count_words(text)
        # First and last 400 chars
        preview = text[:400] + "\n...\n" + text[-400:] if len(text) > 800 else text
        draft_summaries.append(f"## {sid} ({word_count} words)\n{preview}")

    # Build check summary
    check_summary = "## Automated Checks\n"
    for c in cross_checks:
        icon = "PASS" if c.passed else "FAIL"
        check_summary += f"- [{icon}] {c.check_name}: {c.detail}\n"

    l2_summary = "\n## L2 Assessment\n"
    for a in l2_results:
        icon = "PASS" if a.passed else "FAIL"
        l2_summary += f"- [{icon}] {a.criterion}: {a.detail}\n"

    system = (
        "あなたは学術論文のレビュアーです。\n"
        "以下の論文ドラフトの各セクションと自動チェック結果を評価してください。\n"
        "日本語で、建設的かつ具体的なレビューを書いてください。"
    )

    user = (
        f"## RQ\n{rq_title}\n\n"
        + "\n\n".join(draft_summaries)
        + f"\n\n{check_summary}"
        + f"\n{l2_summary}"
        + "\n\n## 指示\n"
        "以下の構成でレビューを書いてください:\n"
        "1. Overall Research Logic — RQ→文献→仮説→方法 の論理フローの評価\n"
        "2. Strengths — 良い点 (3–5 points)\n"
        "3. Weaknesses — 改善が必要な点 (3–5 points)\n"
        "4. Logical Inconsistencies — セクション間の不整合があれば指摘\n"
        "5. Missing Elements — 欠けている要素\n"
        "6. Suggested Improvements — 具体的な改善提案 (actionable)\n"
        "7. Section-by-Section Comments — 各セクションへの短いコメント\n"
        "8. Overall Quality Score — 0–10 で総合評価 (10 = submission-ready)\n"
        "\n簡潔に書いてください。各項目は箇条書きで。"
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
        logger.error("LLM review call failed: %s", e)
        return ""

    text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("LLM review: in=%d out=%d tokens", usage.get("input_tokens", 0), usage.get("output_tokens", 0))
    return text


def _extract_quality_score(llm_review: str) -> float:
    """Extract numeric quality score from LLM review text."""
    # Look for patterns like "7/10", "7.5/10", "Score: 7"
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", llm_review)
    if m:
        return min(10.0, float(m.group(1)))
    m = re.search(r"(?:score|スコア|評価)[：:]\s*(\d+(?:\.\d+)?)", llm_review, re.IGNORECASE)
    if m:
        return min(10.0, float(m.group(1)))
    return 0.0


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def run_review(
    run_dir: Path,
    *,
    llm_client: Any,
) -> ReviewResult:
    """Run full review. Entry point for 099 script."""
    result = ReviewResult()
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        # Load RQ
        rq_path = run_dir / "rq_context.json"
        rq_title = ""
        if rq_path.exists():
            rq_title = json.loads(rq_path.read_text()).get("title", "")
        result.rq_title = rq_title

        # Load outline
        outline: Dict[str, Any] = {}
        outline_path = run_dir / "paper_outline.json"
        if outline_path.exists():
            outline = json.loads(outline_path.read_text())

        outline_specs: Dict[str, Dict] = {}
        for sec in outline.get("sections", []):
            outline_specs[sec.get("section_id", "")] = sec

        # Load drafts
        drafts: Dict[str, str] = {}
        for sid, fname in _DRAFT_FILES.items():
            p = run_dir / fname
            if p.exists():
                drafts[sid] = p.read_text()

        if not drafts:
            result.error = "No draft files found"
            return result

        # 1. Section-level analysis
        for sid, fname in _DRAFT_FILES.items():
            if sid in drafts:
                spec = outline_specs.get(sid, {})
                review = _analyze_section(sid, drafts[sid], spec)
                result.sections.append(review)
            else:
                result.sections.append(SectionReview(section_id=sid, available=False))

        # 2. Cross-section consistency checks
        result.cross_section_checks = [
            _check_rq_consistency(drafts, rq_title),
            _check_hypothesis_methods_alignment(drafts),
            _check_gap_hypothesis_flow(drafts),
            _check_logical_flow(drafts),
        ]

        # 3. L2 assessment
        result.l2_assessment = _assess_l2(result.sections, drafts, result.cross_section_checks)
        result.l2_passed = all(a.passed for a in result.l2_assessment)

        # 4. LLM narrative review
        result.llm_review = _generate_llm_review(
            drafts, rq_title, result.sections,
            result.cross_section_checks, result.l2_assessment,
            llm_client,
        )

        # Extract score from LLM review
        if result.llm_review:
            result.overall_quality_score = _extract_quality_score(result.llm_review)

        result.status = "generated"
        result.metadata = {
            "created_at": now_iso,
            "model": _MODEL,
            "sections_reviewed": len([s for s in result.sections if s.available]),
            "sections_missing": len([s for s in result.sections if not s.available]),
        }

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.error("099: unexpected error: %s", e)

    return result


def render_review_markdown(result: ReviewResult) -> str:
    """Render ReviewResult as human-readable Markdown."""
    parts: List[str] = []
    parts.append("# Research Output Review\n")
    parts.append(f"**RQ**: {result.rq_title}\n")
    parts.append(f"**L2 Working Draft**: {'PASSED' if result.l2_passed else 'NOT YET'}")
    parts.append(f"**Overall Quality Score**: {result.overall_quality_score}/10\n")

    # Section summaries
    parts.append("## Section Summary\n")
    parts.append("| Section | Words | Target | Ratio | Citations | Status |")
    parts.append("|---------|-------|--------|-------|-----------|--------|")
    for s in result.sections:
        if s.available:
            status = "OK" if s.meets_target else f"{'below' if s.word_ratio < 0.8 else 'above'}"
            parts.append(
                f"| {s.section_id} | {s.word_count} | {s.target_words} | "
                f"{s.word_ratio:.2f} | {s.citation_count} | {status} |"
            )
        else:
            parts.append(f"| {s.section_id} | — | — | — | — | MISSING |")

    total = sum(s.word_count for s in result.sections if s.available)
    parts.append(f"\n**Total**: {total} words\n")

    # Cross-section checks
    parts.append("## Cross-Section Checks\n")
    for c in result.cross_section_checks:
        icon = "PASS" if c.passed else "FAIL"
        parts.append(f"- **[{icon}]** {c.check_name}: {c.detail}")

    # L2 Assessment
    parts.append("\n## L2 Working Draft Assessment\n")
    for a in result.l2_assessment:
        icon = "PASS" if a.passed else "FAIL"
        parts.append(f"- **[{icon}]** {a.criterion}: {a.detail}")

    # LLM Review
    if result.llm_review:
        parts.append("\n## Detailed Review (LLM)\n")
        parts.append(result.llm_review)

    return "\n".join(parts) + "\n"
