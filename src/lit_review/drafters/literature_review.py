# src/lit_review/drafters/literature_review.py
"""098 Literature Review Drafter — service logic.

Generates draft_literature_review.md following the outline spec from 094.
Produces a thematic, critical literature review that:

  - Organizes prior work by theoretical streams
  - Integrates established, emerging, and contested findings
  - Identifies gaps, blindspots, and inconsistencies
  - Bridges to the hypotheses and methodology of this study

This is NOT a summary of individual papers.  It is a synthesis that
builds the theoretical foundation for the hypotheses.

Usage::

    from src.lit_review.drafters.literature_review import LiteratureReviewDrafter

    drafter = LiteratureReviewDrafter()
    result = drafter.generate(run_dir, llm_client=client)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from src.lit_review.drafters.base import BaseDrafter, PromptContext


class LiteratureReviewDrafter(BaseDrafter):
    """Generates the Literature Review section of the paper."""

    section_id = "literature_review"
    output_file = "draft_literature_review.md"
    default_max_tokens = 16384

    def required_inputs(self) -> List[str]:
        return ["paper_outline.json", "lit_review.json", "rq_context.json"]

    def build_prompt(self, ctx: PromptContext) -> Tuple[str, str]:
        spec = ctx.outline_spec
        lr = ctx.require("lit_review.json")
        rq = ctx.require("rq_context.json")
        hyp = ctx.input("hypotheses.json")
        landscape = ctx.input("landscape.json")

        target_words = spec.get("target_words", 3000)
        streams = lr.get("theoretical_streams", [])
        findings = lr.get("empirical_findings", {})
        open_questions = lr.get("open_questions", [])

        # --- System prompt ---
        system = (
            "あなたは社会科学分野の学術論文の Literature Review セクションの執筆者です。\n"
            "以下のルールに厳密に従ってください:\n\n"
            "1. 日本語で執筆する\n"
            "2. 先行研究を (著者名, 年) 形式で引用する\n"
            f"3. 目標語数: 約{target_words}語。これは厳守。個別論文を詳述せず、統合的に簡潔に書く\n"
            "4. 個別論文の要約ではなく、テーマ別・論点別の統合的レビューを書く\n"
            "5. 以下の構成で書く:\n"
            "   (a) レビューの目的とスコープ\n"
            "   (b) 理論的ストリームごとの整理 (各ストリームの主要知見と限界)\n"
            "   (c) 実証的知見の統合 (確立された知見 / 新興の知見 / 対立する知見)\n"
            "   (d) 方法論的アプローチの現状と限界\n"
            "   (e) リサーチギャップの体系的整理\n"
            "   (f) 本研究の位置づけと仮説への橋渡し\n"
            "6. Markdown 形式で出力する (# Literature Review から始める)\n"
            "7. Introduction との重複を最小化する (背景説明は省略し、研究の比較・統合・批判に集中)\n"
            "8. セクション末尾で Hypotheses Development に自然に接続する\n"
        )

        # --- User prompt ---
        user_parts: List[str] = []

        # Layer 1: Outline spec (full)
        if spec:
            user_parts.append("## Outline Specification (094)")
            user_parts.append(f"Target words: {target_words}")
            flow = spec.get("argument_flow", [])
            if flow:
                user_parts.append("Argument flow:")
                for i, step in enumerate(flow, 1):
                    user_parts.append(f"  {i}. {step}")
            refs = spec.get("key_references", [])
            if refs:
                user_parts.append(f"Key references to cite: {', '.join(refs)}")
            connects_to = spec.get("connects_to", "")
            if connects_to:
                user_parts.append(f"Must connect to: {connects_to} section")
            user_parts.append("")

        # Layer 2a: RQ (compact — full is in intro)
        user_parts.append(f"## Research Question")
        user_parts.append(f"{rq.get('title', '')}")
        user_parts.append("")

        # Layer 2b: Theoretical streams (full detail)
        user_parts.append(f"## Theoretical Streams ({len(streams)})")
        for stream in streams:
            name = stream.get("name", "")
            desc = stream.get("description", "")
            concepts = stream.get("key_concepts", [])
            papers = stream.get("papers", [])
            user_parts.append(f"\n### {name}")
            user_parts.append(f"Description: {desc}")
            if concepts:
                user_parts.append(f"Key concepts: {', '.join(concepts)}")
            if papers:
                user_parts.append(f"Key papers ({len(papers)}):")
                for p in papers:
                    user_parts.append(f"  - {p[:120] if isinstance(p, str) else str(p)[:120]}")
        user_parts.append("")

        # Layer 2c: Empirical findings (ALL, categorized)
        for category, label in [("established", "Established"), ("emerging", "Emerging"), ("contested", "Contested")]:
            items = findings.get(category, [])
            if not items:
                continue
            user_parts.append(f"\n## {label} Findings ({len(items)})")
            for item in items:
                if category == "contested":
                    user_parts.append(f"\n**Topic**: {item.get('topic', '')}")
                    user_parts.append(f"Disagreement: {item.get('nature_of_disagreement', '')}")
                    for pos in item.get("positions", []):
                        stmt = pos.get("statement", "") if isinstance(pos, dict) else str(pos)
                        paps = pos.get("papers", []) if isinstance(pos, dict) else []
                        user_parts.append(f"  Position: {stmt[:200]}")
                        if paps:
                            user_parts.append(f"  Papers: {', '.join(str(p)[:60] for p in paps[:3])}")
                else:
                    stmt = item.get("statement", item.get("finding", ""))
                    summary = item.get("evidence_summary", "")
                    papers = item.get("supporting_papers", [])
                    strength = item.get("strength", "")
                    user_parts.append(f"\n- **{stmt}** (strength: {strength})")
                    if summary:
                        user_parts.append(f"  Evidence: {summary[:300]}")
                    if papers:
                        user_parts.append(f"  Papers: {', '.join(str(p)[:60] for p in papers[:4])}")
        user_parts.append("")

        # Layer 2d: Open questions / research gaps
        if open_questions:
            user_parts.append(f"## Open Questions ({len(open_questions)})")
            for q in open_questions:
                user_parts.append(f"- {q.get('description', q.get('question', ''))}")
            user_parts.append("")

        # Layer 2e: Landscape data (blindspots, opportunities, methodological landscape)
        if landscape:
            blindspots = landscape.get("blindspots", [])
            if blindspots:
                user_parts.append(f"## Research Blindspots ({len(blindspots)})")
                for b in blindspots:
                    if isinstance(b, dict):
                        user_parts.append(f"- **{b.get('area', '')}**: {b.get('what_is_missing', '')} (severity: {b.get('severity', '')})")
                    else:
                        user_parts.append(f"- {str(b)[:200]}")
                user_parts.append("")

            hotspots = landscape.get("hotspots", [])
            if hotspots:
                user_parts.append(f"## Research Hotspots ({len(hotspots)})")
                for h in hotspots:
                    if isinstance(h, dict):
                        user_parts.append(f"- **{h.get('area', '')}**: {h.get('evidence', '')[:150]} (strength: {h.get('strength', '')})")
                    else:
                        user_parts.append(f"- {str(h)[:200]}")
                user_parts.append("")

            ml = landscape.get("methodological_landscape", {})
            if isinstance(ml, dict):
                quant = ml.get("quantitative", [])
                if quant:
                    user_parts.append("## Methodological Landscape")
                    methods_str = ", ".join(
                        f"{m.get('name', '')} ({m.get('paper_count', '')} papers)"
                        for m in quant if isinstance(m, dict)
                    )
                    user_parts.append(f"Quantitative: {methods_str}")
                    user_parts.append("")

        # Layer 2f: Hypotheses (compact — for showing what gaps this study addresses)
        hypotheses = hyp.get("hypotheses", [])
        if hypotheses:
            user_parts.append(f"## This Study's Hypotheses ({len(hypotheses)}) — for gap-to-contribution mapping")
            for i, h in enumerate(hypotheses):
                user_parts.append(f"H{i + 1}: {h.get('hypothesis_statement', '')[:120]}")
            user_parts.append("")

        # Layer 3: Cross-references (compact)
        if ctx.cross_refs:
            user_parts.append(ctx.cross_refs)
            user_parts.append("")

        # Instruction
        user_parts.append("## 指示")
        user_parts.append(
            "上記の情報をもとに、Literature Review セクションを Markdown で書いてください。\n"
            "個別論文の要約ではなく、テーマ別の統合的レビューにしてください。\n"
            "既存研究の一致点・不一致点・方法論的限界・未解決課題を明示してください。\n"
            "特に Contested Findings と Blindspots を活用して、研究ギャップを体系的に示してください。\n"
            "最後の段落で、これらのギャップに対する本研究の仮説への橋渡しを書いてください。\n"
            "Introduction との重複を避け、先行研究の比較・統合・批判的整理に集中してください。\n"
            f"目標語数: 約{target_words}語。冗長化を避けてください。"
        )

        return system, "\n".join(user_parts)

    def validate_content(self, text: str, outline_spec: Dict[str, Any]) -> List[str]:
        """Literature review-specific quality checks."""
        warnings: List[str] = []
        text_lower = text.lower()

        # Check for multiple theoretical streams (thematic organization)
        stream_markers = ["理論", "theory", "ストリーム", "stream", "系譜", "アプローチ", "perspective"]
        stream_count = sum(1 for m in stream_markers if m in text_lower)
        if stream_count < 2:
            warnings.append("Limited thematic organization detected — expected multiple theoretical streams")

        # Check for gap/limitation/inconsistency language
        gap_markers = [
            "ギャップ", "gap", "限界", "limitation", "不一致", "inconsisten",
            "未解明", "未解決", "blindspot", "対立", "contested", "矛盾",
        ]
        gap_count = sum(1 for m in gap_markers if m in text_lower)
        if gap_count < 2:
            warnings.append("Insufficient gap/inconsistency discussion — literature review should identify research gaps")

        # Check for bridge to hypotheses at end
        last_500 = text[-500:].lower() if len(text) > 500 else text_lower
        bridge_markers = ["仮説", "hypothes", "本研究", "本稿", "次章", "次節", "上記のギャップ"]
        if not any(m in last_500 for m in bridge_markers):
            warnings.append("No bridge to Hypotheses detected in final paragraph")

        # Check minimum paragraph count (lit review should be substantive)
        paragraphs = [p for p in text.split("\n\n") if p.strip() and not p.strip().startswith("#")]
        if len(paragraphs) < 6:
            warnings.append(f"Only {len(paragraphs)} text paragraphs — expected at least 8 for literature review")

        return warnings
