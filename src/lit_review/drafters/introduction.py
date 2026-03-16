# src/lit_review/drafters/introduction.py
"""095 Introduction Drafter — service logic.

Generates draft_introduction.md following the outline spec from 094.
Produces a paper-quality Introduction with the flow:

  1. Research topic background & significance
  2. Problem statement / real-world importance
  3. Prior work limitations & research gap
  4. Research question & contribution (hypotheses preview)
  5. Bridge to subsequent sections (lit review → hypotheses → methods)

Usage::

    from src.lit_review.drafters.introduction import IntroductionDrafter

    drafter = IntroductionDrafter()
    result = drafter.generate(run_dir, llm_client=client)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from src.lit_review.drafters.base import BaseDrafter, PromptContext


class IntroductionDrafter(BaseDrafter):
    """Generates the Introduction section of the paper."""

    section_id = "introduction"
    output_file = "draft_introduction.md"
    default_max_tokens = 8192

    def required_inputs(self) -> List[str]:
        return ["paper_outline.json", "rq_context.json", "lit_review.json", "hypotheses.json"]

    def build_prompt(self, ctx: PromptContext) -> Tuple[str, str]:
        spec = ctx.outline_spec
        rq = ctx.require("rq_context.json")
        lr = ctx.require("lit_review.json")
        hyp = ctx.require("hypotheses.json")

        # --- System prompt ---
        target_words = spec.get("target_words", 2000)
        system = (
            "あなたは社会科学分野の学術論文の Introduction セクションの執筆者です。\n"
            "以下のルールに厳密に従ってください:\n\n"
            "1. 日本語で執筆する\n"
            "2. 学術論文の Introduction として適切な文体・構成を用いる\n"
            "3. 先行研究に言及する際は必ず (著者名, 年) 形式で引用する\n"
            f"4. 目標語数: 約{target_words}語 (日本語)。これは厳守。超過しないよう簡潔に書く\n"
            "5. 以下の5段構成で書く (各段は簡潔に。背景説明を過度に展開しない):\n"
            "   (a) 研究テーマの背景と社会的・学術的重要性\n"
            "   (b) 問題設定 — なぜこの問題が重要か\n"
            "   (c) 既存研究の限界とリサーチギャップ\n"
            "   (d) 本研究の問い・目的・仮説への橋渡し\n"
            "   (e) 論文構成の概要 (Literature Review → Hypotheses → Methods の流れ)\n"
            "6. Markdown 形式で出力する (# Introduction から始める)\n"
            "7. セクション末尾で Literature Review に自然に接続する文を含める\n"
        )

        # --- User prompt: Layer 1 (outline spec) ---
        user_parts: List[str] = []

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

        # --- User prompt: Layer 2 (primary inputs, full) ---

        # RQ context (full)
        user_parts.append(f"## Research Question")
        user_parts.append(f"Title: {rq.get('title', '')}")
        if rq.get("background"):
            user_parts.append(f"Background: {rq['background']}")
        if rq.get("gap"):
            user_parts.append(f"Gap: {rq['gap']}")
        if rq.get("approach"):
            user_parts.append(f"Approach: {rq['approach']}")
        user_parts.append("")

        # Lit review summary (full executive summary + streams)
        user_parts.append("## Literature Review Summary")
        if lr.get("executive_summary"):
            user_parts.append(lr["executive_summary"])
        user_parts.append("")

        # Theoretical streams (full descriptions)
        streams = lr.get("theoretical_streams", [])
        if streams:
            user_parts.append(f"## Theoretical Streams ({len(streams)})")
            for stream in streams:
                name = stream.get("name", "")
                desc = stream.get("description", "")
                user_parts.append(f"### {name}")
                user_parts.append(desc)
            user_parts.append("")

        # Open questions / research gaps
        oqs = lr.get("open_questions", [])
        if oqs:
            user_parts.append(f"## Research Gaps ({len(oqs)})")
            for q in oqs:
                desc = q.get("description", q.get("question", ""))
                user_parts.append(f"- {desc}")
            user_parts.append("")

        # Key empirical findings (for citing established work)
        findings = lr.get("empirical_findings", {})
        established = findings.get("established", [])
        if established:
            user_parts.append(f"## Established Findings ({len(established)})")
            for f in established[:8]:
                finding_text = f.get("finding", f.get("description", ""))
                user_parts.append(f"- {finding_text[:300]}")
            user_parts.append("")

        # Hypotheses (statements only, for contribution preview)
        hypotheses = hyp.get("hypotheses", [])
        if hypotheses:
            user_parts.append(f"## Hypotheses ({len(hypotheses)}) — for contribution preview")
            for i, h in enumerate(hypotheses):
                stmt = h.get("hypothesis_statement", "")
                user_parts.append(f"H{i+1}: {stmt}")
            user_parts.append("")

        # --- User prompt: Layer 3 (cross-references, compact) ---
        if ctx.cross_refs:
            user_parts.append(ctx.cross_refs)
            user_parts.append("")

        # --- Instruction ---
        user_parts.append("## 指示")
        user_parts.append(
            "上記の情報をもとに、学術論文の Introduction セクションを Markdown で書いてください。\n"
            "5段構成 (背景 → 問題設定 → ギャップ → 本研究の目的 → 論文構成) を厳守してください。\n"
            "先行研究は (著者名, 年) 形式で具体的に引用してください。\n"
            "最後の段落で、次の Literature Review セクションへの自然な接続を書いてください。\n"
            f"目標語数: 約{target_words}語。"
        )

        return system, "\n".join(user_parts)

    def validate_content(self, text: str, outline_spec: Dict[str, Any]) -> List[str]:
        """Introduction-specific quality checks."""
        warnings: List[str] = []
        text_lower = text.lower()

        # Check for RQ mention
        if not any(marker in text_lower for marker in ["研究", "リサーチ", "research", "問い", "rq", "question"]):
            warnings.append("No research question reference detected")

        # Check for gap/limitation language
        gap_markers = ["ギャップ", "限界", "不足", "未解明", "gap", "limitation", "insufficient", "未検討"]
        if not any(m in text_lower for m in gap_markers):
            warnings.append("No research gap language detected")

        # Check for section structure markers (should have subsections or clear paragraphs)
        paragraph_count = len([p for p in text.split("\n\n") if p.strip()])
        if paragraph_count < 3:
            warnings.append(f"Only {paragraph_count} paragraphs — expected at least 4-5 for Introduction")

        # Check for bridge to next section
        last_500 = text[-500:].lower() if len(text) > 500 else text.lower()
        bridge_markers = ["文献", "literature", "レビュー", "review", "以下", "構成", "本稿", "次節", "第"]
        if not any(m in last_500 for m in bridge_markers):
            warnings.append("No bridge to Literature Review detected in final paragraph")

        return warnings
