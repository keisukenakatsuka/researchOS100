# src/lit_review/drafters/hypotheses.py
"""096 Hypotheses Drafter — service logic.

Generates draft_hypotheses.md following the outline spec from 094.
Produces a theory-driven Hypotheses Development section that covers
ALL hypotheses with:

  - Theoretical grounding and prior work linkage
  - Formal hypothesis statement
  - Predicted direction and effect size
  - Logical relationship between hypotheses

Usage::

    from src.lit_review.drafters.hypotheses import HypothesesDrafter

    drafter = HypothesesDrafter()
    result = drafter.generate(run_dir, llm_client=client)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from src.lit_review.drafters.base import BaseDrafter, PromptContext


class HypothesesDrafter(BaseDrafter):
    """Generates the Hypotheses Development section of the paper."""

    section_id = "hypotheses"
    output_file = "draft_hypotheses.md"
    default_max_tokens = 16384

    def required_inputs(self) -> List[str]:
        return ["paper_outline.json", "hypotheses.json", "lit_review.json"]

    def build_prompt(self, ctx: PromptContext) -> Tuple[str, str]:
        spec = ctx.outline_spec
        hyp = ctx.require("hypotheses.json")
        lr = ctx.require("lit_review.json")
        asmp = ctx.input("assumptions.json")
        port = ctx.input("hypothesis_portfolio.json")

        hypotheses = hyp.get("hypotheses", [])
        target_words = spec.get("target_words", 3000)

        # Build priority map from portfolio
        priority_map: Dict[str, str] = {}
        for s in port.get("scored_hypotheses", []):
            priority_map[s.get("hypothesis_id", "")] = s.get("recommendation", "")

        # Build assumptions map
        assumptions_map: Dict[str, List[str]] = {}
        for ha in asmp.get("hypothesis_assumptions", []):
            hid = ha.get("hypothesis_id", "")
            assumptions_map[hid] = [
                a.get("assumption_text", a.get("description", ""))
                for a in ha.get("assumptions", [])
            ]

        # --- System prompt ---
        system = (
            "あなたは社会科学分野の学術論文の Hypotheses Development セクションの執筆者です。\n"
            "以下のルールに厳密に従ってください:\n\n"
            "1. 日本語で執筆する\n"
            "2. 学術論文の Hypotheses セクションとして適切な文体を用いる\n"
            "3. 先行研究に言及する際は必ず (著者名, 年) 形式で引用する\n"
            f"4. 目標語数: 約{target_words}語。冗長な繰り返しや過度な修飾を避け、簡潔かつ正確に書く\n"
            "5. 各仮説は以下の構成で書く:\n"
            "   (a) 理論的背景 — どの理論・先行研究から導かれるか\n"
            "   (b) 論理的根拠 — なぜその方向性を予測するか\n"
            "   (c) 正式な仮説文 — 太字で明示 (変数間関係が読めること)\n"
            "   (d) 効果サイズの予測 — 数量的な期待を含む\n"
            f"6. 全 {len(hypotheses)} 仮説を漏れなく扱う (省略禁止)\n"
            "7. 仮説同士の関係性を明示する (主仮説/補助仮説、相互依存 等)\n"
            "8. Markdown 形式で出力する (# Hypotheses Development から始める)\n"
            "9. セクション末尾で Methods に自然に接続する文を含める\n"
            "10. 冗長化を避ける。各仮説の記述は本質に絞り、繰り返しを排除する\n"
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

        # Layer 2: Hypotheses (ALL, full detail, no truncation)
        user_parts.append(f"## Hypotheses ({len(hypotheses)}) — MUST cover ALL")
        high_priority = []
        other = []
        for i, h in enumerate(hypotheses):
            hid = h.get("hypothesis_id", "")
            rec = priority_map.get(hid, "")
            if rec == "high_priority":
                high_priority.append((i, h, hid, rec))
            else:
                other.append((i, h, hid, rec))

        # Present high-priority first
        if high_priority:
            user_parts.append("\n### High Priority Hypotheses")
        for idx, h, hid, rec in high_priority:
            user_parts.append(self._format_hypothesis(idx, h, hid, rec, assumptions_map))

        if other:
            user_parts.append("\n### Other Hypotheses")
        for idx, h, hid, rec in other:
            user_parts.append(self._format_hypothesis(idx, h, hid, rec, assumptions_map))

        user_parts.append("")

        # Layer 2: Theoretical streams (for citation grounding)
        streams = lr.get("theoretical_streams", [])
        if streams:
            user_parts.append(f"## Theoretical Streams ({len(streams)})")
            for stream in streams:
                name = stream.get("name", "")
                desc = stream.get("description", "")
                user_parts.append(f"### {name}")
                user_parts.append(desc)
            user_parts.append("")

        # Established findings (for supporting evidence)
        findings = lr.get("empirical_findings", {})
        established = findings.get("established", [])
        if established:
            user_parts.append(f"## Established Findings ({len(established)})")
            for f in established[:6]:
                user_parts.append(f"- {f.get('finding', f.get('description', ''))[:250]}")
            user_parts.append("")

        # Layer 3: Cross-references (compact)
        if ctx.cross_refs:
            user_parts.append(ctx.cross_refs)
            user_parts.append("")

        # Instruction
        user_parts.append("## 指示")
        user_parts.append(
            "上記の情報をもとに、Hypotheses Development セクションを Markdown で書いてください。\n"
            f"全 {len(hypotheses)} 仮説を漏れなく扱ってください (H1–H{len(hypotheses)})。\n"
            "High priority 仮説はより詳細に記述し、それ以外も理論的根拠を明示してください。\n"
            "仮説同士の関係性（主仮説/補助仮説、理論的つながり）を示してください。\n"
            "各仮説の正式文は **太字** で明示してください。\n"
            "最後の段落で Methods セクションへの自然な接続を書いてください。\n"
            f"目標語数: 約{target_words}語。冗長化を避けてください。"
        )

        return system, "\n".join(user_parts)

    def validate_content(self, text: str, outline_spec: Dict[str, Any]) -> List[str]:
        """Hypotheses-specific quality checks."""
        warnings: List[str] = []
        text_lower = text.lower()

        # Count hypothesis markers (H1, H2, ... or 仮説1, 仮説2, ...)
        h_markers = re.findall(r'\bH\d+\b', text)
        jp_markers = re.findall(r'仮説\s*\d+', text)
        markers = set(h_markers) | set(jp_markers)

        if len(markers) < 5:
            warnings.append(
                f"Only {len(markers)} hypothesis markers found — "
                f"expected at least 9 (H1–H9 or 仮説1–仮説9)"
            )

        # Check for bold hypothesis statements
        bold_count = len(re.findall(r'\*\*[^*]{20,}\*\*', text))
        if bold_count < 3:
            warnings.append(f"Only {bold_count} bold statements — expected formal hypothesis statements in bold")

        # Check for theoretical grounding language
        theory_markers = ["理論", "theory", "理論的", "theoretical", "先行研究", "既存研究"]
        if not any(m in text_lower for m in theory_markers):
            warnings.append("No theoretical grounding language detected")

        # Check for methods bridge at end
        last_500 = text[-500:].lower() if len(text) > 500 else text_lower
        bridge_markers = ["method", "手法", "検証", "実証", "分析", "methods", "次章", "次節"]
        if not any(m in last_500 for m in bridge_markers):
            warnings.append("No bridge to Methods detected in final paragraph")

        return warnings

    @staticmethod
    def _format_hypothesis(
        idx: int,
        h: Dict[str, Any],
        hid: str,
        recommendation: str,
        assumptions_map: Dict[str, List[str]],
    ) -> str:
        """Format a single hypothesis for the prompt."""
        parts = [f"\nH{idx + 1} [{h.get('strategy', '')}] (priority: {recommendation or 'unscored'})"]
        parts.append(f"Statement: {h.get('hypothesis_statement', '')}")
        parts.append(f"Rationale: {h.get('rationale', '')}")
        parts.append(f"Testability: {h.get('testability', '')}")

        if h.get("suggested_test"):
            parts.append(f"Suggested test: {h['suggested_test'][:200]}")

        if h.get("novelty_rationale"):
            parts.append(f"Novelty: {h['novelty_rationale'][:200]}")

        # Include assumptions if available
        assumptions = assumptions_map.get(hid, [])
        if assumptions:
            parts.append(f"Key assumptions ({len(assumptions)}):")
            for a in assumptions[:3]:
                parts.append(f"  - {a[:150]}")

        return "\n".join(parts)
