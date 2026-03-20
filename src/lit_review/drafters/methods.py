# src/lit_review/drafters/methods.py
"""097 Methods Drafter — service logic.

Generates draft_methods.md following the outline spec from 094.
Produces a research methodology section that maps each hypothesis
to its verification design, data, and estimation strategy.

Structure:
  1. Research design overview
  2. Data sources and sample construction
  3. Variable definitions
  4. Estimation strategy (per hypothesis / design)
  5. Robustness and sensitivity analysis
  6. Bridge to results

Usage::

    from src.lit_review.drafters.methods import MethodsDrafter

    drafter = MethodsDrafter()
    result = drafter.generate(run_dir, llm_client=client)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from src.lit_review.drafters.base import BaseDrafter, PromptContext


class MethodsDrafter(BaseDrafter):
    """Generates the Research Methodology section of the paper."""

    section_id = "methods"
    output_file = "draft_methods.md"
    default_max_tokens = 16384

    def required_inputs(self) -> List[str]:
        return [
            "paper_outline.json",
            "hypotheses.json",
            "validation_designs.json",
            "data_requirements.json",
            "method_selection.json",
        ]

    def build_prompt(self, ctx: PromptContext) -> Tuple[str, str]:
        spec = ctx.outline_spec
        hyp = ctx.require("hypotheses.json")
        vd = ctx.require("validation_designs.json")
        dr = ctx.require("data_requirements.json")
        ms = ctx.require("method_selection.json")

        hypotheses = hyp.get("hypotheses", [])
        designs = vd.get("validation_designs", [])
        data_plans = dr.get("data_plans", [])
        selections = ms.get("method_selections", [])
        target_words = spec.get("target_words", 4000)

        # Build hypothesis id → H-label map
        h_labels: Dict[str, str] = {}
        for i, h in enumerate(hypotheses):
            h_labels[h.get("hypothesis_id", "")] = f"H{i + 1}"

        # --- System prompt ---
        system = (
            "あなたは社会科学分野の学術論文の Research Methodology セクションの執筆者です。\n"
            "以下のルールに厳密に従ってください:\n\n"
            "1. 日本語で執筆する\n"
            "2. 学術論文の Methods として適切な文体を用いる\n"
            "3. 各推定手法の原典を (著者名, 年) 形式で引用する (例: DID → Callaway & Sant'Anna (2021), GMM → Arellano & Bond (1991))\n"
            f"4. 目標語数: 約{target_words}語。冗長な繰り返しを避け、簡潔かつ正確に書く\n"
            "5. 以下の構成で書く:\n"
            "   (a) 研究デザインの全体像 — 各仮説にどの検証戦略を対応させるか\n"
            "   (b) データソースとサンプル構築\n"
            "   (c) 変数の操作化定義 (従属・独立・統制)\n"
            "   (d) 推定戦略 — 仮説ごとの識別戦略・推定式を含む\n"
            "   (e) 頑健性確認・感度分析の計画\n"
            "   (f) 分析の限界への言及\n"
            "6. 推定式は数式で示す (Markdown の ``` ブロックか行内で)\n"
            "7. 各検証設計が「どの仮説 (H1–H9) を検証するか」を明示する\n"
            "8. Markdown 形式で出力する (# Research Methodology から始める)\n"
            "9. セクション末尾で Results に自然に接続する文を含める\n"
            "10. 冗長化を避ける。方法の一般論ではなく、この研究固有の設計を書く\n"
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
            user_parts.append("")

        # Layer 2a: Hypotheses — use focused when available
        from src.lit_review.focus import is_focused
        focused = ctx.input("focused_hypotheses.json")
        if is_focused(focused):
            user_parts.append("## Hypotheses to verify (focused)")
            primary = focused["primary"]
            user_parts.append(f"H1 (PRIMARY): {primary.get('hypothesis_statement', '')}")
            if focused.get("has_secondary") and focused.get("secondary"):
                secondary = focused["secondary"]
                user_parts.append(f"H2 (SECONDARY): {secondary.get('hypothesis_statement', '')}")
            user_parts.append("NOTE: Methods section should cover H1/H2 only.")
        else:
            user_parts.append(f"## Hypotheses to verify ({len(hypotheses)})")
            for i, h in enumerate(hypotheses):
                user_parts.append(f"H{i + 1}: {h.get('hypothesis_statement', '')}")
        user_parts.append("")

        # Layer 2b: Validation designs (ALL, full detail)
        user_parts.append(f"## Validation Designs ({len(designs)})")
        for d in designs:
            hid = d.get("hypothesis_id", "")
            label = h_labels.get(hid, hid[:16])
            user_parts.append(f"\n### Design for {label}")
            user_parts.append(f"Type: {d.get('design_type', '')}")
            user_parts.append(f"Identification: {d.get('identification_strategy', '')}")
            user_parts.append(f"DV: {d.get('data_requirements', {}).get('dependent_variable', '')}")
            if d.get("design_description"):
                user_parts.append(f"Description: {d['design_description']}")
            if d.get("identification_rationale"):
                user_parts.append(f"Rationale: {d['identification_rationale']}")
            assumptions = d.get("required_assumptions", [])
            if assumptions:
                user_parts.append(f"Required assumptions:")
                for a in assumptions[:4]:
                    user_parts.append(f"  - {a[:200]}")
            sources = d.get("data_requirements", {}).get("data_sources", [])
            if sources:
                user_parts.append(f"Data sources: {', '.join(sources[:5])}")
        user_parts.append("")

        # Layer 2c: Method selections (ALL, full detail)
        user_parts.append(f"## Method Selections ({len(selections)})")
        for s in selections:
            hid = s.get("hypothesis_id", "")
            label = h_labels.get(hid, hid[:16])
            user_parts.append(f"\n### Methods for {label}")
            user_parts.append(f"Primary: {s.get('primary_method', '')}")
            if s.get("primary_rationale"):
                user_parts.append(f"Primary rationale: {s['primary_rationale'][:300]}")
            user_parts.append(f"Secondary: {s.get('secondary_method', '')}")
            if s.get("secondary_rationale"):
                user_parts.append(f"Secondary rationale: {s['secondary_rationale'][:300]}")
        user_parts.append("")

        # Layer 2d: Data requirements (ALL variables)
        user_parts.append(f"## Data Requirements ({len(data_plans)} plans)")
        total_vars = 0
        for p in data_plans:
            hid = p.get("hypothesis_id", "")
            label = h_labels.get(hid, hid[:16])
            variables = p.get("variables", [])
            total_vars += len(variables)
            user_parts.append(f"\n### Data for {label} ({len(variables)} variables)")
            user_parts.append(f"Identification: {p.get('identification_strategy', '')}")
            user_parts.append(f"Feasibility: {p.get('overall_feasibility', '')}")
            for v in variables:
                name = v.get("name", "")
                defn = v.get("definition", v.get("description", ""))
                role = v.get("role", v.get("type", ""))
                if name:
                    user_parts.append(f"  - {name} [{role}]: {defn[:150]}")
            gaps = p.get("critical_data_gaps", [])
            if gaps:
                user_parts.append(f"Critical gaps: {', '.join(str(g)[:100] for g in gaps[:3])}")
        user_parts.append(f"\nTotal variables: {total_vars}")
        user_parts.append("")

        # Deep literature methods/variables (from 118, if available)
        if ctx.deep_lit:
            for hyp_id, dl in ctx.deep_lit.items():
                mm = dl.get("method_map")
                vm = dl.get("variable_map")
                if mm and mm.get("methods"):
                    user_parts.append(f"## Field Methods (deep lit, {hyp_id[:20]})")
                    for m in mm["methods"][:6]:
                        user_parts.append(f"- {m.get('name', '')} ({m.get('paper_count', 0)} papers)")
                    user_parts.append("")
                if vm and vm.get("variables"):
                    user_parts.append(f"## Field Variables (deep lit)")
                    for vt in ["dependent", "independent"]:
                        vars_list = vm["variables"].get(vt, [])
                        if vars_list:
                            top = [v.get("name", "") for v in vars_list[:4]]
                            user_parts.append(f"- {vt}: {', '.join(top)}")
                    user_parts.append("")

        # Layer 3: Cross-references (compact)
        if ctx.cross_refs:
            user_parts.append(ctx.cross_refs)
            user_parts.append("")

        # Instruction
        user_parts.append("## 指示")
        user_parts.append(
            "上記の情報をもとに、Research Methodology セクションを Markdown で書いてください。\n"
            "各仮説 (H1–H9) がどの検証戦略で検証されるかを明示してください。\n"
            "推定式を含めてください。\n"
            "データソース・サンプル構築・変数定義を具体的に記述してください。\n"
            "頑健性確認（robustness checks）の計画を含めてください。\n"
            "最後の段落で Results セクションへの自然な接続を書いてください。\n"
            f"目標語数: 約{target_words}語。冗長化を避けてください。"
        )

        return system, "\n".join(user_parts)

    def validate_content(self, text: str, outline_spec: Dict[str, Any]) -> List[str]:
        """Methods-specific quality checks."""
        warnings: List[str] = []
        text_lower = text.lower()

        # Check for data/sample description
        data_markers = ["データ", "data", "サンプル", "sample", "観測", "observation", "パネル", "panel"]
        if not any(m in text_lower for m in data_markers):
            warnings.append("No data/sample description detected")

        # Check for variable definitions
        var_markers = ["変数", "variable", "従属", "dependent", "独立", "independent", "統制", "control"]
        if not any(m in text_lower for m in var_markers):
            warnings.append("No variable definitions detected")

        # Check for identification/estimation strategy
        ident_markers = [
            "識別", "identification", "推定", "estimat", "did", "差分", "difference",
            "gmm", "iv", "操作変数", "instrumental", "マッチング", "matching",
        ]
        if not any(m in text_lower for m in ident_markers):
            warnings.append("No identification/estimation strategy detected")

        # Check for robustness discussion
        robust_markers = ["頑健", "robust", "感度", "sensitiv", "代替", "alternative", "プラセボ", "placebo"]
        if not any(m in text_lower for m in robust_markers):
            warnings.append("No robustness/sensitivity analysis discussion detected")

        # Check for hypothesis reference (H1, H2, etc.)
        h_markers = re.findall(r'\bH\d+\b', text)
        if len(set(h_markers)) < 3:
            warnings.append(f"Only {len(set(h_markers))} hypothesis markers — methods should reference hypotheses")

        # Check for estimation equations
        equation_markers = ["=", "β", "α", "γ", "ε", "Σ", "_{", "^{"]
        has_equations = any(m in text for m in equation_markers)
        if not has_equations:
            warnings.append("No estimation equations detected")

        # Check for results bridge at end
        last_500 = text[-500:].lower() if len(text) > 500 else text_lower
        bridge_markers = ["result", "結果", "分析結果", "次章", "次節", "findings"]
        if not any(m in last_500 for m in bridge_markers):
            warnings.append("No bridge to Results detected in final paragraph")

        return warnings
