# src/question/generator.py
"""101 RQ Generator — service logic.

Extracts "RQ seeds" from a completed research run's outputs and
uses LLM to synthesize them into well-formed Research Question candidates.

Seed sources:
  - open_questions from lit_review.json        → gap_driven
  - contested findings from lit_review.json    → resolution_driven
  - research_opportunities from landscape.json → opportunity_driven
  - blindspots from landscape.json             → gap_driven
  - hypotheses source_gaps                     → deepening

Usage::

    from src.question.generator import generate_rq_candidates

    result = generate_rq_candidates(run_dir, llm_client=client)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 8192


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class RQSeed:
    """Raw seed extracted from upstream artifacts."""
    source_type: str          # gap_driven | resolution_driven | opportunity_driven | deepening
    source_artifact: str      # e.g., "lit_review.json:open_questions[0]"
    description: str          # the seed text
    context: str = ""         # additional context (why_unresolved, approach, etc.)
    parent_hypothesis: str = ""  # H1, H2, etc. if applicable


@dataclass
class RQCandidate:
    """A generated Research Question candidate."""
    candidate_id: str
    title: str
    question: str             # full RQ statement
    background: str
    gap: str
    approach: str
    keywords: List[str] = field(default_factory=list)
    source_type: str = ""
    derived_from: str = ""    # which artifact / finding
    rationale: str = ""       # why this is a valuable RQ
    parent_hypothesis: str = ""
    parent_run_id: str = ""
    status: str = "candidate"
    duplicate_flag: bool = False
    duplicate_of: Optional[str] = None


@dataclass
class GeneratorResult:
    status: str = "failed"
    parent_run_id: str = ""
    parent_rq_title: str = ""
    candidates: List[RQCandidate] = field(default_factory=list)
    seeds_extracted: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_run_id": self.parent_run_id,
            "parent_rq_title": self.parent_rq_title,
            "generated_at": self.metadata.get("generated_at", ""),
            "seeds_extracted": self.seeds_extracted,
            "candidates_generated": len(self.candidates),
            "candidates": [asdict(c) for c in self.candidates],
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------
# Seed extraction (deterministic, no LLM)
# ------------------------------------------------------------------

def _extract_seeds(run_dir: Path) -> tuple[List[RQSeed], Dict[str, Any]]:
    """Extract RQ seeds from all available upstream artifacts."""
    seeds: List[RQSeed] = []
    context_data: Dict[str, Any] = {}

    # 1. open_questions from lit_review.json
    lr_path = run_dir / "lit_review.json"
    if lr_path.exists():
        lr = json.loads(lr_path.read_text())
        context_data["lit_review"] = lr
        for i, q in enumerate(lr.get("open_questions", [])):
            desc = q.get("description", q.get("question", ""))
            ctx = q.get("why_unresolved", "")
            approach = q.get("potential_approach", "")
            seeds.append(RQSeed(
                source_type="gap_driven",
                source_artifact=f"lit_review.json:open_questions[{i}]",
                description=desc,
                context=f"Why unresolved: {ctx}. Potential approach: {approach}" if ctx else "",
            ))

        # 2. contested findings
        for i, c in enumerate(lr.get("empirical_findings", {}).get("contested", [])):
            topic = c.get("topic", "")
            disagreement = c.get("nature_of_disagreement", "")
            seeds.append(RQSeed(
                source_type="resolution_driven",
                source_artifact=f"lit_review.json:contested[{i}]",
                description=f"Contested: {topic}",
                context=f"Disagreement: {disagreement}",
            ))

    # 3. blindspots + research_opportunities from landscape.json
    ls_path = run_dir / "landscape.json"
    if ls_path.exists():
        ls = json.loads(ls_path.read_text())
        context_data["landscape"] = ls
        for i, b in enumerate(ls.get("blindspots", [])):
            if isinstance(b, dict):
                area = b.get("area", "")
                missing = b.get("what_is_missing", "")
                seeds.append(RQSeed(
                    source_type="gap_driven",
                    source_artifact=f"landscape.json:blindspots[{i}]",
                    description=f"Blindspot: {area}",
                    context=f"Missing: {missing}",
                ))

        for i, o in enumerate(ls.get("research_opportunities", [])):
            if isinstance(o, dict):
                theme = o.get("theme", "")
                method = o.get("method", "")
                theory = o.get("theory", "")
                seeds.append(RQSeed(
                    source_type="opportunity_driven",
                    source_artifact=f"landscape.json:research_opportunities[{i}]",
                    description=f"Opportunity: {theme}",
                    context=f"Theory: {theory}. Method: {method}",
                ))

    # 4. hypotheses source_gaps
    hyp_path = run_dir / "hypotheses.json"
    if hyp_path.exists():
        hyp = json.loads(hyp_path.read_text())
        context_data["hypotheses"] = hyp
        for i, h in enumerate(hyp.get("hypotheses", [])):
            for g in h.get("source_gaps", []):
                gap_text = g if isinstance(g, str) else str(g)
                seeds.append(RQSeed(
                    source_type="deepening",
                    source_artifact=f"hypotheses.json:hypotheses[{i}].source_gaps",
                    description=f"Deepening: {gap_text}",
                    parent_hypothesis=f"H{i + 1}",
                ))

    # Load parent RQ context
    rq_path = run_dir / "rq_context.json"
    if rq_path.exists():
        context_data["rq_context"] = json.loads(rq_path.read_text())

    return seeds, context_data


def _deduplicate_seeds(seeds: List[RQSeed]) -> List[RQSeed]:
    """Remove near-duplicate seeds (by description similarity)."""
    seen_descriptions: List[str] = []
    unique: List[RQSeed] = []
    for seed in seeds:
        desc_norm = seed.description.lower().strip()
        # Simple check: if first 30 chars match any existing, skip
        is_dup = any(desc_norm[:30] == s[:30] for s in seen_descriptions)
        if not is_dup:
            unique.append(seed)
            seen_descriptions.append(desc_norm)
    return unique


# ------------------------------------------------------------------
# LLM-based RQ synthesis
# ------------------------------------------------------------------

def _generate_id(text: str) -> str:
    """Generate a short stable ID from text."""
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    return f"rqc_{h}"


def _parse_json_from_response(text: str) -> Optional[List[Dict]]:
    """Extract JSON array from LLM response."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "candidates" in result:
            return result["candidates"]
        return None
    except json.JSONDecodeError:
        return None


def _synthesize_candidates(
    seeds: List[RQSeed],
    parent_rq_title: str,
    parent_run_id: str,
    llm_client: Any,
) -> List[RQCandidate]:
    """Use LLM to synthesize seeds into well-formed RQ candidates."""
    # Build seed summary for prompt
    seed_text_parts: List[str] = []
    for i, s in enumerate(seeds):
        parts = [f"Seed {i + 1} [{s.source_type}] ({s.source_artifact})"]
        parts.append(f"  Description: {s.description}")
        if s.context:
            parts.append(f"  Context: {s.context}")
        if s.parent_hypothesis:
            parts.append(f"  Parent hypothesis: {s.parent_hypothesis}")
        seed_text_parts.append("\n".join(parts))

    seed_text = "\n\n".join(seed_text_parts)

    system = (
        "あなたは研究戦略の専門家です。\n"
        "完了した研究から抽出された「研究の種」をもとに、\n"
        "次に取り組むべき新しいResearch Question（RQ）候補を生成してください。\n\n"
        "各RQ候補は以下を満たすこと:\n"
        "- 具体的で検証可能な問い\n"
        "- 元の研究との差別化が明確\n"
        "- 背景・ギャップ・アプローチが記述されている\n"
        "- 出力は JSON 配列のみ"
    )

    user = (
        f"## 親RQ\n{parent_rq_title}\n\n"
        f"## 研究の種 ({len(seeds)} seeds)\n{seed_text}\n\n"
        f"## 指示\n"
        f"上記の種をもとに、次の研究として取り組むべき RQ 候補を JSON 配列で生成してください。\n"
        f"類似する種は統合して 1 つの RQ にまとめてください。\n"
        f"最終的に 3–7 個の RQ 候補を生成してください。\n\n"
        f"各候補の形式:\n"
        f"```json\n"
        f"[\n"
        f"  {{\n"
        f'    "title": "RQの短いタイトル",\n'
        f'    "question": "完全なResearch Question文",\n'
        f'    "background": "この問いが重要な背景",\n'
        f'    "gap": "既存研究で不足している点",\n'
        f'    "approach": "提案するアプローチ",\n'
        f'    "keywords": ["キーワード1", "キーワード2"],\n'
        f'    "source_type": "gap_driven|resolution_driven|opportunity_driven|deepening",\n'
        f'    "derived_from": "どの種から派生したか",\n'
        f'    "rationale": "なぜこのRQが価値があるか",\n'
        f'    "parent_hypothesis": "H1等（該当する場合）"\n'
        f"  }}\n"
        f"]\n"
        f"```\n"
        f"JSON 配列のみ出力。説明文は不要。"
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
        logger.error("LLM call failed: %s", e)
        return []

    text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            text = block.get("text", "")
            break

    usage = resp.get("usage", {})
    logger.info("LLM: in=%d out=%d tokens", usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    raw_candidates = _parse_json_from_response(text)
    if not raw_candidates:
        logger.error("Failed to parse LLM response as JSON")
        return []

    # Convert to RQCandidate
    candidates: List[RQCandidate] = []
    for raw in raw_candidates:
        title = raw.get("title", "")
        cid = _generate_id(f"{parent_run_id}:{title}")
        candidates.append(RQCandidate(
            candidate_id=cid,
            title=title,
            question=raw.get("question", title),
            background=raw.get("background", ""),
            gap=raw.get("gap", ""),
            approach=raw.get("approach", ""),
            keywords=raw.get("keywords", []),
            source_type=raw.get("source_type", ""),
            derived_from=raw.get("derived_from", ""),
            rationale=raw.get("rationale", ""),
            parent_hypothesis=raw.get("parent_hypothesis", ""),
            parent_run_id=parent_run_id,
            status="candidate",
        ))

    return candidates


# ------------------------------------------------------------------
# Markdown rendering
# ------------------------------------------------------------------

def _render_markdown(result: GeneratorResult) -> str:
    parts: List[str] = []
    parts.append("# RQ Candidates\n")
    parts.append(f"**Parent RQ**: {result.parent_rq_title}")
    parts.append(f"**Parent Run**: {result.parent_run_id}")
    parts.append(f"**Seeds extracted**: {result.seeds_extracted}")
    parts.append(f"**Candidates generated**: {len(result.candidates)}\n")

    for i, c in enumerate(result.candidates, 1):
        parts.append(f"## {i}. {c.title}")
        parts.append(f"\n**Question**: {c.question}")
        parts.append(f"\n**Background**: {c.background}")
        parts.append(f"\n**Gap**: {c.gap}")
        parts.append(f"\n**Approach**: {c.approach}")
        parts.append(f"\n**Source**: {c.source_type} ({c.derived_from})")
        if c.parent_hypothesis:
            parts.append(f"**Parent hypothesis**: {c.parent_hypothesis}")
        parts.append(f"**Rationale**: {c.rationale}")
        parts.append(f"**Keywords**: {', '.join(c.keywords)}")
        parts.append(f"**ID**: `{c.candidate_id}`")
        parts.append("")

    return "\n".join(parts) + "\n"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_rq_candidates(
    run_dir: Path,
    *,
    llm_client: Any,
) -> GeneratorResult:
    """Generate RQ candidates from a completed research run."""
    result = GeneratorResult()
    now_iso = datetime.now(timezone.utc).isoformat()
    parent_run_id = run_dir.name

    try:
        # 1. Extract seeds
        seeds, context_data = _extract_seeds(run_dir)
        seeds = _deduplicate_seeds(seeds)
        result.seeds_extracted = len(seeds)
        result.parent_run_id = parent_run_id

        rq_ctx = context_data.get("rq_context", {})
        parent_rq_title = rq_ctx.get("title", "")
        result.parent_rq_title = parent_rq_title

        if not seeds:
            result.error = "No RQ seeds found in run outputs"
            return result

        logger.info("Extracted %d seeds from run %s", len(seeds), parent_run_id)

        # 2. Synthesize candidates via LLM
        candidates = _synthesize_candidates(seeds, parent_rq_title, parent_run_id, llm_client)

        if not candidates:
            result.error = "LLM failed to generate candidates"
            return result

        result.candidates = candidates
        result.status = "generated"
        result.metadata = {
            "generated_at": now_iso,
            "model": _MODEL,
            "seeds_by_type": {
                "gap_driven": sum(1 for s in seeds if s.source_type == "gap_driven"),
                "resolution_driven": sum(1 for s in seeds if s.source_type == "resolution_driven"),
                "opportunity_driven": sum(1 for s in seeds if s.source_type == "opportunity_driven"),
                "deepening": sum(1 for s in seeds if s.source_type == "deepening"),
            },
        }

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.error("101: %s", e)

    return result
