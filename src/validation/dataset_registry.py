# src/validation/dataset_registry.py
"""Dataset Registry Builder (111 core logic).

Extracts dataset mentions from Evidence items, merges seed info from
091 data_requirements, and assesses availability via LLM.

See design.md Section 3.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.validation.ids import dataset_id

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Alternative:
    name: str = ""
    availability_status: str = ""
    cost_tier: str = ""
    coverage_comparison: str = ""
    suitability: str = ""  # full | partial | minimal
    access_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Alternative:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetEntry:
    dataset_id: str = ""
    name: str = ""
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    provider: str = ""
    availability_status: str = ""  # open | restricted | commercial | unavailable
    access_url: str = ""
    cost_tier: str = ""  # free | low | medium | high
    cost_estimate: str = ""
    coverage: Dict[str, str] = field(default_factory=dict)
    update_frequency: str = ""
    data_format: str = ""
    mentioned_in_papers: List[Dict[str, str]] = field(default_factory=list)
    used_by_hypotheses: List[str] = field(default_factory=list)
    alternatives: List[Alternative] = field(default_factory=list)
    acquisition_notes: str = ""
    api_available: bool = False
    bulk_download: bool = False
    scraping_feasibility: str = ""  # recommended | possible | not_recommended
    seeded_from_091: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["alternatives"] = [a.to_dict() for a in self.alternatives]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetEntry:
        alts = [Alternative.from_dict(a) for a in data.pop("alternatives", [])]
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        entry = cls(**fields)
        entry.alternatives = alts
        return entry


@dataclass
class DatasetRegistry:
    run_id: str = ""
    created_at: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    datasets: List[DatasetEntry] = field(default_factory=list)
    open_data_shortcuts: List[Dict[str, Any]] = field(default_factory=list)

    def compute_summary(self) -> None:
        total = len(self.datasets)
        self.summary = {
            "total_datasets": total,
            "open": sum(1 for d in self.datasets if d.availability_status == "open"),
            "restricted": sum(1 for d in self.datasets if d.availability_status == "restricted"),
            "commercial": sum(1 for d in self.datasets if d.availability_status == "commercial"),
            "unavailable": sum(1 for d in self.datasets if d.availability_status == "unavailable"),
        }

    def to_dict(self) -> Dict[str, Any]:
        self.compute_summary()
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "summary": self.summary,
            "datasets": [d.to_dict() for d in self.datasets],
            "open_data_shortcuts": self.open_data_shortcuts,
        }

    def to_markdown(self) -> str:
        self.compute_summary()
        s = self.summary
        lines = [
            f"# Dataset Registry",
            f"Run: {self.run_id} | Created: {self.created_at[:10]}",
            f"",
            f"## Summary",
            f"| Status | Count |",
            f"|--------|-------|",
            f"| Open | {s['open']} |",
            f"| Restricted | {s['restricted']} |",
            f"| Commercial | {s['commercial']} |",
            f"| Unavailable | {s['unavailable']} |",
            f"| **Total** | **{s['total_datasets']}** |",
            f"",
        ]

        # Open data shortcuts
        open_ds = [d for d in self.datasets if d.availability_status == "open"]
        if open_ds:
            lines.append("## Open Data (immediately available)")
            lines.append("")
            for d in open_ds:
                lines.append(f"### {d.name}")
                lines.append(f"- **ID**: `{d.dataset_id}`")
                lines.append(f"- **Provider**: {d.provider}")
                if d.access_url:
                    lines.append(f"- **URL**: {d.access_url}")
                if d.description:
                    lines.append(f"- **Description**: {d.description}")
                if d.coverage:
                    cov = ", ".join(f"{k}: {v}" for k, v in d.coverage.items())
                    lines.append(f"- **Coverage**: {cov}")
                lines.append(f"- **API**: {'Yes' if d.api_available else 'No'} | **Bulk download**: {'Yes' if d.bulk_download else 'No'}")
                lines.append("")

        # Restricted
        restricted = [d for d in self.datasets if d.availability_status == "restricted"]
        if restricted:
            lines.append("## Restricted (registration/application required)")
            lines.append("")
            for d in restricted:
                lines.append(f"### {d.name}")
                lines.append(f"- **ID**: `{d.dataset_id}`")
                lines.append(f"- **Provider**: {d.provider}")
                lines.append(f"- **Cost**: {d.cost_tier} ({d.cost_estimate})" if d.cost_estimate else f"- **Cost**: {d.cost_tier}")
                if d.acquisition_notes:
                    lines.append(f"- **Notes**: {d.acquisition_notes}")
                lines.append("")

        # Commercial
        commercial = [d for d in self.datasets if d.availability_status == "commercial"]
        if commercial:
            lines.append("## Commercial (paid license required)")
            lines.append("")
            for d in commercial:
                lines.append(f"### {d.name}")
                lines.append(f"- **ID**: `{d.dataset_id}`")
                lines.append(f"- **Provider**: {d.provider}")
                lines.append(f"- **Cost**: {d.cost_tier} ({d.cost_estimate})" if d.cost_estimate else f"- **Cost**: {d.cost_tier}")
                if d.acquisition_notes:
                    lines.append(f"- **Notes**: {d.acquisition_notes}")
                if d.alternatives:
                    for alt in d.alternatives:
                        lines.append(f"- **Alternative**: {alt.name} ({alt.availability_status}, suitability: {alt.suitability})")
                lines.append("")

        # Unavailable
        unavailable = [d for d in self.datasets if d.availability_status == "unavailable"]
        if unavailable:
            lines.append("## Unavailable")
            lines.append("")
            for d in unavailable:
                lines.append(f"- **{d.name}** ({d.provider}): {d.acquisition_notes or 'No access path identified'}")
            lines.append("")

        # Full table
        lines.append("## All Datasets")
        lines.append("")
        lines.append("| ID | Name | Provider | Status | Cost |")
        lines.append("|----|------|----------|--------|------|")
        for d in self.datasets:
            cost = d.cost_tier or "-"
            lines.append(f"| `{d.dataset_id}` | {d.name[:50]} | {d.provider[:30]} | {d.availability_status} | {cost} |")
        lines.append("")

        return "\n".join(lines)


# ------------------------------------------------------------------
# Step 1: Extract dataset mentions from evidence + 091 seeds
# ------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
あなたはデータセット・データソースの専門家です。
研究論文の Evidence（知見の抽出結果）を読み、言及されているデータセット・データベース・調査・統計を特定してください。

重要な指示:
- 具体的なデータセット名を正確に抽出してください（例: "Preqin Private Capital Database", "World Bank WDI"）
- 一般的な記述（"secondary data", "publicly available data"）は除外してください
- 同一データセットの異なる表記をグルーピングしてください（aliases）
- 各データセットについて、論文での使用文脈を要約してください

出力は必ず JSON 形式で返してください。"""


def extract_dataset_mentions(
    evidence_items: List[Dict[str, Any]],
    data_requirements: Optional[Dict[str, Any]],
    llm_client: Any,
) -> List[DatasetEntry]:
    """Extract dataset mentions from evidence and merge with 091 seeds."""

    # Step 1a: LLM extraction from evidence
    llm_datasets = _extract_from_evidence_llm(evidence_items, llm_client)

    # Step 1b: Merge 091 seed data
    if data_requirements:
        seed_datasets = _extract_091_seeds(data_requirements)
        llm_datasets = _merge_datasets(llm_datasets, seed_datasets)

    return llm_datasets


def _extract_from_evidence_llm(
    evidence_items: List[Dict[str, Any]],
    llm_client: Any,
) -> List[DatasetEntry]:
    """Use LLM to extract dataset mentions from evidence items."""

    # Build evidence summary for prompt (truncate to avoid token overflow)
    ev_lines = []
    for i, ev in enumerate(evidence_items):
        paper = ev.get("paper_title", "")
        claim = ev.get("claim_or_point", "")
        text = ev.get("evidence_text", "")
        dim = ev.get("dimension", "")
        ev_lines.append(f"[{i+1}] Paper: {paper}\n  Claim: {claim}\n  Evidence: {text}\n  Dimension: {dim}")

    evidence_text = "\n\n".join(ev_lines)[:50_000]

    user_msg = (
        f"## Evidence Items ({len(evidence_items)} items)\n\n"
        f"{evidence_text}\n\n"
        f"## Instructions\n"
        f"上記の Evidence items から言及されているデータセット・データベース・調査・統計を抽出してください。\n"
        f"一般的な記述ではなく、具体的な名称を持つデータソースのみを抽出してください。\n\n"
        f"以下の JSON 形式で出力してください:\n"
        f'{{"datasets": [\n'
        f'  {{\n'
        f'    "name": "データセットの正式名称",\n'
        f'    "aliases": ["略称", "別名"],\n'
        f'    "description": "このデータセットの概要（日本語で1-2文）",\n'
        f'    "provider": "提供元組織",\n'
        f'    "mentioned_in_papers": ["言及されている論文タイトル"],\n'
        f'    "usage_context": "Evidence でどのように使用されているか"\n'
        f'  }}\n'
        f']}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _EXTRACTION_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Dataset extraction LLM call failed: %s", e)
        return []

    resp_text = _extract_text(resp)
    parsed = _parse_json(resp_text)
    if not parsed or "datasets" not in parsed:
        logger.error("Dataset extraction JSON parse failed. Raw: %s", resp_text[:300])
        return []

    usage = resp.get("usage", {})
    logger.info(
        "Extracted %d dataset mentions (in=%d, out=%d tokens)",
        len(parsed["datasets"]),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )

    entries = []
    for raw in parsed["datasets"]:
        name = raw.get("name", "").strip()
        if not name:
            continue
        entries.append(DatasetEntry(
            dataset_id=dataset_id(name),
            name=name,
            aliases=raw.get("aliases", []),
            description=raw.get("description", ""),
            provider=raw.get("provider", ""),
            mentioned_in_papers=[
                {"paper_id": None, "paper_title": t}
                for t in raw.get("mentioned_in_papers", [])
            ],
        ))

    return entries


def _extract_091_seeds(data_requirements: Dict[str, Any]) -> List[DatasetEntry]:
    """Extract seed datasets from 091 data_requirements.json."""
    seeds: Dict[str, DatasetEntry] = {}

    for plan in data_requirements.get("data_plans", []):
        hyp_id = plan.get("hypothesis_id", "")
        for var in plan.get("variables", []):
            for source in _collect_sources(var):
                name = source.get("name", "").strip()
                if not name or _is_calculated(name):
                    continue

                ds_id = dataset_id(name)
                if ds_id in seeds:
                    # Merge hypothesis reference
                    if hyp_id and hyp_id not in seeds[ds_id].used_by_hypotheses:
                        seeds[ds_id].used_by_hypotheses.append(hyp_id)
                    continue

                seeds[ds_id] = DatasetEntry(
                    dataset_id=ds_id,
                    name=name,
                    provider=source.get("provider", ""),
                    availability_status=_map_difficulty(source.get("acquisition_difficulty", "")),
                    cost_tier=_map_cost(source.get("cost_estimate", "")),
                    cost_estimate=source.get("cost_estimate", ""),
                    coverage={"description": source.get("coverage", "")},
                    update_frequency=source.get("update_frequency", ""),
                    acquisition_notes=source.get("limitations", ""),
                    used_by_hypotheses=[hyp_id] if hyp_id else [],
                    seeded_from_091=True,
                )

    return list(seeds.values())


def _collect_sources(var: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect primary + alternative sources from a variable spec."""
    sources = []
    ps = var.get("primary_source")
    if ps and isinstance(ps, dict):
        sources.append(ps)
    for alt in var.get("alternative_sources", []):
        if isinstance(alt, dict):
            sources.append(alt)
    return sources


def _is_calculated(name: str) -> bool:
    """Filter out non-dataset entries like 'Calculated Variable'."""
    lower = name.lower().strip()
    # Exact matches
    if lower in ("calculated variable", "calculated", "計算により作成", "手計算"):
        return True
    # Prefix matches (e.g., "手計算（SWF投資開始年データより）")
    prefixes = ("calculated", "手計算", "計算により")
    return any(lower.startswith(p) for p in prefixes)


def _map_difficulty(difficulty: str) -> str:
    """Map 091 acquisition_difficulty to availability_status."""
    mapping = {"open": "open", "restricted": "restricted", "commercial": "commercial", "unavailable": "unavailable"}
    return mapping.get(difficulty.lower(), "")


def _map_cost(cost: str) -> str:
    """Map 091 cost_estimate to cost_tier."""
    lower = cost.lower() if cost else ""
    if not lower or "free" in lower or "無料" in lower:
        return "free"
    if "low" in lower or "低" in lower:
        return "low"
    if "high" in lower or "高" in lower:
        return "high"
    return "medium"


def _merge_datasets(
    llm_datasets: List[DatasetEntry],
    seed_datasets: List[DatasetEntry],
) -> List[DatasetEntry]:
    """Merge LLM-extracted datasets with 091 seed datasets.

    Seeds provide: availability_status, cost_tier, used_by_hypotheses, coverage.
    LLM provides: mentioned_in_papers, description.
    """
    merged: Dict[str, DatasetEntry] = {}

    # Start with seeds (they have structured metadata)
    for s in seed_datasets:
        merged[s.dataset_id] = s

    # Merge LLM-extracted datasets
    for d in llm_datasets:
        if d.dataset_id in merged:
            existing = merged[d.dataset_id]
            # LLM enriches seed with description and paper mentions
            if d.description and not existing.description:
                existing.description = d.description
            if d.aliases:
                existing.aliases = list(set(existing.aliases + d.aliases))
            if d.mentioned_in_papers:
                existing.mentioned_in_papers = d.mentioned_in_papers
        else:
            merged[d.dataset_id] = d

    return list(merged.values())


# ------------------------------------------------------------------
# Step 2+3: Availability assessment (LLM)
# ------------------------------------------------------------------

_ASSESSMENT_SYSTEM = """\
あなたはデータアクセスとデータベースの専門家です。
各データセットの入手可能性を評価し、具体的なアクセス情報を提供してください。

availability_status の基準:
- open: 無料で公開されている（World Bank, OECD Stats, arXiv 等）
- restricted: 登録・申請が必要だが無料（GEM, 政府統計の個票等）
- commercial: 有料ライセンスが必要（Preqin, PitchBook, Bloomberg 等）
- unavailable: 現時点で入手不可能（非公開データ等）

出力は必ず JSON 形式で返してください。"""


def assess_and_enrich(
    datasets: List[DatasetEntry],
    llm_client: Any,
    batch_size: int = 5,
) -> List[DatasetEntry]:
    """Assess availability and discover alternatives for datasets."""

    # Only assess datasets that don't have status from 091 seeds
    needs_assessment = [d for d in datasets if not d.availability_status]
    has_status = [d for d in datasets if d.availability_status]

    logger.info(
        "Assessment: %d need LLM evaluation, %d already have status from 091",
        len(needs_assessment), len(has_status),
    )

    # Also reassess all via LLM for enrichment (access_url, alternatives, etc.)
    all_for_assessment = datasets

    for batch_start in range(0, len(all_for_assessment), batch_size):
        batch = all_for_assessment[batch_start:batch_start + batch_size]
        _assess_batch(batch, llm_client)

    return datasets


def _assess_batch(datasets: List[DatasetEntry], llm_client: Any) -> None:
    """Assess a batch of datasets via LLM. Mutates datasets in-place."""

    ds_text = ""
    for i, d in enumerate(datasets):
        status_hint = f" (091 seed: {d.availability_status})" if d.seeded_from_091 else ""
        ds_text += (
            f"\n### Dataset {i + 1}\n"
            f"- name: {d.name}\n"
            f"- provider: {d.provider}\n"
            f"- description: {d.description}\n"
            f"- current_status: {d.availability_status or 'unknown'}{status_hint}\n"
        )

    user_msg = (
        f"## Datasets to Assess\n{ds_text}\n\n"
        f"## Instructions\n"
        f"各データセットについて以下を評価してください:\n\n"
        f'{{"assessments": [\n'
        f'  {{\n'
        f'    "dataset_index": 0,\n'
        f'    "availability_status": "open | restricted | commercial | unavailable",\n'
        f'    "access_url": "公式アクセスURL（わかる場合）",\n'
        f'    "cost_tier": "free | low | medium | high",\n'
        f'    "cost_estimate": "具体的なコスト見積もり",\n'
        f'    "data_format": "CSV / API / Web platform 等",\n'
        f'    "api_available": true,\n'
        f'    "bulk_download": true,\n'
        f'    "scraping_feasibility": "recommended | possible | not_recommended",\n'
        f'    "acquisition_notes": "アクセスに関する補足情報",\n'
        f'    "alternatives": [\n'
        f'      {{\n'
        f'        "name": "代替データセット名",\n'
        f'        "availability_status": "open | restricted | commercial",\n'
        f'        "cost_tier": "free | low | medium | high",\n'
        f'        "coverage_comparison": "カバレッジの比較",\n'
        f'        "suitability": "full | partial | minimal",\n'
        f'        "access_url": "URL"\n'
        f'      }}\n'
        f'    ]\n'
        f'  }}\n'
        f']}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 4096,
        "system": _ASSESSMENT_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Assessment LLM call failed: %s", e)
        return

    resp_text = _extract_text(resp)
    parsed = _parse_json(resp_text)
    if not parsed or "assessments" not in parsed:
        logger.error("Assessment JSON parse failed. Raw: %s", resp_text[:300])
        return

    usage = resp.get("usage", {})
    logger.info(
        "Assessed %d datasets (in=%d, out=%d tokens)",
        len(parsed["assessments"]),
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )

    for assessment in parsed["assessments"]:
        idx = assessment.get("dataset_index", -1)
        if 0 <= idx < len(datasets):
            d = datasets[idx]
            d.availability_status = assessment.get("availability_status", d.availability_status or "unavailable")
            if assessment.get("access_url"):
                d.access_url = assessment["access_url"]
            if assessment.get("cost_tier"):
                d.cost_tier = assessment["cost_tier"]
            if assessment.get("cost_estimate"):
                d.cost_estimate = assessment["cost_estimate"]
            if assessment.get("data_format"):
                d.data_format = assessment["data_format"]
            d.api_available = assessment.get("api_available", d.api_available)
            d.bulk_download = assessment.get("bulk_download", d.bulk_download)
            if assessment.get("scraping_feasibility"):
                d.scraping_feasibility = assessment["scraping_feasibility"]
            if assessment.get("acquisition_notes"):
                d.acquisition_notes = assessment["acquisition_notes"]
            # Alternatives
            for alt_raw in assessment.get("alternatives", []):
                d.alternatives.append(Alternative.from_dict(alt_raw))


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
