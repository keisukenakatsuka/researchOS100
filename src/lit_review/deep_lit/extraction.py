# src/lit_review/deep_lit/extraction.py
"""118 Hypothesis Structured Extraction — service logic.

Extracts variables, methods, findings, limitations, and disagreements
from each selected paper (abstract-based), then aggregates into maps.

Usage::

    from src.lit_review.deep_lit.extraction import extract_and_map

    result = extract_and_map(ranked_result, clusters_result, hypothesis, llm_client=client)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.lit_review.deep_lit import (
    _MODEL, parse_json_response, EXTRACTION_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


_EXTRACTION_SYSTEM = """\
あなたは学術文献の構造化抽出の専門家です。
各論文の title と abstract から以下の情報を抽出してください。

抽出項目:
1. variables: dependent / independent / control / instruments に分類
2. methods: estimation strategy, data approach, identification technique
3. findings: key results (direction, effect size if available, confidence)
4. limitations: acknowledged weaknesses
5. disagreements: この論文が他の研究と矛盾・対立する点
6. hypothesis_relevance: この論文が仮説をどう支持/反証するか
7. notes_for_design: 研究設計に役立つ示唆

重要な指針:
- abstract から合理的に推論できる情報は積極的に抽出してください
- 例: "we analyze firm performance" → dependent に "firm performance" を抽出
- 例: "using panel data from 2010-2020" → data_approach に "panel data, 2010-2020" を抽出
- 例: "results show positive impact" → findings に direction: "positive" を抽出
- findings は最低1件は抽出してください。論文の主張や結論が読み取れるはずです
- 本当に何も推論できない場合のみ空にしてください"""


# ------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------

def extract_structured(
    papers: List[Dict[str, Any]],
    hypothesis: Dict[str, Any],
    cluster_assignments: Dict[str, str],
    *,
    llm_client: Any,
) -> List[Dict[str, Any]]:
    """Extract structured information from all selected papers."""
    stmt = hypothesis.get("hypothesis_statement", "")
    extractions = []
    llm_calls = 0

    for i in range(0, len(papers), EXTRACTION_BATCH_SIZE):
        batch = papers[i:i + EXTRACTION_BATCH_SIZE]
        batch_results = _extract_batch(batch, stmt, llm_client)
        llm_calls += 1

        for j, p in enumerate(batch):
            uid = p.get("paper_uid", "")
            ext = batch_results[j] if batch_results and j < len(batch_results) else {}
            extractions.append({
                "paper_uid": uid,
                "title": p.get("title", ""),
                "cluster_id": cluster_assignments.get(uid, ""),
                "variables": ext.get("variables", {"dependent": [], "independent": [], "control": [], "instruments": []}),
                "methods": ext.get("methods", {"estimation": "", "data_approach": "", "identification": ""}),
                "findings": ext.get("findings", []),
                "limitations": ext.get("limitations", []),
                "disagreements": ext.get("disagreements", []),
                "hypothesis_relevance": ext.get("hypothesis_relevance", ""),
                "notes_for_design": ext.get("notes_for_design", ""),
            })

    logger.info("Extracted structured data from %d papers in %d LLM calls",
                len(extractions), llm_calls)
    return extractions


def _extract_batch(
    batch: List[Dict[str, Any]],
    hypothesis_stmt: str,
    llm_client: Any,
) -> Optional[List[Dict]]:
    """Extract structured data from a batch of papers."""
    paper_blocks = []
    for i, p in enumerate(batch):
        title = p.get("title", "")
        abstract = (p.get("abstract", "") or "")[:500]
        year = p.get("year", "")
        paper_blocks.append(f"[P{i}] ({year}) {title}\nAbstract: {abstract}")

    user_msg = (
        f"## Hypothesis\n{hypothesis_stmt}\n\n"
        f"## Papers ({len(batch)})\n\n"
        + "\n\n".join(paper_blocks) + "\n\n"
        f"## Instructions\n"
        f"For each paper, extract structured information.\n"
        f'Output JSON: {{"extractions": [\n'
        f'  {{"paper_index": 0,\n'
        f'   "variables": {{"dependent": [...], "independent": [...], "control": [...], "instruments": [...]}},\n'
        f'   "methods": {{"estimation": "...", "data_approach": "...", "identification": "..."}},\n'
        f'   "findings": [{{"claim": "...", "direction": "positive|negative|null", "effect_size": "...", "confidence": "high|medium|low"}}],\n'
        f'   "limitations": ["..."],\n'
        f'   "disagreements": ["..."],\n'
        f'   "hypothesis_relevance": "...",\n'
        f'   "notes_for_design": "..."}}\n'
        f']}}'
    )

    body = {
        "model": _MODEL,
        "max_tokens": 8192,
        "system": _EXTRACTION_SYSTEM,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        resp = llm_client.messages_create(body=body)
    except Exception as e:
        logger.error("Extraction batch failed: %s", e)
        return None

    resp_text = ""
    for block in resp.get("content", []):
        if block.get("type") == "text":
            resp_text = block.get("text", "")
            break

    parsed = parse_json_response(resp_text)
    if not parsed:
        return None

    results = parsed.get("extractions", [])
    results.sort(key=lambda x: x.get("paper_index", 0))
    return results


# ------------------------------------------------------------------
# Map aggregation
# ------------------------------------------------------------------

def build_variable_map(
    extractions: List[Dict[str, Any]],
    hypothesis_id: str,
) -> Dict[str, Any]:
    """Aggregate variable inventory across all papers."""
    var_types = ["dependent", "independent", "control", "instruments"]
    result: Dict[str, List] = {vt: defaultdict(lambda: {"count": 0, "papers": []}) for vt in var_types}

    for ext in extractions:
        uid = ext.get("paper_uid", "")
        variables = ext.get("variables", {})
        for vt in var_types:
            for var_name in variables.get(vt, []):
                if not var_name:
                    continue
                key = var_name.lower().strip()
                result[vt][key]["count"] += 1
                if uid not in result[vt][key]["papers"]:
                    result[vt][key]["papers"].append(uid)

    # Convert to sorted list format
    output = {"hypothesis_id": hypothesis_id, "variables": {}}
    for vt in var_types:
        items = []
        for name, info in sorted(result[vt].items(), key=lambda x: x[1]["count"], reverse=True):
            items.append({
                "name": name,
                "paper_count": info["count"],
                "proxy_variants": [],
                "measurement_notes": "",
            })
        output["variables"][vt] = items

    return output


def build_method_map(
    extractions: List[Dict[str, Any]],
    hypothesis_id: str,
) -> Dict[str, Any]:
    """Aggregate method inventory across all papers."""
    method_counter: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "papers": [], "variants": set()})

    for ext in extractions:
        uid = ext.get("paper_uid", "")
        methods = ext.get("methods", {})
        for field in ["estimation", "data_approach", "identification"]:
            method = methods.get(field, "")
            if method:
                key = method.lower().strip()
                method_counter[key]["count"] += 1
                if uid not in method_counter[key]["papers"]:
                    method_counter[key]["papers"].append(uid)

    items = []
    for name, info in sorted(method_counter.items(), key=lambda x: x[1]["count"], reverse=True):
        items.append({
            "name": name,
            "paper_count": info["count"],
            "variants": [],
            "strengths": "",
            "limitations": "",
            "key_papers": info["papers"][:5],
        })

    return {"hypothesis_id": hypothesis_id, "methods": items}


def build_finding_map(
    extractions: List[Dict[str, Any]],
    hypothesis_id: str,
) -> Dict[str, Any]:
    """Aggregate findings and identify consensus vs. contested areas."""
    all_findings: List[Dict] = []
    all_disagreements: List[str] = []
    all_limitations: List[str] = []

    for ext in extractions:
        uid = ext.get("paper_uid", "")
        for f in ext.get("findings", []):
            f["paper_uid"] = uid
            all_findings.append(f)
        all_disagreements.extend(ext.get("disagreements", []))
        all_limitations.extend(ext.get("limitations", []))

    # Group findings by direction
    positive = [f for f in all_findings if f.get("direction") == "positive"]
    negative = [f for f in all_findings if f.get("direction") == "negative"]
    null_findings = [f for f in all_findings if f.get("direction") == "null"]

    # Build consensus and contested findings from all_findings
    consensus_findings = _build_consensus(all_findings)
    contested_findings = _build_contested(all_findings, all_disagreements)

    # Count limitation themes
    limitation_counter = Counter(all_limitations)

    return {
        "hypothesis_id": hypothesis_id,
        "total_findings": len(all_findings),
        "consensus_findings": consensus_findings,
        "contested_findings": contested_findings,
        "direction_summary": {
            "positive": len(positive),
            "negative": len(negative),
            "null": len(null_findings),
        },
        "all_findings": all_findings[:200],  # Cap for manageability
        "disagreements": list(set(all_disagreements)),
        "common_limitations": [
            {"limitation": l, "count": c}
            for l, c in limitation_counter.most_common(10)
        ],
        "gaps": [],  # Populated by synthesis step
    }


def _normalize_claim(claim: str) -> str:
    """Normalize a claim string for grouping."""
    return claim.lower().strip().rstrip(".")


def _build_consensus(
    findings: List[Dict[str, Any]],
    min_papers: int = 3,
) -> List[Dict[str, Any]]:
    """Identify consensus findings: claims with same direction from multiple papers.

    Groups findings by normalized claim text. When 3+ papers agree on direction,
    it's consensus. Falls back to direction-level aggregation if no claim-level
    clusters meet the threshold.
    """
    # Group by normalized claim → {direction → [papers]}
    claim_groups: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    claim_original: Dict[str, str] = {}

    for f in findings:
        claim = f.get("claim", "")
        if not claim:
            continue
        key = _normalize_claim(claim)
        direction = f.get("direction", "null")
        uid = f.get("paper_uid", "")
        if uid and uid not in claim_groups[key][direction]:
            claim_groups[key][direction].append(uid)
        if key not in claim_original:
            claim_original[key] = claim

    # Exact claim-level consensus
    consensus = []
    for key, dir_map in claim_groups.items():
        for direction, papers in dir_map.items():
            if len(papers) >= min_papers:
                consensus.append({
                    "claim": claim_original[key],
                    "direction": direction,
                    "paper_count": len(papers),
                    "strength": "strong" if len(papers) >= 5 else "moderate",
                    "key_papers": papers[:5],
                })

    # If no claim-level consensus, fall back to direction-level aggregation
    if not consensus:
        dir_counts: Dict[str, List] = defaultdict(list)
        for f in findings:
            direction = f.get("direction", "null")
            uid = f.get("paper_uid", "")
            if uid:
                dir_counts[direction].append(uid)

        for direction, papers in dir_counts.items():
            unique_papers = list(dict.fromkeys(papers))
            if len(unique_papers) >= min_papers and direction != "null":
                # Pick the most representative claim for this direction
                dir_findings = [f for f in findings if f.get("direction") == direction]
                high_conf = [f for f in dir_findings if f.get("confidence") == "high"]
                representative = (high_conf[0] if high_conf else dir_findings[0]).get("claim", "")

                consensus.append({
                    "claim": representative,
                    "direction": direction,
                    "paper_count": len(unique_papers),
                    "strength": "strong" if len(unique_papers) >= 10 else "moderate",
                    "key_papers": unique_papers[:5],
                })

    consensus.sort(key=lambda x: x.get("paper_count", 0), reverse=True)
    return consensus


def _build_contested(
    findings: List[Dict[str, Any]],
    disagreements: List[str],
    min_each_side: int = 2,
) -> List[Dict[str, Any]]:
    """Identify contested findings: topics where papers disagree on direction.

    A topic is contested when it has both positive and negative findings
    from different papers.
    """
    contested = []

    # Check if findings have both positive and negative directions
    positive_papers = set()
    negative_papers = set()
    for f in findings:
        uid = f.get("paper_uid", "")
        if not uid:
            continue
        if f.get("direction") == "positive":
            positive_papers.add(uid)
        elif f.get("direction") == "negative":
            negative_papers.add(uid)

    if len(positive_papers) >= min_each_side and len(negative_papers) >= min_each_side:
        # There is a directional split — build a contested entry
        pos_claims = [f for f in findings if f.get("direction") == "positive"]
        neg_claims = [f for f in findings if f.get("direction") == "negative"]

        # Pick representative claims from each side
        pos_repr = pos_claims[0].get("claim", "") if pos_claims else ""
        neg_repr = neg_claims[0].get("claim", "") if neg_claims else ""

        contested.append({
            "topic": "Hypothesis mechanism direction",
            "positions": [
                {"claim": pos_repr, "paper_count": len(positive_papers)},
                {"claim": neg_repr, "paper_count": len(negative_papers)},
            ],
        })

    # Add explicit disagreements from extraction
    seen_topics = set()
    for d in disagreements:
        if not d:
            continue
        topic_key = _normalize_claim(d)[:60]
        if topic_key in seen_topics:
            continue
        seen_topics.add(topic_key)
        contested.append({
            "topic": d,
            "positions": [],  # Source-level disagreement, no structured positions
        })

    return contested


# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------

def extract_and_map(
    ranked_result: Dict[str, Any],
    clusters_result: Dict[str, Any],
    hypothesis: Dict[str, Any],
    *,
    llm_client: Any,
) -> Dict[str, Any]:
    """Full extraction pipeline: extract → build maps."""
    hypothesis_id = ranked_result.get("hypothesis_id", "")
    selected = [p for p in ranked_result.get("papers", []) if p.get("selected")]
    assignments = clusters_result.get("paper_cluster_assignments", {})

    extractions = extract_structured(
        selected, hypothesis, assignments,
        llm_client=llm_client,
    )

    variable_map = build_variable_map(extractions, hypothesis_id)
    method_map = build_method_map(extractions, hypothesis_id)
    finding_map = build_finding_map(extractions, hypothesis_id)

    return {
        "extraction": {
            "hypothesis_id": hypothesis_id,
            "papers_processed": len(selected),
            "extractions": extractions,
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": _MODEL,
            },
        },
        "variable_map": variable_map,
        "method_map": method_map,
        "finding_map": finding_map,
    }
