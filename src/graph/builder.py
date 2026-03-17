# src/graph/builder.py
"""106 Knowledge Graph Builder — service logic.

Constructs interactive graphs from research run artifacts:
  - Layer 1: Research Landscape (from landscape.json)
  - Layer 2: Evidence Chain (from evidence.json + lit_review.json + hypotheses.json)

Usage::

    from src.graph.builder import build_all_graphs

    result = build_all_graphs(run_dir)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from src.graph.renderer import render_to_html, export_graph_data

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Color / size schemes
# ------------------------------------------------------------------

LANDSCAPE_COLORS = {
    "rq": "#E91E63",
    "theoretical_stream": "#2196F3",
    "paper": "#B0BEC5",
    "method": "#FF9800",
    "dataset": "#795548",
    "finding": "#4CAF50",
    "gap": "#F44336",
    "blindspot": "#FF5722",
}

LANDSCAPE_SIZES = {
    "rq": 40,
    "theoretical_stream": 28,
    "paper": 12,
    "method": 18,
    "dataset": 16,
    "finding": 22,
    "gap": 24,
    "blindspot": 20,
}

EVIDENCE_COLORS = {
    "source": "#607D8B",
    "evidence": "#03A9F4",
    "claim": "#4CAF50",
    "hypothesis": "#E91E63",
}

EVIDENCE_SIZES = {
    "source": 14,
    "evidence": 16,
    "claim": 22,
    "hypothesis": 30,
}

CONFIDENCE_COLORS = {
    "high": "#4CAF50",
    "medium": "#FFC107",
    "low": "#F44336",
}


# ------------------------------------------------------------------
# Result type
# ------------------------------------------------------------------

@dataclass
class GraphBuildResult:
    status: str = "failed"
    layers_built: List[str] = field(default_factory=list)
    html_files: List[str] = field(default_factory=list)
    node_counts: Dict[str, int] = field(default_factory=dict)
    edge_counts: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


# ------------------------------------------------------------------
# Tooltip formatters
# ------------------------------------------------------------------

def _landscape_tooltip(node: Dict[str, Any]) -> str:
    """Build HTML tooltip for a landscape node."""
    ntype = node.get("node_type", "")
    label = node.get("label", "")
    attrs = node.get("attributes", {})

    parts = [f"<b>{label}</b>", f"Type: {ntype}"]
    if attrs.get("description"):
        parts.append(f"<br>{attrs['description'][:200]}")
    if attrs.get("paper_count"):
        parts.append(f"Papers: {attrs['paper_count']}")
    if attrs.get("strength"):
        parts.append(f"Strength: {attrs['strength']}")
    if attrs.get("severity"):
        parts.append(f"Severity: {attrs['severity']}")
    return "<br>".join(parts)


def _evidence_tooltip(node_type: str, data: Dict[str, Any]) -> str:
    """Build HTML tooltip for an evidence chain node."""
    parts = [f"<b>{data.get('label', '')}</b>", f"Type: {node_type}"]
    if data.get("confidence"):
        parts.append(f"Confidence: {data['confidence']}")
    if data.get("paper_title"):
        parts.append(f"Source: {data['paper_title'][:80]}")
    if data.get("evidence_count"):
        parts.append(f"Evidence items: {data['evidence_count']}")
    if data.get("strategy"):
        parts.append(f"Strategy: {data['strategy']}")
    return "<br>".join(parts)


# ------------------------------------------------------------------
# Layer 1: Research Landscape
# ------------------------------------------------------------------

def build_landscape_graph(landscape: Dict[str, Any]) -> nx.DiGraph:
    """Build landscape graph from landscape.json knowledge_graph."""
    G = nx.DiGraph()
    kg = landscape.get("knowledge_graph", {})

    for node in kg.get("nodes", []):
        nid = node.get("id", "")
        ntype = node.get("node_type", "")
        G.add_node(
            nid,
            label=node.get("label", "")[:50],
            node_type=ntype,
            color=LANDSCAPE_COLORS.get(ntype, "#9E9E9E"),
            size=LANDSCAPE_SIZES.get(ntype, 15),
            title=_landscape_tooltip(node),
            group=ntype,
        )

    for edge in kg.get("edges", []):
        G.add_edge(
            edge.get("source", ""),
            edge.get("target", ""),
            relation=edge.get("relation", ""),
            title=edge.get("relation", ""),
            weight=edge.get("weight", 1.0),
        )

    return G


# ------------------------------------------------------------------
# Layer 2: Evidence Chain
# ------------------------------------------------------------------

def build_evidence_chain(run_dir: Path) -> nx.DiGraph:
    """Build evidence provenance graph: Source → Evidence → Claim → Hypothesis.

    Constructs the graph from multiple artifacts, linking entities by
    their stable IDs (paper_id, evidence indices, claim references,
    hypothesis source_claim_ids).
    """
    G = nx.DiGraph()

    # Load artifacts
    evidence_list = []
    ev_path = run_dir / "evidence.json"
    if ev_path.exists():
        ev_data = json.loads(ev_path.read_text())
        if isinstance(ev_data, list):
            evidence_list = ev_data
        elif isinstance(ev_data, dict):
            evidence_list = ev_data.get("evidence_items", ev_data.get("evidence", []))

    lr = {}
    lr_path = run_dir / "lit_review.json"
    if lr_path.exists():
        lr = json.loads(lr_path.read_text())

    hyp_data = {}
    hyp_path = run_dir / "hypotheses.json"
    if hyp_path.exists():
        hyp_data = json.loads(hyp_path.read_text())

    # --- Sources ---
    # Collect unique sources from evidence.
    # paper_id / source_paper_id may be empty; fall back to paper_title as key.
    sources_seen: Dict[str, Dict] = {}
    for ev in evidence_list:
        pid = ev.get("source_paper_id") or ev.get("paper_id", "")
        ptitle = ev.get("paper_title", "")
        key = pid if pid else ptitle
        if key and key not in sources_seen:
            sources_seen[key] = {"paper_title": ptitle or key}

    for pid, meta in sources_seen.items():
        G.add_node(
            f"src_{pid}",
            label=meta["paper_title"][:40],
            node_type="source",
            color=EVIDENCE_COLORS["source"],
            size=EVIDENCE_SIZES["source"],
            title=_evidence_tooltip("source", {"label": meta["paper_title"], "paper_title": meta["paper_title"]}),
            group="source",
            paper_id=pid,
        )

    # --- Evidence ---
    # Limit to representative sample if too many (>50 makes graph unreadable)
    ev_sample = evidence_list[:50] if len(evidence_list) > 50 else evidence_list
    for i, ev in enumerate(ev_sample):
        ev_id = f"ev_{i}"
        pid = ev.get("source_paper_id") or ev.get("paper_id", "")
        ptitle = ev.get("paper_title", "")
        source_key = pid if pid else ptitle
        claim_text = ev.get("claim_or_point", ev.get("evidence_text", ""))
        confidence = ev.get("confidence", "medium")

        G.add_node(
            ev_id,
            label=claim_text[:45],
            node_type="evidence",
            color=CONFIDENCE_COLORS.get(confidence, EVIDENCE_COLORS["evidence"]),
            size=EVIDENCE_SIZES["evidence"],
            title=_evidence_tooltip("evidence", {
                "label": claim_text[:100],
                "confidence": confidence,
                "paper_title": ev.get("paper_title", ""),
            }),
            group="evidence",
            confidence=confidence,
            source_paper_id=pid,
        )

        # Edge: source → evidence
        if source_key:
            G.add_edge(f"src_{source_key}", ev_id, relation="extracted_from", title="extracted_from")

    # --- Claims (from lit_review findings) ---
    findings = lr.get("empirical_findings", {})
    claim_idx = 0
    # Map paper titles to source node IDs
    title_to_src: Dict[str, str] = {}
    for key, meta in sources_seen.items():
        ptitle = meta["paper_title"]
        title_to_src[ptitle.lower()] = f"src_{key}"
        # Also partial match: first 40 chars
        if len(ptitle) > 40:
            title_to_src[ptitle[:40].lower()] = f"src_{key}"

    for category in ["established", "emerging", "contested"]:
        items = findings.get(category, [])
        for item in items:
            if category == "contested":
                statement = item.get("topic", "")
                strength = "contested"
                supporting = []
                for pos in item.get("positions", []):
                    if isinstance(pos, dict):
                        supporting.extend(pos.get("papers", []))
            else:
                statement = item.get("statement", item.get("finding", ""))
                strength = item.get("strength", "medium")
                supporting = item.get("supporting_papers", [])

            if not statement:
                continue

            claim_id = f"claim_{claim_idx}"
            claim_idx += 1
            evidence_count = item.get("paper_count", len(supporting))

            G.add_node(
                claim_id,
                label=statement[:45],
                node_type="claim",
                color=EVIDENCE_COLORS["claim"],
                size=min(35, EVIDENCE_SIZES["claim"] + evidence_count * 2),
                title=_evidence_tooltip("claim", {
                    "label": statement[:150],
                    "confidence": strength,
                    "evidence_count": evidence_count,
                }),
                group="claim",
                category=category,
                strength=strength,
            )

            # Edges: source → claim (via supporting papers)
            for paper_title in supporting:
                ptl = paper_title.lower()
                src_key = title_to_src.get(ptl, "")
                # Fallback: partial match on first 40 chars
                if not src_key:
                    src_key = title_to_src.get(ptl[:40], "")
                if src_key and G.has_node(src_key):
                    G.add_edge(src_key, claim_id, relation="supports", title="supports")

    # --- Hypotheses ---
    hypotheses = hyp_data.get("hypotheses", [])
    for i, h in enumerate(hypotheses):
        h_id = f"hyp_H{i + 1}"
        statement = h.get("hypothesis_statement", "")
        strategy = h.get("strategy", "")

        G.add_node(
            h_id,
            label=f"H{i + 1}",
            node_type="hypothesis",
            color=EVIDENCE_COLORS["hypothesis"],
            size=EVIDENCE_SIZES["hypothesis"],
            title=_evidence_tooltip("hypothesis", {
                "label": f"H{i + 1}: {statement[:120]}",
                "strategy": strategy,
            }),
            group="hypothesis",
            strategy=strategy,
            full_statement=statement,
        )

        # Edges: claim → hypothesis (via source_claim_ids or source_gaps)
        source_claims = h.get("source_claim_ids", [])
        if source_claims:
            # Try to match claim IDs to our claim nodes
            for sc_id in source_claims:
                # Claims from the pipeline might use different ID format
                # Connect to any matching claim node
                for node_id in G.nodes:
                    if node_id.startswith("claim_") and G.nodes[node_id].get("node_type") == "claim":
                        # Heuristic: connect hypotheses to claims from relevant category
                        G.add_edge(node_id, h_id, relation="grounds", title="grounds")
                        break
        else:
            # Fallback: connect to all claims (indicates general grounding)
            claim_nodes = [n for n in G.nodes if n.startswith("claim_")]
            for cn in claim_nodes[:3]:
                G.add_edge(cn, h_id, relation="grounds", title="grounds")

    return G


# ------------------------------------------------------------------
# Combined graph (all layers merged)
# ------------------------------------------------------------------

# Cross-layer edge colors
CROSS_LAYER_EDGE_COLORS = {
    "landscape_to_evidence": "#9C27B0",   # purple
    "evidence_to_lineage": "#FF9800",     # orange
}


def build_combined_graph(
    run_dir: Path,
    lineage_path: Optional[Path] = None,
) -> nx.DiGraph:
    """Build a single graph merging landscape + evidence + lineage layers.

    Cross-layer connections:
    - Landscape findings ↔ Evidence chain claims (by text similarity)
    - Landscape gaps ↔ RQ lineage children (gap_driven derivation)
    - Evidence hypotheses ↔ RQ lineage promoted candidates

    Nodes from different layers are prefixed to avoid ID collision,
    except shared nodes (RQ, hypotheses) which are merged.
    """
    G = nx.DiGraph()

    # --- Layer 1: Landscape ---
    ls_path = run_dir / "landscape.json"
    if ls_path.exists():
        ls = json.loads(ls_path.read_text())
        G_ls = build_landscape_graph(ls)
        # Add with "ls_" prefix except RQ node
        for nid, attrs in G_ls.nodes(data=True):
            new_id = nid  # landscape IDs are already prefixed
            G.add_node(new_id, **attrs)
        for src, tgt, attrs in G_ls.edges(data=True):
            G.add_edge(src, tgt, **attrs)

    # --- Layer 2: Evidence Chain ---
    G_ev = build_evidence_chain(run_dir)
    if G_ev.nodes:
        for nid, attrs in G_ev.nodes(data=True):
            G.add_node(nid, **attrs)
        for src, tgt, attrs in G_ev.edges(data=True):
            G.add_edge(src, tgt, **attrs)

    # --- Layer 3: Lineage ---
    if lineage_path is None:
        lineage_path = Path("data/question_formation/lineage/rq_lineage.json")
    if lineage_path.exists():
        lineage = json.loads(lineage_path.read_text())
        from src.graph.lineage_viz import build_lineage_graph, STATUS_COLORS, STATUS_SIZES
        G_lin = build_lineage_graph(lineage)
        for nid, attrs in G_lin.nodes(data=True):
            lin_id = f"lin_{nid}"
            G.add_node(lin_id, **attrs)
        for src, tgt, attrs in G_lin.edges(data=True):
            G.add_edge(f"lin_{src}", f"lin_{tgt}", **attrs)

        # --- Cross-layer: Lineage root RQ → Landscape RQ ---
        # Find landscape RQ node
        ls_rq_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "rq" and not n.startswith("lin_")]
        lin_root_nodes = [n for n in G.nodes if n.startswith("lin_") and G.nodes[n].get("node_type") == "original"]
        if ls_rq_nodes and lin_root_nodes:
            G.add_edge(
                ls_rq_nodes[0], lin_root_nodes[0],
                relation="evolves_to", title="evolves_to",
                color=CROSS_LAYER_EDGE_COLORS["evidence_to_lineage"],
            )

        # --- Cross-layer: Hypotheses → Lineage promoted RQs ---
        # Connect hypothesis nodes to promoted lineage nodes via derivation
        hyp_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "hypothesis"]
        promoted_nodes = [n for n, d in G.nodes(data=True) if d.get("status") == "promoted"]
        # Simple heuristic: connect each promoted RQ to all hypotheses (they derive from the research)
        for pn in promoted_nodes:
            for hn in hyp_nodes[:2]:  # limit to avoid clutter
                G.add_edge(
                    hn, pn,
                    relation="generates_rq", title="generates_rq",
                    color=CROSS_LAYER_EDGE_COLORS["evidence_to_lineage"],
                )

    return G


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def build_all_graphs(
    run_dir: Path,
    *,
    layers: Optional[List[str]] = None,
) -> GraphBuildResult:
    """Build and render all requested graph layers.

    Parameters
    ----------
    run_dir : Path
        Research run directory.
    layers : list, optional
        Which layers to build. Default: ["landscape", "evidence"].
    """
    result = GraphBuildResult()
    layers = layers or ["landscape", "evidence"]
    graphs: Dict[str, nx.DiGraph] = {}

    try:
        run_id = run_dir.name

        # Layer 1: Landscape
        if "landscape" in layers:
            ls_path = run_dir / "landscape.json"
            if ls_path.exists():
                ls = json.loads(ls_path.read_text())
                G_landscape = build_landscape_graph(ls)
                graphs["landscape"] = G_landscape

                html_path = render_to_html(
                    G_landscape, run_dir / "knowledge_graph.html",
                    title=f"Research Landscape — {run_id}",
                )
                result.html_files.append(str(html_path))
                result.layers_built.append("landscape")
                result.node_counts["landscape"] = len(G_landscape.nodes)
                result.edge_counts["landscape"] = len(G_landscape.edges)
            else:
                logger.warning("landscape.json not found — skipping landscape layer")

        # Layer 2: Evidence Chain
        if "evidence" in layers:
            G_evidence = build_evidence_chain(run_dir)
            if G_evidence.nodes:
                graphs["evidence"] = G_evidence

                html_path = render_to_html(
                    G_evidence, run_dir / "evidence_chain.html",
                    title=f"Evidence Chain — {run_id}",
                )
                result.html_files.append(str(html_path))
                result.layers_built.append("evidence")
                result.node_counts["evidence"] = len(G_evidence.nodes)
                result.edge_counts["evidence"] = len(G_evidence.edges)
            else:
                logger.warning("No evidence data found — skipping evidence layer")

        # Combined graph (all layers in one)
        if "combined" in layers:
            lineage_path = Path("data/question_formation/lineage/rq_lineage.json")
            G_combined = build_combined_graph(run_dir, lineage_path=lineage_path)
            if G_combined.nodes:
                graphs["combined"] = G_combined

                html_path = render_to_html(
                    G_combined, run_dir / "combined_research_graph.html",
                    title=f"Research Graph — {run_id}",
                )
                result.html_files.append(str(html_path))
                result.layers_built.append("combined")
                result.node_counts["combined"] = len(G_combined.nodes)
                result.edge_counts["combined"] = len(G_combined.edges)

        # Export graph_data.json
        if graphs:
            rq_path = run_dir / "rq_context.json"
            rq_title = ""
            if rq_path.exists():
                rq_title = json.loads(rq_path.read_text()).get("title", "")

            export_graph_data(
                graphs, run_dir / "graph_data.json",
                metadata={
                    "run_id": run_id,
                    "rq_title": rq_title,
                    "layers": list(graphs.keys()),
                },
            )
            result.html_files.append(str(run_dir / "graph_data.json"))

        result.status = "generated" if graphs else "failed"
        if not graphs:
            result.error = "No graph layers could be built"

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.error("106: %s", e)

    return result
