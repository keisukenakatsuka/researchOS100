# src/graph/lineage_viz.py
"""107 RQ Lineage Visualizer — service logic.

Builds a hierarchical graph of RQ lineage (parent → child derivation)
and renders it as interactive HTML.

Usage::

    from src.graph.lineage_viz import build_lineage_graph, render_lineage

    G = build_lineage_graph(lineage_data)
    render_lineage(G, output_path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import networkx as nx

from src.graph.renderer import render_to_html, export_graph_data

logger = logging.getLogger(__name__)

STATUS_COLORS = {
    "active": "#2196F3",
    "promoted": "#4CAF50",
    "candidate": "#9E9E9E",
    "deferred": "#F44336",
    "merged": "#FF9800",
    "completed": "#607D8B",
}

STATUS_SIZES = {
    "active": 35,
    "promoted": 30,
    "candidate": 20,
    "deferred": 18,
    "merged": 18,
    "completed": 25,
}

RELATIONSHIP_COLORS = {
    "resolution": "#9C27B0",
    "branching": "#2196F3",
    "opportunity": "#FF9800",
    "deepening": "#4CAF50",
    "methodological": "#607D8B",
    "merge": "#F44336",
}


def build_lineage_graph(lineage: Dict[str, Any]) -> nx.DiGraph:
    """Build RQ lineage graph from rq_lineage.json data."""
    G = nx.DiGraph()

    for node in lineage.get("nodes", []):
        rq_id = node.get("rq_id", "")
        status = node.get("status", "candidate")
        node_type = node.get("node_type", "generated")
        title = node.get("title", "")

        tooltip = (
            f"<b>{title[:80]}</b><br>"
            f"Type: {node_type}<br>"
            f"Status: {status}<br>"
            f"ID: {rq_id[:20]}"
        )

        G.add_node(
            rq_id,
            label=title[:35],
            node_type=node_type,
            color=STATUS_COLORS.get(status, "#9E9E9E"),
            size=STATUS_SIZES.get(status, 20),
            title=tooltip,
            group=status,
            status=status,
        )

    for edge in lineage.get("edges", []):
        rel = edge.get("relationship_type", "branching")
        G.add_edge(
            edge.get("parent_rq_id", ""),
            edge.get("child_rq_id", ""),
            relation=rel,
            title=rel,
            color=RELATIONSHIP_COLORS.get(rel, "#999999"),
        )

    return G


def render_lineage(
    lineage_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Build and render lineage graph. Returns summary dict."""
    lineage = json.loads(lineage_path.read_text())
    G = build_lineage_graph(lineage)

    render_to_html(
        G, output_path,
        title="RQ Lineage",
        hierarchical=True,
        direction="UD",
    )

    # Also export graph_data
    data_path = output_path.parent / "lineage_graph_data.json"
    export_graph_data(
        {"lineage": G}, data_path,
        metadata={"type": "rq_lineage", "last_updated": lineage.get("last_updated", "")},
    )

    return {
        "nodes": len(G.nodes),
        "edges": len(G.edges),
        "html_path": str(output_path),
        "data_path": str(data_path),
    }
