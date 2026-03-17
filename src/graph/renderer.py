# src/graph/renderer.py
"""Shared Pyvis HTML renderer for knowledge graphs.

Converts a NetworkX DiGraph into an interactive standalone HTML file.

Usage::

    from src.graph.renderer import render_to_html

    render_to_html(G, Path("output.html"), title="Research Landscape")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from pyvis.network import Network

logger = logging.getLogger(__name__)


def render_to_html(
    G: nx.DiGraph,
    output_path: Path,
    *,
    title: str = "Knowledge Graph",
    height: str = "800px",
    physics: bool = True,
    hierarchical: bool = False,
    direction: str = "UD",
) -> Path:
    """Render a NetworkX graph to interactive HTML via Pyvis.

    Parameters
    ----------
    G : nx.DiGraph
        Graph with node/edge attributes (label, color, size, title).
    output_path : Path
        Where to write the HTML file.
    title : str
        Heading displayed in the HTML.
    height : str
        CSS height of the canvas.
    physics : bool
        Enable force-directed physics simulation.
    hierarchical : bool
        Use hierarchical (tree) layout instead of force-directed.
    direction : str
        Hierarchical direction: "UD" (top-down), "LR" (left-right).
    """
    net = Network(
        height=height,
        width="100%",
        directed=True,
        heading=title,
        cdn_resources="remote",
    )

    # Transfer nodes
    for node_id, attrs in G.nodes(data=True):
        net.add_node(
            node_id,
            label=attrs.get("label", str(node_id))[:50],
            color=attrs.get("color", "#9E9E9E"),
            size=attrs.get("size", 15),
            title=attrs.get("title", ""),
            shape=attrs.get("shape", "dot"),
            group=attrs.get("group", attrs.get("node_type", "")),
        )

    # Transfer edges
    for src, tgt, attrs in G.edges(data=True):
        net.add_edge(
            src, tgt,
            title=attrs.get("title", attrs.get("relation", "")),
            width=attrs.get("width", max(1, attrs.get("weight", 1.0))),
            color=attrs.get("color", "#999999"),
            arrows="to",
        )

    # Layout options
    options = {
        "physics": {
            "enabled": physics,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -80, "springLength": 150},
            "stabilization": {"iterations": 200},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "navigationButtons": True,
        },
        "nodes": {
            "font": {"size": 12},
        },
        "edges": {
            "font": {"size": 10, "align": "middle"},
            "smooth": {"type": "cubicBezier"},
        },
    }

    if hierarchical:
        options["layout"] = {
            "hierarchical": {
                "enabled": True,
                "direction": direction,
                "sortMethod": "directed",
                "levelSeparation": 120,
                "nodeSpacing": 150,
            }
        }
        options["physics"]["enabled"] = False

    net.set_options(json.dumps(options))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))

    # Inject filter controls and legend into the HTML
    _inject_controls(output_path, G)

    logger.info("Saved graph HTML: %s (%d nodes, %d edges)", output_path, len(G.nodes), len(G.edges))
    return output_path


def _inject_controls(html_path: Path, G: nx.DiGraph) -> None:
    """Inject node type filter controls and color legend into generated HTML."""
    # Collect node types and their colors
    type_colors: Dict[str, str] = {}
    for _, attrs in G.nodes(data=True):
        ntype = attrs.get("node_type", attrs.get("group", ""))
        color = attrs.get("color", "#9E9E9E")
        if ntype and ntype not in type_colors:
            type_colors[ntype] = color

    if not type_colors:
        return

    # Build legend + filter HTML
    legend_items = []
    for ntype, color in type_colors.items():
        legend_items.append(
            f'<label style="margin-right:12px;cursor:pointer;">'
            f'<input type="checkbox" checked onclick="toggleType(\'{ntype}\')" '
            f'style="margin-right:4px;">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border-radius:50%;vertical-align:middle;'
            f'margin-right:4px;"></span>'
            f'{ntype}</label>'
        )

    controls_html = (
        '<div id="graph-controls" style="position:fixed;top:60px;left:10px;z-index:1000;'
        'background:rgba(255,255,255,0.95);padding:10px 14px;border-radius:8px;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.15);font-family:sans-serif;font-size:13px;">'
        '<div style="margin-bottom:6px;font-weight:bold;">Node Types</div>'
        + "".join(legend_items) +
        '</div>'
        '<script>'
        'var hiddenTypes = {};'
        'function toggleType(t) {'
        '  hiddenTypes[t] = !hiddenTypes[t];'
        '  var nodes = network.body.data.nodes;'
        '  var updates = [];'
        '  nodes.forEach(function(n) {'
        '    if (n.group === t) {'
        '      updates.push({id: n.id, hidden: !!hiddenTypes[t]});'
        '    }'
        '  });'
        '  nodes.update(updates);'
        '}'
        '</script>'
    )

    html = html_path.read_text()
    # Insert before closing </body>
    html = html.replace("</body>", controls_html + "\n</body>")
    html_path.write_text(html)


def export_graph_data(
    graphs: Dict[str, nx.DiGraph],
    output_path: Path,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export multiple graph layers to a single graph_data.json.

    Parameters
    ----------
    graphs : dict
        Mapping of layer_name → NetworkX DiGraph.
    output_path : Path
        Where to write the JSON file.
    metadata : dict, optional
        Additional metadata to include.
    """
    data: Dict[str, Any] = {
        "metadata": metadata or {},
        "layers": {},
    }

    for layer_name, G in graphs.items():
        nodes: List[Dict[str, Any]] = []
        for node_id, attrs in G.nodes(data=True):
            node_data = {"id": str(node_id), **{k: v for k, v in attrs.items()}}
            nodes.append(node_data)

        edges: List[Dict[str, Any]] = []
        for src, tgt, attrs in G.edges(data=True):
            edge_data = {"source": str(src), "target": str(tgt), **{k: v for k, v in attrs.items()}}
            edges.append(edge_data)

        data["layers"][layer_name] = {
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("Saved graph_data.json: %d layers", len(graphs))
    return output_path
