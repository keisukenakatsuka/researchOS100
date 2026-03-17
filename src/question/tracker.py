# src/question/tracker.py
"""104 RQ Evolution Tracker — service logic.

Maintains a stable-ID-based DAG (Directed Acyclic Graph) of RQ lineage:
  parent_rq → child_candidates

All relationships are tracked by ID, never by title string.

The lineage file lives at data/question_formation/lineage/rq_lineage.json
and accumulates edges across multiple runs.

Usage::

    from src.question.tracker import update_lineage

    result = update_lineage(portfolio_path, rq_context_path)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LINEAGE_DIR = Path("data/question_formation/lineage")
_LINEAGE_FILE = "rq_lineage.json"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class LineageNode:
    """A node in the RQ lineage graph."""
    rq_id: str                     # stable ID (candidate_id or notion page id)
    title: str
    node_type: str = "original"    # original | generated | promoted
    status: str = "candidate"      # candidate | evaluated | promoted | deferred | active | completed
    run_id: str = ""               # which run created/processed this RQ
    created_at: str = ""


@dataclass
class LineageEdge:
    """A directed edge: parent → child."""
    parent_rq_id: str
    child_rq_id: str
    relationship_type: str = ""    # deepening | branching | resolution | methodological | opportunity
    derived_from_candidate_id: str = ""
    run_id: str = ""               # the run that generated this edge
    created_at: str = ""


@dataclass
class LineageGraph:
    """Full lineage DAG. Persisted as rq_lineage.json."""
    nodes: List[LineageNode] = field(default_factory=list)
    edges: List[LineageEdge] = field(default_factory=list)
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageGraph":
        nodes = [LineageNode(**n) for n in data.get("nodes", [])]
        edges = [LineageEdge(**e) for e in data.get("edges", [])]
        return cls(nodes=nodes, edges=edges, last_updated=data.get("last_updated", ""))

    def node_ids(self) -> set:
        return {n.rq_id for n in self.nodes}

    def has_edge(self, parent_id: str, child_id: str) -> bool:
        return any(e.parent_rq_id == parent_id and e.child_rq_id == child_id for e in self.edges)


@dataclass
class TrackerResult:
    status: str = "failed"
    nodes_added: int = 0
    edges_added: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------
# Relationship type inference
# ------------------------------------------------------------------

_SOURCE_TO_RELATIONSHIP = {
    "gap_driven": "branching",
    "resolution_driven": "resolution",
    "opportunity_driven": "opportunity",
    "deepening": "deepening",
    "methodological": "methodological",
}


# ------------------------------------------------------------------
# Markdown rendering
# ------------------------------------------------------------------

def _render_markdown(graph: LineageGraph) -> str:
    parts: List[str] = []
    parts.append("# RQ Lineage\n")
    parts.append(f"**Nodes**: {len(graph.nodes)} | **Edges**: {len(graph.edges)}")
    parts.append(f"**Last updated**: {graph.last_updated}\n")

    # Group nodes by type
    originals = [n for n in graph.nodes if n.node_type == "original"]
    generated = [n for n in graph.nodes if n.node_type == "generated"]
    promoted = [n for n in graph.nodes if n.node_type == "promoted"]

    if originals:
        parts.append("## Original RQs\n")
        for n in originals:
            parts.append(f"- `{n.rq_id}` — {n.title} [{n.status}]")

    if promoted:
        parts.append("\n## Promoted RQs\n")
        for n in promoted:
            parts.append(f"- `{n.rq_id}` — {n.title} [{n.status}]")

    if generated:
        parts.append(f"\n## Generated Candidates ({len(generated)})\n")
        for n in generated:
            parts.append(f"- `{n.rq_id}` — {n.title} [{n.status}]")

    # Edge listing
    if graph.edges:
        parts.append("\n## Lineage Edges\n")
        parts.append("| Parent | Child | Relationship | Run |")
        parts.append("|--------|-------|-------------|-----|")
        for e in graph.edges:
            parts.append(f"| `{e.parent_rq_id[:16]}` | `{e.child_rq_id[:16]}` | {e.relationship_type} | {e.run_id[:16]} |")

    # Tree view
    parts.append("\n## Tree View\n")
    parent_to_children: Dict[str, List[LineageEdge]] = {}
    for e in graph.edges:
        parent_to_children.setdefault(e.parent_rq_id, []).append(e)

    node_title = {n.rq_id: n.title for n in graph.nodes}
    roots = [n for n in graph.nodes if not any(e.child_rq_id == n.rq_id for e in graph.edges)]

    def render_tree(rq_id: str, depth: int = 0) -> None:
        indent = "  " * depth
        title = node_title.get(rq_id, rq_id)
        parts.append(f"{indent}- {title} (`{rq_id[:12]}`)")
        for edge in parent_to_children.get(rq_id, []):
            render_tree(edge.child_rq_id, depth + 1)

    for root in roots:
        render_tree(root.rq_id)

    return "\n".join(parts) + "\n"


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def update_lineage(
    portfolio_path: Path,
    rq_context_path: Path,
    *,
    lineage_dir: Optional[Path] = None,
) -> TrackerResult:
    """Update lineage DAG with new parent → child edges.

    No LLM call — purely structural graph operations.
    """
    result = TrackerResult()
    now_iso = datetime.now(timezone.utc).isoformat()
    lineage_dir = lineage_dir or _LINEAGE_DIR
    lineage_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load existing lineage
        lineage_path = lineage_dir / _LINEAGE_FILE
        if lineage_path.exists():
            graph = LineageGraph.from_dict(json.loads(lineage_path.read_text()))
        else:
            graph = LineageGraph()

        # Load parent RQ
        rq_data = json.loads(rq_context_path.read_text())
        parent_rq_id = rq_data.get("rq_id") or f"rq_{rq_context_path.parent.name}"
        parent_title = rq_data.get("title", "")
        parent_run_id = rq_context_path.parent.name

        # Add parent node if not exists
        existing_ids = graph.node_ids()
        nodes_added = 0
        edges_added = 0

        if parent_rq_id not in existing_ids:
            graph.nodes.append(LineageNode(
                rq_id=parent_rq_id,
                title=parent_title,
                node_type="original",
                status="active",
                run_id=parent_run_id,
                created_at=now_iso,
            ))
            nodes_added += 1

        # Load portfolio
        portfolio_data = json.loads(portfolio_path.read_text())
        portfolio = portfolio_data.get("portfolio", [])

        # Also load candidates for source_type
        candidates_path = portfolio_path.parent / "rq_candidates.json"
        source_types: Dict[str, str] = {}
        if candidates_path.exists():
            cand_data = json.loads(candidates_path.read_text())
            for c in cand_data.get("candidates", []):
                source_types[c.get("candidate_id", "")] = c.get("source_type", "")

        # Add child nodes and edges
        for entry in portfolio:
            cid = entry.get("candidate_id", "")
            if not cid:
                continue

            # Determine status from recommendation
            rec = entry.get("recommendation", "")
            status_map = {
                "promote": "promoted",
                "refine": "candidate",
                "defer": "deferred",
                "merge": "merged",
            }
            child_status = status_map.get(rec, "candidate")
            node_type = "promoted" if rec == "promote" else "generated"

            # Add child node
            if cid not in graph.node_ids():
                graph.nodes.append(LineageNode(
                    rq_id=cid,
                    title=entry.get("title", ""),
                    node_type=node_type,
                    status=child_status,
                    run_id=parent_run_id,
                    created_at=now_iso,
                ))
                nodes_added += 1

            # Add edge (parent → child)
            if not graph.has_edge(parent_rq_id, cid):
                source = source_types.get(cid, "")
                rel_type = _SOURCE_TO_RELATIONSHIP.get(source, "branching")

                # Handle merge: edge goes from merge target, not parent
                if rec == "merge" and entry.get("merge_target_id"):
                    # Still add parent → child edge for lineage
                    pass

                graph.edges.append(LineageEdge(
                    parent_rq_id=parent_rq_id,
                    child_rq_id=cid,
                    relationship_type=rel_type,
                    derived_from_candidate_id=cid,
                    run_id=parent_run_id,
                    created_at=now_iso,
                ))
                edges_added += 1

            # If merge, also add merge edge
            if rec == "merge" and entry.get("merge_target_id"):
                merge_target = entry["merge_target_id"]
                if not graph.has_edge(merge_target, cid):
                    graph.edges.append(LineageEdge(
                        parent_rq_id=merge_target,
                        child_rq_id=cid,
                        relationship_type="merge",
                        derived_from_candidate_id=cid,
                        run_id=parent_run_id,
                        created_at=now_iso,
                    ))
                    edges_added += 1

        graph.last_updated = now_iso

        # Save
        lineage_path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2))
        logger.info("Lineage updated: +%d nodes, +%d edges (total: %d nodes, %d edges)",
                     nodes_added, edges_added, len(graph.nodes), len(graph.edges))

        # Save markdown
        md_path = lineage_dir / "rq_lineage.md"
        md_path.write_text(_render_markdown(graph))

        result.status = "generated"
        result.nodes_added = nodes_added
        result.edges_added = edges_added
        result.total_nodes = len(graph.nodes)
        result.total_edges = len(graph.edges)

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        logger.error("104: %s", e)

    return result
