"""Paper CRUD and _registry.json management."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.paper.models import (
    PAPER_TYPES,
    STAGES,
    STATUSES,
    Paper,
    RegistryEntry,
    _now_iso,
)

logger = logging.getLogger(__name__)

PAPERS_DIR = Path("data/papers")
REGISTRY_FILE = PAPERS_DIR / "_registry.json"


def _paper_dir(paper_id: str) -> Path:
    return PAPERS_DIR / paper_id


def _paper_json(paper_id: str) -> Path:
    return _paper_dir(paper_id) / "paper.json"


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def create_paper(
    paper_id: str,
    title: str,
    paper_type: str = "empirical",
    *,
    rq: str = "",
    target_journal: str = "",
    authors: Optional[List[str]] = None,
    stage: str = "idea",
    status: str = "active",
    data_dir: str = "",
    tags: Optional[List[str]] = None,
    notes: str = "",
) -> Paper:
    """Create a new paper with its directory and initial files."""
    if paper_type not in PAPER_TYPES:
        raise ValueError(f"Invalid paper_type '{paper_type}'. Must be one of {PAPER_TYPES}")
    if stage not in STAGES:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of {STAGES}")
    if status not in STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {STATUSES}")

    d = _paper_dir(paper_id)
    if d.exists():
        raise FileExistsError(f"Paper '{paper_id}' already exists at {d}")

    d.mkdir(parents=True, exist_ok=False)

    paper = Paper(
        paper_id=paper_id,
        title=title,
        paper_type=paper_type,
        rq=rq,
        target_journal=target_journal,
        authors=authors or [],
        current_stage=stage,
        current_status=status,
        data_dir=data_dir,
        tags=tags or [],
        notes=notes,
    )
    paper.save(_paper_json(paper_id))

    # Initialize empty log/data files
    (d / "stages.jsonl").write_text("")
    (d / "decisions.jsonl").write_text("")
    (d / "tasks.json").write_text(json.dumps({"tasks": []}, indent=2) + "\n")

    _sync_registry_entry(paper)
    logger.info("Created paper '%s' at %s", paper_id, d)
    return paper


def get_paper(paper_id: str) -> Paper:
    """Load a paper by ID."""
    path = _paper_json(paper_id)
    if not path.exists():
        raise FileNotFoundError(f"Paper '{paper_id}' not found at {path}")
    return Paper.load(path)


def list_papers() -> List[RegistryEntry]:
    """List all papers from _registry.json."""
    if not REGISTRY_FILE.exists():
        return []
    data = json.loads(REGISTRY_FILE.read_text())
    return [RegistryEntry(**e) for e in data.get("papers", [])]


def update_paper(paper_id: str, **fields: Any) -> Paper:
    """Update paper metadata fields."""
    paper = get_paper(paper_id)
    for key, value in fields.items():
        if not hasattr(paper, key):
            raise ValueError(f"Unknown paper field: '{key}'")
        setattr(paper, key, value)
    paper.updated_at = _now_iso()
    paper.save(_paper_json(paper_id))
    _sync_registry_entry(paper)
    return paper


def link_run(paper_id: str, run_id: str) -> Paper:
    """Link a run_id to a paper. Validates run directory exists."""
    run_dir = Path("data/lit_review") / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}. "
            f"Expected data/lit_review/{run_id}/"
        )
    paper = get_paper(paper_id)
    if run_id not in paper.run_ids:
        paper.run_ids.append(run_id)
        paper.updated_at = _now_iso()
        paper.save(_paper_json(paper_id))
        _sync_registry_entry(paper)
        logger.info("Linked run '%s' to paper '%s'", run_id, paper_id)
    else:
        logger.info("Run '%s' already linked to paper '%s'", run_id, paper_id)
    return paper


# ---------------------------------------------------------------------------
# Registry sync
# ---------------------------------------------------------------------------


def _sync_registry_entry(paper: Paper) -> None:
    """Update _registry.json with paper's current state."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    if REGISTRY_FILE.exists():
        data = json.loads(REGISTRY_FILE.read_text())
        entries = data.get("papers", [])

    # Count open tasks
    tasks_path = _paper_dir(paper.paper_id) / "tasks.json"
    open_tasks = 0
    if tasks_path.exists():
        tasks_data = json.loads(tasks_path.read_text())
        open_tasks = sum(
            1 for t in tasks_data.get("tasks", [])
            if t.get("status") in ("open", "in_progress", "blocked")
        )

    new_entry = RegistryEntry(
        paper_id=paper.paper_id,
        title=paper.title,
        paper_type=paper.paper_type,
        current_stage=paper.current_stage,
        current_status=paper.current_status,
        open_tasks=open_tasks,
        updated_at=paper.updated_at,
    ).to_dict()

    # Replace or append
    found = False
    for i, e in enumerate(entries):
        if e["paper_id"] == paper.paper_id:
            entries[i] = new_entry
            found = True
            break
    if not found:
        entries.append(new_entry)

    REGISTRY_FILE.write_text(
        json.dumps({"papers": entries}, indent=2, ensure_ascii=False) + "\n"
    )


def rebuild_registry() -> int:
    """Rebuild _registry.json from all paper.json files. Returns count."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    for d in sorted(PAPERS_DIR.iterdir()):
        pj = d / "paper.json"
        if d.is_dir() and pj.exists():
            paper = Paper.load(pj)
            _sync_registry_entry(paper)
    # Re-read after sync
    if REGISTRY_FILE.exists():
        data = json.loads(REGISTRY_FILE.read_text())
        return len(data.get("papers", []))
    return 0
