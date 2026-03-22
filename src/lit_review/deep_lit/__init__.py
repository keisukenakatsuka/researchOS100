# src/lit_review/deep_lit/__init__.py
"""Literature Deep Dive — hypothesis-scoped deep literature retrieval and structuring.

Scripts 114-119 form a pipeline that, for each focused hypothesis (H1/H2),
retrieves 100-150 papers, clusters them, extracts structured information,
and synthesizes a deep literature base.

Shared utilities for paper identity, title normalization, and hypothesis loading
are defined here for use by all deep_lit modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"

# Batch sizes
SCORING_BATCH_SIZE = 25
EXTRACTION_BATCH_SIZE = 10

# Retrieval defaults
DEFAULT_MAX_PER_QUERY = 100
DEFAULT_MIN_PAPERS = 100
DEFAULT_MAX_PAPERS = 150

# Clustering defaults
DEFAULT_MIN_CLUSTERS = 4
DEFAULT_MAX_CLUSTERS = 10


# ------------------------------------------------------------------
# Paper UID
# ------------------------------------------------------------------

def paper_uid(paper: Dict[str, Any]) -> str:
    """Generate a stable unique ID for a paper.

    Priority: DOI > arXiv ID > Semantic Scholar ID > title hash.
    """
    doi = (paper.get("doi") or "").strip()
    if doi:
        return f"doi:{doi.lower()}"

    arxiv = (paper.get("arxiv_id") or "").strip()
    if arxiv:
        return f"arxiv:{arxiv}"

    s2_id = (paper.get("source_id") or paper.get("paperId") or "").strip()
    if s2_id and paper.get("source") == "semantic_scholar":
        return f"s2:{s2_id}"

    # Fallback: title hash
    title = normalize_title(paper.get("title", ""))
    if title:
        h = hashlib.sha256(title.encode()).hexdigest()[:12]
        return f"title:{h}"

    return ""


# ------------------------------------------------------------------
# Title normalization
# ------------------------------------------------------------------

def normalize_title(title: str) -> str:
    """Normalize a paper title for dedup comparison.

    Lowercases, strips punctuation, collapses whitespace, removes accents.
    """
    if not title:
        return ""
    # Normalize unicode
    title = unicodedata.normalize("NFKD", title)
    # Remove accent marks
    title = "".join(c for c in title if not unicodedata.combining(c))
    # Lowercase
    title = title.lower()
    # Remove punctuation
    title = re.sub(r"[^\w\s]", "", title)
    # Collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()
    return title


# ------------------------------------------------------------------
# Hypothesis loading
# ------------------------------------------------------------------

def load_hypotheses_for_deep_lit(run_dir: Path) -> List[Dict[str, Any]]:
    """Load focused hypotheses for deep literature processing.

    Returns a list of hypothesis dicts (H1, and H2 if present).
    Each dict has hypothesis_id, hypothesis_statement, and full metadata.
    """
    focused_path = run_dir / "focused_hypotheses.json"
    if not focused_path.exists():
        raise FileNotFoundError("focused_hypotheses.json not found — run 089c first")

    focused = json.loads(focused_path.read_text())
    hypotheses = []

    primary = focused.get("primary")
    if not primary:
        raise ValueError("focused_hypotheses.json has no primary hypothesis")
    hypotheses.append(primary)

    if focused.get("has_secondary") and focused.get("secondary"):
        hypotheses.append(focused["secondary"])

    logger.info("Loaded %d focused hypotheses for deep literature", len(hypotheses))
    return hypotheses


def hyp_lit_dir(run_dir: Path, hypothesis_id: str) -> Path:
    """Get or create the per-hypothesis literature directory."""
    d = run_dir / "hyp_literature" / hypothesis_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------------

def parse_json_response(text: str) -> Optional[Any]:
    """Extract JSON from LLM response (handles ```json blocks)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------
# Downstream loading
# ------------------------------------------------------------------

def load_deep_lit_outputs(
    run_dir: Path,
    hypothesis_id: str,
) -> Optional[Dict[str, Any]]:
    """Load deep literature outputs for a hypothesis if available.

    Returns a dict with all available deep_lit artifacts, or None
    if the deep lit pipeline hasn't been run for this hypothesis.

    Downstream scripts use this to enrich their context when deep lit
    data is available, falling back to existing data when not.
    """
    lit_dir = run_dir / "hyp_literature" / hypothesis_id
    if not lit_dir.exists():
        return None

    result: Dict[str, Any] = {"hypothesis_id": hypothesis_id}

    file_keys = [
        ("synthesis", "hyp_literature_synthesis.json"),
        ("variable_map", "hyp_variable_map.json"),
        ("method_map", "hyp_method_map.json"),
        ("finding_map", "hyp_finding_map.json"),
        ("clusters", "hyp_clusters.json"),
        ("papers_ranked", "hyp_papers_ranked.json"),
    ]

    loaded = 0
    for key, fname in file_keys:
        path = lit_dir / fname
        if path.exists():
            try:
                result[key] = json.loads(path.read_text())
                loaded += 1
            except json.JSONDecodeError:
                logger.warning("Failed to parse %s", path)
                result[key] = None
        else:
            result[key] = None

    if loaded == 0:
        return None

    logger.info("Loaded deep_lit outputs for %s: %d/%d files",
                hypothesis_id, loaded, len(file_keys))
    return result
