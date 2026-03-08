"""Deep Research Engine — Block 2 service logic.

Provides run_id generation and JSON I/O helpers shared across
pipeline steps 067-072.
"""

from __future__ import annotations

import json
import random
import string
from datetime import datetime
from pathlib import Path

# -- constants --------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "deep_research"

STEP_FILES = {
    "067": "plan.json",
    "068": "sources.json",
    "069": "evidence.json",
    "070": "credibility.json",
    "071": "claims.json",
    "072": "memo.json",
}


# -- run_id -----------------------------------------------------------

def generate_run_id() -> str:
    """Generate a unique run_id in the format ``dr_YYYYMMDD_xxxxx``."""
    date_part = datetime.now().strftime("%Y%m%d")
    random_part = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"dr_{date_part}_{random_part}"


def run_short(run_id: str) -> str:
    """Extract the last 5 characters of a run_id for entity ID generation."""
    return run_id[-5:]


# -- entity ID generators --------------------------------------------

def make_source_id(run_id: str, seq: int) -> str:
    return f"src_{run_short(run_id)}_{seq:03d}"


def make_evidence_id(run_id: str, seq: int) -> str:
    return f"ev_{run_short(run_id)}_{seq:03d}"


def make_claim_id(run_id: str, seq: int) -> str:
    return f"cl_{run_short(run_id)}_{seq:03d}"


def make_memo_id(run_id: str, seq: int) -> str:
    return f"memo_{run_short(run_id)}_{seq:03d}"


# -- JSON I/O --------------------------------------------------------

def _run_dir(run_id: str) -> Path:
    """Return the data directory for a given run, creating it if needed."""
    d = DATA_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_step_output(run_id: str, step: str, data: dict) -> Path:
    """Save a pipeline step's output as JSON.

    Args:
        run_id: The research run identifier.
        step: Pipeline step number (e.g. "067", "068").
        data: Dict to serialise.

    Returns:
        Path to the written file.
    """
    filename = STEP_FILES.get(step, f"{step}.json")
    path = _run_dir(run_id) / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_step_output(run_id: str, step: str) -> dict:
    """Load a pipeline step's output JSON.

    Args:
        run_id: The research run identifier.
        step: Pipeline step number (e.g. "067", "068").

    Returns:
        Parsed dict.

    Raises:
        FileNotFoundError: If the step output file does not exist.
    """
    filename = STEP_FILES.get(step, f"{step}.json")
    path = DATA_DIR / run_id / filename
    return json.loads(path.read_text(encoding="utf-8"))
