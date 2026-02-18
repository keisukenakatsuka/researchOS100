# src/run_metadata.py
"""Run-level infrastructure: output paths, logging, and provenance metadata.

Pure helpers for the "output side" of a pipeline run — where to write,
how to log, and how to stamp each run for reproducibility.

Usage::

    from src.run_metadata import get_output_dir, setup_logging, RunMetadata

    setup_logging()
    out = get_output_dir("047_weekly_papers_review", "2026-W07")
    meta = RunMetadata.build(
        notebook="047_weekly_papers_review",
        week_id="2026-W07",
        date_from="2026-02-08T23:35:52+00:00",
        date_to="2026-02-15T23:35:52+00:00",
        counts={"papers_fetched": 45},
    )
    meta.save(out / "run_metadata.json")

Migration note
--------------
When this package is renamed ``src/`` → ``researchos/``,
update consumer imports:  ``from researchos.run_metadata import ...``
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Output paths
# ----------------------------------------------------------------

def get_output_dir(
    notebook_name: str,
    week_id: str,
    *,
    base: str = "outputs",
    category: str = "weekly",
    create: bool = True,
) -> Path:
    """Return (and optionally create) the canonical output directory.

    Default pattern: ``outputs/weekly/<week_id>/<notebook_name>/``
    """
    p = Path(base) / category / week_id / notebook_name
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a standard format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ----------------------------------------------------------------
# Git helpers (private)
# ----------------------------------------------------------------

def _git_sha(*, short: bool = True) -> str:
    """Return the current HEAD SHA, or ``"unknown"`` if git is unavailable."""
    try:
        cmd = ["git", "rev-parse"]
        if short:
            cmd.append("--short")
        cmd.append("HEAD")
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """Return ``True`` if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return result.returncode != 0
    except Exception:
        return False


# ----------------------------------------------------------------
# Run metadata  (git SHA, week_id, counts — for reproducibility)
# ----------------------------------------------------------------

@dataclass(frozen=True)
class RunMetadata:
    """Immutable snapshot of run provenance written alongside outputs.

    ``date_from`` and ``date_to`` record the exact query window that was
    sent to Notion (ISO 8601 with timezone), so anyone looking at the
    output later can see *precisely* which time range was covered.

    Example usage::

        from src.run_metadata import RunMetadata
        from src.time import get_week_context

        wk = get_week_context()
        meta = RunMetadata.build(
            notebook="040_weekly_papers_review",
            week_id=wk.week_id,
            date_from=wk.date_from_iso,
            date_to=wk.date_to_iso,
            counts={"papers_total": 45, "read": 9, "keep": 18, "skip": 18},
        )
        meta.save(output_dir / "run_metadata.json")
    """
    notebook: str
    week_id: str
    git_sha: str
    git_dirty: bool
    run_ts_utc: str
    date_from: str = ""    # ISO 8601 with tz  (query window start)
    date_to: str = ""      # ISO 8601 with tz  (query window end)
    counts: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- factories ----

    @classmethod
    def build(
        cls,
        *,
        notebook: str,
        week_id: str,
        date_from: str = "",
        date_to: str = "",
        counts: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "RunMetadata":
        """Create a :class:`RunMetadata` by auto-detecting git state.

        Parameters
        ----------
        date_from, date_to:
            ISO 8601 strings **with timezone** describing the query window.
            Pass ``wk.date_from_iso`` / ``wk.date_to_iso`` from a
            :class:`WeekContext`, or compute explicitly for custom windows.
        """
        return cls(
            notebook=notebook,
            week_id=week_id,
            git_sha=_git_sha(short=True),
            git_dirty=_git_dirty(),
            run_ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            date_from=date_from,
            date_to=date_to,
            counts=counts or {},
            extra=extra or {},
        )

    # ---- I/O ----

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        """Write metadata JSON to *path*.  Parent dirs are created."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")
        logger.debug("Run metadata saved to %s", p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "RunMetadata":
        """Read metadata JSON back into a :class:`RunMetadata`."""
        data = json.loads(Path(path).read_text())
        return cls(**data)
