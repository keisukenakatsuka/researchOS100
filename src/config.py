# src/config.py
"""Centralized environment loading and configuration access.

This module is the single entry-point for environment variables,
Notion credentials, and database IDs.  Time/week utilities and
run-metadata helpers live in their own modules but are re-exported
here for backward compatibility.

Usage (from a notebook)::

    from src.config import load_env, get_week_context, get_output_dir

    load_env()                          # loads notebooks/env.txt
    wk = get_week_context()             # WeekContext dataclass
    out = get_output_dir("040_weekly_papers_review", wk.week_id)

Migration note
--------------
When this package is renamed ``src/`` → ``researchos/``,
update consumer imports:  ``from researchos.config import ...``
Internal code uses ``from src.`` today — one-time find-replace is enough.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Env loading
# ----------------------------------------------------------------

_ENV_LOADED = False


def load_env(env_path: Optional[str | Path] = None) -> None:
    """Load environment variables from *env_path* (default: ``notebooks/env.txt``).

    Safe to call multiple times — only the first call actually loads.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    if env_path is None:
        # Resolve relative to this file:  src/config.py  →  <project>/notebooks/env.txt
        env_path = Path(__file__).resolve().parent.parent / "notebooks" / "env.txt"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")

    load_dotenv(dotenv_path=str(env_path))
    _ENV_LOADED = True
    logger.debug("Loaded env from %s", env_path)


def _require_env(name: str) -> str:
    """Return env var value or raise."""
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


# ----------------------------------------------------------------
# Notion env helpers
# ----------------------------------------------------------------

def get_notion_token() -> str:
    return _require_env("NOTION_TOKEN")


def get_notion_version(default: str = "2025-09-03") -> str:
    return os.getenv("NOTION_VERSION", default).strip()


def get_db_id(env_name: str) -> str:
    """Get a Notion database ID by env-var name, e.g. ``get_db_id("NOTION_LIT_DB_ID")``."""
    return _require_env(env_name)


def get_optional_db_id(env_name: str, default: str = "") -> str:
    return os.getenv(env_name, default).strip()


def is_notion_writeback_enabled() -> bool:
    return os.getenv("ENABLE_NOTION_WRITEBACK", "false").strip().lower() == "true"


def is_notion_writeback_dryrun() -> bool:
    return os.getenv("ENABLE_NOTION_WRITEBACK_DRYRUN", "false").strip().lower() == "true"


# ----------------------------------------------------------------
# Re-exports  (backward compatibility)
#
# All consumers can continue to write:
#     from src.config import WeekContext, RunMetadata, get_output_dir, ...
# ----------------------------------------------------------------

from src.time import (  # noqa: E402, F401
    WeekContext,
    get_iso_week_context,
    get_week_context,
)
from src.run_metadata import (  # noqa: E402, F401
    RunMetadata,
    get_output_dir,
    setup_logging,
)
