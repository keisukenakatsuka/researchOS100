# src/tools/github_timer_sync.py
"""DEPRECATED — all logic has moved to ``src.scripts.053_github_timer_sync``.

This module re-exports public symbols for backward compatibility.
New code should import from ``src.scripts.053_github_timer_sync`` directly.

Usage::

    python -m src.scripts.053_github_timer_sync --run
"""

import importlib as _importlib

_mod = _importlib.import_module("src.scripts.053_github_timer_sync")

# Constants
JST = _mod.JST
PRIVATE_REPO = _mod.PRIVATE_REPO
PROD_REPO = _mod.PROD_REPO
LOG_PATH = _mod.LOG_PATH
LOCK_PATH = _mod.LOCK_PATH
DEFAULT_INCLUDE_GLOBS = _mod.DEFAULT_INCLUDE_GLOBS

# Git helpers
git_has_changes = _mod.git_has_changes
git_changed_paths = _mod.git_changed_paths
git_add_all = _mod.git_add_all
git_commit = _mod.git_commit
git_push = _mod.git_push

# File copy
collect_eligible_files = _mod.collect_eligible_files
copy_files_to_prod = _mod.copy_files_to_prod

# Phases
run_private_phase = _mod.run_private_phase
run_prod_phase = _mod.run_prod_phase

# CLI
build_parser = _mod.build_parser
main = _mod.main
