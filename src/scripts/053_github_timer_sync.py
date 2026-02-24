#!/usr/bin/env python
# src/scripts/053_github_timer_sync.py
"""Notion-driven GitHub timer sync: private → production.

Reads enabled rows from the GITHUB_TIMER_DB Notion database and:
  - Phase A: commit & push private repo, update Notion timestamps
  - Phase B+C: copy eligible files to production, commit & push, update Notion
  - Phase D (src-sync): scan ``src/`` for .py files not covered by Notion rows,
    copy missing or outdated files from private → production

Designed for scheduled execution via macOS launchd.

Running ``python -m src.scripts.053_github_timer_sync --run``
executes the full pipeline.

Usage::

    python -m src.scripts.053_github_timer_sync --dry-run
    python -m src.scripts.053_github_timer_sync --run
    python -m src.scripts.053_github_timer_sync --run --only private
    python -m src.scripts.053_github_timer_sync --run --only src-sync
    python -m src.scripts.053_github_timer_sync --run --name-contains "044"
    python -m src.scripts.053_github_timer_sync --env notebooks/env.txt --run -v
"""

from __future__ import annotations

import argparse
import fcntl
import filecmp
import fnmatch
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env, get_db_id, setup_logging
from src.notion.client import build_notion_client_from_env, NotionDataSourceResolver
from src.notion.github_timer_repo import GitHubTimerRepo

logger = logging.getLogger("053_github_timer_sync")

SCRIPT_NAME = "053_github_timer_sync"

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------

JST = ZoneInfo("Asia/Tokyo")
PRIVATE_REPO = Path.home() / "projects" / "researchOS100-private"
PROD_REPO = Path.home() / "projects" / "researchOS100"
LOG_PATH = Path.home() / "Library" / "Logs" / "researchos_github_timer.log"
LOCK_PATH = Path("/tmp/researchos_github_timer.lock")
DEFAULT_INCLUDE_GLOBS = ["**/*.py", "**/*.ipynb"]
# Always excluded regardless of row config
_ALWAYS_EXCLUDE_DIRS = {".ipynb_checkpoints", "__pycache__", ".git"}


# ----------------------------------------------------------------
# Execution lock (prevent overlapping runs)
# ----------------------------------------------------------------

class _ExecutionLock:
    """File-based exclusive lock using ``fcntl.flock``.

    Usage::

        lock = _ExecutionLock(LOCK_PATH)
        if not lock.acquire():
            sys.exit(0)
        try:
            ...  # main work
        finally:
            lock.release()
    """

    def __init__(self, path: Path):
        self._path = path
        self._fp: Any = None

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True on success."""
        try:
            self._fp = open(self._path, "w")
            fcntl.flock(self._fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fp.write(str(os.getpid()))
            self._fp.flush()
            return True
        except (OSError, IOError):
            if self._fp:
                self._fp.close()
                self._fp = None
            return False

    def release(self) -> None:
        """Release the lock and remove the lock file."""
        if self._fp:
            try:
                fcntl.flock(self._fp, fcntl.LOCK_UN)
                self._fp.close()
            except (OSError, IOError):
                pass
            self._fp = None
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass


# ----------------------------------------------------------------
# Dual logging (console + file)
# ----------------------------------------------------------------

def setup_dual_logging(verbose: bool) -> None:
    """Configure root logger with both console and file handlers.

    Console: INFO (or DEBUG if verbose).
    File: always DEBUG, appending to LOG_PATH.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove existing handlers (avoid duplicates on re-run)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # File handler
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(LOG_PATH), mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        ))
        root.addHandler(fh)
    except OSError as e:
        console.setLevel(logging.DEBUG)
        logger.warning("Could not open log file %s: %s", LOG_PATH, e)


# ----------------------------------------------------------------
# Git helpers (subprocess-based)
# ----------------------------------------------------------------

def git_has_changes(repo_dir: Path) -> bool:
    """Return True if the working tree has uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def git_changed_paths(repo_dir: Path) -> Set[str]:
    """Return set of changed file paths (relative to repo root).

    Includes staged, unstaged, and untracked files.

    Uses ``git status --porcelain`` which outputs lines like::

        XY PATH          (normal)
        XY OLD -> NEW    (rename)

    where XY is a 2-char status field followed by a space.  However,
    some git versions/configs may vary the spacing.  To be robust, we
    strip the leading status characters by finding the first non-status
    character rather than relying on a fixed offset.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    paths: Set[str] = set()
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        # Strip leading status chars (XY) and whitespace to get the path.
        # Standard format: "XY PATH" where XY is 2 chars + 1 space = 3.
        # Robust: lstrip status chars then whitespace.
        path_part = line.lstrip(" MADRCU?!").strip()
        if not path_part:
            continue
        # Handle renames: "old -> new"
        if " -> " in path_part:
            path_part = path_part.split(" -> ")[-1]
        # Remove surrounding quotes (git quotes paths with special chars)
        if path_part.startswith('"') and path_part.endswith('"'):
            path_part = path_part[1:-1]
        paths.add(path_part)
    return paths


def git_add_all(repo_dir: Path) -> None:
    """Run ``git add -A``."""
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    logger.debug("git add -A in %s", repo_dir.name)


def git_commit(repo_dir: Path, message: str) -> bool:
    """Run ``git commit -m ...``. Returns True on success."""
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Committed in %s: %s", repo_dir.name, message[:120])
        return True
    else:
        # returncode=1 means "nothing to commit"
        logger.warning(
            "git commit failed (rc=%d) in %s: %s",
            result.returncode, repo_dir.name, result.stderr.strip()[:200],
        )
        return False


def git_push(repo_dir: Path) -> bool:
    """Run ``git push``. Returns True on success."""
    result = subprocess.run(
        ["git", "push"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Pushed %s", repo_dir.name)
        return True
    else:
        logger.error(
            "git push failed (rc=%d) in %s: %s",
            result.returncode, repo_dir.name, result.stderr.strip()[:200],
        )
        return False


# ----------------------------------------------------------------
# Path safety validation
# ----------------------------------------------------------------

def _validate_path_safety(
    src_path: str,
    dst_path: str,
    row_name: str,
) -> bool:
    """Validate that src_path and dst_path stay within their repo roots.

    Prevents directory traversal via ``../`` or symlink escape.

    Returns True if both paths are safe, False otherwise.
    """
    try:
        src_abs = (PRIVATE_REPO / src_path).resolve()
        dst_abs = (PROD_REPO / dst_path).resolve()
    except (OSError, ValueError) as e:
        logger.error("Row %s: path resolution failed: %s", row_name, e)
        return False

    private_root = PRIVATE_REPO.resolve()
    prod_root = PROD_REPO.resolve()

    if not str(src_abs).startswith(str(private_root) + "/") and src_abs != private_root:
        logger.error(
            "Row %s: src_path %r escapes PRIVATE_REPO (%s → %s) — skipping",
            row_name, src_path, private_root, src_abs,
        )
        return False

    if not str(dst_abs).startswith(str(prod_root) + "/") and dst_abs != prod_root:
        logger.error(
            "Row %s: dst_path %r escapes PROD_REPO (%s → %s) — skipping",
            row_name, dst_path, prod_root, dst_abs,
        )
        return False

    return True


# ----------------------------------------------------------------
# File copy logic
# ----------------------------------------------------------------

def _matches_any_glob(path: Path, globs: List[str]) -> bool:
    """Check if a path matches any of the given glob patterns."""
    path_str = str(path)
    for pattern in globs:
        if fnmatch.fnmatch(path_str, pattern):
            return True
        # Also match against just the filename for simple patterns
        if fnmatch.fnmatch(path.name, pattern):
            return True
    return False


def collect_eligible_files(
    src_root: Path,
    src_path: str,
    *,
    include_globs: List[str],
    exclude_globs: List[str],
    since: Optional[datetime],
    now_jst: datetime,
) -> List[Path]:
    """Return list of files eligible for copying to production.

    Supports both **directory** and **single-file** ``src_path`` values:

    - Directory: scans recursively using ``include_globs`` and returns
      paths relative to ``src_root / src_path``.
    - Single file: returns ``[Path(filename)]`` if the file passes the
      modification-time check, ignoring glob patterns (the user
      explicitly listed this file).

    Parameters
    ----------
    src_root:
        The private repo root.
    src_path:
        Relative path within the repo for this row's source.
        May be a directory (e.g. ``"src/scripts"``) or a single file
        (e.g. ``"src/scripts/047_weekly_papers_review.py"``).
    include_globs:
        Glob patterns to include (e.g. ``["**/*.py", "**/*.ipynb"]``).
        Ignored when ``src_path`` is a single file.
    exclude_globs:
        Glob patterns to exclude.
        Ignored when ``src_path`` is a single file.
    since:
        Only include files modified after this time (JST-aware).
        If None (first run), include all matching files.
    now_jst:
        Current time in JST (for logging).

    Returns
    -------
    list[Path]
        Relative paths of eligible files.  For a single-file src_path,
        the relative path is just the filename (e.g. ``Path("047.py")``).
    """
    base = src_root / src_path
    if not base.exists():
        logger.warning("src_path does not exist: %s", base)
        return []

    # ---- Single-file mode ----
    if base.is_file():
        if since is not None:
            mtime = datetime.fromtimestamp(base.stat().st_mtime, tz=JST)
            if mtime <= since:
                return []
        # Return just the filename as the relative path
        return [Path(base.name)]

    # ---- Directory mode ----
    eligible: List[Path] = []

    for pattern in include_globs:
        for file_path in base.glob(pattern):
            if not file_path.is_file():
                continue

            # Skip always-excluded directories
            if _ALWAYS_EXCLUDE_DIRS & set(file_path.relative_to(base).parts):
                continue

            rel = file_path.relative_to(base)

            # Exclude check
            if exclude_globs and _matches_any_glob(rel, exclude_globs):
                logger.debug("Excluded: %s", rel)
                continue

            # Modification time check (always tz-aware via JST)
            if since is not None:
                mtime = datetime.fromtimestamp(
                    file_path.stat().st_mtime, tz=JST,
                )
                if mtime <= since:
                    continue

            eligible.append(rel)

    # Deduplicate (multiple include globs may match same file)
    seen: set[str] = set()
    deduped: List[Path] = []
    for p in eligible:
        key = str(p)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return sorted(deduped)


def copy_files_to_prod(
    files: List[Path],
    src_root: Path,
    src_path: str,
    dst_root: Path,
    dst_path: str,
) -> int:
    """Copy files from private to production repo.

    Supports both directory and single-file ``src_path``/``dst_path``:

    - **Directory mode**: ``files`` contains paths relative to the
      directory.  Each is appended to ``src_base``/``dst_base``.
    - **Single-file mode**: ``src_path`` and ``dst_path`` are file paths.
      ``files`` contains just ``[Path(filename)]``.  The source is
      ``src_root / src_path`` and the destination is ``dst_root / dst_path``
      directly (not ``dst_base / rel``).

    Preserves relative directory structure in directory mode.  Never
    deletes.

    Returns count of files copied.
    """
    copied = 0
    src_base = src_root / src_path
    dst_base = dst_root / dst_path

    # Detect single-file mode: src_path points to a file, not a directory
    single_file = src_base.is_file()

    for rel in files:
        if single_file:
            src_file = src_base
            dst_file = dst_base
        else:
            src_file = src_base / rel
            dst_file = dst_base / rel

        try:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_file), str(dst_file))
            copied += 1
            logger.debug("Copied: %s → %s", src_file, dst_file)
        except OSError as e:
            logger.error("Failed to copy %s: %s", src_file, e)

    return copied


# ----------------------------------------------------------------
# Row helpers
# ----------------------------------------------------------------

def _parse_globs(raw: Optional[str]) -> List[str]:
    """Parse comma-separated glob patterns from a Notion rich_text value."""
    if not raw or not raw.strip():
        return []
    return [g.strip() for g in raw.split(",") if g.strip()]


def _parse_iso_datetime(raw: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 datetime string into a JST-aware datetime."""
    if not raw or not raw.strip():
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=JST)
        return dt
    except (ValueError, TypeError):
        logger.warning("Could not parse datetime: %r", raw)
        return None


def _row_src_overlaps_changes(row: Dict[str, Any], changed_paths: Set[str]) -> bool:
    """Check if any changed path falls under this row's src_path."""
    src_path = (row.get("src_path") or "").strip().rstrip("/")
    if not src_path:
        return False
    for cp in changed_paths:
        if cp.startswith(src_path + "/") or cp == src_path:
            return True
    return False


def _build_private_commit_msg(
    affected_rows: List[Dict[str, Any]],
) -> str:
    """Build commit message for private phase.

    Format:
    - Single row:  ``Day044 updated``
    - Multi row:   ``Private update: Day044, Day045``
    - With flags:  ``... | NO_PRIVATE_COMMENT: Day045``
    """
    names = [r.get("Name", "?") for r in affected_rows]

    if len(names) == 1:
        msg = f"{names[0]} updated"
    else:
        msg = "Private update: " + ", ".join(names)

    no_comment = [
        r.get("Name", "?")
        for r in affected_rows
        if r.get("private_comment_needed", False)
    ]
    if no_comment:
        msg += " | NO_PRIVATE_COMMENT: " + ", ".join(no_comment)

    return msg


def _build_prod_commit_msg(
    involved_rows: List[Dict[str, Any]],
) -> str:
    """Build commit message for production phase.

    Format: ``Github timer update: Day044, Day045 | NO_PROD_COMMENT: Day045``
    """
    names = [r.get("Name", "?") for r in involved_rows]
    msg = "Github timer update: " + ", ".join(names)

    no_comment = [
        r.get("Name", "?")
        for r in involved_rows
        if r.get("prod_comment_needed", False)
    ]
    if no_comment:
        msg += " | NO_PROD_COMMENT: " + ", ".join(no_comment)

    return msg


# ----------------------------------------------------------------
# Phase A — Private Repo
# ----------------------------------------------------------------

def run_private_phase(
    rows: List[Dict[str, Any]],
    repo: Any,  # GitHubTimerRepo
    now_jst: datetime,
    *,
    dry_run: bool,
) -> None:
    """Phase A: commit & push private repo, update Notion.

    1. Check for changes in PRIVATE_REPO
    2. If changes exist, determine which rows' src_path overlaps
    3. Commit & push
    4. After Phase A completes (even with no git changes):
       - last_checked_at        ← now_jst  (all rows)
       - last_private_run_at    ← now_jst  (all rows)
       - private_comment_needed ← false    (all rows)
    """
    logger.info("=== Phase A: Private Repo ===")

    changed_paths = git_changed_paths(PRIVATE_REPO)
    logger.info("Changed paths in private repo: %d", len(changed_paths))
    for cp in sorted(changed_paths)[:20]:
        logger.debug("  changed: %s", cp)

    # Determine which rows had changes (used for commit message only)
    affected_rows = [r for r in rows if _row_src_overlaps_changes(r, changed_paths)] if changed_paths else []

    # Diagnostic: log overlap evaluation per row
    for row in rows:
        name = row.get("Name", "?")
        src_path = (row.get("src_path") or "").strip().rstrip("/")
        overlaps = row in affected_rows
        logger.debug(
            "  row %-40s src_path=%-50s overlaps_changes=%s",
            name, src_path, overlaps,
        )

    # ---- dry-run reporting ----
    if dry_run:
        logger.info("--- Dry-run: Private Phase Summary ---")
        for row in rows:
            name = row.get("Name", "?")
            is_affected = row in affected_rows
            status = "AFFECTED (file changes)" if is_affected else "no file changes"
            logger.info("  [%s] %s", status, name)
        logger.info("Notion updates that would occur:")
        logger.info("  last_checked_at:         %d rows (all)", len(rows))
        logger.info("  last_private_run_at:     %d rows (all)", len(rows))
        logger.info("  private_comment_needed:  %d rows → unchecked (all)", len(rows))
        if changed_paths and affected_rows:
            msg = _build_private_commit_msg(affected_rows)
            logger.info("Commit message: %s", msg)
        elif changed_paths:
            logger.info("Commit message: Github timer: non-tracked changes")
        else:
            logger.info("No commit (no changes)")
        logger.info("--- End Dry-run Summary ---")
        return

    # ---- live: handle no changes ----
    if not changed_paths:
        logger.info("No changes in private repo — skipping commit")
        # Still update last_checked_at and last_private_run_at
        # (the sync job entered the DB page successfully)
        _update_checked_all(rows, repo, now_jst)
        _update_private_run_all(rows, repo, now_jst)
        return

    # ---- live: commit & push ----
    if affected_rows:
        msg = _build_private_commit_msg(affected_rows)
    else:
        logger.info("Changes found but not in any tracked src_path")
        msg = "Github timer: non-tracked changes"

    git_add_all(PRIVATE_REPO)
    if not git_commit(PRIVATE_REPO, msg):
        logger.error("Private commit failed — skipping push and Notion updates")
        return

    if not git_push(PRIVATE_REPO):
        logger.error("Private push failed — skipping Notion updates")
        return

    # ---- Notion updates (only after successful push) ----
    # Update last_checked_at for all rows
    _update_checked_all(rows, repo, now_jst)

    # Update last_private_run_at and clear private_comment_needed for ALL rows.
    _update_private_run_all(rows, repo, now_jst)


def _update_checked_all(
    rows: List[Dict[str, Any]],
    repo: Any,
    now_jst: datetime,
) -> None:
    """Update last_checked_at for all rows (sync job ran)."""
    for row in rows:
        page_id = row.get("notion_page_id", "")
        name = row.get("Name", "?")
        try:
            repo.update_checked(page_id=page_id, now_jst=now_jst)
            logger.info("Updated last_checked_at for %s", name)
        except Exception as e:
            logger.error("Failed to update last_checked_at for %s: %s", name, e)


def _update_private_run_all(
    rows: List[Dict[str, Any]],
    repo: Any,
    now_jst: datetime,
) -> None:
    """Update last_private_run_at and clear private_comment_needed for ALL rows.

    Called every time Phase A completes successfully (regardless of
    whether git changes existed).  The sync job entered the DB page,
    so every enabled row gets ``last_private_run_at = now_jst`` and
    ``private_comment_needed = false``.
    """
    for row in rows:
        page_id = row.get("notion_page_id", "")
        name = row.get("Name", "?")
        try:
            repo.update_private_changed(page_id=page_id, now_jst=now_jst)
            logger.info("Updated last_private_run_at for %s", name)
            logger.info("Cleared private_comment_needed for %s", name)
        except Exception as e:
            logger.error("Failed to update private timestamps for %s: %s", name, e)


# ----------------------------------------------------------------
# Phase B+C — Production
# ----------------------------------------------------------------

def run_prod_phase(
    rows: List[Dict[str, Any]],
    repo: Any,  # GitHubTimerRepo
    now_jst: datetime,
    *,
    dry_run: bool,
    force: bool,
) -> None:
    """Phase B+C: copy files, commit & push prod repo, update Notion.

    0. Safety check: abort if prod repo is dirty (unless --force)
    1. For each row: validate paths, collect eligible files, copy to prod
    2. Commit & push production repo
    3. Only after successful push: update Notion timestamps
       - last_prod_run_at and prod_comment_needed for all rows
         that passed the prod_due_at gate (regardless of file changes)
    """
    logger.info("=== Phase B+C: Production ===")

    # Safety check: abort if prod repo is dirty
    if git_has_changes(PROD_REPO):
        if force:
            logger.warning(
                "ABORT: Production repository has uncommitted changes. "
                "Proceeding anyway because --force was provided."
            )
        else:
            logger.error(
                "ABORT: Production repository has uncommitted changes. "
                "Use --force to override."
            )
            return

    involved_rows: List[Dict[str, Any]] = []  # rows that had files copied
    eligible_rows: List[Dict[str, Any]] = []  # rows that passed prod_due_at gate
    total_copied = 0
    skipped: List[tuple[str, str]] = []  # (name, reason) for dry-run

    for row in rows:
        name = row.get("Name", "?")
        src_path = (row.get("src_path") or "").strip()
        dst_path = (row.get("dst_path") or "").strip()

        if not src_path or not dst_path:
            reason = "missing src_path or dst_path"
            logger.warning("Row %s: %s — skipping", name, reason)
            skipped.append((name, reason))
            continue

        # Path safety validation
        if not _validate_path_safety(src_path, dst_path, name):
            skipped.append((name, "invalid path (directory traversal)"))
            continue

        # Gate check: prod_due_at
        prod_due_at = _parse_iso_datetime(row.get("prod_due_at"))
        logger.debug(
            "Row %s: prod_due_at=%s  now_jst=%s  gate_passed=%s",
            name,
            prod_due_at.isoformat(timespec="seconds") if prod_due_at else "None",
            now_jst.isoformat(timespec="seconds"),
            prod_due_at is None or now_jst >= prod_due_at,
        )
        if prod_due_at is not None and now_jst < prod_due_at:
            reason = f"prod_due_at in future ({prod_due_at.isoformat(timespec='seconds')})"
            logger.info("Row %s: %s — skipping", name, reason)
            skipped.append((name, reason))
            continue

        # Row passed the prod_due_at gate → mark as eligible for Notion update
        eligible_rows.append(row)

        # Determine include/exclude globs
        include_globs = _parse_globs(row.get("include_globs")) or DEFAULT_INCLUDE_GLOBS
        exclude_globs = _parse_globs(row.get("exclude_globs"))

        # Determine since timestamp
        since = _parse_iso_datetime(row.get("last_prod_run_at"))
        logger.debug(
            "Row %s: last_prod_run_at=%s  include=%s  exclude=%s",
            name,
            since.isoformat(timespec="seconds") if since else "None",
            include_globs, exclude_globs,
        )

        try:
            files = collect_eligible_files(
                PRIVATE_REPO,
                src_path,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
                since=since,
                now_jst=now_jst,
            )
        except Exception as e:
            logger.error("Row %s: error collecting files: %s", name, e)
            skipped.append((name, f"error: {e}"))
            continue

        if not files:
            logger.info("Row %s: no eligible files (already up-to-date)", name)
            continue

        logger.info("Row %s: %d eligible files", name, len(files))
        for f in files[:10]:
            logger.debug("  %s", f)
        if len(files) > 10:
            logger.debug("  ... and %d more", len(files) - 10)

        if dry_run:
            involved_rows.append(row)
            total_copied += len(files)
            continue

        try:
            n = copy_files_to_prod(files, PRIVATE_REPO, src_path, PROD_REPO, dst_path)
            if n > 0:
                involved_rows.append(row)
                total_copied += n
                logger.info("Row %s: copied %d files", name, n)
        except Exception as e:
            logger.error("Row %s: error copying files: %s", name, e)

    logger.info(
        "Production phase: %d eligible (passed gate), %d with files copied, %d skipped",
        len(eligible_rows), len(involved_rows), len(skipped),
    )

    # ---- dry-run reporting ----
    if dry_run:
        logger.info("--- Dry-run: Production Phase Summary ---")
        for row in involved_rows:
            name = row.get("Name", "?")
            logger.info("  [COPY] %s", name)
        for row in eligible_rows:
            if row not in involved_rows:
                name = row.get("Name", "?")
                logger.info("  [GATE-PASSED, no files] %s", name)
        for name, reason in skipped:
            logger.info("  [SKIP] %s — %s", name, reason)
        if involved_rows:
            msg = _build_prod_commit_msg(involved_rows)
            logger.info("Commit message: %s", msg)
            logger.info("Total files to copy: %d", total_copied)
        else:
            logger.info("No files to copy — no commit")
        logger.info("Notion updates (last_prod_run_at): %d rows (all gate-passed)", len(eligible_rows))
        logger.info("--- End Dry-run Summary ---")
        return

    # ---- live: commit & push (only if files were actually copied) ----
    if involved_rows:
        msg = _build_prod_commit_msg(involved_rows)

        git_add_all(PROD_REPO)
        if not git_commit(PROD_REPO, msg):
            logger.error("Prod commit failed — skipping push and Notion updates")
            return

        if not git_push(PROD_REPO):
            logger.error("Prod push failed — skipping Notion updates")
            return

    # ---- Notion updates for ALL rows that passed the prod_due_at gate ----
    # Even rows with no files to copy get their timestamps updated,
    # because the gate condition (now >= prod_due_at) was satisfied.
    if not eligible_rows:
        logger.info("No rows passed prod_due_at gate — no Notion updates")
        return

    for row in eligible_rows:
        page_id = row.get("notion_page_id", "")
        name = row.get("Name", "?")
        try:
            repo.update_prod_run(page_id=page_id, now_jst=now_jst)
            logger.info("Updated last_prod_run_at + cleared prod_comment_needed for %s", name)
        except Exception as e:
            logger.error("Failed to update prod timestamps for %s: %s", name, e)


# ----------------------------------------------------------------
# Phase D — src/ file sync (non-Notion-managed)
# ----------------------------------------------------------------

_SRC_SYNC_INCLUDE_GLOBS = ["**/*.py"]
_SRC_SYNC_EXCLUDE_DIRS = {".ipynb_checkpoints", "__pycache__", ".git", "__pycache__"}


def _collect_notion_covered_paths(rows: List[Dict[str, Any]]) -> Set[str]:
    """Return set of relative paths (under PRIVATE_REPO) covered by Notion rows.

    Includes both directory-level and single-file ``src_path`` values.
    Used to skip files that Phase B+C already handles.
    """
    covered: Set[str] = set()
    for row in rows:
        src_path = (row.get("src_path") or "").strip().rstrip("/")
        if src_path:
            covered.add(src_path)
    return covered


def _is_under_any(rel_str: str, prefixes: Set[str]) -> bool:
    """Return True if *rel_str* is under (or equal to) any prefix."""
    for prefix in prefixes:
        if rel_str == prefix or rel_str.startswith(prefix + "/"):
            return True
    return False


def _scan_src_py_files(repo_root: Path) -> List[Path]:
    """Return sorted list of .py files under ``repo_root/src/``.

    Excludes ``__pycache__``, ``.ipynb_checkpoints``, ``.git``.
    Paths returned are relative to *repo_root* (e.g. ``src/config.py``).
    """
    src_dir = repo_root / "src"
    if not src_dir.is_dir():
        return []

    results: List[Path] = []
    for py_file in src_dir.rglob("*.py"):
        if not py_file.is_file():
            continue
        rel = py_file.relative_to(repo_root)
        # skip excluded dirs
        if _SRC_SYNC_EXCLUDE_DIRS & set(rel.parts):
            continue
        results.append(rel)
    return sorted(results)


def run_src_sync_phase(
    rows: List[Dict[str, Any]],
    *,
    dry_run: bool,
    force: bool,
) -> None:
    """Phase D: sync ``src/`` .py files from private → production.

    For each ``.py`` file under ``PRIVATE_REPO/src/``:
    1. Skip if the file is covered by an enabled Notion row's ``src_path``
    2. If the file is **missing** in production → copy
    3. If the file **exists** but content differs and private is newer → copy
    4. Otherwise → skip (up-to-date)

    This phase only touches files **not** managed by Notion rows, so it
    complements Phase B+C rather than duplicating it.
    """
    logger.info("=== Phase D: src/ File Sync ===")

    # Determine which paths Notion rows already cover
    notion_covered = _collect_notion_covered_paths(rows)
    if notion_covered:
        logger.info("Notion-covered paths (%d): %s",
                     len(notion_covered), ", ".join(sorted(notion_covered)))

    # Safety: check prod repo is clean (same as Phase B+C)
    if not dry_run and git_has_changes(PROD_REPO):
        if force:
            logger.warning(
                "Production repo has uncommitted changes. "
                "Proceeding anyway because --force was provided."
            )
        else:
            logger.error(
                "Production repo has uncommitted changes — skipping src-sync. "
                "Use --force to override."
            )
            return

    # Scan private src/
    private_files = _scan_src_py_files(PRIVATE_REPO)
    logger.info("Private src/ .py files: %d", len(private_files))

    # Categorise
    to_copy: List[Path] = []       # missing or outdated in prod
    skipped_notion: List[Path] = []
    skipped_current: List[Path] = []

    for rel in private_files:
        rel_str = str(rel)

        # Skip if covered by a Notion row
        if _is_under_any(rel_str, notion_covered):
            skipped_notion.append(rel)
            continue

        priv_file = PRIVATE_REPO / rel
        prod_file = PROD_REPO / rel

        if not prod_file.exists():
            # Missing in production → copy
            to_copy.append(rel)
            logger.debug("  MISSING in prod: %s", rel)
            continue

        # Both exist — compare content
        if filecmp.cmp(str(priv_file), str(prod_file), shallow=False):
            # Identical content
            skipped_current.append(rel)
            continue

        # Content differs — check if private is newer
        priv_mtime = priv_file.stat().st_mtime
        prod_mtime = prod_file.stat().st_mtime
        if priv_mtime > prod_mtime:
            to_copy.append(rel)
            logger.debug("  OUTDATED in prod: %s (priv %.0f > prod %.0f)",
                         rel, priv_mtime, prod_mtime)
        else:
            # Prod is newer or same mtime but different content — skip (safety)
            skipped_current.append(rel)
            logger.debug("  SKIP (prod newer/same): %s", rel)

    # ---- Summary ----
    logger.info("src-sync: %d to copy, %d skipped (Notion-managed), %d up-to-date",
                len(to_copy), len(skipped_notion), len(skipped_current))

    if not to_copy:
        logger.info("Nothing to sync — production src/ is up-to-date")
        return

    for f in to_copy:
        prod_exists = (PROD_REPO / f).exists()
        tag = "UPDATE" if prod_exists else "NEW"
        logger.info("  [%s] %s", tag, f)

    # ---- Dry-run ----
    if dry_run:
        logger.info("--- Dry-run: src-sync would copy %d files ---", len(to_copy))
        return

    # ---- Live: copy files ----
    copied = 0
    for rel in to_copy:
        src_file = PRIVATE_REPO / rel
        dst_file = PROD_REPO / rel
        try:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_file), str(dst_file))
            copied += 1
            logger.debug("Copied: %s", rel)
        except OSError as e:
            logger.error("Failed to copy %s: %s", rel, e)

    logger.info("src-sync: copied %d / %d files", copied, len(to_copy))

    if copied == 0:
        return

    # ---- Commit & push production ----
    msg = f"src-sync: {copied} file(s) synced from private"
    git_add_all(PROD_REPO)
    if not git_commit(PROD_REPO, msg):
        logger.error("src-sync commit failed — skipping push")
        return

    if not git_push(PROD_REPO):
        logger.error("src-sync push failed")
        return

    logger.info("src-sync: committed and pushed %d files", copied)


# ================================================================
# CLI
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SCRIPT_NAME,
        description="Notion-driven GitHub timer sync: private → production",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Read-only mode (default). No git commit, push, Notion updates, or file copies.",
    )
    mode.add_argument(
        "--run",
        dest="dry_run",
        action="store_false",
        help="Live mode. Commits, pushes, copies files, updates Notion.",
    )
    p.add_argument(
        "--only",
        choices=["private", "prod", "src-sync", "all"],
        default="all",
        help="Which phase(s) to run (default: all).",
    )
    p.add_argument(
        "--name-contains",
        default="",
        help="Filter rows by name substring.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow running production phase even if prod repo is dirty.",
    )
    p.add_argument(
        "--env",
        default="notebooks/env.txt",
        help="Path to env.txt file (default: notebooks/env.txt).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG level on console).",
    )
    return p


def main(argv: List[str] | None = None) -> None:
    """Entry point: parse args, acquire lock, load env, fetch rows, run phases."""
    args = build_parser().parse_args(argv)
    setup_dual_logging(args.verbose)

    mode_label = "DRY-RUN" if args.dry_run else "LIVE"
    logger.info("=" * 60)
    logger.info("%s — %s mode", SCRIPT_NAME, mode_label)
    logger.info("=" * 60)

    # ---- execution lock ----
    lock = _ExecutionLock(LOCK_PATH)
    if not lock.acquire():
        logger.info("Another instance is already running. Exiting.")
        return

    try:
        _run_with_lock(args, mode_label)
    finally:
        lock.release()


def _run_with_lock(args: argparse.Namespace, mode_label: str) -> None:
    """Core logic, called while holding the execution lock."""

    # ---- load env (consistent with 052 pattern) ----
    load_env(args.env)
    logger.info("Env loaded from %s", args.env)

    # ---- Notion client + resolver ----
    client = build_notion_client_from_env()
    resolver = NotionDataSourceResolver(client)

    db_id = get_db_id("NOTION_GITHUB_TIMER_DB_ID")
    resolved = resolver.resolve_once(name="GITHUB_TIMER", database_id=db_id)
    logger.info(
        "Resolved GITHUB_TIMER_DB: db=%s ds=%s",
        resolved.database_id[:12] + "...",
        resolved.data_source_id[:12] + "...",
    )

    repo = GitHubTimerRepo(
        client=client,
        database_id=resolved.database_id,
        data_source_id=resolved.data_source_id,
    )

    # ---- schema validation ----
    repo.ensure_schema()

    # ---- fetch enabled rows ----
    rows = repo.fetch_enabled_rows(name_contains=args.name_contains)
    if not rows:
        logger.info("No enabled rows found — nothing to do")
        return

    logger.info("Enabled rows: %d", len(rows))
    for r in rows:
        logger.info("  %s (src=%s dst=%s)", r.get("Name"), r.get("src_path"), r.get("dst_path"))

    # ---- timestamp (always tz-aware JST) ----
    now_jst = datetime.now(JST)
    logger.info("Now (JST): %s", now_jst.isoformat(timespec="seconds"))

    # ---- run phases ----
    if args.only in ("private", "all"):
        run_private_phase(rows, repo, now_jst, dry_run=args.dry_run)

    if args.only in ("prod", "all"):
        run_prod_phase(rows, repo, now_jst, dry_run=args.dry_run, force=args.force)

    if args.only in ("src-sync", "all"):
        run_src_sync_phase(rows, dry_run=args.dry_run, force=args.force)

    logger.info("=" * 60)
    logger.info("%s completed (%s)", SCRIPT_NAME, mode_label)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
