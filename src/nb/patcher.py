# src/nb/patcher.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import hashlib
import shutil
import time

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


@dataclass
class PatchResult:
    ok: bool
    error: Optional[str] = None
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None
    diff_path: Optional[str] = None


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _trim(s: Any, n: int = 1200) -> str:
    t = "" if s is None else str(s)
    return t if len(t) <= n else t[:n] + "...(truncated)"


def patch_cell_source_with_artifacts(
    *,
    notebook_path: str,
    new_source: str,
    artifacts,
    backup_dir,
    backup_tag: str,
    mode: str = "REPLACE",
    cell_index=None,
    cell_type: str = "code",
    diff_name_hint: str | None = None,
) -> PatchResult:
    mode_u = (mode or "REPLACE").upper()

    # ✅ HARD GUARD: REPLACE/INSERT は cell_index 必須
    if mode_u in ("REPLACE", "INSERT") and cell_index is None:
        return PatchResult(ok=False, error=f"patcher: cell_index is required for mode={mode_u}")

    # ✅ normalize to int (fail fast with clear error)
    if cell_index is not None:
        try:
            cell_index = int(cell_index)
        except Exception:
            return PatchResult(ok=False, error=f"patcher: invalid cell_index={cell_index!r}")

    try:
        nb_path = Path(notebook_path).expanduser().resolve()
        if not nb_path.exists():
            raise FileNotFoundError(f"notebook not found: {nb_path}")

        # ---- BACKUP (pre-patch snapshot) ----
        backup_dir_p = Path(backup_dir).expanduser().resolve()
        backup_dir_p.mkdir(parents=True, exist_ok=True)
        if backup_tag:
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_name = f"{nb_path.stem}.{backup_tag}.{ts}.bak.ipynb"
            shutil.copy2(nb_path, backup_dir_p / backup_name)

        # ---- READ NOTEBOOK ----
        nb = nbformat.read(str(nb_path), as_version=4)

        mode_u = (mode or "REPLACE").upper()
        cell_type_l = (cell_type or "code").lower()

        if not str(new_source or "").strip():
            raise ValueError("new_source is empty")

        new_cell = new_markdown_cell(new_source) if cell_type_l == "markdown" else new_code_cell(new_source)

        before_hash: Optional[str] = None
        after_hash: Optional[str] = None
        applied_index: Optional[int] = None

        # ---- APPLY PATCH ----
        if mode_u == "REPLACE":
            # cell_index は上で int に正規化済み（None も除外済み）
            idx = int(cell_index)
            if idx < 0:
                raise IndexError(f"cell_index out of range: {idx} (cells={len(nb.cells)})")

            # ✅ before_hash は “idx” で取る（applied_index はまだ None）
            if idx < len(nb.cells):
                prev_src = (
                    nb.cells[idx].get("source")
                    if isinstance(nb.cells[idx], dict)
                    else getattr(nb.cells[idx], "source", "")
                ) or ""
                before_hash = _hash(prev_src)
            else:
                before_hash = None

            # ✅ Auto-extend notebook cells if needed
            if idx >= len(nb.cells):
                while len(nb.cells) <= idx:
                    nb.cells.append(new_markdown_cell("") if cell_type_l == "markdown" else new_code_cell(""))

            nb.cells[idx] = new_cell
            applied_index = idx

        elif mode_u == "APPEND":
            nb.cells.append(new_cell)
            applied_index = len(nb.cells) - 1

        elif mode_u == "INSERT":
            # cell_index は上で int に正規化済み（None も除外済み）
            idx = int(cell_index)
            if idx < 0 or idx > len(nb.cells):
                raise IndexError(f"cell_index out of range: {idx} (cells={len(nb.cells)})")
            nb.cells.insert(idx, new_cell)
            applied_index = idx

        else:
            raise ValueError(f"Unknown patch mode: {mode_u}")

        # Ensure every cell has an id (nbformat v4 expects stable ids)
        for c in nb.cells:
            if "id" not in c:
                c["id"] = nbformat.uuid4()

        # Compute after_hash from actual notebook cell content (post-normalization)
        if applied_index is not None and 0 <= applied_index < len(nb.cells):
            final_src = (nb.cells[applied_index].get("source") if isinstance(nb.cells[applied_index], dict) else getattr(nb.cells[applied_index], "source", "")) or ""
            after_hash = _hash(final_src)
        else:
            # fallback (shouldn't happen)
            after_hash = _hash(new_source)

        # ---- WRITE NOTEBOOK ----
        nbformat.write(nb, str(nb_path))

        # ---- ARTIFACT MARKERS ----
        base = getattr(artifacts, "base_dir", None)
        diff_path = None
        if base is not None:
            base_dir = Path(base)
            base_dir.mkdir(parents=True, exist_ok=True)

            hint = (diff_name_hint or "patch").strip().replace(" ", "_")
            ts = time.strftime("%Y%m%d_%H%M%S")
            marker_path = base_dir / f"{hint}.cell{applied_index}. {ts}.applied".replace(" ", "_")

            marker_path.write_text(
                "\n".join(
                    [
                        "PATCH_APPLIED",
                        f"notebook_path={nb_path}",
                        f"mode={mode_u}",
                        f"cell_index={applied_index}",
                        f"cell_type={cell_type_l}",
                        f"before_hash={before_hash or ''}",
                        f"after_hash={after_hash}",
                        "---- new_source_preview ----",
                        _trim(new_source, 2000),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            diff_path = str(marker_path)

        return PatchResult(
            ok=True,
            before_hash=before_hash,
            after_hash=after_hash,
            diff_path=diff_path,
        )

    except Exception as e:
        return PatchResult(ok=False, error=f"{type(e).__name__}: {e}")
