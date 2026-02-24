# src/pdf/pipeline.py
"""Single-PDF processing pipeline and batch orchestration.

Ported from notebook 031 cells 09 (single PDF) and 10 (batch loop).
"""
from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.pdf.drive import upload_pdf_and_slide
from src.pdf.metadata import extract_pdf_metadata
from src.pdf.notion_adapter import (
    check_for_duplicate,
    create_notion_paper_record,
)
from src.pdf.slides import generate_slide_artifact

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Config flags
# ----------------------------------------------------------------

SKIP_SLIDE_IF_DUPLICATE = True
REQUIRE_OPENAI_FOR_FIELDS = False
ENABLE_DRIVE_UPLOAD = True


# ----------------------------------------------------------------
# Drive upload + Notion link update
# ----------------------------------------------------------------


def _maybe_drive_upload_and_update_notion(
    *,
    status: str,
    page_id: Optional[str],
    pdf_path: Path,
    slide_path: Optional[Path],
    metadata: Dict[str, Any],
    drive_service: Any,
    drive_folder_id: str,
    update_paper_links_fn: Optional[Callable],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ran": False,
        "uploaded": None,
        "pdf_url": None,
        "slide_url": None,
        "notion_update_result": None,
    }

    if status != "success" or not page_id:
        return out
    if drive_service is None or not drive_folder_id:
        return out

    base_title = (metadata.get("title") or pdf_path.stem).strip() or "paper"
    uploaded = upload_pdf_and_slide(
        drive_service=drive_service,
        pdf_path=pdf_path,
        slide_path=slide_path,
        drive_folder_id=drive_folder_id,
        base_title=base_title,
    )

    pdf_url = (uploaded.get("pdf") or {}).get("url")
    slide_url = (uploaded.get("slide") or {}).get("url")

    if update_paper_links_fn and callable(update_paper_links_fn):
        upd = update_paper_links_fn(
            page_id=page_id,
            pdf_link=pdf_url,
            slide1_url=slide_url,
        )
        out["notion_update_result"] = upd

    out.update(ran=True, uploaded=uploaded, pdf_url=pdf_url, slide_url=slide_url)
    return out


# ----------------------------------------------------------------
# Single PDF processing
# ----------------------------------------------------------------


def process_single_pdf(
    pdf_path: Path,
    *,
    run_id: str,
    inbox_path: Path,
    processed_path: Path,
    failed_path: Path,
    slides_path: Path,
    artifacts_path: Path,
    openai_client: Any = None,
    openai_model: str = "gpt-4.1-mini",
    openai_temperature: float = 0.2,
    gemini_client: Any = None,
    drive_service: Any = None,
    drive_folder_id: str = "",
    # Notion wrappers
    create_paper_inbox_fn: Optional[Callable] = None,
    find_duplicate_paper_fn: Optional[Callable] = None,
    find_duplicate_by_doi_fn: Optional[Callable] = None,
    find_duplicate_by_arxiv_id_fn: Optional[Callable] = None,
    find_duplicate_by_title_fn: Optional[Callable] = None,
    update_paper_links_fn: Optional[Callable] = None,
    # Options
    dry_run: bool = False,
    enable_drive_upload: bool = True,
    skip_slide_if_duplicate: bool = True,
) -> Dict[str, Any]:
    """Process a single PDF through the full pipeline.

    Stages: metadata -> dedup -> slide -> notion -> drive -> move file.
    """
    start_time = time.time()

    result: Dict[str, Any] = {
        "pdf_name": pdf_path.name,
        "status": "failed",
        "page_id": None,
        "duplicate_info": None,
        "slide_path": None,
        "moved_to": None,
        "errors": [],
        "duration_seconds": 0.0,
        "timestamp": None,
        "dedup_key": None,
        "source_uid": None,
        "notion_payload_preview": None,
        "drive": None,
    }

    logger.info("[%s] Processing PDF: %s", run_id, pdf_path.name)

    metadata: Optional[Dict[str, Any]] = None
    slide_path_result: Optional[Path] = None

    try:
        # --- Stage 1: Metadata extraction ---
        logger.info("[%s] Stage 1/5: Extracting metadata...", run_id)
        metadata = extract_pdf_metadata(
            pdf_path,
            openai_client=openai_client,
            openai_model=openai_model,
            openai_temperature=openai_temperature,
        )

        if metadata.get("extraction_errors"):
            for err in metadata["extraction_errors"]:
                result["errors"].append(f"metadata: {err}")

        if not (metadata.get("title") or "").strip():
            result["errors"].append("No title resolved.")
            return result

        # --- Stage 2: Dedup ---
        logger.info("[%s] Stage 2/5: Checking for duplicates...", run_id)
        try:
            dup = check_for_duplicate(
                metadata,
                find_duplicate_paper=find_duplicate_paper_fn,
                find_duplicate_by_doi=find_duplicate_by_doi_fn,
                find_duplicate_by_arxiv_id=find_duplicate_by_arxiv_id_fn,
                find_duplicate_by_title=find_duplicate_by_title_fn,
            )
            if dup.get("is_duplicate"):
                result["status"] = "duplicate"
                result["duplicate_info"] = dup
                logger.info(
                    "[%s] DUPLICATE type=%s page_id=%s",
                    run_id, dup.get("match_type"), dup.get("page_id"),
                )
            if dup.get("error"):
                result["errors"].append(f"dedup: {dup['error']}")
        except Exception as e:
            result["errors"].append(f"dedup exception: {e}")
            logger.warning("[%s] dedup exception: %s", run_id, e)

        # --- Stage 3: Slide ---
        if result["status"] == "duplicate" and skip_slide_if_duplicate:
            logger.info("[%s] Stage 3/5: Skipping slide (duplicate)", run_id)
        elif gemini_client is None:
            logger.info("[%s] Stage 3/5: Skipping slide (no Gemini client)", run_id)
        elif dry_run:
            logger.info("[%s] Stage 3/5: Skipping slide (dry-run)", run_id)
        else:
            logger.info("[%s] Stage 3/5: Generating slide...", run_id)
            try:
                sp = generate_slide_artifact(
                    pdf_path,
                    metadata,
                    slides_path=slides_path,
                    gemini_client=gemini_client,
                    openai_client=openai_client,
                    openai_model=openai_model,
                    openai_temperature=openai_temperature,
                )
                if sp and sp.exists():
                    slide_path_result = sp
                    result["slide_path"] = str(sp)
                    logger.info("[%s] Slide OK: %s", run_id, sp.name)
                else:
                    result["errors"].append("slide: generation returned None")
            except Exception as e:
                result["errors"].append(f"slide exception: {e}")
                logger.warning("[%s] slide exception: %s", run_id, e)

        # --- Stage 4: Notion ---
        if result["status"] == "duplicate":
            logger.info("[%s] Stage 4/5: Skipping Notion (duplicate)", run_id)
        elif dry_run:
            logger.info("[%s] Stage 4/5: Skipping Notion (dry-run)", run_id)
            result["status"] = "success"
        elif create_paper_inbox_fn is None:
            logger.warning("[%s] Stage 4/5: Skipping Notion (no create_paper_inbox)", run_id)
            result["errors"].append("notion: create_paper_inbox not available")
            result["status"] = "success"
        else:
            logger.info("[%s] Stage 4/5: Creating Notion record...", run_id)
            try:
                creation = create_notion_paper_record(
                    metadata=metadata,
                    pdf_path=pdf_path,
                    slide_path=slide_path_result,
                    artifacts_path=artifacts_path,
                    run_id=run_id,
                    create_paper_inbox_fn=create_paper_inbox_fn,
                    openai_client=openai_client,
                    openai_model=openai_model,
                    openai_temperature=openai_temperature,
                )
                result["dedup_key"] = creation.get("dedup_key")
                result["source_uid"] = creation.get("source_uid")
                result["notion_payload_preview"] = creation.get("payload_preview")

                if creation.get("success"):
                    result["status"] = "success"
                    result["page_id"] = creation.get("page_id")
                else:
                    result["errors"].append(f"notion: {creation.get('error')}")
                    result["status"] = "failed"
            except Exception as e:
                result["errors"].append(f"notion exception: {e}")
                result["status"] = "failed"

        # --- Stage 4.5: Drive upload ---
        if metadata is None:
            metadata = {}
        if enable_drive_upload and not dry_run:
            try:
                drive_out = _maybe_drive_upload_and_update_notion(
                    status=result["status"],
                    page_id=result.get("page_id"),
                    pdf_path=pdf_path,
                    slide_path=slide_path_result,
                    metadata=metadata,
                    drive_service=drive_service,
                    drive_folder_id=drive_folder_id,
                    update_paper_links_fn=update_paper_links_fn,
                )
                result["drive"] = drive_out
                if drive_out.get("ran"):
                    logger.info("[%s] Drive upload & Notion link update OK", run_id)
            except Exception as e:
                result["errors"].append(f"drive exception: {e}")
                logger.warning("[%s] drive exception: %s", run_id, e)

        # --- Stage 5: Move file ---
        if dry_run:
            logger.info("[%s] Stage 5/5: Skipping file move (dry-run)", run_id)
        else:
            logger.info("[%s] Stage 5/5: Moving PDF...", run_id)
            try:
                target_dir = processed_path if result["status"] in ("success", "duplicate") else failed_path
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_path = target_dir / f"{ts}_{pdf_path.name}"
                shutil.move(str(pdf_path), str(target_path))
                result["moved_to"] = str(target_path)
                logger.info("[%s] Moved to: %s", run_id, target_path)
            except Exception as e:
                result["errors"].append(f"move exception: {e}")
                logger.error("[%s] move exception: %s", run_id, e)

    finally:
        result["duration_seconds"] = round(time.time() - start_time, 2)
        result["timestamp"] = datetime.now().isoformat()
        logger.info(
            "[%s] Done: %s status=%s duration=%.2fs errors=%d",
            run_id, pdf_path.name, result["status"],
            result["duration_seconds"], len(result["errors"]),
        )

    return result


# ----------------------------------------------------------------
# Batch processing
# ----------------------------------------------------------------


def run_batch(
    *,
    inbox_path: Path,
    processed_path: Path,
    failed_path: Path,
    slides_path: Path,
    artifacts_path: Path,
    run_id: str,
    limit: int = 0,
    # Clients
    openai_client: Any = None,
    openai_model: str = "gpt-4.1-mini",
    openai_temperature: float = 0.2,
    gemini_client: Any = None,
    drive_service: Any = None,
    drive_folder_id: str = "",
    # Notion wrappers
    create_paper_inbox_fn: Optional[Callable] = None,
    find_duplicate_paper_fn: Optional[Callable] = None,
    find_duplicate_by_doi_fn: Optional[Callable] = None,
    find_duplicate_by_arxiv_id_fn: Optional[Callable] = None,
    find_duplicate_by_title_fn: Optional[Callable] = None,
    update_paper_links_fn: Optional[Callable] = None,
    # Options
    dry_run: bool = False,
    enable_drive_upload: bool = True,
) -> Dict[str, Any]:
    """Process all PDFs in the inbox directory.

    Returns
    -------
    dict
        Run summary with statistics and per-file results.
    """
    # Discover PDFs
    candidates = list(inbox_path.glob("*.pdf")) + list(inbox_path.glob("*.PDF"))
    pdf_files = sorted(
        {p.expanduser().resolve() for p in candidates if p.is_file()},
        key=lambda p: p.name.lower(),
    )

    if limit > 0:
        pdf_files = pdf_files[:limit]

    logger.info("[%s] Found %d PDF(s) in inbox: %s", run_id, len(pdf_files), inbox_path)

    run_summary: Dict[str, Any] = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "total_files": len(pdf_files),
        "processed": 0,
        "success": 0,
        "duplicates": 0,
        "failures": 0,
        "total_errors": 0,
        "total_duration_seconds": 0.0,
        "results": [],
    }

    batch_start = time.time()

    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info("[%s] [%d/%d] %s", run_id, i, len(pdf_files), pdf_path.name)

        try:
            result = process_single_pdf(
                pdf_path,
                run_id=run_id,
                inbox_path=inbox_path,
                processed_path=processed_path,
                failed_path=failed_path,
                slides_path=slides_path,
                artifacts_path=artifacts_path,
                openai_client=openai_client,
                openai_model=openai_model,
                openai_temperature=openai_temperature,
                gemini_client=gemini_client,
                drive_service=drive_service,
                drive_folder_id=drive_folder_id,
                create_paper_inbox_fn=create_paper_inbox_fn,
                find_duplicate_paper_fn=find_duplicate_paper_fn,
                find_duplicate_by_doi_fn=find_duplicate_by_doi_fn,
                find_duplicate_by_arxiv_id_fn=find_duplicate_by_arxiv_id_fn,
                find_duplicate_by_title_fn=find_duplicate_by_title_fn,
                update_paper_links_fn=update_paper_links_fn,
                dry_run=dry_run,
                enable_drive_upload=enable_drive_upload,
            )
        except Exception as e:
            logger.error("[%s] Unexpected error processing %s: %s", run_id, pdf_path.name, e)
            result = {
                "pdf_name": pdf_path.name,
                "status": "failed",
                "page_id": None,
                "duplicate_info": None,
                "slide_path": None,
                "moved_to": None,
                "errors": [f"Unexpected error: {e}"],
                "duration_seconds": 0.0,
                "timestamp": datetime.now().isoformat(),
            }

        run_summary["results"].append(result)
        run_summary["processed"] += 1

        st = result.get("status")
        if st == "success":
            run_summary["success"] += 1
        elif st == "duplicate":
            run_summary["duplicates"] += 1
        else:
            run_summary["failures"] += 1

        errs = result.get("errors") or []
        run_summary["total_errors"] += len(errs)
        run_summary["total_duration_seconds"] += float(result.get("duration_seconds") or 0.0)

    batch_duration = round(time.time() - batch_start, 2)
    run_summary["end_time"] = datetime.now().isoformat()
    run_summary["batch_duration_seconds"] = batch_duration
    run_summary["total_duration_seconds"] = round(run_summary["total_duration_seconds"], 2)
    run_summary["avg_duration_per_file"] = round(
        (run_summary["total_duration_seconds"] / run_summary["processed"])
        if run_summary["processed"]
        else 0.0,
        2,
    )

    logger.info(
        "[%s] BATCH COMPLETE | files=%d success=%d dup=%d fail=%d errors=%d duration=%.2fs",
        run_id,
        run_summary["total_files"],
        run_summary["success"],
        run_summary["duplicates"],
        run_summary["failures"],
        run_summary["total_errors"],
        batch_duration,
    )

    return run_summary
