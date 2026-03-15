#!/usr/bin/env python
# src/scripts/031_pdf_inbox_processor.py
"""PDF Inbox Processor — batch-process PDFs into Notion Papers DB.

Pipeline:
1. Scan inbox directory for PDF files
2. For each PDF:
   a. Extract metadata (title, authors, DOI, arXiv ID) with local heuristics + OpenAI repair
   b. Check for duplicates in Notion (DOI -> arXiv -> title)
   c. Generate one-slide visual summary (OpenAI structured text + Gemini image)
   d. Create Notion Papers record (status=INBOX) with rich fields
   e. Upload PDF + slide to Google Drive, update Notion with links
   f. Move file to processed/ or failed/
3. Generate run summary and Slack snippet

Ported from notebooks/031_pdf_inbox_processor.ipynb.

Usage::

    # Show help
    python -m src.scripts.031_pdf_inbox_processor --help

    # Dry-run (no Notion writes, no Drive uploads, no file moves)
    python -m src.scripts.031_pdf_inbox_processor --dry-run

    # Process with custom paths
    python -m src.scripts.031_pdf_inbox_processor --inbox-dir data/downloads --artifacts-dir artifacts

    # Limit to first 3 PDFs
    python -m src.scripts.031_pdf_inbox_processor --limit 3

    # Skip Drive upload / Skip slide generation
    python -m src.scripts.031_pdf_inbox_processor --no-drive --no-slides

    # Custom Drive auth paths
    python -m src.scripts.031_pdf_inbox_processor --drive-token notebooks/token.json --drive-credentials notebooks/client_secret.json

    # Require Drive (fail-fast if Drive auth fails)
    python -m src.scripts.031_pdf_inbox_processor --require-drive

    # Verbose logging
    python -m src.scripts.031_pdf_inbox_processor -v
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env, setup_logging

logger = logging.getLogger("031_pdf_inbox_processor")

SCRIPT_NAME = "031_pdf_inbox_processor"


# ================================================================
# Client initialization
# ================================================================


def _init_openai_client(model: str = "gpt-4.1-mini"):
    """Initialize OpenAI client from environment."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; LLM features disabled")
        return None, model, 0.2

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    model = os.getenv("OPENAI_MODEL", model).strip()
    logger.info("OpenAI client initialized (model=%s)", model)
    return client, model, temperature


def _init_gemini_client():
    """Initialize Gemini client from environment."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; slide generation disabled")
        return None

    # Clear GOOGLE_API_KEY to avoid conflicts
    os.environ.pop("GOOGLE_API_KEY", None)

    from google import genai

    client = genai.Client(api_key=api_key)
    logger.info("Gemini client initialized")
    return client


def _init_drive_service(
    *,
    drive_token: str | None = None,
    drive_credentials: str | None = None,
    require: bool = False,
):
    """Initialize Google Drive service.

    Parameters
    ----------
    drive_token : str or None
        Explicit path to token.json. Falls back to DRIVE_TOKEN_PATH env,
        then notebooks/token.json.
    drive_credentials : str or None
        Explicit path to client_secret.json. Falls back to
        DRIVE_CLIENT_SECRET_PATH env, then notebooks/client_secret.json.
    require : bool
        If True, raise on failure instead of returning None.

    Returns
    -------
    google Drive service or None
    """
    from src.pdf.drive import (
        build_drive_service,
        resolve_client_secret_path,
        resolve_token_path,
    )

    tp = resolve_token_path(drive_token)
    csp = resolve_client_secret_path(drive_credentials)

    logger.info("Drive: token_path=%s (exists=%s)", tp, tp.exists())
    logger.info("Drive: client_secret_path=%s (exists=%s)", csp, csp.exists())

    if not tp.exists() and not csp.exists():
        msg = (
            f"Drive auth files not found: token={tp}, secret={csp}. "
            "Use --drive-token / --drive-credentials or set DRIVE_TOKEN_PATH / DRIVE_CLIENT_SECRET_PATH."
        )
        if require:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        logger.info("Drive: DISABLED")
        return None

    try:
        service = build_drive_service(token_path=tp, client_secret_path=csp)
        logger.info("Drive: ENABLED (service built)")
        return service
    except Exception as e:
        msg = f"Drive auth failed: {e}"
        if require:
            raise RuntimeError(msg) from e
        logger.warning(msg)
        logger.info("Drive: DISABLED")
        return None


def _verify_drive(drive_service, drive_folder_id: str, *, require: bool = False) -> bool:
    """Run Drive preflight check. Returns True if ok."""
    from src.pdf.drive import verify_drive_connection

    check = verify_drive_connection(drive_service, drive_folder_id)
    if check["ok"]:
        return True

    msg = f"Drive preflight failed: {check.get('error')}"
    if require:
        raise RuntimeError(msg)
    logger.warning(msg)
    return False


def _init_notion_wrappers():
    """Initialize Notion client and return wrapper functions.

    Uses build_notion_client_from_env() + NotionDataSourceResolver to build
    wrappers compatible with src.pdf.pipeline / src.pdf.notion_adapter.
    No dependency on 029 notebook.
    """
    wrappers = {
        "create_paper_inbox": None,
        "find_duplicate_paper": None,
        "find_duplicate_by_doi": None,
        "find_duplicate_by_arxiv_id": None,
        "find_duplicate_by_title": None,
        "update_paper_links": None,
    }

    try:
        notion_token = os.getenv("NOTION_TOKEN", "").strip() or os.getenv("NOTION_API_KEY", "").strip()
        if not notion_token:
            logger.warning("NOTION_TOKEN not set; Notion features disabled")
            return wrappers

        lit_db_id = os.getenv("NOTION_LIT_DB_ID", "").strip()
        if not lit_db_id:
            logger.warning("NOTION_LIT_DB_ID not set; Notion features disabled")
            return wrappers

        from src.notion import build_notion_client_from_env, NotionDataSourceResolver

        client = build_notion_client_from_env()
        resolver = NotionDataSourceResolver(client)
        resolved = resolver.resolve_once(name="LIT_DB", database_id=lit_db_id)
        data_source_id = resolved.data_source_id

        logger.info(
            "Notion client initialized: LIT_DB database_id=%s data_source_id=%s",
            lit_db_id[:12], data_source_id[:12],
        )

        # --- Notion property helpers ---
        def _rich_text(text):
            if not text:
                return {"rich_text": []}
            return {"rich_text": [{"text": {"content": str(text)[:2000]}}]}

        def _title(text):
            return {"title": [{"text": {"content": str(text or "")[:2000]}}]}

        def _select(value):
            if not value:
                return {"select": None}
            return {"select": {"name": str(value)}}

        def _multi_select(values):
            if not values:
                return {"multi_select": []}
            return {"multi_select": [{"name": str(v)} for v in values if v]}

        def _url(value):
            if not value:
                return {"url": None}
            return {"url": str(value)}

        def _date(value):
            if not value:
                return {"date": None}
            return {"date": {"start": str(value)}}

        # --- Wrapper: create_paper_inbox ---
        def create_paper_inbox(
            *, name, authors_year="", pdf_link=None, tags=None,
            status="INBOX", pdf_status="LOCAL", dedup_key="",
            source_uid="", run_id="", slide1_url=None, extra=None,
        ):
            notion_fields = (extra or {}).get("notion_fields") or {}

            properties = {
                "Name": _title(name),
                "Authors & Year": _rich_text(authors_year),
                "Status": _select(status),
                "PDF Status": _select(pdf_status),
                "Dedup Key": _rich_text(dedup_key),
                "Source UID": _rich_text(source_uid),
                "Run ID": _rich_text(run_id),
                "Ingested At": _date(datetime.today().date().isoformat()),
            }
            if pdf_link:
                properties["PDF Link"] = _url(pdf_link)
            if slide1_url:
                properties["Slide 1 URL"] = _url(slide1_url)
            if tags:
                properties["Tags"] = _multi_select(tags)

            for field, prop in [
                ("core_idea", "Core Idea"), ("findings", "Findings"),
                ("methods", "Methods"), ("notes", "Notes"),
                ("datasets", "Datasets"),
            ]:
                val = notion_fields.get(field)
                if val:
                    properties[prop] = _rich_text(val)
            if notion_fields.get("type"):
                properties["Type"] = _rich_text(notion_fields["type"])
            if notion_fields.get("source"):
                properties["Source"] = _rich_text(notion_fields["source"])

            result = client.create_page(parent_db_id=lit_db_id, properties=properties)
            page_id = result.get("id")
            return {"page_id": page_id, "id": page_id}

        # --- Wrapper: find_duplicate_by_doi ---
        def find_duplicate_by_doi(doi):
            if not doi:
                return None
            uid = f"doi:{doi.lower().strip()}"
            pages = client.query_data_source(
                data_source_id=data_source_id,
                filter={"property": "Source UID", "rich_text": {"equals": uid}},
                page_size=1, fetch_all=False,
            )
            return pages[0] if pages else None

        # --- Wrapper: find_duplicate_by_arxiv_id ---
        def find_duplicate_by_arxiv_id(arxiv_id):
            if not arxiv_id:
                return None
            uid = f"arxiv:{arxiv_id.lower().strip()}"
            pages = client.query_data_source(
                data_source_id=data_source_id,
                filter={"property": "Source UID", "rich_text": {"equals": uid}},
                page_size=1, fetch_all=False,
            )
            return pages[0] if pages else None

        # --- Wrapper: find_duplicate_by_title ---
        def find_duplicate_by_title(title_norm):
            if not title_norm:
                return None
            pages = client.query_data_source(
                data_source_id=data_source_id,
                filter={"property": "Name", "title": {"equals": title_norm}},
                page_size=1, fetch_all=False,
            )
            return pages[0] if pages else None

        # --- Wrapper: find_duplicate_paper (generic) ---
        def find_duplicate_paper(*, name, dedup_key, source_uid, pdf_link):
            if dedup_key:
                pages = client.query_data_source(
                    data_source_id=data_source_id,
                    filter={"property": "Dedup Key", "rich_text": {"equals": dedup_key}},
                    page_size=1, fetch_all=False,
                )
                if pages:
                    return (True, pages[0].get("id"), "dedup_key")
            if source_uid:
                pages = client.query_data_source(
                    data_source_id=data_source_id,
                    filter={"property": "Source UID", "rich_text": {"equals": source_uid}},
                    page_size=1, fetch_all=False,
                )
                if pages:
                    return (True, pages[0].get("id"), "source_uid")
            return (False, None, None)

        # --- Wrapper: update_paper_links ---
        def update_paper_links(page_id, pdf_link, slide1_url):
            properties = {}
            if pdf_link:
                properties["PDF Link"] = _url(pdf_link)
            if slide1_url:
                properties["Slide 1 URL"] = _url(slide1_url)
            if not properties:
                return {}
            return client.update_page(page_id=page_id, properties=properties)

        wrappers["create_paper_inbox"] = create_paper_inbox
        wrappers["find_duplicate_paper"] = find_duplicate_paper
        wrappers["find_duplicate_by_doi"] = find_duplicate_by_doi
        wrappers["find_duplicate_by_arxiv_id"] = find_duplicate_by_arxiv_id
        wrappers["find_duplicate_by_title"] = find_duplicate_by_title
        wrappers["update_paper_links"] = update_paper_links

        for name in wrappers:
            if wrappers[name] is not None:
                logger.info("Notion wrapper loaded: %s", name)

    except Exception as e:
        logger.warning("Notion wrapper initialization failed: %s", e)

    return wrappers


# ================================================================
# Main pipeline
# ================================================================


def run_pipeline(
    *,
    inbox_dir: str = "data/downloads",
    artifacts_dir: str = "artifacts",
    dry_run: bool = False,
    limit: int = 0,
    no_drive: bool = False,
    no_slides: bool = False,
    require_drive: bool = False,
    drive_token: str | None = None,
    drive_credentials: str | None = None,
    openai_model: str = "gpt-4.1-mini",
    verbose: bool = False,
) -> dict:
    """Execute the PDF inbox processing pipeline.

    Parameters
    ----------
    inbox_dir : str
        Path to the inbox directory containing PDF files.
    artifacts_dir : str
        Path to the artifacts directory for slide outputs.
    dry_run : bool
        If True, skip Notion writes, Drive uploads, and file moves.
    limit : int
        Max number of PDFs to process (0 = unlimited).
    no_drive : bool
        Skip Google Drive upload.
    no_slides : bool
        Skip slide generation.
    require_drive : bool
        Fail-fast if Drive auth or preflight fails.
    drive_token : str or None
        Explicit path to Drive token.json.
    drive_credentials : str or None
        Explicit path to Drive client_secret.json.
    openai_model : str
        OpenAI model for metadata repair and summaries.
    verbose : bool
        Enable DEBUG logging.

    Returns
    -------
    dict
        Run summary with statistics.
    """
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)
    try:
        load_env()
    except FileNotFoundError:
        if dry_run:
            logger.warning("env.txt not found; continuing in dry-run mode without env vars")
        else:
            raise

    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    logger.info("[%s] Starting PDF inbox processor", run_id)
    if dry_run:
        logger.info("[%s] DRY-RUN mode: no Notion writes, no Drive uploads, no file moves", run_id)

    # --- Resolve paths ---
    inbox_path = Path(inbox_dir).expanduser().resolve()
    artifacts_path = Path(artifacts_dir).expanduser().resolve()

    inbox_path.mkdir(parents=True, exist_ok=True)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    processed_path = inbox_path / "processed"
    failed_path = inbox_path / "failed"
    dupe_path = processed_path / "duplicates"
    slides_path = artifacts_path / "slides"

    for d in [processed_path, failed_path, dupe_path, slides_path]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Inbox: %s", run_id, inbox_path)
    logger.info("[%s] Artifacts: %s", run_id, artifacts_path)

    # --- Initialize clients ---
    openai_client, openai_model, openai_temperature = _init_openai_client(model=openai_model)

    gemini_client = None
    if not no_slides and not dry_run:
        gemini_client = _init_gemini_client()

    # --- Drive initialization + preflight ---
    drive_service = None
    drive_folder_id = ""
    if not no_drive and not dry_run:
        drive_service = _init_drive_service(
            drive_token=drive_token,
            drive_credentials=drive_credentials,
            require=require_drive,
        )
        if drive_service:
            drive_folder_id = os.getenv("DRIVE_FOLDER_ID", "").strip()
            if not drive_folder_id:
                msg = "DRIVE_FOLDER_ID not set in env"
                if require_drive:
                    raise RuntimeError(msg)
                logger.warning("%s; Drive upload disabled", msg)
                drive_service = None
            else:
                logger.info("Drive: DRIVE_FOLDER_ID=%s", drive_folder_id)
                ok = _verify_drive(drive_service, drive_folder_id, require=require_drive)
                if not ok:
                    drive_service = None
    else:
        reason = "dry-run" if dry_run else "--no-drive"
        logger.info("Drive: DISABLED (%s)", reason)

    # --- Initialize Notion wrappers ---
    notion = {"create_paper_inbox": None, "find_duplicate_paper": None,
              "find_duplicate_by_doi": None, "find_duplicate_by_arxiv_id": None,
              "find_duplicate_by_title": None, "update_paper_links": None}
    if not dry_run:
        notion = _init_notion_wrappers()

    # --- Run batch ---
    from src.pdf.pipeline import run_batch

    run_summary = run_batch(
        inbox_path=inbox_path,
        processed_path=processed_path,
        failed_path=failed_path,
        slides_path=slides_path,
        artifacts_path=artifacts_path,
        run_id=run_id,
        limit=limit,
        openai_client=openai_client,
        openai_model=openai_model,
        openai_temperature=openai_temperature,
        gemini_client=gemini_client,
        drive_service=drive_service,
        drive_folder_id=drive_folder_id,
        create_paper_inbox_fn=notion.get("create_paper_inbox"),
        find_duplicate_paper_fn=notion.get("find_duplicate_paper"),
        find_duplicate_by_doi_fn=notion.get("find_duplicate_by_doi"),
        find_duplicate_by_arxiv_id_fn=notion.get("find_duplicate_by_arxiv_id"),
        find_duplicate_by_title_fn=notion.get("find_duplicate_by_title"),
        update_paper_links_fn=notion.get("update_paper_links"),
        dry_run=dry_run,
        enable_drive_upload=(not no_drive and not dry_run),
    )

    # --- Generate report ---
    from src.pdf.report import (
        build_formatted_summary,
        build_slack_snippet,
        build_summary_text,
        save_json_summary,
    )

    formatted = build_formatted_summary(run_summary)
    summary_text = build_summary_text(formatted)
    slack_snippet = build_slack_snippet(formatted)

    # Print summary
    logger.info("\n%s", summary_text)
    logger.info("Slack snippet:\n%s", slack_snippet)

    # Save JSON summary
    if not dry_run:
        save_json_summary(formatted, summary_text, artifacts_path, run_id)

    return run_summary


# ================================================================
# CLI
# ================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.scripts.031_pdf_inbox_processor",
        description="Process PDFs from an inbox folder into Notion Papers DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m src.scripts.031_pdf_inbox_processor --dry-run
  python -m src.scripts.031_pdf_inbox_processor --inbox-dir data/downloads --limit 3
  python -m src.scripts.031_pdf_inbox_processor --no-drive --no-slides
  python -m src.scripts.031_pdf_inbox_processor --require-drive -v
""",
    )
    parser.add_argument(
        "--inbox-dir",
        default="data/downloads",
        help="Path to inbox directory containing PDFs (default: data/downloads)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Path to artifacts directory for slides/summaries (default: artifacts)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Notion writes, Drive uploads, and file moves",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of PDFs to process (0 = unlimited, default: 0)",
    )

    # --- Drive options ---
    drive_group = parser.add_argument_group("Google Drive")
    drive_group.add_argument(
        "--no-drive",
        action="store_true",
        help="Skip Google Drive upload entirely",
    )
    drive_group.add_argument(
        "--require-drive",
        action="store_true",
        help="Fail-fast if Drive auth or preflight fails (default: graceful fallback)",
    )
    drive_group.add_argument(
        "--drive-token",
        default=None,
        metavar="PATH",
        help="Path to Drive OAuth token.json (default: DRIVE_TOKEN_PATH env or notebooks/token.json)",
    )
    drive_group.add_argument(
        "--drive-credentials",
        default=None,
        metavar="PATH",
        help="Path to Drive OAuth client_secret.json (default: DRIVE_CLIENT_SECRET_PATH env or notebooks/client_secret.json)",
    )

    parser.add_argument(
        "--no-slides",
        action="store_true",
        help="Skip slide generation (Gemini)",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4.1-mini",
        help="OpenAI model for metadata repair and summaries (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to env file (default: notebooks/env.txt)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load env from custom path if specified
    if args.env:
        load_env(env_path=args.env)

    try:
        summary = run_pipeline(
            inbox_dir=args.inbox_dir,
            artifacts_dir=args.artifacts_dir,
            dry_run=args.dry_run,
            limit=args.limit,
            no_drive=args.no_drive,
            no_slides=args.no_slides,
            require_drive=args.require_drive,
            drive_token=args.drive_token,
            drive_credentials=args.drive_credentials,
            openai_model=args.openai_model,
            verbose=args.verbose,
        )

        # Exit with error code if there were failures
        failures = summary.get("failures", 0)
        if failures > 0:
            logger.warning("Completed with %d failure(s)", failures)
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    main()
