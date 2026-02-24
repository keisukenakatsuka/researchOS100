# src/pdf/drive.py
"""Google Drive authentication and file upload helpers.

Ported from notebook 031 cells 01 (auth) and 09 (upload).

Authentication file resolution order:
1. Explicit paths passed to build_drive_service()
2. Environment variables: DRIVE_TOKEN_PATH, DRIVE_CLIENT_SECRET_PATH
3. Default: notebooks/token.json, notebooks/client_secret.json (relative to project root)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]

# Resolve project root: src/pdf/drive.py -> src/pdf -> src -> <project_root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default auth file locations (relative to project root)
# google_token.json has full drive scope; token.json only has drive.file scope
_DEFAULT_TOKEN_PATH = _PROJECT_ROOT / "notebooks" / "google_token.json"
_DEFAULT_CLIENT_SECRET_PATH = _PROJECT_ROOT / "notebooks" / "client_secret.json"


# ----------------------------------------------------------------
# Auth file resolution
# ----------------------------------------------------------------


def resolve_token_path(explicit: Optional[str] = None) -> Path:
    """Resolve token path: explicit arg > env var > default notebooks/token.json."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_val = os.getenv("DRIVE_TOKEN_PATH", "").strip()
    if env_val:
        return Path(env_val).expanduser().resolve()
    return _DEFAULT_TOKEN_PATH


def resolve_client_secret_path(explicit: Optional[str] = None) -> Path:
    """Resolve client secret path: explicit arg > env var > default notebooks/client_secret.json."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_val = os.getenv("DRIVE_CLIENT_SECRET_PATH", "").strip()
    if env_val:
        return Path(env_val).expanduser().resolve()
    return _DEFAULT_CLIENT_SECRET_PATH


# ----------------------------------------------------------------
# Authentication
# ----------------------------------------------------------------


def build_drive_service(
    token_path: Optional[Path] = None,
    client_secret_path: Optional[Path] = None,
) -> Any:
    """Build an authenticated Google Drive service.

    Attempts to load credentials from *token_path*; if expired, refreshes.
    If no valid token exists, runs the OAuth flow using *client_secret_path*.

    Parameters
    ----------
    token_path : Path, optional
        Resolved token file path. If None, uses resolve_token_path().
    client_secret_path : Path, optional
        Resolved client secret path. If None, uses resolve_client_secret_path().
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    if token_path is None:
        token_path = resolve_token_path()
    if client_secret_path is None:
        client_secret_path = resolve_client_secret_path()

    logger.info("Drive auth: token_path=%s (exists=%s)", token_path, token_path.exists())
    logger.info("Drive auth: client_secret_path=%s (exists=%s)", client_secret_path, client_secret_path.exists())

    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.valid:
            logger.info("Drive auth: token is valid")
        elif creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                token_path.write_text(creds.to_json(), encoding="utf-8")
                logger.info("Drive auth: token refreshed and saved")
            except Exception as e:
                logger.warning("Drive auth: token refresh failed: %s", e)
                creds = None
        else:
            logger.warning("Drive auth: token exists but is invalid (no refresh_token)")
            creds = None

    if creds is None:
        if not client_secret_path.exists():
            raise FileNotFoundError(
                f"Drive auth: client secret not found at {client_secret_path.resolve()}. "
                "Provide via --drive-credentials or DRIVE_CLIENT_SECRET_PATH env var."
            )
        logger.info("Drive auth: running OAuth flow (browser will open)...")
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), SCOPES)
        creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        logger.info("Drive auth: new token saved to %s", token_path)

    service = build("drive", "v3", credentials=creds)
    logger.info("Drive auth: service built successfully")
    return service


# ----------------------------------------------------------------
# Preflight verification
# ----------------------------------------------------------------


def verify_drive_connection(drive_service: Any, folder_id: str) -> Dict[str, Any]:
    """Verify Drive connection by listing the target folder.

    Returns
    -------
    dict
        Keys: ok (bool), folder_name (str or None), file_count (int), error (str or None)
    """
    result: Dict[str, Any] = {
        "ok": False,
        "folder_name": None,
        "file_count": 0,
        "error": None,
    }
    try:
        # Get folder metadata (supportsAllDrives for Shared Drives)
        folder = drive_service.files().get(
            fileId=folder_id, fields="id,name", supportsAllDrives=True
        ).execute()
        result["folder_name"] = folder.get("name")

        # Count files in folder (lightweight query, max 5)
        resp = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            pageSize=5,
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        result["file_count"] = len(resp.get("files", []))
        result["ok"] = True

        logger.info(
            "Drive preflight OK: folder='%s' (id=%s), %d+ file(s) visible",
            result["folder_name"], folder_id, result["file_count"],
        )
    except Exception as e:
        result["error"] = str(e)
        logger.error("Drive preflight FAILED: %s", e)

    return result


# ----------------------------------------------------------------
# Upload helpers
# ----------------------------------------------------------------


def _drive_safe_title(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "_").replace(":", "").replace("\n", " ")
    return " ".join(s.split())


def _drive_title(base_title: str, suffix: str, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _drive_safe_title(base_title)[:120] or "paper"
    return f"{base}__{suffix}__{ts}.{ext}"


def _ensure_shareable(drive_service: Any, file_id: str) -> None:
    drive_service.permissions().create(
        fileId=file_id,
        body={"role": "reader", "type": "anyone"},
    ).execute()


def _drive_file_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def upload_file_to_drive(
    drive_service: Any,
    local_path: Path,
    drive_folder_id: str,
    title: str,
    mimetype: str,
) -> Dict[str, Any]:
    """Upload a single file to Drive and make it shareable. Returns {file_id, name, url}."""
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(str(local_path), mimetype=mimetype, resumable=True)
    created = (
        drive_service.files()
        .create(
            body={"name": title, "parents": [drive_folder_id]},
            media_body=media,
            fields="id,name",
        )
        .execute()
    )
    file_id = created["id"]
    _ensure_shareable(drive_service, file_id)
    return {"file_id": file_id, "name": created["name"], "url": _drive_file_url(file_id)}


def upload_pdf_and_slide(
    drive_service: Any,
    pdf_path: Path,
    slide_path: Optional[Path],
    drive_folder_id: str,
    base_title: str,
) -> Dict[str, Any]:
    """Upload a PDF (and optionally its slide) to Drive. Returns {pdf: {...}, slide: {...}}."""
    out: Dict[str, Any] = {"pdf": None, "slide": None}

    if pdf_path and pdf_path.exists():
        pdf_title = _drive_title(base_title, "paper", "pdf")
        out["pdf"] = upload_file_to_drive(
            drive_service, pdf_path, drive_folder_id, pdf_title, "application/pdf"
        )

    if slide_path and slide_path.exists():
        ext = slide_path.suffix.lstrip(".").lower() or "png"
        slide_title = _drive_title(base_title, "slide", ext)
        out["slide"] = upload_file_to_drive(
            drive_service, slide_path, drive_folder_id, slide_title, f"image/{ext}"
        )

    return out
