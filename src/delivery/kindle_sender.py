# src/delivery/kindle_sender.py
"""Kindle SMTP Sender — send .docx to Kindle via email.

Usage::

    from src.delivery.kindle_sender import send_to_kindle

    send_to_kindle(Path("review_bundle.docx"), "user@kindle.com")
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger(__name__)


def send_to_kindle(docx_path: Path, kindle_email: str) -> None:
    """Send a .docx file to a Kindle email address via SMTP.

    Requires environment variables:
        SMTP_HOST (default: smtp.gmail.com)
        SMTP_PORT (default: 587)
        SMTP_USER
        SMTP_PASS
        MAIL_FROM
    """
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    mail_from = os.environ.get("MAIL_FROM", "")

    if not smtp_user or not smtp_pass:
        raise ValueError("SMTP_USER and SMTP_PASS environment variables are required")
    if not mail_from:
        raise ValueError("MAIL_FROM environment variable is required")
    if not docx_path.exists():
        raise FileNotFoundError(f"File not found: {docx_path}")

    # Build email
    msg = EmailMessage()
    msg["Subject"] = f"[researchOS] Review Bundle — {docx_path.stem}"
    msg["From"] = mail_from
    msg["To"] = kindle_email
    msg.set_content("Review bundle attached. Sent from researchOS.")

    # Attach .docx
    with open(docx_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=docx_path.name,
        )

    # Send via SMTP
    logger.info("Connecting to %s:%d", smtp_host, smtp_port)
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

    logger.info("Sent %s to %s", docx_path.name, kindle_email)
