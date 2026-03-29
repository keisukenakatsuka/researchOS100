#!/usr/bin/env python
"""120 Review Bundle & Kindle Delivery — assemble and send research artifacts.

Generates a review_bundle.docx from paper_draft.md, review_report.md, and
evidence_sufficiency.json, optionally sending it to a Kindle email address.

Usage::

    # Generate docx only
    python -m src.scripts.120_review_bundle_kindle --run-id <id>

    # Generate and send to Kindle
    python -m src.scripts.120_review_bundle_kindle --run-id <id> --send

    # Custom output directory
    python -m src.scripts.120_review_bundle_kindle --run-id <id> --output-dir /tmp

    # Override Kindle email
    python -m src.scripts.120_review_bundle_kindle --run-id <id> --send --kindle-email user@kindle.com
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env

logger = logging.getLogger("120_review_bundle_kindle")

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.scripts.120_review_bundle_kindle",
        description="Review Bundle & Kindle Delivery — assemble research artifacts into .docx",
    )
    p.add_argument("--run-id", type=str, required=True,
                    help="Run ID to bundle (e.g., 20260326_004645_c22a6849)")
    p.add_argument("--send", action="store_true",
                    help="Send docx to Kindle via SMTP (requires SMTP env vars)")
    p.add_argument("--kindle-email", type=str, default=None,
                    help="Override KINDLE_EMAIL env var")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Override output directory (default: run directory)")
    p.add_argument("-v", "--verbose", action="store_true",
                    help="Enable DEBUG logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    load_env()

    # Resolve run directory
    run_dir = _DATA_DIR / args.run_id
    if not run_dir.exists():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None

    # Build docx
    from src.delivery.review_bundle import build_review_bundle

    logger.info("=== 120 Review Bundle === run_id=%s", args.run_id)
    docx_path = build_review_bundle(run_dir, output_dir=output_dir)

    print(f"\n{'=' * 60}")
    print(f"120 Review Bundle — Generated")
    print(f"{'=' * 60}")
    print(f"  Run ID:  {args.run_id}")
    print(f"  Output:  {docx_path}")
    print(f"  Size:    {docx_path.stat().st_size / 1024:.1f} KB")

    # Send to Kindle
    if args.send:
        kindle_email = args.kindle_email or os.environ.get("KINDLE_EMAIL", "")
        if not kindle_email:
            logger.error("No Kindle email specified. Use --kindle-email or set KINDLE_EMAIL env var.")
            sys.exit(1)

        from src.delivery.kindle_sender import send_to_kindle

        logger.info("Sending to Kindle: %s", kindle_email)
        try:
            send_to_kindle(docx_path, kindle_email)
            print(f"  Sent to: {kindle_email}")
        except Exception as e:
            logger.error("Failed to send: %s", e)
            print(f"  ERROR:   Failed to send — {e}")
            sys.exit(1)

    logger.info("=== 120 Done ===")


if __name__ == "__main__":
    main()
