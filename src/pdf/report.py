# src/pdf/report.py
"""Run summary generation and Slack snippet formatting.

Ported from notebook 031 cells 11 and 12.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def build_formatted_summary(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build a structured summary from the raw run_summary dict."""
    processing_results = run_summary.get("results", [])

    formatted: Dict[str, Any] = {
        "run_metadata": {
            "run_id": run_summary.get("run_id"),
            "start_time": run_summary.get("start_time"),
            "end_time": run_summary.get("end_time"),
            "duration_seconds": run_summary.get("batch_duration_seconds", 0.0),
        },
        "statistics": {
            "total_files": run_summary.get("total_files", 0),
            "processed": run_summary.get("processed", 0),
            "success": run_summary.get("success", 0),
            "duplicates": run_summary.get("duplicates", 0),
            "failures": run_summary.get("failures", 0),
            "total_errors": run_summary.get("total_errors", 0),
            "avg_duration_per_file": run_summary.get("avg_duration_per_file", 0.0),
        },
        "success_details": [],
        "duplicate_details": [],
        "failure_details": [],
        "error_summary": [],
        "recommendations": [],
    }

    for result in processing_results:
        if not isinstance(result, dict):
            continue
        pdf_name = result.get("pdf_name", "(unknown)")
        status = result.get("status", "unknown")
        page_id = result.get("page_id")
        errors = result.get("errors", []) or []
        duration = float(result.get("duration_seconds") or 0.0)

        if status == "success":
            formatted["success_details"].append(
                {"pdf_name": pdf_name, "page_id": page_id, "duration_seconds": duration}
            )
        elif status == "duplicate":
            dup_info = result.get("duplicate_info", {}) or {}
            formatted["duplicate_details"].append({
                "pdf_name": pdf_name,
                "match_type": dup_info.get("match_type", "unknown"),
                "existing_page_id": dup_info.get("page_id"),
                "existing_title": (dup_info.get("title", "(unknown)") or "(unknown)")[:50],
            })
        else:
            formatted["failure_details"].append(
                {"pdf_name": pdf_name, "errors": errors, "duration_seconds": duration}
            )
            for err in errors:
                formatted["error_summary"].append(
                    {"pdf_name": pdf_name, "error": str(err)[:200]}
                )

    stats = formatted["statistics"]
    if stats["failures"] > 0:
        formatted["recommendations"].append(
            f"Review {stats['failures']} failed file(s) in failed/ directory"
        )
    if stats["duplicates"] > 0:
        formatted["recommendations"].append(
            f"{stats['duplicates']} duplicate(s) found; verify if unexpected"
        )
    if stats["total_files"] == 0:
        formatted["recommendations"].append("Inbox was empty; no action required")
    if stats["total_errors"] > stats["failures"]:
        formatted["recommendations"].append(
            "Multiple errors per file detected; check for systemic issues"
        )

    return formatted


def build_summary_text(formatted: Dict[str, Any]) -> str:
    """Generate a plaintext summary report."""
    stats = formatted["statistics"]
    meta = formatted["run_metadata"]

    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("PDF INBOX PROCESSING SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Run ID: {meta.get('run_id')}")
    lines.append(f"Start: {meta.get('start_time')}")
    lines.append(f"End: {meta.get('end_time')}")
    lines.append(f"Duration: {meta.get('duration_seconds', 0):.2f}s")
    lines.append("")
    lines.append("STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total files: {stats['total_files']}")
    lines.append(f"Processed: {stats['processed']}")
    lines.append(f"Success: {stats['success']}")
    lines.append(f"Duplicates: {stats['duplicates']}")
    lines.append(f"Failures: {stats['failures']}")
    lines.append(f"Total errors: {stats['total_errors']}")
    lines.append(f"Avg time/file: {stats['avg_duration_per_file']:.2f}s")
    lines.append("")

    if formatted["success_details"]:
        lines.append("SUCCESSFUL IMPORTS")
        lines.append("-" * 40)
        for i, item in enumerate(formatted["success_details"], 1):
            lines.append(f"{i}. {item['pdf_name']}  (page_id={item['page_id']})")
        lines.append("")

    if formatted["duplicate_details"]:
        lines.append("DUPLICATES SKIPPED")
        lines.append("-" * 40)
        for i, item in enumerate(formatted["duplicate_details"], 1):
            lines.append(
                f"{i}. {item['pdf_name']}  (match={item['match_type']})"
            )
        lines.append("")

    if formatted["failure_details"]:
        lines.append("FAILURES")
        lines.append("-" * 40)
        for i, item in enumerate(formatted["failure_details"], 1):
            lines.append(f"{i}. {item['pdf_name']}")
            for err in item["errors"]:
                lines.append(f"   ERROR: {str(err)[:150]}")
        lines.append("")

    if formatted["recommendations"]:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for i, rec in enumerate(formatted["recommendations"], 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def build_slack_snippet(formatted: Dict[str, Any]) -> str:
    """Generate a Slack-friendly markdown snippet."""
    stats = formatted["statistics"]
    meta = formatted["run_metadata"]

    total = int(stats.get("total_files", 0) or 0)
    success = int(stats.get("success", 0) or 0)
    duplicates = int(stats.get("duplicates", 0) or 0)
    failures = int(stats.get("failures", 0) or 0)
    duration = float(meta.get("duration_seconds", 0.0) or 0.0)
    avg = float(stats.get("avg_duration_per_file", 0.0) or 0.0)

    lines: List[str] = []
    lines.append("*PDF Inbox Processing Summary*")
    lines.append(f"`Run ID: {meta.get('run_id')}`")
    lines.append("")
    lines.append("*Statistics:*")
    lines.append(f"Total files: {total}")
    lines.append(f"Success: {success}")
    lines.append(f"Duplicates: {duplicates}")
    lines.append(f"Failures: {failures}")
    lines.append(f"Duration: {duration:.1f}s (avg {avg:.1f}s/file)")
    lines.append("")

    success_details = formatted.get("success_details", []) or []
    if success_details:
        lines.append(f"*Imported {len(success_details)} paper(s):*")
        for i, item in enumerate(success_details[:5], 1):
            lines.append(f"{i}. `{item.get('pdf_name')}`")
        if len(success_details) > 5:
            lines.append(f"   (+{len(success_details) - 5} more)")
        lines.append("")

    lines.append(f"_Completed at {meta.get('end_time')}_")
    return "\n".join(lines)


def save_json_summary(
    formatted: Dict[str, Any],
    summary_text: str,
    artifacts_path: Path,
    run_id: str,
) -> Path:
    """Save a JSON summary file to artifacts/summaries/."""
    summaries_path = artifacts_path / "summaries"
    summaries_path.mkdir(parents=True, exist_ok=True)

    path = summaries_path / f"{run_id}_summary.json"

    data = {
        "run_metadata": formatted.get("run_metadata"),
        "statistics": formatted.get("statistics"),
        "success_details": formatted.get("success_details"),
        "duplicate_details": formatted.get("duplicate_details"),
        "failure_details": formatted.get("failure_details"),
        "error_summary": formatted.get("error_summary"),
        "recommendations": formatted.get("recommendations"),
        "summary_text": summary_text,
    }

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON summary saved: %s", path)
    return path
