# src/daily/session_targets.py
"""076 Session to Targets — service logic.

Scans recent 073 research sessions, extracts monitoring target candidates
via LLM, validates (especially People), checks for duplicates, and
registers new targets in MONITORING_TARGETS_DB.

Design deviation from design.md §2.4.1:
  session.json does NOT contain memo_title, memo_summary, intent, topic,
  subtype, or claims (they are stripped by _save_session). Instead, 076
  reads per-run intermediate files: plan.json, memo.json, claims.json
  from data/deep_research/{run_id}/. This avoids modifying 073 code.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("076_session_to_targets")

# -- constants ---------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SESSIONS_DIR = _PROJECT_ROOT / "data" / "deep_research_sessions"
_RUNS_DIR = _PROJECT_ROOT / "data" / "deep_research"
_LLM_MODEL = "claude-sonnet-4-20250514"
_NOTION_DELAY_SEC = 0.5

# topic/subtype → type mapping (design §2.4.2)
_TOPIC_TYPE_MAP = {
    ("company", "startup"): "Startup",
    ("company", "vc"): "VC",
    ("organization", "vc"): "VC",
    ("person", ""): "People",
    ("policy", ""): "Policy",
}

_EXTRACT_SYSTEM = """\
Extract monitoring target candidates from research results.

For each entity found, return:
- name: Full official name (e.g., "Sequoia Capital", not just "Sequoia")
- type: One of VC / Startup / People / Policy
- search_keywords: 2-5 search terms for news monitoring
- confidence: high / medium / low
- affiliation: Organization name (required for People type, optional for others)

Rules:
- VC: venture capital firms, LP investors, fund managers
- Startup: startups, growth-stage companies (NOT large public companies like Toyota, Google)
- People: individuals in VC, startup, policy roles (MUST include full name AND affiliation)
- Policy: specific regulations, government programs, policy initiatives (NOT broad topics like "AI policy")

Do NOT extract:
- Broad industry categories (e.g., "AI industry", "healthcare sector")
- Well-known public companies (Fortune 500, Nikkei 225)
- World leaders, heads of state, or Fortune 500 CEOs

Return ONLY a JSON object: {"candidates": [...]}
No markdown fences. No extra text."""


# -- Session scanning --------------------------------------------------------


def scan_recent_sessions(
    *,
    hours: int = 24,
    sessions_dir: Path = _SESSIONS_DIR,
) -> List[Dict[str, Any]]:
    """Collect session.json files from the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    sessions: List[Dict[str, Any]] = []

    if not sessions_dir.exists():
        logger.warning("Sessions directory not found: %s", sessions_dir)
        return sessions

    for sess_dir in sorted(sessions_dir.iterdir()):
        if not sess_dir.is_dir() or sess_dir.name.startswith("."):
            continue
        session_file = sess_dir / "session.json"
        if not session_file.exists():
            continue

        try:
            data = json.loads(session_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", session_file, e)
            continue

        # Check recency
        created_at = data.get("created_at", "")
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                if created_dt < cutoff:
                    continue
            except ValueError:
                pass

        if data.get("status") != "completed":
            continue

        sessions.append(data)

    logger.info("Found %d completed sessions in last %d hours", len(sessions), hours)
    return sessions


# -- Per-run data loading (reads intermediate files) -------------------------


def _load_run_data(run_id: str, runs_dir: Path = _RUNS_DIR) -> Dict[str, Any]:
    """Load plan.json, memo.json, claims.json for a run."""
    run_dir = runs_dir / run_id
    result: Dict[str, Any] = {
        "intent": "general_research",
        "topic": "",
        "subtype": "",
        "memo_title": "",
        "memo_summary": "",
        "claims": [],
    }

    # plan.json
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            result["intent"] = plan.get("intent", "general_research")
            result["topic"] = plan.get("topic", "")
            result["subtype"] = plan.get("subtype", "")
        except (json.JSONDecodeError, OSError):
            pass

    # memo.json
    memo_path = run_dir / "memo.json"
    if memo_path.exists():
        try:
            memo = json.loads(memo_path.read_text(encoding="utf-8"))
            result["memo_title"] = memo.get("title", "")
            result["memo_summary"] = memo.get("summary", "")
        except (json.JSONDecodeError, OSError):
            pass

    # claims.json
    claims_path = run_dir / "claims.json"
    if claims_path.exists():
        try:
            claims_data = json.loads(claims_path.read_text(encoding="utf-8"))
            result["claims"] = [
                {"statement": c.get("statement", ""), "confidence": c.get("confidence", "")}
                for c in claims_data.get("claims", [])
            ]
        except (json.JSONDecodeError, OSError):
            pass

    return result


# -- LLM entity extraction --------------------------------------------------


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON response, tolerating markdown fences."""
    text = text.strip()
    m = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_target_candidates(
    run: Dict[str, Any],
    run_data: Dict[str, Any],
    llm_client: Any,
    *,
    model: str = _LLM_MODEL,
) -> List[Dict[str, Any]]:
    """Extract monitoring target candidates from a single run via LLM.

    Uses memo_title + memo_summary + claims as input.
    Applies topic/subtype → type hint.
    """
    memo_title = run_data.get("memo_title", "")
    memo_summary = run_data.get("memo_summary", "")
    claims = run_data.get("claims", [])

    if not memo_title and not memo_summary:
        return []

    # Build input text
    parts = []
    if memo_title:
        parts.append(f"Title: {memo_title}")
    if memo_summary:
        parts.append(f"Summary: {memo_summary[:5000]}")
    if claims:
        claims_text = "\n".join(
            f"- [{c.get('confidence', '?')}] {c.get('statement', '')}"
            for c in claims[:10]
        )
        parts.append(f"Key Claims:\n{claims_text}")

    # Add type hint from topic/subtype
    topic = run_data.get("topic", "")
    subtype = run_data.get("subtype", "")
    type_hint = _infer_type_from_topic(topic, subtype)
    if type_hint:
        parts.append(f"\nHint: The research topic suggests type={type_hint}")

    user_content = "\n\n".join(parts)

    body = {
        "model": model,
        "max_tokens": 1024,
        "system": _EXTRACT_SYSTEM,
        "messages": [{"role": "user", "content": user_content}],
    }

    try:
        resp = llm_client.messages_create(body=body)
        resp_text = ""
        for block in resp.get("content", []):
            if block.get("type") == "text":
                resp_text = block.get("text", "")
                break

        parsed = _parse_json_response(resp_text)
        if parsed and "candidates" in parsed:
            return parsed["candidates"]
    except Exception as e:
        logger.warning("LLM extraction failed for run %s: %s", run.get("run_id"), e)

    return []


def _infer_type_from_topic(topic: str, subtype: str) -> str:
    """Map topic/subtype to a suggested type."""
    topic = topic.lower().strip()
    subtype = subtype.lower().strip()

    # Exact match
    key = (topic, subtype)
    if key in _TOPIC_TYPE_MAP:
        return _TOPIC_TYPE_MAP[key]

    # Wildcard match (topic, "")
    for (t, s), v in _TOPIC_TYPE_MAP.items():
        if t == topic and s == "":
            return v

    return ""


# -- People validation -------------------------------------------------------


def validate_people_candidate(candidate: Dict[str, Any]) -> bool:
    """Validate a People-type candidate against anti-false-positive rules.

    Rules (design §2.4.3):
    1. Full name required (given + family, at least 2 name parts)
    2. Affiliation or role must be specified
    3. Public figures (heads of state, Fortune 500 CEOs) excluded
    """
    name = (candidate.get("name") or "").strip()
    affiliation = (candidate.get("affiliation") or "").strip()

    # Rule 1: Full name check
    if not _is_full_name(name):
        logger.debug("People rejected (not full name): %s", name)
        return False

    # Rule 2: Affiliation check — also accept if search_keywords contain org info
    keywords = candidate.get("search_keywords", "")
    if isinstance(keywords, list):
        keywords = " ".join(keywords)
    if not affiliation and not keywords:
        logger.debug("People rejected (no affiliation): %s", name)
        return False

    # Rule 3: Public figure filter
    if _is_public_figure(name):
        logger.debug("People rejected (public figure): %s", name)
        return False

    return True


def _is_full_name(name: str) -> bool:
    """Check if a name has at least 2 parts (given + family).

    Handles both Western (space-separated) and CJK (2+ characters) names.
    """
    if not name:
        return False

    # CJK name: at least 3 characters (e.g., 溝口剛) to exclude surname-only (e.g., 鈴木)
    # Or 2+ CJK chars with a space separator (e.g., "溝口 剛")
    cjk_chars = sum(1 for c in name if unicodedata.category(c).startswith("Lo"))
    if cjk_chars >= 3:
        return True
    if cjk_chars >= 2 and " " in name:
        return True

    # Western name: at least 2 space-separated parts
    parts = [p for p in name.split() if len(p) > 1]
    return len(parts) >= 2


# Common public figure patterns — not exhaustive, just high-noise targets
_PUBLIC_FIGURE_TITLES = frozenset({
    "president", "prime minister", "chancellor", "premier",
    "大統領", "首相", "総理", "国家主席",
})

_PUBLIC_FIGURE_NAMES = frozenset({
    # Keep this small — only the highest-noise names
    "elon musk", "jeff bezos", "mark zuckerberg", "tim cook",
    "satya nadella", "sundar pichai", "jensen huang",
})


def _is_public_figure(name: str) -> bool:
    """Check if a name matches known high-noise public figures."""
    name_lower = name.lower().strip()

    if name_lower in _PUBLIC_FIGURE_NAMES:
        return True

    for title in _PUBLIC_FIGURE_TITLES:
        if title in name_lower:
            return True

    return False


# -- Duplicate checking ------------------------------------------------------


def normalize_name(name: str) -> str:
    """Normalize a name for duplicate comparison.

    Lowercase, strip common suffixes (Inc., Co., Ltd., etc.),
    normalize whitespace.
    """
    s = name.lower().strip()
    # Remove common corporate suffixes
    s = re.sub(r"\s*(inc\.?|co\.?|ltd\.?|llc\.?|corp\.?|corporation|株式会社|有限会社)\s*$", "", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_duplicate(
    candidate_name: str,
    existing_targets: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Check if a candidate name duplicates an existing target.

    Returns the matching target dict if duplicate, None otherwise.
    """
    normalized = normalize_name(candidate_name)
    if not normalized:
        return None

    for target in existing_targets:
        existing_normalized = normalize_name(target.get("name", ""))
        if not existing_normalized:
            continue
        # Exact match
        if normalized == existing_normalized:
            return target
        # Containment (e.g., "Sequoia Capital" ↔ "Sequoia Capital Japan")
        if normalized in existing_normalized or existing_normalized in normalized:
            return target

    return None


# -- Target registration -----------------------------------------------------


def register_target(
    notion_client: Any,
    targets_db_id: str,
    candidate: Dict[str, Any],
    session_id: str,
) -> Optional[str]:
    """Register a new target in MONITORING_TARGETS_DB.

    Returns the created page_id, or None on failure.
    """
    name = (candidate.get("name") or "").strip()
    target_type = (candidate.get("type") or "Unknown").strip()
    keywords = candidate.get("search_keywords", "")
    if isinstance(keywords, list):
        keywords = ", ".join(keywords)

    properties = {
        "Name": {"title": [{"type": "text", "text": {"content": name[:2000]}}]},
        "Type": {"select": {"name": target_type}},
        "Status": {"select": {"name": "Active"}},
        "Priority": {"select": {"name": "Medium"}},
        "Cadence": {"select": {"name": "Weekly"}},
        "Enabled": {"checkbox": True},
        "Search Keywords": {
            "rich_text": [{"type": "text", "text": {"content": keywords[:2000]}}]
        },
        "Consecutive Misses": {"number": 0},
        "Created By": {"select": {"name": "076_session"}},
        "Source Session": {
            "rich_text": [{"type": "text", "text": {"content": session_id}}]
        },
    }

    try:
        result = notion_client.create_page(
            parent_db_id=targets_db_id,
            properties=properties,
        )
        page_id = result.get("id", "")
        logger.info("Registered target: %s (type=%s, page_id=%s)", name, target_type, page_id[:12])
        return page_id
    except Exception as e:
        logger.error("Failed to register target '%s': %s", name, e)
        return None


# -- Batch processing --------------------------------------------------------


def run_session_to_targets(
    *,
    notion_client: Any,
    targets_db_id: str,
    targets_data_source_id: str,
    llm_client: Any,
    hours: int = 24,
    dry_run: bool = False,
    limit: int = 0,
    sessions_dir: Path = _SESSIONS_DIR,
    runs_dir: Path = _RUNS_DIR,
) -> Dict[str, Any]:
    """Run the session-to-targets pipeline.

    Args:
        limit: Max number of targets to register (0 = unlimited).

    Returns a summary dict with counts and per-candidate results.
    """
    from src.notion.targets_normalize import normalize_targets

    summary: Dict[str, Any] = {
        "sessions_scanned": 0,
        "runs_processed": 0,
        "candidates_extracted": 0,
        "duplicates_skipped": 0,
        "people_rejected": 0,
        "low_confidence_skipped": 0,
        "registered": 0,
        "errors": 0,
        "results": [],
        "dry_run": dry_run,
    }

    # Scan recent sessions
    sessions = scan_recent_sessions(hours=hours, sessions_dir=sessions_dir)
    summary["sessions_scanned"] = len(sessions)

    if not sessions:
        logger.info("No recent sessions found (last %d hours)", hours)
        return summary

    # Fetch existing targets for dedup
    pages = notion_client.query_data_source(
        data_source_id=targets_data_source_id,
        fetch_all=True,
    )
    existing_targets = normalize_targets(pages)
    logger.info("Loaded %d existing targets for dedup", len(existing_targets))

    # Track newly registered names within this run for cross-session dedup
    newly_registered: List[Dict[str, Any]] = []

    for session in sessions:
        session_id = session.get("session_id", "unknown")
        runs = session.get("runs", [])
        logger.info("Processing session %s (%d runs)", session_id, len(runs))

        for run in runs:
            if run.get("status") != "completed":
                continue

            run_id = run.get("run_id", "")
            if not run_id:
                continue

            summary["runs_processed"] += 1

            # Load per-run intermediate data
            run_data = _load_run_data(run_id, runs_dir=runs_dir)

            if not run_data.get("memo_title") and not run_data.get("memo_summary"):
                logger.debug("Run %s: no memo data, skipping", run_id)
                continue

            # Extract candidates via LLM
            if dry_run:
                # In dry-run, still call LLM for extraction but don't register
                candidates = extract_target_candidates(run, run_data, llm_client)
            else:
                candidates = extract_target_candidates(run, run_data, llm_client)

            for candidate in candidates:
                summary["candidates_extracted"] += 1
                cand_name = (candidate.get("name") or "").strip()
                cand_type = (candidate.get("type") or "").strip()
                confidence = (candidate.get("confidence") or "").lower().strip()

                result_entry: Dict[str, Any] = {
                    "session_id": session_id,
                    "run_id": run_id,
                    "name": cand_name,
                    "type": cand_type,
                    "confidence": confidence,
                    "action": None,
                    "reason": None,
                }

                # Filter: confidence threshold
                if confidence not in ("high", "medium"):
                    summary["low_confidence_skipped"] += 1
                    result_entry["action"] = "skipped"
                    result_entry["reason"] = f"low confidence ({confidence})"
                    summary["results"].append(result_entry)
                    continue

                # Filter: People validation
                if cand_type == "People":
                    if not validate_people_candidate(candidate):
                        summary["people_rejected"] += 1
                        result_entry["action"] = "skipped"
                        result_entry["reason"] = "People validation failed"
                        summary["results"].append(result_entry)
                        continue

                # Filter: duplicate check (against existing + newly registered)
                all_targets = existing_targets + newly_registered
                dup = check_duplicate(cand_name, all_targets)
                if dup:
                    summary["duplicates_skipped"] += 1
                    result_entry["action"] = "skipped"
                    result_entry["reason"] = f"duplicate of '{dup.get('name', '?')}'"
                    summary["results"].append(result_entry)
                    continue

                # Limit check
                if limit > 0 and summary["registered"] >= limit:
                    result_entry["action"] = "skipped"
                    result_entry["reason"] = f"limit reached ({limit})"
                    summary["results"].append(result_entry)
                    continue

                # Register
                if dry_run:
                    result_entry["action"] = "would_register"
                    result_entry["reason"] = "dry-run"
                    logger.info(
                        "  [DRY-RUN] Would register: %s (type=%s, confidence=%s)",
                        cand_name, cand_type, confidence,
                    )
                else:
                    page_id = register_target(
                        notion_client, targets_db_id, candidate, session_id,
                    )
                    if page_id:
                        result_entry["action"] = "registered"
                        result_entry["page_id"] = page_id
                        summary["registered"] += 1
                        # Add to dedup list
                        newly_registered.append({
                            "name": cand_name,
                            "type": cand_type,
                            "page_id": page_id,
                        })
                    else:
                        result_entry["action"] = "error"
                        result_entry["reason"] = "registration failed"
                        summary["errors"] += 1

                    time.sleep(_NOTION_DELAY_SEC)

                summary["results"].append(result_entry)

    return summary
