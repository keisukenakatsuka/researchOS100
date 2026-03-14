# src/daily/news_monitor.py
"""075 Smart News Monitor — service logic.

Queries MONITORING_TARGETS_DB for targets due today, runs Google CSE +
NewsAPI searches, writes events to EVENTS_DB, and applies cadence
state transitions (Weekly ↔ Monthly → Paused).

State machine is execution-count based: Consecutive Misses tracks how
many search executions returned zero news, not calendar time.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.notion.targets_schema import (
    MONTHLY_CHECK_INTERVAL_DAYS,
    MONTHLY_TO_PAUSED_MISSES,
    WEEKLY_TO_MONTHLY_MISSES,
)

logger = logging.getLogger("075_smart_news_monitor")

# -- constants ---------------------------------------------------------------

_NOTION_DELAY_SEC = 0.5
_SEARCH_DELAY_SEC = 1.0


# -- Check Day ---------------------------------------------------------------


def assign_check_day(target_id: str) -> int:
    """Deterministic check day (0=Mon .. 6=Sun) from target_id hash."""
    hash_val = int(hashlib.md5(target_id.encode()).hexdigest(), 16)
    return hash_val % 7


def compute_next_check(cadence: str, target_id: str, now: date) -> Optional[date]:
    """Compute the next check date for a target.

    Weekly/Daily: next occurrence of this target's check day.
    Monthly: now + 30 days.
    Other (Paused/Archived): None.
    """
    cadence_l = (cadence or "").strip().lower()
    if cadence_l in ("weekly", "daily"):
        check_day = assign_check_day(target_id)
        days_ahead = (check_day - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # already ran today → next week
        return now + timedelta(days=days_ahead)
    elif cadence_l == "monthly":
        return now + timedelta(days=MONTHLY_CHECK_INTERVAL_DAYS)
    return None


# -- Target selection --------------------------------------------------------


def get_todays_targets(
    targets: List[Dict[str, Any]],
    today: date,
) -> List[Dict[str, Any]]:
    """Filter targets that are due for checking today.

    A target is due if:
    - enabled=True AND status=Active
    - next_check <= today  OR  next_check is empty (never checked)
    """
    due = []
    for t in targets:
        if not t.get("enabled"):
            continue
        status = (t.get("status") or "").strip().lower()
        if status != "active":
            continue
        next_check = t.get("next_check", "")
        if next_check:
            try:
                nc_date = date.fromisoformat(next_check)
                if nc_date > today:
                    continue
            except ValueError:
                pass  # invalid date → treat as due
        due.append(t)
    logger.info("Targets due today (%s): %d", today.isoformat(), len(due))
    return due


# -- Search execution -------------------------------------------------------


def search_target(
    target: Dict[str, Any],
    *,
    google_client: Any,
    news_client: Any,
    num_results: int = 5,
    days_back: int = 7,
) -> List[Dict[str, Any]]:
    """Run Google CSE + NewsAPI for a target. Returns merged results.

    Each result has: title, url, source, date, snippet.
    """
    keywords = target.get("search_keywords", "") or target.get("name", "")
    if not keywords:
        return []

    results: List[Dict[str, Any]] = []

    # Google CSE
    try:
        cse_results = google_client.search(keywords, num=num_results)
        for r in cse_results:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "source": "Google",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "snippet": r.get("snippet", ""),
            })
    except Exception as e:
        logger.warning("[%s] Google CSE failed: %s", target["name"][:40], e)

    time.sleep(_SEARCH_DELAY_SEC)

    # NewsAPI
    try:
        news_results = news_client.search(keywords, days_back=days_back, page_size=num_results)
        for r in news_results:
            pub_date = (r.get("publishedAt") or "")[:10]
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "source": r.get("source", "NewsAPI"),
                "date": pub_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "snippet": r.get("description", ""),
            })
    except Exception as e:
        logger.warning("[%s] NewsAPI failed: %s", target["name"][:40], e)

    return results


# -- Relevance filter --------------------------------------------------------

_RELEVANCE_MIN_SCORE = 2


def _tokenize_lower(text: str) -> set[str]:
    """Split text into lowercase tokens for matching."""
    import re as _re
    return {t.lower() for t in _re.split(r'[\s,.\-/;:?!！？「」（）()\[\]+]+', text) if len(t) >= 2}


def score_result_relevance(
    result: Dict[str, Any],
    target: Dict[str, Any],
) -> int:
    """Score how relevant a search result is to the target.

    Scoring:
    - target name tokens found in title or snippet → +2 per token hit
    - target search_keywords tokens found → +1 per token hit

    Returns the total score.
    """
    title_snippet = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

    score = 0

    # Target name tokens
    name = target.get("name", "")
    name_tokens = _tokenize_lower(name)
    for tok in name_tokens:
        if tok in title_snippet:
            score += 2

    # Search keywords tokens
    kw = target.get("search_keywords", "")
    if kw:
        kw_tokens = _tokenize_lower(kw) - name_tokens  # avoid double counting
        for tok in kw_tokens:
            if tok in title_snippet:
                score += 1

    return score


def filter_relevant_results(
    results: List[Dict[str, Any]],
    target: Dict[str, Any],
    *,
    min_score: int = _RELEVANCE_MIN_SCORE,
) -> List[Dict[str, Any]]:
    """Filter search results by relevance to the target."""
    relevant = []
    for r in results:
        s = score_result_relevance(r, target)
        if s >= min_score:
            r["relevance_score"] = s
            relevant.append(r)
        else:
            logger.debug(
                "[%s] Filtered out (score=%d): %s",
                target["name"][:30], s, r.get("title", "")[:60],
            )
    if len(results) != len(relevant):
        logger.info(
            "[%s] Relevance filter: %d → %d results",
            target["name"][:30], len(results), len(relevant),
        )
    return relevant


# -- Dedup -------------------------------------------------------------------


def compute_dedup_key(title: str, event_date: str, source_url: str) -> str:
    """SHA256 of title|date|source_url, truncated to 16 chars."""
    h_in = f"{title}|{event_date}|{source_url}".encode("utf-8")
    return hashlib.sha256(h_in).hexdigest()[:16]


def dedup_results(
    results: List[Dict[str, Any]],
    existing_keys: set[str],
) -> List[Dict[str, Any]]:
    """Remove results whose dedup_key already exists."""
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for r in results:
        key = compute_dedup_key(r["title"], r["date"], r["url"])
        if key in existing_keys or key in seen:
            continue
        seen.add(key)
        r["dedup_key"] = key
        unique.append(r)
    return unique


# -- Event creation ----------------------------------------------------------


def create_events(
    notion_client: Any,
    events_db_id: str,
    target: Dict[str, Any],
    results: List[Dict[str, Any]],
    *,
    run_id: str,
) -> int:
    """Write search results as events to EVENTS_DB. Returns count created."""
    now = datetime.now(timezone.utc)
    created = 0

    for r in results:
        event_date_str = r.get("date", now.strftime("%Y-%m-%d"))
        try:
            event_dt = datetime.strptime(event_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            event_dt = now

        properties = {
            "Name": {"title": [{"type": "text", "text": {"content": r["title"][:2000]}}]},
            "Date": {"date": {"start": event_date_str}},
            "Detected At": {"date": {"start": now.isoformat()}},
            "Event Type": {"select": {"name": _infer_event_type(target)}},
            "Source": {"select": {"name": r.get("source", "Unknown")[:100]}},
            "Source URL": {"url": r.get("url") or None},
            "Summary": {"rich_text": [{"type": "text", "text": {"content": (r.get("snippet") or "")[:2000]}}]},
            "Confidence": {"number": 0.7},
            "Status": {"select": {"name": "active"}},
            "Dedup Key": {"rich_text": [{"type": "text", "text": {"content": r.get("dedup_key", "")}}]},
            "Run ID": {"rich_text": [{"type": "text", "text": {"content": run_id}}]},
            "Ingested At": {"date": {"start": now.isoformat()}},
            "Action Needed": {"checkbox": False},
        }

        # Link to target
        target_page_id = target.get("page_id", "")
        if target_page_id:
            properties["Target"] = {"relation": [{"id": target_page_id}]}

        try:
            notion_client.create_page(parent_db_id=events_db_id, properties=properties)
            created += 1
        except Exception as e:
            logger.warning("Failed to create event for '%s': %s", r["title"][:40], e)

    return created


def _infer_event_type(target: Dict[str, Any]) -> str:
    """Map target type to event type."""
    t = (target.get("type") or "").strip()
    return {
        "VC": "VC News",
        "Startup": "Startup News",
        "Policy": "Policy News",
        "People": "People News",
    }.get(t, "Market News")


# -- Cadence state machine --------------------------------------------------


def apply_cadence_transition(
    target: Dict[str, Any],
    news_found: bool,
) -> Dict[str, Any]:
    """Apply cadence state machine rules. Returns dict of fields to update.

    State transitions (execution-count based):
    - W0 + news → W0, misses=0, last_hit_at=now
    - W0 + no news → misses+1; if misses>=3 → M0
    - M0 + news → W0, misses=0, last_hit_at=now
    - M0 + no news → misses+1; if misses>=3 → Paused
    """
    cadence_raw = (target.get("cadence") or "Weekly").strip()
    cadence = cadence_raw.lower()
    status = (target.get("status") or "Active").strip().lower()
    misses = target.get("consecutive_misses", 0) or 0
    today = date.today()
    now_iso = today.isoformat()

    updates: Dict[str, Any] = {
        "last_checked": now_iso,
    }

    if news_found:
        updates["consecutive_misses"] = 0
        updates["last_hit_at"] = now_iso
        if cadence == "monthly" and status == "active":
            # Monthly → Weekly promotion
            updates["cadence"] = "Weekly"
            updates["cadence_reason"] = "news found → weekly"
            updates["next_check"] = compute_next_check("Weekly", target["page_id"], today)
            logger.info("[%s] Monthly → Weekly (news found)", target["name"][:40])
        else:
            updates["next_check"] = compute_next_check(cadence_raw, target["page_id"], today)
    else:
        new_misses = misses + 1
        updates["consecutive_misses"] = new_misses

        if cadence in ("weekly", "daily") and new_misses >= WEEKLY_TO_MONTHLY_MISSES:
            # Weekly → Monthly demotion
            updates["cadence"] = "Monthly"
            updates["consecutive_misses"] = 0
            updates["cadence_reason"] = f"{WEEKLY_TO_MONTHLY_MISSES} weekly misses → monthly"
            updates["next_check"] = compute_next_check("Monthly", target["page_id"], today)
            logger.info("[%s] Weekly → Monthly (%d misses)", target["name"][:40], new_misses)
        elif cadence == "monthly" and new_misses >= MONTHLY_TO_PAUSED_MISSES:
            # Monthly → Paused
            updates["status"] = "Paused"
            updates["cadence_reason"] = f"{MONTHLY_TO_PAUSED_MISSES} monthly misses → paused"
            updates["next_check"] = None
            logger.info("[%s] Monthly → Paused (%d misses)", target["name"][:40], new_misses)
        else:
            updates["next_check"] = compute_next_check(cadence_raw, target["page_id"], today)

    return updates


# -- Notion target update ----------------------------------------------------


def update_target_operational(
    notion_client: Any,
    target_page_id: str,
    updates: Dict[str, Any],
) -> bool:
    """Update MONITORING_TARGETS_DB operational fields."""
    properties: Dict[str, Any] = {}

    if "last_checked" in updates:
        val = updates["last_checked"]
        properties["Last Checked"] = {"date": {"start": val}} if val else {"date": None}

    if "next_check" in updates:
        val = updates["next_check"]
        if isinstance(val, date):
            val = val.isoformat()
        properties["Next Check"] = {"date": {"start": val}} if val else {"date": None}

    if "last_hit_at" in updates:
        val = updates["last_hit_at"]
        properties["Last Hit At"] = {"date": {"start": val}} if val else {"date": None}

    if "consecutive_misses" in updates:
        properties["Consecutive Misses"] = {"number": float(updates["consecutive_misses"])}

    if "cadence_reason" in updates:
        properties["Cadence Reason"] = {
            "rich_text": [{"type": "text", "text": {"content": updates["cadence_reason"][:2000]}}]
        }

    if "cadence" in updates:
        properties["Cadence"] = {"select": {"name": updates["cadence"]}}

    if "status" in updates:
        properties["Status"] = {"select": {"name": updates["status"]}}

    if not properties:
        return True

    try:
        notion_client.update_page(page_id=target_page_id, properties=properties)
        return True
    except Exception as e:
        logger.error("Failed to update target %s: %s", target_page_id, e)
        return False


# -- Fetch existing dedup keys -----------------------------------------------


def fetch_recent_dedup_keys(
    notion_client: Any,
    events_data_source_id: str,
    days_back: int = 7,
) -> set[str]:
    """Fetch dedup keys from recent events for pre-dedup."""
    since = (date.today() - timedelta(days=days_back)).isoformat()
    filt = {
        "property": "Ingested At",
        "date": {"on_or_after": since},
    }
    try:
        pages = notion_client.query_data_source(
            data_source_id=events_data_source_id,
            filter=filt,
            fetch_all=True,
        )
    except Exception as e:
        logger.warning("Failed to fetch recent events for dedup: %s", e)
        return set()

    from src.notion.properties import extract_property_value as ev

    keys: set[str] = set()
    for page in pages:
        dk = ev(page, "Dedup Key") or ""
        if dk:
            keys.add(dk)
    logger.info("Fetched %d existing dedup keys (last %d days)", len(keys), days_back)
    return keys


# -- Batch processing --------------------------------------------------------


def run_smart_monitor(
    *,
    notion_client: Any,
    targets_data_source_id: str,
    events_db_id: str,
    events_data_source_id: str,
    google_client: Any,
    news_client: Any,
    dry_run: bool = False,
    limit: int = 0,
    target_type: str = "",
    run_id: str = "",
) -> Dict[str, Any]:
    """Run the smart news monitoring pipeline.

    Returns a summary dict with counts and per-target results.
    """
    from src.notion.targets_normalize import normalize_targets

    summary: Dict[str, Any] = {
        "total_targets": 0,
        "due_today": 0,
        "searched": 0,
        "events_created": 0,
        "promotions": 0,
        "demotions": 0,
        "paused": 0,
        "errors": 0,
        "results": [],
        "dry_run": dry_run,
    }

    # Fetch all targets
    pages = notion_client.query_data_source(
        data_source_id=targets_data_source_id,
        filter={"property": "Enabled", "checkbox": {"equals": True}},
        fetch_all=True,
    )
    all_targets = normalize_targets(pages)
    summary["total_targets"] = len(all_targets)

    # Filter by type if specified
    if target_type:
        all_targets = [t for t in all_targets if t.get("type", "").lower() == target_type.lower()]
        logger.info("Filtered to type=%s: %d targets", target_type, len(all_targets))

    # Get today's targets
    today = date.today()
    due = get_todays_targets(all_targets, today)
    if limit > 0:
        due = due[:limit]
    summary["due_today"] = len(due)

    if not due:
        logger.info("No targets due today")
        return summary

    if dry_run:
        logger.info("=== DRY-RUN: %d targets due ===", len(due))
        for t in due:
            next_check = compute_next_check(t.get("cadence", "Weekly"), t["page_id"], today)
            # Simulate transition for display
            transition = apply_cadence_transition(t, news_found=False)
            logger.info(
                "  [%s] type=%s cadence=%s misses=%d next_check=%s → would_become: cadence=%s status=%s",
                t["name"][:40],
                t.get("type", "?"),
                t.get("cadence", "?"),
                t.get("consecutive_misses", 0),
                next_check,
                transition.get("cadence", t.get("cadence")),
                transition.get("status", t.get("status")),
            )
            summary["results"].append({
                "target_name": t["name"],
                "type": t.get("type"),
                "cadence": t.get("cadence"),
                "consecutive_misses": t.get("consecutive_misses", 0),
                "simulated_no_news_transition": transition,
                "dry_run": True,
            })
        return summary

    # Fetch existing dedup keys for pre-dedup
    existing_keys = fetch_recent_dedup_keys(notion_client, events_data_source_id)

    if not run_id:
        run_id = f"075_{today.isoformat()}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

    # Process each target
    for i, target in enumerate(due, 1):
        logger.info("--- Processing %d/%d: %s ---", i, len(due), target["name"][:40])

        try:
            # Search
            results = search_target(
                target,
                google_client=google_client,
                news_client=news_client,
            )

            # Relevance filter — drop results unrelated to the target
            results = filter_relevant_results(results, target)

            # Dedup
            unique_results = dedup_results(results, existing_keys)
            news_found = len(unique_results) > 0

            # Create events
            events_created = 0
            if unique_results:
                events_created = create_events(
                    notion_client, events_db_id, target, unique_results, run_id=run_id,
                )
                # Add new keys to existing set for in-run dedup
                for r in unique_results:
                    if "dedup_key" in r:
                        existing_keys.add(r["dedup_key"])

            # Cadence transition
            transition = apply_cadence_transition(target, news_found)

            # Update target in Notion
            updated = update_target_operational(
                notion_client, target["page_id"], transition,
            )

            # Track stats
            summary["searched"] += 1
            summary["events_created"] += events_created

            if transition.get("cadence") != target.get("cadence"):
                if transition.get("cadence") == "Weekly":
                    summary["promotions"] += 1
                elif transition.get("cadence") == "Monthly":
                    summary["demotions"] += 1
            if transition.get("status") == "Paused":
                summary["paused"] += 1
            if not updated:
                summary["errors"] += 1

            summary["results"].append({
                "target_name": target["name"],
                "type": target.get("type"),
                "results_found": len(results),
                "unique_results": len(unique_results),
                "events_created": events_created,
                "news_found": news_found,
                "transition": transition,
            })

            logger.info(
                "[%s] found=%d unique=%d events=%d → misses=%s cadence=%s",
                target["name"][:40],
                len(results),
                len(unique_results),
                events_created,
                transition.get("consecutive_misses"),
                transition.get("cadence", target.get("cadence")),
            )

        except Exception as e:
            logger.error("[%s] Processing failed: %s", target["name"][:40], e)
            summary["errors"] += 1
            summary["results"].append({
                "target_name": target["name"],
                "error": str(e),
            })

        # Notion API rate limit
        time.sleep(_NOTION_DELAY_SEC)

    return summary


# -- Migration ---------------------------------------------------------------


def migrate_targets(
    *,
    notion_client: Any,
    targets_data_source_id: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """One-shot migration: set Next Check + Cadence=Weekly for Active targets.

    Normalizes legacy cadence values (DAILY/WEEKLY → Weekly) and computes
    hash-based Next Check dates for even distribution across the week.
    """
    from src.notion.targets_normalize import normalize_targets

    pages = notion_client.query_data_source(
        data_source_id=targets_data_source_id,
        filter={"property": "Enabled", "checkbox": {"equals": True}},
        fetch_all=True,
    )
    all_targets = normalize_targets(pages)
    today = date.today()

    migrated = 0
    skipped = 0

    for t in all_targets:
        status = (t.get("status") or "").strip().lower()
        if status != "active":
            skipped += 1
            continue

        cadence_raw = (t.get("cadence") or "").strip().lower()
        # Normalize: DAILY/WEEKLY/empty → Weekly
        new_cadence = "Weekly"
        if cadence_raw == "monthly":
            new_cadence = "Monthly"

        next_check = compute_next_check(new_cadence, t["page_id"], today)

        logger.info(
            "  [%s] %s → cadence=%s next_check=%s",
            t["name"][:40],
            cadence_raw or "(empty)",
            new_cadence,
            next_check,
        )

        if not dry_run:
            update_target_operational(
                notion_client,
                t["page_id"],
                {
                    "cadence": new_cadence,
                    "next_check": next_check,
                    "consecutive_misses": 0,
                },
            )
            time.sleep(_NOTION_DELAY_SEC)

        migrated += 1

    summary = {
        "total": len(all_targets),
        "migrated": migrated,
        "skipped": skipped,
        "dry_run": dry_run,
    }
    logger.info(
        "Migration complete: %d migrated, %d skipped (dry_run=%s)",
        migrated, skipped, dry_run,
    )
    return summary
