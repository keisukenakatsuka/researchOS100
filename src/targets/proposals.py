# src/targets/proposals.py
"""Rule-based proposal generation for weekly targets review.

Pure functions that take enriched target records (with metrics) and events,
and produce keep/drop/review decisions, priority/cadence suggestions, and
keyword tuning proposals — all with explicit reason strings.

No LLM, no embeddings, no write-back. Designed to be transparent and
easy to extend.

Usage::

    from src.targets.proposals import propose_target_actions

    proposals = propose_target_actions(enriched_targets, events)
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set

from src.notion.targets_schema import (
    CADENCE_OPTIONS,
    DOWNGRADE_EFFECTIVENESS_THRESHOLD,
    HIGH_NOISE_THRESHOLD,
    KEYWORD_MAX_SUGGESTIONS,
    KEYWORD_MIN_EVENT_COUNT,
    KEYWORD_MIN_TOKEN_LEN,
    KEYWORD_NOISE_CONFIDENCE_THRESHOLD,
    KEYWORD_NOISE_RATIO_THRESHOLD,
    KEYWORD_STOPWORDS,
    MIN_SIGNAL_THRESHOLD,
    NO_EVENT_DAYS_THRESHOLD,
    PRIORITY_ORDER,
    REMOVAL_EFFECTIVENESS_THRESHOLD,
    UPGRADE_EFFECTIVENESS_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Priority / cadence helpers
# ----------------------------------------------------------------

def _priority_index(priority: str) -> int:
    p = (priority or "").strip()
    return PRIORITY_ORDER.index(p) if p in PRIORITY_ORDER else 1  # default Medium

def _adjust_priority(current: str, direction: str) -> str:
    idx = _priority_index(current)
    if direction == "up" and idx < len(PRIORITY_ORDER) - 1:
        return PRIORITY_ORDER[idx + 1]
    if direction == "down" and idx > 0:
        return PRIORITY_ORDER[idx - 1]
    return current

def _cadence_index(cadence: str) -> int:
    c = (cadence or "").strip()
    return CADENCE_OPTIONS.index(c) if c in CADENCE_OPTIONS else 1  # default Weekly

def _adjust_cadence(current: str, direction: str) -> str:
    """Adjust cadence. 'up' = more frequent (lower index), 'down' = less frequent."""
    idx = _cadence_index(current)
    if direction == "up" and idx > 0:
        return CADENCE_OPTIONS[idx - 1]
    if direction == "down" and idx < len(CADENCE_OPTIONS) - 1:
        return CADENCE_OPTIONS[idx + 1]
    return current


# ----------------------------------------------------------------
# Tokenization  (with noise controls)
# ----------------------------------------------------------------

_NUM_RE = re.compile(r"^\d+$")


def _tokenize_for_keywords(text: str) -> List[str]:
    """Tokenize text into clean keyword tokens.

    Noise controls:
    - lowercase, strip punctuation
    - ignore tokens < KEYWORD_MIN_TOKEN_LEN chars
    - ignore pure numbers
    - ignore URL fragments
    - ignore stopwords (EN + JA)
    """
    # Split on whitespace + common punctuation, keeping unicode for JA
    raw_tokens = re.findall(r"[\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+",
                            (text or "").lower())
    clean = []
    for t in raw_tokens:
        if len(t) < KEYWORD_MIN_TOKEN_LEN:
            continue
        if _NUM_RE.match(t):
            continue
        if t in KEYWORD_STOPWORDS:
            continue
        clean.append(t)
    return clean


# ----------------------------------------------------------------
# Priority & cadence proposals
# ----------------------------------------------------------------

def propose_priority_and_cadence(
    target: Dict[str, Any],
    *,
    min_signal: float = MIN_SIGNAL_THRESHOLD,
    max_noise: float = HIGH_NOISE_THRESHOLD,
) -> Dict[str, Any]:
    """Propose action + priority/cadence adjustments for a single target.

    Returns a dict with: action, proposed_priority, proposed_cadence, reason.
    """
    sig = target.get("signal_score", 0.0)
    noise = target.get("noise_score", 0.0)
    n_events = target.get("number_of_events", 0)
    days_since = target.get("days_since_last_event")
    error_count = target.get("error_count", 0)
    share_action = target.get("share_action_needed", 0.0)
    current_priority = target.get("priority", "Medium")
    current_cadence = target.get("cadence", "Weekly")

    action = "keep"
    reason_parts: List[str] = []
    proposed_priority = current_priority
    proposed_cadence = current_cadence

    # --- Rule A: Disable / remove candidates ---
    if sig < REMOVAL_EFFECTIVENESS_THRESHOLD:
        action = "drop_candidate"
        reason_parts.append(f"very low signal ({sig:.2f} < {REMOVAL_EFFECTIVENESS_THRESHOLD})")
    elif noise > 0.80 and n_events < 3:
        action = "drop_candidate"
        reason_parts.append(f"high noise ({noise:.2f}) with few events ({n_events})")
    elif n_events == 0 and sig < DOWNGRADE_EFFECTIVENESS_THRESHOLD and (
        days_since is None or days_since >= NO_EVENT_DAYS_THRESHOLD
    ):
        action = "review"
        reason_parts.append(
            f"no events recently (days_since={days_since}) "
            f"and low signal ({sig:.2f})"
        )

    # --- Rule B: Upgrade candidates ---
    if action == "keep" and sig >= UPGRADE_EFFECTIVENESS_THRESHOLD:
        proposed_priority = _adjust_priority(current_priority, "up")
        if proposed_priority != current_priority:
            reason_parts.append(
                f"high signal ({sig:.2f}) → suggest priority "
                f"{current_priority} → {proposed_priority}"
            )

    # --- Rule C: Downgrade candidates ---
    if action == "keep" and sig < DOWNGRADE_EFFECTIVENESS_THRESHOLD and noise > max_noise:
        proposed_priority = _adjust_priority(current_priority, "down")
        action = "review"
        reason_parts.append(
            f"low signal ({sig:.2f}) + high noise ({noise:.2f}) → suggest priority "
            f"{current_priority} → {proposed_priority}"
        )

    # --- Rule D: Cadence adjustments ---
    if action in ("keep", "review"):
        # Increase frequency for high-signal targets
        if sig >= 0.70 and n_events >= 3 and (share_action >= 0.30 or sig >= 0.80):
            new_cadence = _adjust_cadence(current_cadence, "up")
            if new_cadence != current_cadence:
                proposed_cadence = new_cadence
                reason_parts.append(
                    f"strong signal ({sig:.2f}, {n_events} events) → "
                    f"cadence {current_cadence} → {proposed_cadence}"
                )
        # Decrease frequency for low-signal / dormant
        elif n_events == 0 or (sig < min_signal and noise > max_noise):
            new_cadence = _adjust_cadence(current_cadence, "down")
            if new_cadence != current_cadence:
                proposed_cadence = new_cadence
                reason_parts.append(
                    f"low activity (events={n_events}, signal={sig:.2f}) → "
                    f"cadence {current_cadence} → {proposed_cadence}"
                )

    # --- Rule E: Error-count operational review ---
    if error_count and error_count >= 3:
        if action == "keep":
            action = "review"
        reason_parts.append(f"error_count={error_count} — operational review needed")

    if not reason_parts:
        reason_parts.append(f"signal={sig:.2f}, noise={noise:.2f}, events={n_events} — no changes")

    return {
        "action": action,
        "proposed_priority": proposed_priority,
        "proposed_cadence": proposed_cadence,
        "reason": "; ".join(reason_parts),
    }


# ----------------------------------------------------------------
# Keyword tuning proposals
# ----------------------------------------------------------------

def propose_keyword_tuning(
    target: Dict[str, Any],
    related_events: List[Dict[str, Any]],
    *,
    keyword_min_count: int = KEYWORD_MIN_EVENT_COUNT,
    max_suggestions: int = KEYWORD_MAX_SUGGESTIONS,
) -> Dict[str, Any]:
    """Propose keyword changes for a single target based on linked events.

    Returns a dict with:
    - ``keywords_to_add``: top-N frequent tokens not in current search_keywords
    - ``keywords_stale``: current keywords absent from events (stale candidates only)
    - ``keywords_to_exclude``: tokens correlated with noise (low-confidence events)
    """
    # Parse current keywords
    raw_kw = target.get("search_keywords", "") or ""
    current_tokens = set(_tokenize_for_keywords(raw_kw))

    # Tokenize event text
    token_events: Dict[str, List[str]] = defaultdict(list)  # token → event titles
    token_confidence: Dict[str, List[float]] = defaultdict(list)
    total_events = len(related_events)

    for ev in related_events:
        title = ev.get("title", "")
        summary = ev.get("summary_text", "")
        tokens = _tokenize_for_keywords(f"{title} {summary}")
        conf = ev.get("confidence", 0.5)
        try:
            conf = float(conf)
        except (ValueError, TypeError):
            conf = 0.5

        for t in set(tokens):  # unique per event
            token_events[t].append(title[:80])
            token_confidence[t].append(conf)

    # --- keywords_to_add ---
    # Frequent tokens NOT in current keywords
    candidates = []
    for token, titles in token_events.items():
        if token in current_tokens:
            continue
        count = len(titles)
        if count < keyword_min_count:
            continue
        candidates.append({
            "keyword": token,
            "count": count,
            "sample_events": titles[:3],
            "reason": f"appeared in {count}/{total_events} events, not in current keywords",
        })
    candidates.sort(key=lambda c: -c["count"])
    keywords_to_add = candidates[:max_suggestions]

    # --- keywords_stale ---
    # Current keywords with 0 appearances (stale candidates, NOT removal)
    event_token_set = set(token_events.keys())
    keywords_stale = []
    for kw in sorted(current_tokens):
        if kw not in event_token_set:
            keywords_stale.append({
                "keyword": kw,
                "reason": (
                    f"appeared 0 times in {total_events} events this week; "
                    f"consider reviewing if trend persists across multiple weeks"
                ),
            })

    # --- keywords_to_exclude ---
    # Tokens where most appearances are in low-confidence events
    keywords_to_exclude = []
    for token, confs in token_confidence.items():
        if len(confs) < keyword_min_count:
            continue
        low_count = sum(1 for c in confs if c < KEYWORD_NOISE_CONFIDENCE_THRESHOLD)
        ratio = low_count / len(confs)
        if ratio >= KEYWORD_NOISE_RATIO_THRESHOLD:
            keywords_to_exclude.append({
                "keyword": token,
                "total_appearances": len(confs),
                "low_confidence_appearances": low_count,
                "noise_ratio": round(ratio, 2),
                "reason": (
                    f"{low_count}/{len(confs)} appearances are low-confidence "
                    f"(< {KEYWORD_NOISE_CONFIDENCE_THRESHOLD}); noise ratio {ratio:.0%}"
                ),
            })
    keywords_to_exclude.sort(key=lambda x: -x["noise_ratio"])
    keywords_to_exclude = keywords_to_exclude[:max_suggestions]

    return {
        "target_id": target.get("page_id", ""),
        "target_name": target.get("name", ""),
        "current_keywords": sorted(current_tokens),
        "total_linked_events": total_events,
        "keywords_to_add": keywords_to_add,
        "keywords_stale": keywords_stale,
        "keywords_to_exclude": keywords_to_exclude,
    }


# ----------------------------------------------------------------
# Top-level orchestrator
# ----------------------------------------------------------------

def propose_target_actions(
    enriched_targets: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    *,
    min_signal: float = MIN_SIGNAL_THRESHOLD,
    max_noise: float = HIGH_NOISE_THRESHOLD,
    keyword_min_count: int = KEYWORD_MIN_EVENT_COUNT,
    max_keyword_suggestions: int = KEYWORD_MAX_SUGGESTIONS,
) -> List[Dict[str, Any]]:
    """Generate review proposals for all targets.

    Returns a list of per-target proposal dicts combining:
    - priority/cadence suggestions
    - keyword tuning suggestions
    - evidence (linked event IDs, scores)

    Parameters
    ----------
    enriched_targets:
        Output of ``compute_target_metrics()``.
    events:
        Full events list (from 048 output). Used for keyword analysis.
    """
    # Build target_id → events index (reuse normalised IDs)
    from src.notion.targets_normalize import _normalize_uuid
    events_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        tids = ev.get("target_ids") or []
        if isinstance(tids, str):
            tids = [t.strip() for t in tids.split(",") if t.strip()]
        for tid in tids:
            events_by_target[_normalize_uuid(tid)].append(ev)

    proposals = []
    for t in enriched_targets:
        pid = _normalize_uuid(t["page_id"])

        # Priority & cadence
        pc = propose_priority_and_cadence(t, min_signal=min_signal, max_noise=max_noise)

        # Keyword tuning
        linked_events = events_by_target.get(pid, [])
        kw = propose_keyword_tuning(
            t, linked_events,
            keyword_min_count=keyword_min_count,
            max_suggestions=max_keyword_suggestions,
        )

        proposals.append({
            # Identity
            "target_id": pid,
            "target_name": t.get("name", ""),
            "target_type": t.get("type", ""),
            # Current state
            "current_priority": t.get("priority", ""),
            "current_cadence": t.get("cadence", ""),
            # Metrics
            "signal_score": t.get("signal_score", 0.0),
            "noise_score": t.get("noise_score", 0.0),
            "number_of_events": t.get("number_of_events", 0),
            "days_since_last_event": t.get("days_since_last_event"),
            "recency_score": t.get("recency_score", 0.0),
            # Decisions
            "action": pc["action"],
            "proposed_priority": pc["proposed_priority"],
            "proposed_cadence": pc["proposed_cadence"],
            "reason": pc["reason"],
            # Keyword suggestions
            "keyword_suggestions": kw,
            # Evidence
            "linked_event_ids": t.get("linked_event_ids", []),
        })

    # Tally
    action_counts: Dict[str, int] = defaultdict(int)
    for p in proposals:
        action_counts[p["action"]] += 1
    logger.info("Proposals: %s", dict(action_counts))

    return proposals
