# src/discovery/entities.py
"""Entity extraction from events and papers — pure functions, no I/O.

Extracts candidate entity names (organizations, people, policy bodies,
products) from unstructured text fields.  Two complementary strategies:

1. **Capitalized-phrase extraction** — multi-token proper nouns
   ("Sam Altman", "Andreessen Horowitz", "European Commission").
2. **High-frequency token extraction** — single tokens that recur
   across multiple records (mirrors 050 keyword tokenization).

Both strategies share the noise controls from
:mod:`src.notion.targets_schema` (stopwords, min length, URL filtering).

Usage::

    from src.discovery.entities import (
        extract_entities_from_events,
        extract_entities_from_papers,
        merge_raw_entities,
    )

    raw_ev = extract_entities_from_events(events)
    raw_pa = extract_entities_from_papers(papers)
    merged = merge_raw_entities(raw_ev + raw_pa)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.notion.targets_schema import (
    KEYWORD_MIN_TOKEN_LEN,
    KEYWORD_STOPWORDS,
)

# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------

# Entity extraction controls
ENTITY_MAX_NGRAM: int = 4          # capitalized phrases up to N tokens
ENTITY_MIN_PHRASE_LEN: int = 2     # min chars for a phrase (after strip)
ENTITY_SINGLE_TOKEN_MIN_COUNT: int = 2  # single caps tokens need ≥N mentions

# Additional stopwords specific to entity extraction
# (broader than keyword stopwords — filters generic nouns that look like names)
_ENTITY_EXTRA_STOPS = frozenset({
    # Common title words that get capitalized (full + abbreviated)
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # Common section headers / boilerplate
    "breaking", "exclusive", "update", "report", "analysis", "notice",
    "correction", "guidance", "pinned", "applications", "agreements",
    "sources", "reuters", "bloomberg", "associated", "press",
    "photo", "image", "video", "read", "more", "click", "here",
    # Generic business terms that appear capitalised
    "ceo", "cto", "cfo", "coo", "vp", "svp", "evp",
    "inc", "ltd", "corp", "llc", "plc",
    # Common words that get capitalised at sentence start
    "new", "says", "said", "set", "get", "use", "now", "just", "its",
    "one", "two", "three", "four", "five", "most", "each", "next",
    "last", "first", "year", "week", "month", "day", "time",
    # Short abbreviations / noise fragments common in news/policy text
    "par", "vis", "pro", "ads", "fig", "ref", "vol", "nos", "iss",
    "news", "data", "info", "tech", "dept", "sect", "app", "dev",
    "note", "memo", "file", "list", "item", "page", "link",
    # Geographic false positives
    "delhi",
})

_ALL_STOPS = KEYWORD_STOPWORDS | _ENTITY_EXTRA_STOPS

_NUM_RE = re.compile(r"^\d+$")
_URL_FRAG_RE = re.compile(r"^(https?|www|com|org|net|html|php|jpg|png|gif)$", re.I)

# Pattern for capitalized phrases in Latin text
# Matches sequences of Capitalized words (Title Case), allowing hyphens/apostrophes
_CAP_WORD = r"[A-Z][a-z''\-]*[a-z]+"  # e.g. "Altman", "O'Brien"
_CAP_ABBR = r"[A-Z]{2,}"               # e.g. "NVIDIA", "OECD", "AI"
_CAP_MIXED = r"[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*"  # e.g. "OpenAI", "DeepSeek"
_CAP_TOKEN = rf"(?:{_CAP_WORD}|{_CAP_ABBR}|{_CAP_MIXED})"

# Multi-token: 1-4 capitalized tokens, optionally joined by "of", "and", "the", "for", "de", "du"
_LINK_WORDS = r"(?:of|and|the|for|de|du|&)"
_CAP_PHRASE_RE = re.compile(
    rf"(?<![.\w])"  # not preceded by word char or period (avoids mid-sentence)
    rf"({_CAP_TOKEN}"
    rf"(?:\s+(?:{_LINK_WORDS}\s+)?{_CAP_TOKEN})"
    rf"{{0,{ENTITY_MAX_NGRAM - 1}}})"
    rf"(?![.\w])",  # not followed by word char
)

# Japanese org/person patterns (simple: katakana sequences ≥3 chars)
_JA_KATA_RE = re.compile(r"[\u30A0-\u30FF]{3,}")
_JA_KANJI_ORG_RE = re.compile(r"[\u4E00-\u9FFF]{2,}(?:省|庁|機構|委員会|研究所|大学|銀行|基金)")


# ----------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------

def _normalize_entity_name(name: str) -> str:
    """Canonical form for dedup: strip, collapse whitespace, lowercase."""
    return " ".join(name.split()).lower().strip()


def _is_noise_token(tok: str) -> bool:
    """True if token should be filtered out."""
    low = tok.lower()
    if len(low) < KEYWORD_MIN_TOKEN_LEN:
        return True
    if _NUM_RE.match(low):
        return True
    if _URL_FRAG_RE.match(low):
        return True
    if low in _ALL_STOPS:
        return True
    return False


def _extract_cap_phrases(text: str) -> List[str]:
    """Extract capitalized multi-token phrases from text."""
    if not text:
        return []
    phrases = []
    for m in _CAP_PHRASE_RE.finditer(text):
        phrase = m.group(1).strip()
        # Filter: all tokens must pass noise check individually
        tokens = phrase.split()
        # Allow link words through (of, and, the, for)
        content_tokens = [t for t in tokens if t.lower() not in
                          {"of", "and", "the", "for", "de", "du", "&"}]
        if not content_tokens:
            continue
        # At least one content token must not be noise
        if all(_is_noise_token(t) for t in content_tokens):
            continue
        if len(phrase) < ENTITY_MIN_PHRASE_LEN:
            continue
        phrases.append(phrase)
    return phrases


def _extract_ja_entities(text: str) -> List[str]:
    """Extract Japanese entity candidates (katakana sequences, kanji orgs)."""
    if not text:
        return []
    entities = []
    for m in _JA_KATA_RE.finditer(text):
        entities.append(m.group(0))
    for m in _JA_KANJI_ORG_RE.finditer(text):
        entities.append(m.group(0))
    return entities


# ----------------------------------------------------------------
# Public extraction functions
# ----------------------------------------------------------------

def extract_entities_from_events(
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract raw entity mentions from 048 events.

    Returns a list of raw mention dicts::

        {
            "entity_name": "Sam Altman",
            "source_type": "event",
            "source_id": "<page_id>",
            "source_title": "...",
            "event_type": "PEOPLE",
            "confidence": 0.8,
            "action_needed": true,
        }
    """
    mentions: List[Dict[str, Any]] = []

    for ev in events:
        title = ev.get("title", "") or ""
        summary = ev.get("summary_text", "") or ""
        page_id = ev.get("page_id", "")
        event_type = ev.get("event_type", "")
        confidence = ev.get("confidence", 0.5)
        action_needed = ev.get("action_needed", False)

        # Extract from title + summary
        text_fields = [title, summary]
        seen_in_event: Set[str] = set()

        for text in text_fields:
            # Capitalized phrases (Latin)
            for phrase in _extract_cap_phrases(text):
                norm = _normalize_entity_name(phrase)
                if norm in seen_in_event:
                    continue
                seen_in_event.add(norm)
                mentions.append({
                    "entity_name": phrase,
                    "entity_name_normalized": norm,
                    "source_type": "event",
                    "source_id": page_id,
                    "source_title": title[:120],
                    "event_type": event_type,
                    "confidence": confidence,
                    "action_needed": action_needed,
                })

            # Japanese entities
            for ja_ent in _extract_ja_entities(text):
                norm = _normalize_entity_name(ja_ent)
                if norm in seen_in_event:
                    continue
                seen_in_event.add(norm)
                mentions.append({
                    "entity_name": ja_ent,
                    "entity_name_normalized": norm,
                    "source_type": "event",
                    "source_id": page_id,
                    "source_title": title[:120],
                    "event_type": event_type,
                    "confidence": confidence,
                    "action_needed": action_needed,
                })

    return mentions


def extract_entities_from_papers(
    papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract raw entity mentions from 047 papers.

    Returns mention dicts similar to events but with paper-specific fields.
    """
    mentions: List[Dict[str, Any]] = []

    for paper in papers:
        name = paper.get("Name", "") or ""
        tags = paper.get("Tags", "") or ""
        core_idea = paper.get("Core Idea", "") or ""
        findings = paper.get("Findings", "") or ""
        page_id = paper.get("notion_page_id", "")

        text_fields = [name, tags, core_idea, findings]
        seen_in_paper: Set[str] = set()

        for text in text_fields:
            for phrase in _extract_cap_phrases(text):
                norm = _normalize_entity_name(phrase)
                if norm in seen_in_paper:
                    continue
                seen_in_paper.add(norm)
                mentions.append({
                    "entity_name": phrase,
                    "entity_name_normalized": norm,
                    "source_type": "paper",
                    "source_id": page_id,
                    "source_title": name[:120],
                    "event_type": "",
                    "confidence": 1.0,  # papers are curated
                    "action_needed": False,
                })

            for ja_ent in _extract_ja_entities(text):
                norm = _normalize_entity_name(ja_ent)
                if norm in seen_in_paper:
                    continue
                seen_in_paper.add(norm)
                mentions.append({
                    "entity_name": ja_ent,
                    "entity_name_normalized": norm,
                    "source_type": "paper",
                    "source_id": page_id,
                    "source_title": name[:120],
                    "event_type": "",
                    "confidence": 1.0,
                    "action_needed": False,
                })

    return mentions


def merge_raw_entities(
    mentions: List[Dict[str, Any]],
    *,
    min_count: int = 1,
) -> List[Dict[str, Any]]:
    """Merge raw mentions by normalized name.

    Returns a list of merged entity dicts::

        {
            "candidate_name": "Sam Altman",            # best-cased form
            "candidate_name_normalized": "sam altman",
            "mention_count": 5,
            "source_count": 3,                         # unique source IDs
            "event_types": ["PEOPLE", "STARTUP"],      # distinct types
            "source_types": ["event", "paper"],
            "avg_confidence": 0.75,
            "action_needed_count": 2,
            "sample_event_titles": ["...", "...", "..."],
            "sample_paper_titles": ["..."],
            "event_ids": ["id1", "id2", ...],
        }
    """
    # Group by normalized name
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in mentions:
        groups[m["entity_name_normalized"]].append(m)

    merged = []
    for norm_name, group in groups.items():
        count = len(group)
        if count < min_count:
            continue

        # Pick the best-cased display name: prefer the most common casing
        name_counter = Counter(m["entity_name"] for m in group)
        best_name = name_counter.most_common(1)[0][0]

        # Unique sources
        source_ids = set(m["source_id"] for m in group if m["source_id"])
        event_types = sorted(set(m["event_type"] for m in group if m["event_type"]))
        source_types = sorted(set(m["source_type"] for m in group))

        # Confidence
        confs = [m["confidence"] for m in group]
        avg_conf = sum(confs) / len(confs) if confs else 0.5

        # Action needed
        action_count = sum(1 for m in group if m.get("action_needed"))

        # Sample titles (deduped, capped)
        ev_titles = list(dict.fromkeys(
            m["source_title"] for m in group
            if m["source_type"] == "event" and m["source_title"]
        ))[:3]
        pa_titles = list(dict.fromkeys(
            m["source_title"] for m in group
            if m["source_type"] == "paper" and m["source_title"]
        ))[:2]

        # Event IDs
        ev_ids = list(dict.fromkeys(
            m["source_id"] for m in group
            if m["source_type"] == "event" and m["source_id"]
        ))

        merged.append({
            "candidate_name": best_name,
            "candidate_name_normalized": norm_name,
            "aliases": [],             # populated by _dedup_substrings
            "mention_count": count,
            "source_count": len(source_ids),
            "event_types": event_types,
            "source_types": source_types,
            "avg_confidence": round(avg_conf, 4),
            "action_needed_count": action_count,
            "sample_event_titles": ev_titles,
            "sample_paper_titles": pa_titles,
            "event_ids": ev_ids,
        })

    # Substring dedup: absorb *safe* variants only (same-entity phrases),
    # preserving aliases for traceability.
    merged = _dedup_substrings(merged)

    return merged


# ----------------------------------------------------------------
# Substring dedup — safe merging with aliases
# ----------------------------------------------------------------

# Tokens that signal the longer phrase is a *different entity* from
# the shorter substring and must NOT be absorbed.
# e.g. "OpenAI" ≠ "OpenAI CEO Sam Altman" (org ≠ person phrase).
_CROSS_ENTITY_TOKENS = frozenset({
    "ceo", "cto", "cfo", "coo", "founder", "president", "chairman",
    "director", "chief", "head", "vp", "partner", "manager",
    "ventures", "capital", "partners", "fund", "group", "holdings",
    "institute", "commission", "agency", "ministry", "university",
})


def _is_safe_to_merge(shorter: str, longer: str) -> bool:
    """Return True if *longer* is a benign variant of *shorter*.

    We refuse to merge when the extra tokens in *longer* contain
    cross-entity words (job titles, org suffixes) that indicate the
    longer phrase refers to a **different entity class**.

    Examples that should NOT merge:
      "openai" → "openai ceo sam altman"   (org ≠ person)
      "nvidia" → "nvidia ceo jensen huang" (org ≠ person)
      "seligman" → "seligman ventures"     (person? ≠ vc firm)

    Examples that SHOULD merge (safe variants of the same entity):
      "chatgpt" → "chatgpt advertising"    (product + noise)
      "nih"     → "nih guide for grants"   (org + boilerplate)
      "india ai"→ "india ai impact summit" (event name variant)
    """
    # Tokens in the longer phrase that are NOT in the shorter one
    short_tokens = set(shorter.split())
    long_tokens = longer.split()
    extra_tokens = [t for t in long_tokens if t not in short_tokens]

    # If any extra token is a cross-entity signal, refuse merge
    for tok in extra_tokens:
        if tok in _CROSS_ENTITY_TOKENS:
            return False

    # If the extra tokens contain what looks like a person name
    # (a capitalized word that isn't a stopword), and the shorter
    # entity is a single token (likely an org), refuse merge
    if len(short_tokens) == 1 and len(extra_tokens) >= 1:
        # Check if any extra token could be a person name component
        # Person-name tokens: not in stopwords, not a known org suffix
        for tok in extra_tokens:
            if tok not in _ALL_STOPS and tok not in _CROSS_ENTITY_TOKENS:
                # Looks like a proper noun — might be a person name
                # Only refuse if the shorter token is clearly an org/product
                # (single capitalized word or known product)
                pass  # We already block via _CROSS_ENTITY_TOKENS above

    return True


def _dedup_substrings(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold safe substring variants into the shorter canonical form.

    When "ChatGPT" and "ChatGPT Advertising" both exist:
    - Keep "ChatGPT" (canonical)
    - Absorb "ChatGPT Advertising" (noise variant)
    - Record "ChatGPT Advertising" in the ``aliases`` field

    When "OpenAI" and "OpenAI CEO Sam Altman" both exist:
    - Keep BOTH as separate entities (cross-entity: org vs person)
    - No absorption

    Absorbed variants are stored in ``aliases`` for traceability.
    """
    if not entities:
        return entities

    by_name = {e["candidate_name_normalized"]: e for e in entities}
    names = sorted(by_name.keys(), key=len)

    absorbed: Set[str] = set()

    for i, shorter in enumerate(names):
        if shorter in absorbed:
            continue
        for longer in names[i + 1:]:
            if longer in absorbed:
                continue
            # Check if shorter is a substring of longer
            if shorter not in longer:
                continue

            # Safety check: refuse cross-entity merges
            if not _is_safe_to_merge(shorter, longer):
                continue

            # --- Safe to absorb ---
            short_ent = by_name[shorter]
            long_ent = by_name[longer]

            # Record alias
            short_ent["aliases"].append(long_ent["candidate_name"])

            # Merge unique event IDs
            existing_ids = set(short_ent["event_ids"])
            for eid in long_ent["event_ids"]:
                if eid not in existing_ids:
                    short_ent["event_ids"].append(eid)
                    existing_ids.add(eid)

            # Merge source count (max) and mention count (sum)
            short_ent["source_count"] = max(
                short_ent["source_count"], long_ent["source_count"],
            )
            short_ent["mention_count"] += long_ent["mention_count"]

            # Merge event types
            merged_types = set(short_ent["event_types"]) | set(long_ent["event_types"])
            short_ent["event_types"] = sorted(merged_types)

            # Merge sample titles (capped at 3 event + 2 paper)
            existing_ev = set(short_ent["sample_event_titles"])
            for t in long_ent["sample_event_titles"]:
                if t not in existing_ev and len(short_ent["sample_event_titles"]) < 3:
                    short_ent["sample_event_titles"].append(t)

            existing_pa = set(short_ent["sample_paper_titles"])
            for t in long_ent["sample_paper_titles"]:
                if t not in existing_pa and len(short_ent["sample_paper_titles"]) < 2:
                    short_ent["sample_paper_titles"].append(t)

            # Also absorb the longer entity's own aliases
            for alias in long_ent.get("aliases", []):
                if alias not in short_ent["aliases"]:
                    short_ent["aliases"].append(alias)

            absorbed.add(longer)

    return [e for e in entities if e["candidate_name_normalized"] not in absorbed]
