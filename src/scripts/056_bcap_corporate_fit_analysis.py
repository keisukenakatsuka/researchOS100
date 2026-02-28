#!/usr/bin/env python
"""BCAP x Corporate Strategic Fit Analysis.

Analyses fit between a corporate's historical investments and B Capital's
portfolio to identify co-investment opportunities.

Pipeline
--------
1. Load B Capital portfolio CSV and CB Insights corporate investment CSV(s).
2. Build text descriptors for each company.
3. Embed all descriptors (OpenAI ``text-embedding-3-small``).
4. Coarse-filter pairs by cosine similarity on embeddings.
5. Shortlisted pairs are scored by LLM (similarity 0-1 + rationale).
6. Matched pairs (above threshold) are clustered into strategic themes.
7. Write text + JSON report.

Usage::

    # Full run
    python -m src.scripts.056_bcap_corporate_fit_analysis \\
        --cbi "CBI Search Results (0).csv" --run

    # Dry-run (parse only, no OpenAI)
    python -m src.scripts.056_bcap_corporate_fit_analysis \\
        --cbi "CBI Search Results (0).csv" --dry-run

    # Multiple CBI files
    python -m src.scripts.056_bcap_corporate_fit_analysis \\
        --cbi "CBI Search Results (0).csv" "CBI Search Results (1).csv" --run

    # Custom threshold and limits
    python -m src.scripts.056_bcap_corporate_fit_analysis \\
        --cbi "CBI Search Results (0).csv" --threshold 0.8 \\
        --max-cbi 50 --max-bcap 100 --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project-root path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.run_metadata import setup_logging

# load_env() relies on notebooks/env.txt which may not exist in worktrees.
# This script only needs OPENAI_API_KEY, so we try load_env but don't fail.
def _try_load_env() -> None:
    try:
        from src.config import load_env
        load_env()
    except FileNotFoundError:
        pass

logger = logging.getLogger("056_bcap_corporate_fit_analysis")
SCRIPT_NAME = "056_bcap_corporate_fit_analysis"
JST = ZoneInfo("Asia/Tokyo")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DOWNLOADS_DIR = Path("/Users/yuetoya/Downloads")
DEFAULT_BCAP_FILENAME = "BCAP.csv"
DEFAULT_THRESHOLD = 0.75
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
COARSE_TOPK_PER_CBI = 10  # top-K BCAP candidates per CBI company
COARSE_SIM_FLOOR = 0.30   # minimum cosine sim to even consider


# ============================================================================
# Data models
# ============================================================================

@dataclass
class Company:
    """Unified representation for either CBI or BCAP company."""
    name: str
    source: str  # "cbi" or "bcap"
    description: str
    fallback_text: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def text_for_embedding(self) -> str:
        return self.description if self.description.strip() else self.fallback_text

    @property
    def pair_key_part(self) -> str:
        return f"{self.source}::{self.name}"


@dataclass
class SimilarityResult:
    """Result of an LLM-scored similarity comparison."""
    cbi_name: str
    bcap_name: str
    score: float
    explanation: str
    is_match: bool


@dataclass
class ThemeGroup:
    """A thematic cluster of matched companies."""
    label: str
    description: str
    cbi_companies: List[str]
    bcap_companies: List[str]
    rationale: str


# ============================================================================
# Loading
# ============================================================================

def load_bcap(path: Path) -> pd.DataFrame:
    """Load B Capital portfolio CSV."""
    logger.info("Loading BCAP portfolio from %s", path)
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"Cannot decode BCAP file: {path}")

    logger.info("  BCAP rows: %d, columns: %s", len(df), list(df.columns))
    return df


def load_cbi(paths: Sequence[Path], max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load one or more CB Insights CSVs and concatenate."""
    frames: List[pd.DataFrame] = []
    for p in paths:
        logger.info("Loading CBI file: %s", p)
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(p, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise RuntimeError(f"Cannot decode CBI file: {p}")
        logger.info("  CBI rows: %d, columns: %s", len(df), list(df.columns))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.drop_duplicates(subset=["Companies"], keep="first", inplace=True)
    if max_rows:
        combined = combined.head(max_rows)
    logger.info("CBI total after dedup: %d companies", len(combined))
    return combined


def _safe_str(val: Any) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def build_cbi_companies(df: pd.DataFrame) -> List[Company]:
    """Convert CBI DataFrame rows into Company objects."""
    companies: List[Company] = []
    for _, row in df.iterrows():
        name = _safe_str(row.get("Companies", ""))
        if not name:
            continue
        desc = _safe_str(row.get("Description", ""))
        fallback_parts = [
            name,
            _safe_str(row.get("Industry", "")),
            _safe_str(row.get("Markets", "")),
        ]
        fallback = " | ".join(p for p in fallback_parts if p)
        companies.append(Company(
            name=name,
            source="cbi",
            description=desc,
            fallback_text=fallback,
            raw=row.to_dict(),
        ))
    return companies


def build_bcap_companies(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
) -> List[Company]:
    """Convert BCAP DataFrame rows into Company objects."""
    companies: List[Company] = []
    for _, row in df.iterrows():
        name = _safe_str(row.get("company_name", ""))
        if not name:
            continue
        desc = _safe_str(row.get("one_line_overview", ""))
        fallback_parts = [
            name,
            _safe_str(row.get("verticals / sectors", "")),
        ]
        fallback = " | ".join(p for p in fallback_parts if p)
        companies.append(Company(
            name=name,
            source="bcap",
            description=desc,
            fallback_text=fallback,
            raw=row.to_dict(),
        ))
    if max_rows:
        companies = companies[:max_rows]
    return companies


# ============================================================================
# Embedding (coarse filter)
# ============================================================================

def _get_openai_client():
    """Lazy-build a raw openai.OpenAI client for embedding calls."""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required but not set.")
    return OpenAI(api_key=api_key, timeout=120)


def embed_texts(
    client,
    texts: List[str],
    *,
    model: str = EMBEDDING_MODEL,
    batch_size: int = 100,
) -> np.ndarray:
    """Embed a list of texts, returning (N, dim) numpy array."""
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Replace empty strings to avoid API error
        batch = [t if t.strip() else "N/A" for t in batch]
        logger.info("  Embedding batch %d–%d / %d", i, i + len(batch), len(texts))
        resp = client.embeddings.create(model=model, input=batch)
        for item in resp.data:
            all_embeddings.append(item.embedding)
    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of a and b -> (len_a, len_b)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def coarse_filter(
    cbi_companies: List[Company],
    bcap_companies: List[Company],
    cbi_embeddings: np.ndarray,
    bcap_embeddings: np.ndarray,
    *,
    top_k: int = COARSE_TOPK_PER_CBI,
    sim_floor: float = COARSE_SIM_FLOOR,
) -> List[Tuple[int, int, float]]:
    """Return shortlisted (cbi_idx, bcap_idx, cosine_sim) triples."""
    sim_matrix = cosine_similarity_matrix(cbi_embeddings, bcap_embeddings)
    pairs: List[Tuple[int, int, float]] = []
    seen = set()
    for ci in range(len(cbi_companies)):
        row = sim_matrix[ci]
        top_indices = np.argsort(row)[::-1][:top_k]
        for bi in range(len(top_indices)):
            idx = top_indices[bi]
            sim = float(row[idx])
            if sim < sim_floor:
                continue
            key = (ci, idx)
            if key not in seen:
                seen.add(key)
                pairs.append((ci, idx, sim))
    logger.info("Coarse filter: %d candidate pairs (top_k=%d, floor=%.2f)",
                len(pairs), top_k, sim_floor)
    return pairs


# ============================================================================
# LLM similarity scoring
# ============================================================================

SIMILARITY_SYSTEM_PROMPT = """\
You are an expert venture capital analyst specializing in strategic corporate-VC alignment.

Given two companies — one from a corporate investor's portfolio and one from B Capital's portfolio — evaluate how similar they are in terms of business focus, target market, technology domain, and strategic relevance.

Return a JSON object:
{
  "similarity_score": <float 0.0-1.0>,
  "explanation": "<1-3 sentence rationale>"
}

Scoring guide:
- 0.9-1.0: Nearly identical business / direct competitors
- 0.7-0.9: Same domain, complementary or overlapping focus
- 0.5-0.7: Adjacent space, some strategic connection
- 0.3-0.5: Loosely related industry
- 0.0-0.3: Different domains
"""


def _pair_cache_key(cbi_name: str, bcap_name: str) -> str:
    blob = json.dumps({"cbi": cbi_name, "bcap": bcap_name}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_similarity_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load JSONL cache file -> dict keyed by pair hash."""
    cache: Dict[str, Dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            cache[entry["key"]] = entry
        except (json.JSONDecodeError, KeyError):
            continue
    logger.info("Loaded %d cached similarity results from %s", len(cache), cache_path)
    return cache


def append_similarity_cache(cache_path: Path, entry: Dict[str, Any]) -> None:
    """Append one JSONL entry to cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def score_pair_llm(
    client,
    cbi: Company,
    bcap: Company,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> SimilarityResult:
    """Score a single CBI-BCAP pair via LLM."""
    user_prompt = (
        f"Corporate Investment:\n"
        f"  Name: {cbi.name}\n"
        f"  Description: {cbi.text_for_embedding}\n\n"
        f"B Capital Portfolio Company:\n"
        f"  Name: {bcap.name}\n"
        f"  Description: {bcap.text_for_embedding}\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SIMILARITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    score = float(parsed.get("similarity_score", 0.0))
    explanation = parsed.get("explanation", "")
    return SimilarityResult(
        cbi_name=cbi.name,
        bcap_name=bcap.name,
        score=score,
        explanation=explanation,
        is_match=False,  # caller sets threshold
    )


def score_shortlisted_pairs(
    client,
    cbi_companies: List[Company],
    bcap_companies: List[Company],
    shortlist: List[Tuple[int, int, float]],
    *,
    threshold: float,
    cache_path: Path,
) -> List[SimilarityResult]:
    """Score all shortlisted pairs via LLM, using cache."""
    cache = load_similarity_cache(cache_path)
    results: List[SimilarityResult] = []
    new_calls = 0

    for ci, bi, cosine_sim in shortlist:
        cbi = cbi_companies[ci]
        bcap = bcap_companies[bi]
        key = _pair_cache_key(cbi.name, bcap.name)

        if key in cache:
            entry = cache[key]
            sr = SimilarityResult(
                cbi_name=cbi.name,
                bcap_name=bcap.name,
                score=entry["score"],
                explanation=entry["explanation"],
                is_match=entry["score"] >= threshold,
            )
            results.append(sr)
            continue

        sr = score_pair_llm(client, cbi, bcap)
        sr = SimilarityResult(
            cbi_name=sr.cbi_name,
            bcap_name=sr.bcap_name,
            score=sr.score,
            explanation=sr.explanation,
            is_match=sr.score >= threshold,
        )
        results.append(sr)
        new_calls += 1

        # Cache
        entry = {
            "key": key,
            "cbi_name": cbi.name,
            "bcap_name": bcap.name,
            "score": sr.score,
            "explanation": sr.explanation,
            "cosine_sim": cosine_sim,
        }
        append_similarity_cache(cache_path, entry)
        cache[key] = entry

        # Mild rate-limit courtesy
        if new_calls % 20 == 0:
            logger.info("  Scored %d pairs so far (%d new API calls)", len(results), new_calls)
            time.sleep(0.5)

    logger.info("Scoring complete: %d pairs, %d new API calls, %d cached",
                len(results), new_calls, len(results) - new_calls)
    return results


# ============================================================================
# Thematic clustering
# ============================================================================

THEME_SYSTEM_PROMPT = """\
You are a senior strategy consultant analyzing corporate-VC investment alignment.

Given a list of matched company pairs between a corporate investor and B Capital,
group them into 3-7 strategic themes. Each theme should represent a coherent
business domain or strategic area.

Return a JSON object:
{
  "themes": [
    {
      "label": "<short theme name, 3-6 words>",
      "description": "<1-2 sentence description of the theme>",
      "cbi_companies": ["Company A", "Company B"],
      "bcap_companies": ["Company X", "Company Y"],
      "rationale": "<1-2 sentence rationale for this grouping>"
    }
  ],
  "strategic_observations": [
    "<observation 1>",
    "<observation 2>"
  ],
  "alignment_level": "High" | "Medium" | "Low",
  "key_insight": "<2-4 sentence narrative summary of the strategic relationship>"
}
"""


def cluster_into_themes(
    client,
    matches: List[SimilarityResult],
    cbi_companies: List[Company],
    bcap_companies: List[Company],
    *,
    model: str = "gpt-4o",
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Use LLM to cluster matched pairs into strategic themes."""
    cbi_lookup = {c.name: c for c in cbi_companies}
    bcap_lookup = {c.name: c for c in bcap_companies}

    pairs_text = []
    for m in matches:
        cbi_desc = cbi_lookup.get(m.cbi_name, Company(m.cbi_name, "cbi", "", "")).text_for_embedding
        bcap_desc = bcap_lookup.get(m.bcap_name, Company(m.bcap_name, "bcap", "", "")).text_for_embedding
        pairs_text.append(
            f"- Corporate: {m.cbi_name} (score={m.score:.2f})\n"
            f"    Desc: {cbi_desc[:300]}\n"
            f"  BCAP: {m.bcap_name}\n"
            f"    Desc: {bcap_desc[:300]}\n"
            f"  Rationale: {m.explanation}"
        )

    user_prompt = (
        f"Below are {len(matches)} matched investment pairs between a corporate "
        f"investor and B Capital's portfolio.\n\n"
        + "\n\n".join(pairs_text)
        + "\n\nPlease group these into strategic themes."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": THEME_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    return json.loads(raw)


# ============================================================================
# Report generation
# ============================================================================

def generate_text_report(
    *,
    matches: List[SimilarityResult],
    themes_data: Dict[str, Any],
    total_cbi: int,
    total_bcap: int,
    threshold: float,
    generated_at: str,
) -> str:
    """Build the text report string."""
    lines: List[str] = []

    alignment = themes_data.get("alignment_level", "N/A")
    key_insight = themes_data.get("key_insight", "N/A")
    themes = themes_data.get("themes", [])
    observations = themes_data.get("strategic_observations", [])

    # Header
    lines.append("BCAP x Corporate Strategic Fit Analysis")
    lines.append(f"Generated at: {generated_at}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("1. Executive Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"- Total Corporate Investments Reviewed: {total_cbi}")
    lines.append(f"- Total B Capital Portfolio Companies: {total_bcap}")
    lines.append(f"- Total Matched Pairs (above threshold {threshold}): {len(matches)}")
    lines.append(f"- Overall Observed Strategic Alignment Level: {alignment}")
    lines.append("")
    lines.append(f"Key Insight: {key_insight}")
    lines.append("")

    # Top matches
    lines.append("-" * 60)
    lines.append("2. Top Similar Investment Matches")
    lines.append("-" * 60)
    lines.append("")

    sorted_matches = sorted(matches, key=lambda m: m.score, reverse=True)
    for i, m in enumerate(sorted_matches[:20], 1):
        lines.append(f"Match #{i}")
        lines.append(f"  Corporate Company: {m.cbi_name}")
        lines.append(f"  B Capital Company: {m.bcap_name}")
        lines.append(f"  Similarity Score: {m.score:.2f}")
        lines.append(f"  LLM Rationale: {m.explanation}")
        lines.append("")

    # Themes
    lines.append("-" * 60)
    lines.append("3. Thematic Clustering of Matched Investments")
    lines.append("-" * 60)
    lines.append("")

    for i, theme in enumerate(themes, 1):
        label = theme.get("label", f"Theme {i}")
        desc = theme.get("description", "")
        cbi_cos = theme.get("cbi_companies", [])
        bcap_cos = theme.get("bcap_companies", [])
        rationale = theme.get("rationale", "")

        lines.append(f"Theme {i}: {label}")
        lines.append(f"  Description: {desc}")
        lines.append("")
        lines.append("  Included Corporate Investments:")
        for c in cbi_cos:
            lines.append(f"    - {c}")
        lines.append("")
        lines.append("  Included BCAP Investments:")
        for c in bcap_cos:
            lines.append(f"    - {c}")
        lines.append("")
        lines.append(f"  Strategic Interpretation: {rationale}")
        lines.append("")

    # Strategic observations
    lines.append("-" * 60)
    lines.append("4. Strategic Observations & Implications")
    lines.append("-" * 60)
    lines.append("")
    for obs in observations:
        lines.append(f"- {obs}")
    lines.append("")

    # Appendix
    lines.append("-" * 60)
    lines.append("5. Appendix")
    lines.append("-" * 60)
    lines.append("")
    lines.append(f"- Threshold used: {threshold}")
    lines.append(f"- Total pairs evaluated by LLM: varies (coarse-filtered first)")
    lines.append(f"- Matches above threshold: {len(matches)}")
    if matches:
        scores = [m.score for m in matches]
        lines.append(f"- Score range: {min(scores):.2f} – {max(scores):.2f}")
        lines.append(f"- Mean score (matched only): {sum(scores)/len(scores):.2f}")
    lines.append("")

    lines.append("Full list of matched pairs:")
    for m in sorted_matches:
        lines.append(f"  {m.cbi_name}  <-->  {m.bcap_name}  (score={m.score:.2f})")
    lines.append("")

    return "\n".join(lines)


def generate_json_report(
    *,
    matches: List[SimilarityResult],
    themes_data: Dict[str, Any],
    total_cbi: int,
    total_bcap: int,
    threshold: float,
    generated_at: str,
) -> Dict[str, Any]:
    """Build the JSON report dict."""
    return {
        "generated_at": generated_at,
        "total_cbi": total_cbi,
        "total_bcap": total_bcap,
        "threshold": threshold,
        "total_matches": len(matches),
        "alignment_level": themes_data.get("alignment_level", "N/A"),
        "key_insight": themes_data.get("key_insight", ""),
        "matches": [
            {
                "cbi_name": m.cbi_name,
                "bcap_name": m.bcap_name,
                "score": m.score,
                "explanation": m.explanation,
            }
            for m in sorted(matches, key=lambda m: m.score, reverse=True)
        ],
        "themes": themes_data.get("themes", []),
        "strategic_observations": themes_data.get("strategic_observations", []),
    }


# ============================================================================
# Pipeline
# ============================================================================

def run_pipeline(
    *,
    cbi_filenames: List[str],
    bcap_filename: str = DEFAULT_BCAP_FILENAME,
    downloads_dir: Path = DEFAULT_DOWNLOADS_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    max_cbi: Optional[int] = None,
    max_bcap: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute the BCAP x Corporate fit analysis pipeline.

    Returns dict with summary counts and output paths.
    """
    _try_load_env()
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    now_jst = datetime.now(tz=JST)
    generated_at = now_jst.strftime("%Y-%m-%d %H:%M:%S")
    ts_slug = now_jst.strftime("%Y%m%d_%H%M%S")

    # ── Load data ──────────────────────────────────────────────
    bcap_path = downloads_dir / bcap_filename
    cbi_paths = [downloads_dir / fn for fn in cbi_filenames]

    for p in [bcap_path] + cbi_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    bcap_df = load_bcap(bcap_path)
    cbi_df = load_cbi(cbi_paths, max_rows=max_cbi)

    bcap_companies = build_bcap_companies(bcap_df, max_rows=max_bcap)
    cbi_companies = build_cbi_companies(cbi_df)

    logger.info("Companies: %d CBI, %d BCAP", len(cbi_companies), len(bcap_companies))

    if dry_run:
        logger.info("=== DRY RUN — no OpenAI calls ===")
        logger.info("CBI sample: %s", [c.name for c in cbi_companies[:5]])
        logger.info("BCAP sample: %s", [c.name for c in bcap_companies[:5]])
        return {
            "mode": "dry_run",
            "cbi_count": len(cbi_companies),
            "bcap_count": len(bcap_companies),
        }

    # ── Embeddings (coarse filter) ─────────────────────────────
    client = _get_openai_client()

    logger.info("Computing embeddings for CBI companies ...")
    cbi_texts = [c.text_for_embedding for c in cbi_companies]
    cbi_embeddings = embed_texts(client, cbi_texts)

    logger.info("Computing embeddings for BCAP companies ...")
    bcap_texts = [c.text_for_embedding for c in bcap_companies]
    bcap_embeddings = embed_texts(client, bcap_texts)

    shortlist = coarse_filter(
        cbi_companies, bcap_companies,
        cbi_embeddings, bcap_embeddings,
    )
    logger.info("Shortlisted %d pairs for LLM scoring", len(shortlist))

    # ── LLM similarity scoring ─────────────────────────────────
    cache_path = downloads_dir / f".bcap_similarity_cache.jsonl"
    results = score_shortlisted_pairs(
        client, cbi_companies, bcap_companies, shortlist,
        threshold=threshold,
        cache_path=cache_path,
    )

    matches = [r for r in results if r.is_match]
    logger.info("Matches above threshold %.2f: %d / %d scored",
                threshold, len(matches), len(results))

    # ── Thematic clustering ────────────────────────────────────
    if matches:
        logger.info("Clustering %d matches into themes ...", len(matches))
        themes_data = cluster_into_themes(
            client, matches, cbi_companies, bcap_companies,
        )
    else:
        logger.warning("No matches found. Skipping thematic clustering.")
        themes_data = {
            "themes": [],
            "strategic_observations": ["No significant overlap detected."],
            "alignment_level": "Low",
            "key_insight": "No matches were found above the similarity threshold.",
        }

    # ── Generate reports ───────────────────────────────────────
    text_report = generate_text_report(
        matches=matches,
        themes_data=themes_data,
        total_cbi=len(cbi_companies),
        total_bcap=len(bcap_companies),
        threshold=threshold,
        generated_at=generated_at,
    )

    json_report = generate_json_report(
        matches=matches,
        themes_data=themes_data,
        total_cbi=len(cbi_companies),
        total_bcap=len(bcap_companies),
        threshold=threshold,
        generated_at=generated_at,
    )

    # Write outputs
    txt_path = downloads_dir / f"bcap_fit_report_{ts_slug}.txt"
    json_path = downloads_dir / f"bcap_fit_report_{ts_slug}.json"

    txt_path.write_text(text_report, encoding="utf-8")
    logger.info("Text report written: %s", txt_path)

    json_path.write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("JSON report written: %s", json_path)

    summary = {
        "cbi_count": len(cbi_companies),
        "bcap_count": len(bcap_companies),
        "pairs_scored": len(results),
        "matches": len(matches),
        "themes": len(themes_data.get("themes", [])),
        "txt_report": str(txt_path),
        "json_report": str(json_path),
    }
    logger.info("Pipeline complete: %s", json.dumps(summary, indent=2))
    return summary


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BCAP x Corporate Strategic Fit Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cbi", nargs="+", required=True,
        help='CBI CSV filename(s), e.g. "CBI Search Results (0).csv"',
    )
    parser.add_argument(
        "--bcap", default=DEFAULT_BCAP_FILENAME,
        help=f"BCAP CSV filename (default: {DEFAULT_BCAP_FILENAME})",
    )
    parser.add_argument(
        "--downloads-dir", default=str(DEFAULT_DOWNLOADS_DIR),
        help=f"Directory for input/output (default: {DEFAULT_DOWNLOADS_DIR})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Similarity threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--max-cbi", type=int, default=None,
        help="Max CBI companies to process",
    )
    parser.add_argument(
        "--max-bcap", type=int, default=None,
        help="Max BCAP companies to process",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Parse inputs only, no OpenAI calls",
    )
    parser.add_argument(
        "--run", action="store_true", default=False,
        help="Actually run the pipeline (required)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if not args.run and not args.dry_run:
        parser.print_help()
        print("\nPass --run or --dry-run to execute.")
        sys.exit(0)

    result = run_pipeline(
        cbi_filenames=args.cbi,
        bcap_filename=args.bcap,
        downloads_dir=Path(args.downloads_dir),
        threshold=args.threshold,
        max_cbi=args.max_cbi,
        max_bcap=args.max_bcap,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
