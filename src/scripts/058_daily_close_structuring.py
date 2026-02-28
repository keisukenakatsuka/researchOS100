#!/usr/bin/env python
# src/scripts/058_daily_close_structuring.py
"""Daily Close Structuring — LLM-based meaning extraction engine.

Transforms the raw evening close log (057 output) into structured,
forward-looking insight using LLM semantic extraction.

Pipeline:
1. Load 057 output from data/daily/close_raw/YYYY-MM-DD/
2. Send raw close log to LLM for meaning extraction (Japanese output)
3. Validate returned JSON strictly
4. Build CloseStructured with:
   - structured_summary (3-6 sentence Japanese summary)
   - provisional_top3 (forward-looking, actionable priorities)
   - friction_blockers (structural/emotional friction)
   - open_questions (explicit + implicit uncertainties)
5. Persist to data/daily/close_structured/YYYY-MM-DD/
6. Upsert to Notion (Structured Summary, Provisional Top 3, etc.)
7. Set Stage = structured

All text output is in Japanese.  No free-text parsing — strict JSON only.

Usage::

    # Structure today's close log
    python -m src.scripts.058_daily_close_structuring

    # Structure a specific date
    python -m src.scripts.058_daily_close_structuring --date 2026-02-23

    # Verbose logging
    python -m src.scripts.058_daily_close_structuring --date 2026-02-23 -v
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_env, setup_logging, RunMetadata, get_iso_week_context
from src.daily.models import CloseRawInput, CloseStructured, StructuredItem
from src.daily.io import (
    CLOSE_RAW_DIR, CLOSE_STRUCTURED_DIR,
    daily_output_dir, save_json, load_json,
)

logger = logging.getLogger("058_daily_close_structuring")

SCRIPT_NAME = "058_daily_close_structuring"
JST = ZoneInfo("Asia/Tokyo")

# LLM cache directory
_LLM_CACHE_DIR = Path("data/cache/daily_structuring")

# Default model for meaning extraction
_DEFAULT_MODEL = "gpt-4o"


# ── LLM Prompt (Japanese) ────────────────────────────────────────

_SYSTEM_PROMPT = """\
あなたは日本語で書く「意味抽出エンジン」です。
ユーザーの一日の振り返りログ（生テキスト）を読み、そこから構造化された洞察を抽出してください。

出力ルール:
- すべてのテキストは日本語で書くこと
- マークダウンのヘッダー（#, ##）は使用しないこと
- 番号付きリスト（1. 2.）は使用しないこと（自然な日本語文の一部でない限り）
- 簡潔で明確な文のみ

抽出の指針:

structured_summary:
- その日の活動・出来事・感情を3〜6文で要約
- 単なる事実の列挙ではなく、文脈と意味を含めること

provisional_top_3:
- 明日以降に取るべき具体的なアクションを3つ抽出
- 「誰が」「何を」「どのような文脈で」を含む具体的な表現にすること
- ログに明示されていなくても、暗示されているコミットメントやアクションを推論すること
- ヘッダーや要約文をそのままコピーしないこと
- 例: 「平井さんとの打ち合わせに向けた論点整理（2026年戦略）」

friction_blockers:
- 感情的な摩擦、構造的な不確実性、健康上の制約、戦略的な曖昧さを抽出
- 簡潔な箇条書き形式

open_questions:
- 明示的な疑問と暗示的な不確実性を抽出
- 疑問文の形式で書くこと

出力形式（必ずこのJSON構造で返すこと）:
{
  "structured_summary": "3〜6文の日本語要約",
  "provisional_top_3": ["具体的アクション1", "具体的アクション2", "具体的アクション3"],
  "friction_blockers": ["摩擦1", "摩擦2"],
  "open_questions": ["未解決の問い1", "未解決の問い2"]
}"""


def _build_user_prompt(raw_text: str, date_iso: str) -> str:
    """Build the user prompt for LLM extraction."""
    return (
        f"日付: {date_iso}\n\n"
        f"一日の振り返りログ:\n{raw_text}\n\n"
        "上記のログから意味を抽出し、JSON形式で返してください。"
    )


# ── Helpers ──────────────────────────────────────────────────────

def _input_hash(text: str) -> str:
    """SHA-256 hash of input text for LLM cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_close_raw(date_iso: str) -> CloseRawInput:
    """Load 057 output for a given date."""
    raw_dir = CLOSE_RAW_DIR / date_iso
    raw_path = raw_dir / "close_raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No 057 output for {date_iso}. Run 057_daily_close_input first.\n"
            f"Expected: {raw_path}"
        )
    data = load_json(raw_path)
    return CloseRawInput.from_dict(data)


# ── JSON Validation ──────────────────────────────────────────────

class ExtractionValidationError(ValueError):
    """Raised when LLM extraction output fails validation."""


def _validate_extraction(data: dict) -> dict:
    """Validate the LLM extraction result against required schema.

    Raises ExtractionValidationError if validation fails.
    Returns the validated data dict.
    """
    required_keys = ["structured_summary", "provisional_top_3", "friction_blockers", "open_questions"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ExtractionValidationError(f"Missing required keys: {missing}")

    if not isinstance(data["structured_summary"], str) or not data["structured_summary"].strip():
        raise ExtractionValidationError("structured_summary must be a non-empty string")

    if not isinstance(data["provisional_top_3"], list) or len(data["provisional_top_3"]) < 1:
        raise ExtractionValidationError("provisional_top_3 must be a non-empty list")

    for i, item in enumerate(data["provisional_top_3"]):
        if not isinstance(item, str) or not item.strip():
            raise ExtractionValidationError(f"provisional_top_3[{i}] must be a non-empty string")

    if not isinstance(data["friction_blockers"], list):
        raise ExtractionValidationError("friction_blockers must be a list")

    if not isinstance(data["open_questions"], list):
        raise ExtractionValidationError("open_questions must be a list")

    return data


# ── LLM Extraction Layer ────────────────────────────────────────
#
# Uses LLMRouter.call() — the only public call method on LLMRouter.
# File: src/llm/router.py, class LLMRouter, method call() at line 110.
# Signature: call(*, task_type, system, user, model_override, temperature_override, use_cache)
# Returns: OpenAIResult with .parsed (dict)
#
# There is NO .chat(), .generate(), .complete(), or .run() on LLMRouter.
# ─────────────────────────────────────────────────────────────────

def _extract_with_llm(
    close_raw: CloseRawInput,
    *,
    model: str = _DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Call LLM for meaning extraction. Returns validated JSON dict.

    Uses LLMRouter.call(task_type=TASK_REASONING, ...) via OpenAI JSON mode.
    Raises on failure — no silent fallback to rule-based.
    """
    cache_key = _input_hash(close_raw.raw_text)
    cache_path = _LLM_CACHE_DIR / f"{close_raw.date}_{cache_key}.json"

    if cache_path.exists():
        logger.info("LLM cache hit: %s", cache_path)
        cached = load_json(cache_path)
        return _validate_extraction(cached)

    from src.llm.router import build_router_from_env, TASK_REASONING

    router = build_router_from_env(cache_dir=_LLM_CACHE_DIR)
    user_prompt = _build_user_prompt(close_raw.raw_text, close_raw.date)

    logger.info("Calling LLM for meaning extraction (model=%s)", model)

    # Call the correct method: LLMRouter.call()
    # NOT router.chat() — that method does not exist.
    result = router.call(
        task_type=TASK_REASONING,
        system=_SYSTEM_PROMPT,
        user=user_prompt,
        model_override=model,
        temperature_override=0.3,
        use_cache=False,  # We handle caching ourselves
    )

    data = result.parsed
    logger.info("LLM returned JSON with keys: %s", list(data.keys()))

    validated = _validate_extraction(data)

    # Cache the validated result
    _LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_json(cache_path, validated)
    logger.info("LLM extraction cached: %s", cache_path)

    return validated


# ── Build CloseStructured ────────────────────────────────────────

def _build_close_structured(
    close_raw: CloseRawInput,
    extraction: Dict[str, Any],
) -> CloseStructured:
    """Build a CloseStructured from the raw input and LLM extraction.

    The extraction dict (from LLM) populates:
      - structured_summary
      - provisional_top3
      - friction_blockers
      - open_questions

    These are the values that flow directly into the Notion PATCH.
    """
    # Build minimal items from raw text for backward compatibility
    # (downstream scripts 059/060 may still iterate over items)
    items = []
    if close_raw.raw_text:
        items.append(StructuredItem(
            text=close_raw.raw_text[:500],
            category="Progress",
        ))

    return CloseStructured(
        date=close_raw.date,
        items=items,
        structured_summary=extraction["structured_summary"],
        provisional_top3=extraction["provisional_top_3"],
        value_domains=close_raw.value_domains or [],
        research_candidates=[],
        contact_candidates=[],
        open_questions=extraction["open_questions"],
        friction_blockers=extraction["friction_blockers"],
        stage="structured",
    )


# ── Pipeline ─────────────────────────────────────────────────────

def run_pipeline(
    *,
    date_override: Optional[str] = None,
    verbose: bool = False,
    dry_run: bool = False,
    model: str = _DEFAULT_MODEL,
) -> dict:
    """Execute the daily close structuring pipeline."""
    load_env()
    setup_logging(level=logging.DEBUG if verbose else logging.INFO)

    now_jst = datetime.now(tz=JST)
    date_iso = date_override or now_jst.date().isoformat()
    wk = get_iso_week_context(tz=JST)

    logger.info("Starting %s date=%s model=%s", SCRIPT_NAME, date_iso, model)

    # ── 1. Load 057 output ──
    close_raw = _load_close_raw(date_iso)
    logger.info("Loaded close_raw: %d chars", len(close_raw.raw_text))

    # ── 2. LLM Meaning Extraction (no fallback) ──
    try:
        extraction = _extract_with_llm(close_raw, model=model)
    except Exception:
        logger.error(
            "LLM extraction FAILED. Full traceback:\n%s",
            traceback.format_exc(),
        )
        raise

    method = "llm"
    logger.info("LLM extraction complete (method=%s)", method)
    logger.info("  structured_summary: %d chars", len(extraction["structured_summary"]))
    logger.info("  provisional_top_3: %d items", len(extraction["provisional_top_3"]))
    for i, item in enumerate(extraction["provisional_top_3"]):
        logger.info("    [%d] %s", i + 1, item)
    logger.info("  friction_blockers: %d items", len(extraction["friction_blockers"]))
    logger.info("  open_questions: %d items", len(extraction["open_questions"]))

    # ── 3. Build CloseStructured ──
    structured = _build_close_structured(close_raw, extraction)

    # ── 4. Save output ──
    out_dir = daily_output_dir(CLOSE_STRUCTURED_DIR, date_iso)
    out_path = save_json(out_dir / "close_structured.json", structured.to_dict())
    logger.info("Saved close_structured.json -> %s", out_path)

    # ── 5. Notion upsert — structured layer ──
    # Uses the LLM-extracted values directly. No re-derivation.
    from src.notion.daily_schema import build_daily_log_properties
    from src.notion.daily_upsert import safe_truncate, upsert_daily_log

    notion_props = build_daily_log_properties(
        title=f"Daily Log {date_iso}",
        date=date_iso,
        # 057 raw layer — write canonical raw_text so Notion matches local JSON
        raw_close_log=safe_truncate(close_raw.raw_text),
        satisfaction=close_raw.satisfaction,
        energy_level=close_raw.energy_level or "",
        # LLM-extracted structured_summary (NOT raw item text)
        structured_summary=safe_truncate(
            structured.structured_summary
        ),
        # LLM-extracted friction_blockers
        friction_blockers=safe_truncate(
            "\n".join(structured.friction_blockers)
        ),
        # LLM-extracted open_questions
        open_questions=safe_truncate(
            "\n".join(structured.open_questions)
        ),
        # LLM-extracted provisional_top3 — plain newline-joined, NO numbering
        provisional_top3=safe_truncate(
            "\n".join(structured.provisional_top3)
        ),
        value_domains=structured.value_domains or None,
        stage="structured",
    )
    logger.info(
        "Including 057 raw layer in Notion upsert: raw_close_log=%d chars, "
        "satisfaction=%s, energy=%s",
        len(close_raw.raw_text), close_raw.satisfaction, close_raw.energy_level,
    )

    # Log the exact Provisional Top 3 going into the PATCH
    logger.info(
        "Notion payload Provisional Top 3:\n%s",
        "\n".join(structured.provisional_top3),
    )

    if dry_run:
        logger.info("[dry-run] Would upsert Notion Daily Log for %s", date_iso)
        notion_result = {"ok": True, "action": "dry-run", "date": date_iso}
    else:
        notion_result = upsert_daily_log(
            date_iso=date_iso,
            properties=notion_props,
            log_label="058_structured",
        )

    # ── 6. Run metadata ──
    meta = RunMetadata.build(
        notebook=SCRIPT_NAME,
        week_id=wk.week_id,
        counts={
            "provisional_top3": len(structured.provisional_top3),
            "friction_blockers": len(structured.friction_blockers),
            "open_questions": len(structured.open_questions),
            "summary_chars": len(structured.structured_summary),
            "method": method,
            "model": model,
            "stage": "structured",
        },
    )
    meta.save(out_dir / "run_metadata.json")

    result = {
        "date": date_iso,
        "output_dir": str(out_dir),
        "method": method,
        "model": model,
        "structured_summary": structured.structured_summary,
        "provisional_top3": structured.provisional_top3,
        "friction_blockers": structured.friction_blockers,
        "open_questions": structured.open_questions,
        "value_domains": structured.value_domains,
        "stage": "structured",
    }
    if notion_result:
        result["notion"] = notion_result

    print(f"\n058 Structured -> {out_dir}")
    print(f"  Method: {method}")
    print(f"  Summary: {structured.structured_summary[:80]}...")
    print(f"  Provisional Top 3:")
    for i, t in enumerate(structured.provisional_top3, 1):
        print(f"    {i}. {t}")
    print(f"  Friction / Blockers: {len(structured.friction_blockers)}")
    print(f"  Open Questions: {len(structured.open_questions)}")
    print(f"  Stage: structured")
    if notion_result and notion_result.get("ok"):
        print(f"  Notion: {notion_result.get('action', '')} -> {notion_result.get('page_url', '')}")
    elif notion_result:
        print(f"  Notion: FAILED -> {notion_result.get('error', 'unknown')}")

    return result


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="058 Daily Close Structuring — LLM-based meaning extraction",
    )
    parser.add_argument("--date", type=str, default=None, help="Override date (YYYY-MM-DD)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Skip Notion writes")
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL,
                        help=f"LLM model (default: {_DEFAULT_MODEL})")
    args = parser.parse_args()

    result = run_pipeline(
        date_override=args.date,
        verbose=args.verbose,
        dry_run=args.dry_run,
        model=args.model,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
