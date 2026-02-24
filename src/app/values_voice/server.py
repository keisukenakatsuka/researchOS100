# src/app/values_voice/server.py
"""FastAPI backend for Values Voice Reflection.

API routes:
  GET  /api/domains          — load value domains from values.json
  POST /api/translate        — translate domain content via LLMRouter (cached)
  POST /api/transcribe       — audio blob -> Whisper + LLM refinement
  POST /api/tts              — text -> speech audio via LLMRouter
  POST /api/submit           — save AlignmentEntry locally (no Notion write)
  POST /api/submit-all       — batch write all entries to Notion (Alignment Log + Codex)
  GET  /api/config           — current voice/session config

Translation:
  Codex stays canonical English.  When --lang ja, the /api/translate
  endpoint dynamically translates domain_label, value_definition, and
  reflection_questions via LLMRouter.  Translations are cached in-memory
  for the session.  In --dry-run mode a deterministic dummy is returned.

All OpenAI calls go through LLMRouter (no direct client access).
Reuses existing modules: schema, router, alignment_repo.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("values_voice_server")

JST = ZoneInfo("Asia/Tokyo")

# ----------------------------------------------------------------
# App state (initialized via create_app)
# ----------------------------------------------------------------

_state = {
    "router": None,
    "voice_config": None,
    "record": None,
    "language": "ja",
    "write_notion": False,
    "dry_run": False,
    "review_type": "Daily",
    "output_dir": None,
}

# In-memory translation cache: domain_id -> translated dict.
# Persists for the lifetime of the server process (one session).
_translation_cache: dict[str, dict] = {}


def create_app(
    *,
    values_json_path: Optional[str] = None,
    language: str = "ja",
    write_notion: bool = False,
    dry_run: bool = False,
    review_type: str = "Daily",
) -> FastAPI:
    """Factory: build and configure the FastAPI app."""
    from src.config import get_iso_week_context, get_output_dir, load_env, setup_logging
    from src.values.schema import value_record_from_dict
    from src.values.voice import VoiceConfig

    load_env()
    setup_logging(level=logging.INFO)

    voice_config = VoiceConfig(language=language)
    _state["voice_config"] = voice_config
    _state["language"] = language
    _state["write_notion"] = write_notion
    _state["dry_run"] = dry_run
    _state["review_type"] = review_type

    # Load value domains.
    record = _load_value_record(values_json_path)
    _state["record"] = record
    logger.info("Loaded %d value domains", len(record.domains))

    # Output directory.
    wk = get_iso_week_context(tz=JST)
    out_dir = get_output_dir("055_values_voice_reflection", wk.week_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)
    _state["output_dir"] = out_dir

    # LLM Router (skip in dry-run).
    if not dry_run:
        from src.llm.router import build_router_from_env
        _state["router"] = build_router_from_env()
        logger.info("LLM router initialized")
    else:
        logger.info("[dry-run] LLM router skipped")

    # Build FastAPI app.
    app = FastAPI(title="Values Voice Reflection", version="1.0.0")

    # Mount static files (frontend).
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Register routes.
    _register_routes(app)

    return app


# ----------------------------------------------------------------
# Value record loading (same logic as 055 CLI)
# ----------------------------------------------------------------

def _load_value_record(values_json_path: Optional[str]):
    """Load ValueRecord from explicit path or latest 054 output."""
    from src.values.schema import value_record_from_dict

    if values_json_path:
        path = Path(values_json_path)
    else:
        path = _find_latest_054_output()

    if path is None or not path.exists():
        raise FileNotFoundError(
            "Cannot find 054 values.json. "
            "Run 054_values_scale_setup first, or use --values-json <path>."
        )
    logger.info("Loading value domains from: %s", path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return value_record_from_dict(data)


def _find_latest_054_output() -> Optional[Path]:
    base = Path("outputs/weekly")
    if not base.exists():
        return None
    week_dirs = sorted(base.iterdir(), reverse=True)
    for week_dir in week_dirs:
        candidate = week_dir / "054_values_scale_setup" / "values.json"
        if candidate.exists():
            return candidate
    return None


# ----------------------------------------------------------------
# API routes
# ----------------------------------------------------------------

def _register_routes(app: FastAPI) -> None:

    @app.get("/")
    async def index():
        """Serve the frontend (no cache so changes take effect immediately)."""
        static_dir = Path(__file__).parent / "static"
        return FileResponse(
            str(static_dir / "index.html"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/api/config")
    async def get_config():
        """Return current session configuration."""
        vc = _state["voice_config"]
        return {
            "language": vc.language,
            "language_display": vc.display_name,
            "write_notion": _state["write_notion"],
            "dry_run": _state["dry_run"],
            "review_type": _state["review_type"],
        }

    @app.get("/api/domains")
    async def get_domains():
        """Return all value domains with reflection questions."""
        record = _state["record"]
        domains = []
        for d in record.domains:
            domains.append({
                "domain_id": d.domain_id,
                "domain_label": d.domain_label,
                "value_definition": d.value_definition,
                "reflection_questions": list(d.reflection_questions),
                "behavioral_translation": d.behavioral_translation,
                "misalignment_description": d.misalignment_description,
            })
        return {"domains": domains, "total": len(domains)}

    @app.post("/api/translate")
    async def translate_domain(domain_id: str = Form(...)):
        """Translate a domain's display content to the session language.

        Translates domain_label, value_definition, and reflection_questions
        via LLMRouter.  Results are cached in-memory for the session.
        Codex data is never modified.

        In dry-run mode, returns deterministic dummy translations.
        When language is 'en', returns original content unchanged.
        """
        language = _state["language"]

        # English = no translation needed.
        if language == "en":
            record = _state["record"]
            d = record.get_domain(domain_id)
            if d is None:
                raise HTTPException(404, f"Unknown domain: {domain_id}")
            return {
                "domain_id": domain_id,
                "domain_label": d.domain_label,
                "value_definition": d.value_definition,
                "reflection_questions": list(d.reflection_questions),
            }

        # Check cache.
        if domain_id in _translation_cache:
            logger.debug("Translation cache hit: %s", domain_id)
            return _translation_cache[domain_id]

        # Look up original.
        record = _state["record"]
        d = record.get_domain(domain_id)
        if d is None:
            raise HTTPException(404, f"Unknown domain: {domain_id}")

        # Japanese: prefer canonical hand-curated translations.
        # These are authoritative for the 12 Value Evaluation Scale
        # domains and must NOT be auto-translated from English.
        if language == "ja" and domain_id in _JA_TRANSLATIONS:
            result = _build_canonical_ja_translation(d)
            _translation_cache[domain_id] = result
            logger.info("Using canonical Japanese for domain %s", domain_id)
            return result

        # Dry-run: deterministic dummy (covers non-ja languages too).
        if _state["dry_run"]:
            result = _build_dry_run_translation(d)
            _translation_cache[domain_id] = result
            return result

        # Live: translate via LLMRouter (only for unknown domains
        # or non-ja languages).
        router = _state["router"]
        if router is None:
            raise HTTPException(500, "LLM router not initialized")

        result = _translate_domain_via_llm(d, router)
        _translation_cache[domain_id] = result
        logger.info("Translated domain %s -> %s via LLM", domain_id, language)
        return result

    @app.post("/api/transcribe")
    async def transcribe_audio(audio: UploadFile = File(...)):
        """Transcribe uploaded audio via Whisper."""
        router = _state["router"]
        voice_config = _state["voice_config"]

        if _state["dry_run"]:
            if _state["language"] == "ja":
                return {"transcript": "これはドライランのダミー文字起こしです。実際のセッションでは、音声入力がここに表示されます。"}
            return {"transcript": "(Dry-run dummy transcript. In a real session, your voice input will appear here.)"}

        if router is None:
            raise HTTPException(500, "LLM router not initialized")

        # Save uploaded audio to temp file.
        suffix = ".wav"
        if audio.content_type and "webm" in audio.content_type:
            suffix = ".webm"
        elif audio.content_type and "ogg" in audio.content_type:
            suffix = ".ogg"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            transcript = router.transcribe_audio(
                tmp_path, voice_config=voice_config,
            )
            # Refine: add punctuation, remove fillers, smooth phrasing.
            refined = _refine_transcript(transcript, router)
            return {"transcript": refined}
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            raise HTTPException(500, f"Transcription failed: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.post("/api/tts")
    async def text_to_speech(text: str = Form(...)):
        """Generate TTS audio and return as downloadable file."""
        router = _state["router"]
        voice_config = _state["voice_config"]

        if _state["dry_run"]:
            raise HTTPException(400, "TTS disabled in dry-run mode")

        if router is None:
            raise HTTPException(500, "LLM router not initialized")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            router.speak_text(text, tmp_path, voice_config=voice_config)
            return FileResponse(
                str(tmp_path),
                media_type="audio/mpeg",
                filename="question.mp3",
            )
        except Exception as e:
            logger.error("TTS failed: %s", e)
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(500, f"TTS failed: {e}")

    @app.post("/api/submit")
    async def submit_reflection(
        domain_id: str = Form(...),
        importance_score: int = Form(...),
        alignment_score: int = Form(...),
        transcript: str = Form(""),
        question_text: str = Form(""),
    ):
        """Create an AlignmentEntry and optionally write to Notion."""
        from src.notion.alignment_repo import validate_score_range
        from src.values.schema import AlignmentEntry

        # Validate scores.
        try:
            validate_score_range(importance_score, "importance_score")
            validate_score_range(alignment_score, "alignment_score")
        except ValueError as e:
            raise HTTPException(422, str(e))

        now_jst = datetime.now(tz=JST)
        date_iso = now_jst.date().isoformat()

        entry = AlignmentEntry(
            date_iso=date_iso,
            review_type=_state["review_type"],
            domain_id=domain_id,
            importance_score=importance_score,
            alignment_score=alignment_score,
            reflection_text="",
            transcript=transcript,
        )

        result = {
            "domain_id": entry.domain_id,
            "importance_score": entry.importance_score,
            "alignment_score": entry.alignment_score,
            "gap_score": entry.gap_score,
            "significant_gap": entry.significant_gap,
            "date": date_iso,
        }

        # Notion write is deferred to /api/submit-all (batch send from results screen).
        # Save per-domain JSON locally.
        out_dir = _state["output_dir"]
        if out_dir:
            domains_dir = out_dir / "domains"
            domains_dir.mkdir(exist_ok=True)
            domain_path = domains_dir / f"{domain_id}.json"
            domain_path.write_text(json.dumps({
                **result,
                "transcript": transcript,
                "question_text": question_text,
            }, indent=2, ensure_ascii=False))

        return result

    @app.post("/api/submit-all")
    async def submit_all_to_notion(request: Request):
        """Batch write all entries to Notion (Alignment Log + Values Codex).

        Called from the results screen "Send to Notion" button.
        Expects JSON body: { entries: [{ domain_id, importance_score, alignment_score, transcript }] }
        """
        from src.notion.alignment_repo import validate_score_range
        from src.values.schema import AlignmentEntry

        if _state["dry_run"]:
            return JSONResponse({"ok": False, "error": "Notion write disabled in dry-run mode"})

        if not _state["write_notion"]:
            return JSONResponse({"ok": False, "error": "Notion write not enabled (start with --write)"})

        body = await request.json()
        entries = body.get("entries", [])
        if not entries:
            return JSONResponse({"ok": False, "error": "No entries provided"})

        now_jst = datetime.now(tz=JST)
        date_iso = now_jst.date().isoformat()

        domain_results = []
        for raw in entries:
            domain_id = raw.get("domain_id", "")
            imp = raw.get("importance_score", 0)
            align = raw.get("alignment_score", 0)
            transcript = raw.get("transcript", "")

            dr = {"domain_id": domain_id, "alignment_written": False, "codex_written": False, "errors": []}

            # Validate scores.
            try:
                validate_score_range(imp, "importance_score")
                validate_score_range(align, "alignment_score")
            except ValueError as e:
                dr["errors"].append(str(e))
                domain_results.append(dr)
                continue

            # Auto-generate AI summary and next adjustment via LLM.
            router = _state["router"]
            record = _state["record"]
            domain_obj = record.get_domain(domain_id) if record else None
            domain_label = domain_obj.domain_label if domain_obj else domain_id

            ai_summary = ""
            next_adjustment = ""
            if router is not None:
                ai_summary = _generate_ai_summary(transcript, domain_id, router)
                next_adjustment = _generate_next_adjustment(
                    domain_id, domain_label, imp, align, transcript, router,
                )

            entry = AlignmentEntry(
                date_iso=date_iso,
                review_type=_state["review_type"],
                domain_id=domain_id,
                importance_score=imp,
                alignment_score=align,
                reflection_text="",
                transcript=transcript,
                ai_summary=ai_summary,
                next_adjustment=next_adjustment,
            )

            # 1) Write to ROS_Alignment_Log (with Domain relation).
            try:
                _write_single_entry(entry)
                dr["alignment_written"] = True
                logger.info("Batch: wrote alignment entry for %s", domain_id)
            except Exception as e:
                logger.error("Batch: alignment write failed for %s: %s", domain_id, e)
                dr["errors"].append(f"Alignment Log: {e}")

            # 2) Write to ROS_Values_Codex.
            try:
                _write_codex_entry(domain_id)
                dr["codex_written"] = True
                logger.info("Batch: wrote codex entry for %s", domain_id)
            except Exception as e:
                logger.error("Batch: codex write failed for %s: %s", domain_id, e)
                dr["errors"].append(f"Values Codex: {e}")

            domain_results.append(dr)

        return JSONResponse({"ok": True, "results": domain_results})


# ----------------------------------------------------------------
# Translation helpers
# ----------------------------------------------------------------

# Full deterministic Japanese translations for dry-run mode.
# These are hand-curated translations so the UI feels 100% Japanese
# even without an LLM call.  Keyed by domain_id.
_JA_TRANSLATIONS: dict[str, dict] = {
    "family": {
        "domain_label": "家族",
        "value_definition":
            "家族の中での自分の在り方について振り返りましょう。",
        "reflection_questions": [
            "どのような父／母、息子／娘、兄弟姉妹、叔父／叔母でありたいか？",
            "家族のなかでどのようにふるまいたいか？",
            "家族とどのような関係性を築きたいのか？",
            "もし今のあなたが「理想の自分」だとしたら、家族に対してどのように接するか？",
        ],
    },
    "marriage_romantic": {
        "domain_label": "結婚・恋愛",
        "value_definition":
            "親密なパートナーとの関係性について振り返りましょう。",
        "reflection_questions": [
            "親密な相手に対してどのような夫／妻／パートナーでありたいか？",
            "相手とどのような関係を育てたいか？",
            "どのような関係性を作りたいのか？",
            "もし今のあなたが「理想の自分」だとしたら、結婚相手や恋愛相手にどのように接するか？",
        ],
    },
    "parenting": {
        "domain_label": "子育て",
        "value_definition":
            "親としての自分の在り方について振り返りましょう。",
        "reflection_questions": [
            "どのような親になりたいか？",
            "子供とどんな関係を結びたいか？",
            "子供と接するなかでどんな個性を育てたいか？",
            "もし今のあなたが「理想の自分」だとしたら、子供にどのように接するか？",
            "子供からどのように見られたいか？",
        ],
    },
    "friendships_interpersonal": {
        "domain_label": "友人・対人関係",
        "value_definition":
            "友人や周囲の人との関わり方について振り返りましょう。",
        "reflection_questions": [
            "どのような友情を育てたいか？",
            "友人関係のなかに自分のどのような特徴・資質を活かしたいか？",
            "自分が相手にとって最高の友人だとしたら、どのようにふるまうか？",
        ],
    },
    "career_work": {
        "domain_label": "キャリア・仕事",
        "value_definition":
            "仕事やキャリアに対する自分の姿勢について振り返りましょう。",
        "reflection_questions": [
            "自分は仕事のどういった点に重きを置いているか？",
            "仕事をもっと意味あるものにするにはどうすればいいか？",
            "いまの暮らしが理想の状態だったとしたら、自分のどのような資質を仕事に活かしたいか？",
            "職場や仕事のパートナーとどのような関係を築きたいか？",
        ],
    },
    "personal_growth": {
        "domain_label": "自己成長",
        "value_definition":
            "学びや自己成長に対する自分の姿勢について振り返りましょう。",
        "reflection_questions": [
            "もっと知りたいことはなにか？",
            "学習や教育について、自分はどんな点に重きを置いているのか？",
            "習得してみたい新しいスキルはなにか？",
            "自分が成長するためにどんな資質を活かしたいか？",
        ],
    },
    "leisure": {
        "domain_label": "余暇・レジャー",
        "value_definition":
            "遊びやリラックスの時間について振り返りましょう。",
        "reflection_questions": [
            "どんな趣味や遊びをしてみたいか？",
            "自分がリラックスできるのはどんなことか？",
            "どんなときに新しい活動に挑戦してみたいか？",
        ],
    },
    "spirituality": {
        "domain_label": "スピリチュアリティ",
        "value_definition":
            "人智を超えたものや哲学的な問いとの関わりについて振り返りましょう。",
        "reflection_questions": [
            "宗教、大自然、宇宙のような「人智を超えたもの」に対してどのような関係を築きたいか？",
            "（無信心の場合は宗教は無視して構いません）",
            "どのような哲学的な疑問に興味があるか？",
        ],
    },
    "community_social": {
        "domain_label": "コミュニティ・社会生活",
        "value_definition":
            "コミュニティや社会への関わり方について振り返りましょう。",
        "reflection_questions": [
            "どのようなコミュニティの一員でありたいか？",
            "地域社会にどのように貢献したいか？",
            "自分の居場所をどのように作りたいか？",
        ],
    },
    "health": {
        "domain_label": "健康",
        "value_definition":
            "身体の健康に対する自分の姿勢について振り返りましょう。",
        "reflection_questions": [
            "身体の健康について何に重きを置いているか？",
            "自分の身体をどのようにケアしたいか？",
        ],
    },
    "environment": {
        "domain_label": "環境",
        "value_definition":
            "地球環境に対する自分の姿勢について振り返りましょう。",
        "reflection_questions": [
            "地球環境について何に重きを置いているか？",
            "環境改善のために貢献したいことはあるか？",
        ],
    },
    "creative_arts": {
        "domain_label": "芸術",
        "value_definition":
            "芸術やアートとの関わり方について振り返りましょう。",
        "reflection_questions": [
            "絵画、音楽、文学、アートとどのような関係を築きたいか？",
            "どのような芸術に触れていたいか？",
            "参加してみたい芸術活動はあるか？",
        ],
    },
}


def _build_canonical_ja_translation(d) -> dict:
    """Return canonical hand-curated Japanese for a known domain.

    These are the authoritative Japanese prompts for the 12 Value
    Evaluation Scale domains.  They must NOT be auto-translated.
    Caller must check ``d.domain_id in _JA_TRANSLATIONS`` first.
    """
    ja = _JA_TRANSLATIONS[d.domain_id]
    return {
        "domain_id": d.domain_id,
        "domain_label": ja["domain_label"],
        "value_definition": ja["value_definition"],
        "reflection_questions": list(ja["reflection_questions"]),
    }


def _build_dry_run_translation(d) -> dict:
    """Return a deterministic Japanese translation for dry-run mode.

    Delegates to canonical Japanese for known domains.
    Falls back to English originals for unknown domains.
    """
    if d.domain_id in _JA_TRANSLATIONS:
        return _build_canonical_ja_translation(d)
    # Unknown domain: fall back to original English.
    return {
        "domain_id": d.domain_id,
        "domain_label": d.domain_label,
        "value_definition": d.value_definition,
        "reflection_questions": list(d.reflection_questions) if d.reflection_questions else [],
    }


def _translate_domain_via_llm(d, router) -> dict:
    """Translate a single domain's display fields via LLMRouter."""
    voice_config = _state["voice_config"]
    language = _state["language"]

    questions_block = ""
    if d.reflection_questions:
        questions_block = "\n".join(
            f"  Q{i+1}: {q}" for i, q in enumerate(d.reflection_questions)
        )

    system = (
        "You are a professional translator. "
        "Translate the following value domain content accurately and naturally. "
        "Preserve the meaning, tone, and nuance. "
        "The definition uses the identity format 'I am someone who...' — "
        "keep this structure in the target language. "
        "Return valid JSON with keys: domain_label, value_definition, reflection_questions (array)."
    )
    user_prompt = (
        f"Translate this to {voice_config.display_name}:\n\n"
        f"Domain Label: {d.domain_label}\n"
        f"Value Definition: {d.value_definition}\n"
    )
    if questions_block:
        user_prompt += f"Reflection Questions:\n{questions_block}\n"
    else:
        user_prompt += "Reflection Questions: (none)\n"

    try:
        result = router.call_voice_processing(
            system=system,
            user=user_prompt,
            voice_config=voice_config,
            use_cache=True,
        )
        parsed = result.parsed
        return {
            "domain_id": d.domain_id,
            "domain_label": parsed.get("domain_label", d.domain_label),
            "value_definition": parsed.get("value_definition", d.value_definition),
            "reflection_questions": parsed.get("reflection_questions", list(d.reflection_questions)),
        }
    except Exception as e:
        logger.error("Translation failed for %s: %s", d.domain_id, e)
        # Fallback: return originals so the UI doesn't break.
        return {
            "domain_id": d.domain_id,
            "domain_label": d.domain_label,
            "value_definition": d.value_definition,
            "reflection_questions": list(d.reflection_questions),
        }


# ----------------------------------------------------------------
# Notion write helper
# ----------------------------------------------------------------

def _write_single_entry(entry) -> None:
    """Write one AlignmentEntry to ROS_Alignment_Log with Domain relation."""
    from src.config import get_db_id
    from src.notion.alignment_repo import AlignmentLogRepo
    from src.notion.client import NotionDataSourceResolver, build_notion_client_from_env
    from src.notion.truncation import TruncationTracker
    from src.notion.values_repo import ValuesCodexRepo

    logger.info("Notion write: starting for domain=%s", entry.domain_id)

    client = build_notion_client_from_env()
    logger.info("Notion write: client initialized")

    db_id = get_db_id("NOTION_ROS_Alignment_Log_ID")
    logger.info("Notion write: database_id=%s", db_id)

    resolver = NotionDataSourceResolver(client=client)
    resolved = resolver.resolve_once(
        name="ROS_Alignment_Log",
        database_id=db_id,
    )
    data_source_id = resolved.data_source_id
    logger.info("Notion write: resolved data_source_id=%s", data_source_id)

    repo = AlignmentLogRepo(
        client=client,
        database_id=db_id,
        data_source_id=data_source_id,
    )
    repo.ensure_schema()
    logger.info("Notion write: schema validated")

    # Resolve Domain relation → codex_page_id.
    codex_page_id = _resolve_codex_page_id(client, resolver, entry.domain_id)

    tracker = TruncationTracker()
    repo.create_validated_entry(
        entry=entry,
        codex_page_id=codex_page_id,
        tracker=tracker,
    )
    logger.info(
        "Notion write: entry created for domain=%s (codex_page_id=%s)",
        entry.domain_id, codex_page_id or "NONE",
    )


def _write_codex_entry(domain_id: str) -> None:
    """Write one domain to ROS_Values_Codex (upsert by quarter:domain_id)."""
    from src.config import get_db_id
    from src.notion.client import NotionDataSourceResolver, build_notion_client_from_env
    from src.notion.truncation import TruncationTracker
    from src.notion.values_repo import ValuesCodexRepo

    record = _state["record"]
    domain = record.get_domain(domain_id)
    if domain is None:
        raise ValueError(f"Unknown domain_id: {domain_id}")

    logger.info("Codex write: starting for domain=%s", domain_id)

    client = build_notion_client_from_env()
    codex_db_id = get_db_id("NOTION_ROS_Values_Codex_ID")

    resolver = NotionDataSourceResolver(client=client)
    resolved = resolver.resolve_once(
        name="ROS_Values_Codex",
        database_id=codex_db_id,
    )
    data_source_id = resolved.data_source_id
    logger.info("Codex write: resolved data_source_id=%s", data_source_id)

    repo = ValuesCodexRepo(
        client=client,
        database_id=codex_db_id,
        data_source_id=data_source_id,
    )
    repo.ensure_schema()

    tracker = TruncationTracker()
    now_jst = datetime.now(tz=JST)
    now_iso = now_jst.isoformat(timespec="seconds")
    review_quarter = record.review_quarter

    key, props = repo.build_domain_properties(
        domain=domain,
        review_quarter=review_quarter,
        now_iso=now_iso,
        tracker=tracker,
    )
    repo.upsert_domain(key=key, properties=props)
    logger.info("Codex write: upserted domain=%s (key=%s)", domain_id, key)


# ----------------------------------------------------------------
# Domain relation resolver
# ----------------------------------------------------------------

def _resolve_codex_page_id(client, resolver, domain_id: str) -> Optional[str]:
    """Look up the ROS_Values_Codex page ID for a domain.

    Returns the Notion page ID or None if not found.
    Logs warnings on failure — never raises.
    """
    from src.config import get_db_id
    from src.notion.values_repo import ValuesCodexRepo

    record = _state["record"]
    review_quarter = record.review_quarter
    key = f"{review_quarter}:{domain_id}"

    try:
        codex_db_id = get_db_id("NOTION_ROS_Values_Codex_ID")
        resolved = resolver.resolve_once(
            name="ROS_Values_Codex",
            database_id=codex_db_id,
        )
        repo = ValuesCodexRepo(
            client=client,
            database_id=codex_db_id,
            data_source_id=resolved.data_source_id,
        )
        page_id = repo.fetch_domain_page_id(key)
        if page_id:
            logger.info("Domain relation resolved: %s -> %s", key, page_id)
        else:
            logger.warning("Domain relation: no codex page found for key=%s", key)
        return page_id
    except Exception as e:
        logger.warning("Domain relation resolution failed for %s: %s", domain_id, e)
        return None


# ----------------------------------------------------------------
# AI Summary + Next Adjustment generators
# ----------------------------------------------------------------

def _generate_ai_summary(transcript: str, domain_id: str, router) -> str:
    """Generate a concise AI summary (1-3 sentences) from transcript.

    Uses LLMRouter (JSON mode). Falls back to empty string on error.
    """
    if not transcript or not transcript.strip():
        return ""

    voice_config = _state["voice_config"]
    language = _state["language"]

    if language == "ja":
        system = (
            "あなたは内省コーチです。以下の振り返り音声の書き起こしを、"
            "1〜3文で簡潔に要約してください。\n"
            "・本人が語った核心的な気づきや感情を捉える\n"
            "・内容や意味を変えない\n"
            "・ラベルや前置きは不要\n"
            '・JSON形式 {"summary": "要約テキスト"} で返す'
        )
    else:
        system = (
            "You are a reflection coach. Summarize the following voice reflection "
            "transcript in 1-3 concise sentences.\n"
            "- Capture the key insight or feeling expressed\n"
            "- Do NOT change the meaning\n"
            "- No labels or prefixes\n"
            '- Return as JSON: {"summary": "summary text here"}'
        )

    try:
        result = router.call_voice_processing(
            system=system,
            user=transcript,
            voice_config=voice_config,
        )
        summary = result.parsed.get("summary", "").strip()
        logger.info("AI summary generated for %s: %d chars", domain_id, len(summary))
        return summary
    except Exception as e:
        logger.warning("AI summary generation failed for %s: %s", domain_id, e)
        return ""


def _generate_next_adjustment(
    domain_id: str, domain_label: str,
    importance: int, alignment: int, transcript: str, router,
) -> str:
    """Generate a single concrete next-adjustment sentence.

    Focuses on behavioral change, not abstract advice.
    Uses LLMRouter (JSON mode). Falls back to empty string on error.
    """
    gap = importance - alignment
    voice_config = _state["voice_config"]
    language = _state["language"]

    if language == "ja":
        system = (
            "あなたは行動変容コーチです。\n"
            "以下の価値観ドメインについて、重要度と一致度のギャップに基づいて、"
            "今週できる具体的な行動提案を1文で生成してください。\n"
            "・抽象的なアドバイスではなく、具体的な行動を提案する\n"
            "・「〜する」の形で終わる\n"
            '・JSON形式 {"adjustment": "行動提案テキスト"} で返す'
        )
        user_prompt = (
            f"ドメイン: {domain_label}\n"
            f"重要度: {importance}/5\n"
            f"一致度: {alignment}/5\n"
            f"ギャップ: {gap}\n"
        )
    else:
        system = (
            "You are a behavioral change coach.\n"
            "Based on the gap between importance and alignment for this value domain, "
            "generate ONE concrete, actionable adjustment for this week.\n"
            "- Focus on a specific behavior, not abstract advice\n"
            "- Keep it to one sentence\n"
            '- Return as JSON: {"adjustment": "adjustment text here"}'
        )
        user_prompt = (
            f"Domain: {domain_label}\n"
            f"Importance: {importance}/5\n"
            f"Alignment: {alignment}/5\n"
            f"Gap: {gap}\n"
        )

    if transcript:
        user_prompt += f"\nReflection transcript:\n{transcript[:500]}"

    try:
        result = router.call_voice_processing(
            system=system,
            user=user_prompt,
            voice_config=voice_config,
        )
        adjustment = result.parsed.get("adjustment", "").strip()
        logger.info("Next adjustment generated for %s: %s", domain_id, adjustment[:80])
        return adjustment
    except Exception as e:
        logger.warning("Next adjustment generation failed for %s: %s", domain_id, e)
        return ""


# ----------------------------------------------------------------
# Transcript refinement helper
# ----------------------------------------------------------------

def _refine_transcript(raw_transcript: str, router) -> str:
    """Refine Whisper output: add punctuation, remove fillers, smooth phrasing.

    Uses LLMRouter.call_voice_processing() with a system prompt that
    preserves meaning while making the text more readable.
    Falls back to the raw transcript on error.
    """
    if not raw_transcript or not raw_transcript.strip():
        return raw_transcript

    voice_config = _state["voice_config"]
    language = _state["language"]

    if language == "ja":
        system = (
            "あなたは日本語の文章校正者です。"
            "以下の音声書き起こしテキストを、意味を一切変えずに読みやすく整えてください。\n"
            "・句読点（。、）を適切に付ける\n"
            "・「えー」「あのー」「まあ」などのフィラーを除去する\n"
            "・文として自然になるように整える\n"
            "・内容や意味は絶対に変えない\n"
            '・結果はJSON形式 {"refined": "整形後のテキスト"} で返す'
        )
    else:
        system = (
            "You are a text editor. Clean up the following speech transcript "
            "without changing its meaning:\n"
            "- Add proper punctuation\n"
            "- Remove filler words (um, uh, like, you know)\n"
            "- Make it read naturally as written text\n"
            "- Do NOT change the meaning or add content\n"
            '- Return as JSON: {"refined": "cleaned text here"}'
        )

    try:
        result = router.call_voice_processing(
            system=system,
            user=raw_transcript,
            voice_config=voice_config,
        )
        # Router uses call_json → result.parsed is a dict.
        refined = result.parsed.get("refined", "").strip()
        if refined:
            logger.info("Transcript refined: %d -> %d chars", len(raw_transcript), len(refined))
            return refined
        return raw_transcript
    except Exception as e:
        logger.warning("Transcript refinement failed, using raw: %s", e)
        return raw_transcript
