# src/app/values_voice/server.py
"""FastAPI backend for Values Voice Reflection.

API routes:
  GET  /api/domains          — load value domains from values.json
  POST /api/translate        — translate domain content via LLMRouter (cached)
  POST /api/transcribe       — audio blob -> OpenAI Whisper via LLMRouter
  POST /api/tts              — text -> speech audio via LLMRouter
  POST /api/submit           — create AlignmentEntry + write via AlignmentLogRepo
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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
        """Serve the frontend."""
        static_dir = Path(__file__).parent / "static"
        return FileResponse(str(static_dir / "index.html"))

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

        # Dry-run: deterministic dummy.
        if _state["dry_run"]:
            result = _build_dry_run_translation(d)
            _translation_cache[domain_id] = result
            return result

        # Live: translate via LLMRouter.
        router = _state["router"]
        if router is None:
            raise HTTPException(500, "LLM router not initialized")

        result = _translate_domain_via_llm(d, router)
        _translation_cache[domain_id] = result
        logger.info("Translated domain %s -> %s", domain_id, language)
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
            return {"transcript": transcript}
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
            "notion_written": False,
        }

        # Write to Notion if enabled.
        if _state["write_notion"] and not _state["dry_run"]:
            try:
                _write_single_entry(entry)
                result["notion_written"] = True
                logger.info(
                    "Wrote alignment entry: %s (imp=%d, align=%d)",
                    domain_id, importance_score, alignment_score,
                )
            except Exception as e:
                logger.error("Notion write failed: %s", e)
                result["notion_error"] = str(e)
        elif _state["dry_run"]:
            logger.info("[dry-run] Notion write skipped for %s", domain_id)

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
            "私は、家族の中で地に足のついた、今この瞬間に向き合い、感情的にも寄り添える存在でありたい。",
        "reflection_questions": [
            "家族の一員として、どのような存在でありたいですか？",
            "家族の中で、どんな「居方」を体現したいですか？",
            "自分が望む家族の文化を創っていますか？　それとも、惰性で受け継いでいますか？",
        ],
    },
    "marriage_romantic": {
        "domain_label": "結婚・恋愛関係",
        "value_definition":
            "私は、正直に向き合い、意図的に弱さを見せ、共に進化するパートナーシップを通じて深い親密さを築く人間でありたい。",
        "reflection_questions": [
            "どんなパートナーでありたいですか？",
            "理想の自分だったら、パートナーとどう向き合いますか？",
            "言うべきなのに避けていることは何ですか？",
        ],
    },
    "parenting": {
        "domain_label": "子育て",
        "value_definition":
            "私は、子どもたちが自分らしく成長できるよう、安定していて好奇心に満ち、感情的に安全な環境を作る人間でありたい。",
        "reflection_questions": [
            "どんな親でありたいですか？",
            "子どもたちの中で、どんな資質を育てたいですか？",
            "子どもに受け継いでほしい感情パターンは、今の自分がモデルになっていますか？",
        ],
    },
    "friendships_interpersonal": {
        "domain_label": "友人・対人関係",
        "value_definition":
            "私は、自分の強みを活かして有意義に関わる、互恵的で正直な少数の人間関係を大切にする人間でありたい。",
        "reflection_questions": [
            "どんな友人でありたいですか？",
            "人間関係の中で、自分の強みをどう活かしたいですか？",
            "自分が望む友人像を、自分自身が体現できていますか？",
        ],
    },
    "career_work": {
        "domain_label": "キャリア・仕事",
        "value_definition":
            "私は、スキル・評判・経済的自立が時間とともに積み上がる、意味のある仕事を築く人間でありたい。",
        "reflection_questions": [
            "人生の中で、仕事はどんな役割を果たしていますか？",
            "仕事を通じて、どのように貢献したいですか？",
            "積み上がるスキルを構築していますか？　それとも、ただタスクをこなしていますか？",
        ],
    },
    "personal_growth": {
        "domain_label": "自己成長",
        "value_definition":
            "私は、知的好奇心を持ち続け、反証となる証拠を求め、学びを行動変容に変える人間でありたい。",
        "reflection_questions": [
            "何を学び続けたいですか？",
            "どんな資質を身につけたいですか？",
            "今週、実際に考え方や行動を変えるような学びはありましたか？",
        ],
    },
    "leisure": {
        "domain_label": "余暇",
        "value_definition":
            "私は、遊び・新鮮さ・自発性を、良い人生のオプションではなく不可欠な要素として守る人間でありたい。",
        "reflection_questions": [
            "どんな活動がリラックスと喜びをもたらしますか？",
            "どうやってエネルギーを回復していますか？",
            "休息を計画的に取っていますか？　それとも、ただ崩れ落ちるように休んでいますか？",
        ],
    },
    "spirituality": {
        "domain_label": "精神性",
        "value_definition":
            "私は、内なる静けさ・俯瞰する視点・日常を超えた意味との関係を育む人間でありたい。",
        "reflection_questions": [
            "宗教・哲学・より深い意味との関係はどのようなものですか？",
            "人生に対して主体的に応答していますか？　それとも、ただ反応していますか？",
            "もし恐れがなかったら、何を違うようにしますか？",
        ],
    },
    "community_social": {
        "domain_label": "コミュニティ・社会生活",
        "value_definition":
            "私は、身近な範囲を超えたコミュニティに貢献し、個々のやり取りを超えて残る価値を創る人間でありたい。",
        "reflection_questions": [
            "どんなコミュニティの一員でありたいですか？",
            "地域にどのように貢献したいですか？",
            "今、誰かが必要としている知識を自分は持っていますか？",
        ],
    },
    "health": {
        "domain_label": "健康",
        "value_definition":
            "私は、心身の健康を、他の全てを支える交渉の余地のない基盤として扱う人間でありたい。",
        "reflection_questions": [
            "自分の身体をどのようにケアしたいですか？",
            "長期的な活力を支えるライフスタイルとは？",
            "睡眠は自分の能力を守っていますか？　それとも蝕んでいますか？",
        ],
    },
    "environment": {
        "domain_label": "環境",
        "value_definition":
            "私は、集中・落ち着き・自分が望む暮らしを支えるように物理的環境をデザインする人間でありたい。",
        "reflection_questions": [
            "自分の物理的な環境や地球とどう向き合いたいですか？",
            "環境を変えることで解消できる日常の摩擦は何ですか？",
            "自分のワークスペースは、良い仕事をしたいと思わせますか？　それとも逃げ出したくなりますか？",
        ],
    },
    "creative_arts": {
        "domain_label": "創造・芸術",
        "value_definition":
            "私は、自己表現と意味の創造の実践として、オリジナルの作品を創り世に出す人間でありたい。",
        "reflection_questions": [
            "芸術的な表現とどのように関わっていますか？",
            "人生の中で、創造性はどんな役割を果たしていますか？",
            "創っている量より、消費している量の方が多くありませんか？",
        ],
    },
}


def _build_dry_run_translation(d) -> dict:
    """Return a deterministic Japanese translation for dry-run mode.

    Uses hand-curated translations so the UI feels fully Japanese
    without any LLM call.  No "(dry-run)" prefix — the header badge
    already indicates dry-run status.
    """
    ja = _JA_TRANSLATIONS.get(d.domain_id)
    if ja:
        return {
            "domain_id": d.domain_id,
            "domain_label": ja["domain_label"],
            "value_definition": ja["value_definition"],
            "reflection_questions": list(ja["reflection_questions"]),
        }
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
    """Write one AlignmentEntry to ROS_Alignment_Log."""
    from src.config import get_db_id
    from src.notion.alignment_repo import AlignmentLogRepo
    from src.notion.client import NotionDataSourceResolver, build_notion_client_from_env
    from src.notion.truncation import TruncationTracker

    client = build_notion_client_from_env()
    db_id = get_db_id("NOTION_ROS_Alignment_Log_ID")
    resolver = NotionDataSourceResolver(client=client)
    data_source_id = resolver.resolve(db_id)

    repo = AlignmentLogRepo(
        client=client,
        database_id=db_id,
        data_source_id=data_source_id,
    )
    repo.ensure_schema()

    tracker = TruncationTracker()
    repo.create_validated_entry(entry=entry, tracker=tracker)
