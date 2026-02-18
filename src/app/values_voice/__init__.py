# src/app/values_voice/__init__.py
"""Values Voice Reflection — browser-based UI.

Provides a FastAPI server + single-page frontend for interactive
voice reflection sessions.  Reuses all existing modules:
- src.values.schema (ValueDomain, AlignmentEntry)
- src.values.voice  (VoiceConfig, ReflectionResponse)
- src.llm.router    (LLMRouter — Whisper, TTS, summarization)
- src.notion.alignment_repo (AlignmentLogRepo)

Run via::

    python -m src.app.values_voice --run --lang ja
"""
