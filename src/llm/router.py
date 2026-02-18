# src/llm/router.py
"""LLM routing layer — route tasks to the appropriate model provider.

Architecture Decision
---------------------
- **Refinement tasks** -> OpenAI models
- **Voice tasks** -> OpenAI models (language-aware, default: Japanese)
- No direct LLM calls outside this router in business logic.

Voice Language Support
----------------------
The router's ``call_voice_processing()`` method accepts an optional
``voice_config`` parameter.  When provided, the language instruction
from ``VoiceConfig.system_prompt_language_instruction`` is prepended
to the system prompt, ensuring LLM responses match the user's language.

Default language is Japanese (ja).

Usage::

    from src.llm.router import LLMRouter, build_router_from_env

    router = build_router_from_env()
    result = router.call_refinement(system=..., user=...)

    # Voice with Japanese (default)
    from src.values.voice import VoiceConfig
    result = router.call_voice_processing(
        system="...", user="...",
        voice_config=VoiceConfig(),  # Japanese
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.llm.openai_client import (
    OpenAIClient,
    OpenAIResult,
    build_openai_client_from_env,
)

logger = logging.getLogger(__name__)

TASK_REFINEMENT = "refinement"
TASK_VOICE = "voice"
TASK_CLASSIFICATION = "classification"
TASK_REASONING = "reasoning"


@dataclass(frozen=True)
class RouteConfig:
    """Configuration for a single LLM route."""
    provider: str
    model: str
    temperature: float = 0.2
    description: str = ""


DEFAULT_ROUTES: Dict[str, RouteConfig] = {
    TASK_REFINEMENT: RouteConfig(
        provider="openai", model="gpt-4o", temperature=0.3,
        description="Value domain refinement, structured diff generation",
    ),
    TASK_VOICE: RouteConfig(
        provider="openai", model="gpt-4o-mini", temperature=0.2,
        description="Voice transcription processing, audio analysis",
    ),
    TASK_CLASSIFICATION: RouteConfig(
        provider="openai", model="gpt-4o-mini", temperature=0.2,
        description="Paper/event classification and clustering",
    ),
    TASK_REASONING: RouteConfig(
        provider="openai", model="gpt-4o", temperature=0.4,
        description="High-level design reasoning",
    ),
}


class LLMRouter:
    """Routes LLM calls to the appropriate provider based on task type."""

    def __init__(
        self,
        *,
        openai_client: OpenAIClient,
        routes: Optional[Dict[str, RouteConfig]] = None,
    ):
        self._openai = openai_client
        self._routes = routes or dict(DEFAULT_ROUTES)

    def get_route(self, task_type: str) -> RouteConfig:
        if task_type not in self._routes:
            raise ValueError(
                f"Unknown task type: {task_type!r}. "
                f"Available: {sorted(self._routes.keys())}"
            )
        return self._routes[task_type]

    def _resolve_client(self, route: RouteConfig) -> OpenAIClient:
        if route.provider == "openai":
            return self._openai
        raise ValueError(f"Unsupported provider: {route.provider!r}")

    def call(
        self,
        *,
        task_type: str,
        system: str,
        user: str,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        use_cache: bool = True,
    ) -> OpenAIResult:
        route = self.get_route(task_type)
        client = self._resolve_client(route)
        model = model_override or route.model
        temperature = temperature_override if temperature_override is not None else route.temperature
        logger.info("LLM route: task=%s provider=%s model=%s", task_type, route.provider, model)
        return client.call_json(
            system=system, user=user, model=model,
            temperature=temperature, use_cache=use_cache,
        )

    def call_refinement(self, *, system: str, user: str, **kwargs: Any) -> OpenAIResult:
        return self.call(task_type=TASK_REFINEMENT, system=system, user=user, **kwargs)

    def call_voice_processing(
        self,
        *,
        system: str,
        user: str,
        voice_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> OpenAIResult:
        """Call voice processing route with optional language-aware prompt.

        Parameters
        ----------
        system : str
            Base system prompt.
        user : str
            User prompt.
        voice_config : VoiceConfig or None
            When provided, the language instruction is prepended to the
            system prompt.  Default language is Japanese.
        """
        if voice_config is not None:
            lang_instruction = voice_config.system_prompt_language_instruction
            system = f"{lang_instruction}\n\n{system}"
            logger.info(
                "Voice processing: language=%s (%s)",
                voice_config.language, voice_config.display_name,
            )
        return self.call(task_type=TASK_VOICE, system=system, user=user, **kwargs)

    def transcribe_audio(
        self,
        audio_path: Path,
        *,
        voice_config: Optional[Any] = None,
    ) -> str:
        """Transcribe an audio file via OpenAI Whisper.

        Parameters
        ----------
        audio_path : Path
            Path to audio file (.wav, .mp3, etc.).
        voice_config : VoiceConfig or None
            When provided, ``stt_language_hint`` is passed to Whisper.

        Returns
        -------
        str
            Transcribed text.
        """
        stt_model = "whisper-1"
        language = None
        if voice_config is not None:
            stt_model = voice_config.stt_model
            language = voice_config.stt_language_hint
            logger.info(
                "Transcribing audio: model=%s language=%s file=%s",
                stt_model, language, audio_path.name,
            )
        else:
            logger.info("Transcribing audio: model=%s file=%s", stt_model, audio_path.name)

        with open(audio_path, "rb") as f:
            kwargs: Dict[str, Any] = {"model": stt_model, "file": f}
            if language:
                kwargs["language"] = language
            result = self._openai.raw_client.audio.transcriptions.create(**kwargs)
        return result.text

    def speak_text(
        self,
        text: str,
        output_path: Path,
        *,
        voice_config: Optional[Any] = None,
    ) -> Path:
        """Generate speech audio via OpenAI TTS.

        Parameters
        ----------
        text : str
            Text to speak.
        output_path : Path
            Where to write the audio file.
        voice_config : VoiceConfig or None
            TTS model and voice settings.

        Returns
        -------
        Path
            Path to generated audio file.
        """
        tts_model = "tts-1"
        tts_voice = "alloy"
        if voice_config is not None:
            tts_model = voice_config.tts_model
            tts_voice = voice_config.tts_voice
        logger.info("TTS: model=%s voice=%s -> %s", tts_model, tts_voice, output_path.name)

        response = self._openai.raw_client.audio.speech.create(
            model=tts_model,
            voice=tts_voice,
            input=text,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        response.stream_to_file(str(output_path))
        return output_path

    @property
    def usage_summary(self) -> str:
        return self._openai.usage_summary()


def build_router_from_env(
    *,
    refinement_model: Optional[str] = None,
    voice_model: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> LLMRouter:
    """Build an LLMRouter from environment variables."""
    openai_client = build_openai_client_from_env(cache_dir=cache_dir)
    routes = dict(DEFAULT_ROUTES)

    ref_model = refinement_model or os.getenv("LLM_REFINEMENT_MODEL", "").strip()
    if ref_model:
        existing = routes[TASK_REFINEMENT]
        routes[TASK_REFINEMENT] = RouteConfig(
            provider=existing.provider, model=ref_model,
            temperature=existing.temperature, description=existing.description,
        )

    voice_m = voice_model or os.getenv("LLM_VOICE_MODEL", "").strip()
    if voice_m:
        existing = routes[TASK_VOICE]
        routes[TASK_VOICE] = RouteConfig(
            provider=existing.provider, model=voice_m,
            temperature=existing.temperature, description=existing.description,
        )

    return LLMRouter(openai_client=openai_client, routes=routes)
