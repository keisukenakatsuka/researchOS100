# src/values/voice.py
"""Voice + avatar interface for value reflection sessions.

Provides:
- VoiceConfig: language/locale settings for voice I/O (default: Japanese)
- ReflectionQuestion / ReflectionResponse / ReflectionLog: session data
- Stub functions for future voice reflection pipeline

Language Support
----------------
Default language is Japanese (``ja``).  This affects:
- STT (Speech-to-Text) input language
- TTS (Text-to-Speech) output language
- LLM system prompts for voice processing
- Reflection question generation language

Override via:
- ``VoiceConfig(language="en")`` in code
- ``--lang en`` on the CLI
- ``VOICE_LANGUAGE=en`` in .env

Supported languages: ja (Japanese), en (English).
Additional languages can be added to ``SUPPORTED_LANGUAGES``.

NOT FULLY IMPLEMENTED — core functions raise NotImplementedError.
Data structures and configuration are production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


# ----------------------------------------------------------------
# Language configuration
# ----------------------------------------------------------------

SUPPORTED_LANGUAGES: Tuple[str, ...] = ("ja", "en")
"""ISO 639-1 language codes supported by the voice interface."""

DEFAULT_LANGUAGE: str = "ja"
"""Default voice language — Japanese."""

LanguageCode = Literal["ja", "en"]


@dataclass(frozen=True)
class VoiceConfig:
    """Configuration for voice input and output.

    Parameters
    ----------
    language : str
        ISO 639-1 language code. Default: "ja" (Japanese).
    stt_model : str
        Speech-to-Text model identifier. Default: "whisper-1".
    tts_model : str
        Text-to-Speech model identifier. Default: "tts-1".
    tts_voice : str
        TTS voice preset. Default: "alloy".
    speaking_rate : float
        TTS speaking rate multiplier. Default: 1.0.
    """

    language: str = DEFAULT_LANGUAGE
    stt_model: str = "whisper-1"
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    speaking_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {self.language!r}. "
                f"Supported: {SUPPORTED_LANGUAGES}"
            )
        if self.speaking_rate <= 0:
            raise ValueError(
                f"speaking_rate must be positive, got {self.speaking_rate}"
            )

    @property
    def is_japanese(self) -> bool:
        """True when language is Japanese."""
        return self.language == "ja"

    @property
    def display_name(self) -> str:
        """Human-readable language name."""
        names = {"ja": "Japanese", "en": "English"}
        return names.get(self.language, self.language)

    @property
    def stt_language_hint(self) -> str:
        """Language hint for Whisper STT API."""
        return self.language

    @property
    def system_prompt_language_instruction(self) -> str:
        """Instruction to include in LLM system prompts for voice tasks."""
        if self.language == "ja":
            return (
                "You MUST respond entirely in Japanese (日本語). "
                "Use natural, conversational Japanese appropriate for "
                "a personal reflection session. Avoid overly formal "
                "keigo unless the context requires it."
            )
        return (
            "Respond in English. Use natural, conversational tone "
            "appropriate for a personal reflection session."
        )


# ----------------------------------------------------------------
# Session data structures
# ----------------------------------------------------------------

@dataclass(frozen=True)
class ReflectionQuestion:
    """A question posed to the user during a voice reflection session."""
    domain_id: str
    question_text: str
    follow_up: str = ""
    language: str = DEFAULT_LANGUAGE


@dataclass(frozen=True)
class ReflectionResponse:
    """User's voice response to a reflection question."""
    domain_id: str
    question_text: str
    transcript: str
    audio_url: str = ""
    language: str = DEFAULT_LANGUAGE


@dataclass(frozen=True)
class ReflectionLog:
    """Complete log of a voice reflection session."""
    date_iso: str
    review_type: str
    responses: Tuple[ReflectionResponse, ...]
    ai_summary: str = ""
    language: str = DEFAULT_LANGUAGE


# ----------------------------------------------------------------
# Voice reflection functions (stubs)
# ----------------------------------------------------------------

def generate_reflection_questions(
    domain_id: str,
    *,
    count: int = 3,
    voice_config: Optional[VoiceConfig] = None,
) -> list[ReflectionQuestion]:
    """Generate reflection questions for a domain in the configured language.

    When ``voice_config.language`` is "ja" (default), questions are generated
    in Japanese.  When "en", in English.

    Parameters
    ----------
    domain_id : str
        The value domain to generate questions for.
    count : int
        Number of questions to generate (default: 3).
    voice_config : VoiceConfig or None
        Voice configuration. Defaults to ``VoiceConfig()`` (Japanese).

    Returns
    -------
    list[ReflectionQuestion]
        Generated questions in the target language.

    Raises
    ------
    NotImplementedError
        Voice reflection pipeline is not yet implemented.
    """
    config = voice_config or VoiceConfig()
    raise NotImplementedError(
        f"Voice reflection interface not yet implemented "
        f"(language={config.language}). "
        f"See src/values/voice.py for the planned API contract."
    )


def classify_transcript_to_domain(
    transcript: str,
    *,
    voice_config: Optional[VoiceConfig] = None,
) -> str:
    """Classify a voice transcript to the most relevant domain.

    Supports transcripts in any configured language. When the voice
    config language is "ja", the LLM prompt is tuned for Japanese
    transcript analysis.

    Parameters
    ----------
    transcript : str
        Raw voice transcript text.
    voice_config : VoiceConfig or None
        Voice configuration. Defaults to ``VoiceConfig()`` (Japanese).

    Returns
    -------
    str
        The domain_id that best matches the transcript content.

    Raises
    ------
    NotImplementedError
        Voice transcript classification is not yet implemented.
    """
    config = voice_config or VoiceConfig()
    raise NotImplementedError(
        f"Voice transcript classification not yet implemented "
        f"(language={config.language}). "
        f"See src/values/voice.py for the planned API contract."
    )


def summarize_reflection_session(
    responses: Tuple[ReflectionResponse, ...],
    *,
    voice_config: Optional[VoiceConfig] = None,
    router: Optional[object] = None,
) -> str:
    """Generate an AI summary of a complete reflection session.

    Produces a summary in the configured language. For Japanese sessions,
    the summary is written in natural Japanese.

    All LLM calls go through the ``router`` parameter (LLMRouter).

    Parameters
    ----------
    responses : tuple of ReflectionResponse
        All responses from the session.
    voice_config : VoiceConfig or None
        Voice configuration. Defaults to ``VoiceConfig()`` (Japanese).
    router : LLMRouter or None
        LLM router for summarization call.  Required for actual
        summarization.  When None, returns a placeholder string.

    Returns
    -------
    str
        AI-generated summary of the session in the target language.
    """
    config = voice_config or VoiceConfig()

    if not responses:
        return ""

    if router is None:
        return "[AI Summary unavailable — no LLM router provided]"

    # Build transcript block for the LLM.
    transcript_lines = []
    for resp in responses:
        transcript_lines.append(
            f"Domain: {resp.domain_id}\n"
            f"Question: {resp.question_text}\n"
            f"Response: {resp.transcript}"
        )
    transcript_block = "\n---\n".join(transcript_lines)

    system = (
        "You are a reflective coaching assistant. "
        "The user has completed a value reflection session where they "
        "answered questions about different life domains. "
        "Summarize the session: highlight key themes, areas of strength, "
        "and areas where the user expressed tension or misalignment. "
        "Be concise (3–5 sentences). "
        "This summary is clearly AI-generated and should be labeled as such."
    )
    user_prompt = (
        f"Here are the reflection transcripts from today's session:\n\n"
        f"{transcript_block}\n\n"
        f"Please provide a brief summary."
    )

    result = router.call_voice_processing(
        system=system,
        user=user_prompt,
        voice_config=config,
    )
    # The router returns JSON; extract the summary text.
    summary = result.parsed.get("summary", "")
    if not summary:
        # Fallback: use raw text if JSON structure doesn't match.
        summary = result.raw_text
    return f"[AI-Generated Summary]\n{summary}"
