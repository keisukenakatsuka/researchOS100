# src/values/audio.py
"""Audio I/O for voice reflection sessions.

Provides:
- ``record_audio()``: microphone recording (Enter to start/stop)
- ``parse_score_from_text()``: extract 1–5 score from transcript
- ``prompt_score()``: voice-first score prompt with keyboard fallback

All OpenAI calls (Whisper, TTS) go through ``LLMRouter`` — never
call the OpenAI client directly from this module.

Dependencies: sounddevice, numpy (for mic recording).
"""

from __future__ import annotations

import logging
import re
import threading
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from src.llm.router import LLMRouter
    from src.values.voice import VoiceConfig

logger = logging.getLogger(__name__)

# Sample rate for mic recording (Whisper handles 16 kHz well).
SAMPLE_RATE = 16000
CHANNELS = 1


# ----------------------------------------------------------------
# Microphone recording
# ----------------------------------------------------------------

def record_audio(output_path: Path, *, sample_rate: int = SAMPLE_RATE) -> Path:
    """Record audio from the system microphone.

    Press Enter to start recording, then Enter again to stop.
    Saves a 16-bit mono WAV file.

    Parameters
    ----------
    output_path : Path
        Where to write the .wav file.
    sample_rate : int
        Recording sample rate (default: 16000).

    Returns
    -------
    Path
        The written file path.
    """
    import sounddevice as sd

    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []

    def _callback(indata, frame_count, time_info, status):
        if status:
            logger.debug("sounddevice status: %s", status)
        frames.append(indata.copy())

    input("[mic] Press Enter to START recording...")
    logger.info("Recording started (press Enter to stop)")

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="int16",
        callback=_callback,
    )
    stream.start()

    input("[mic] Press Enter to STOP recording...")
    stream.stop()
    stream.close()

    if not frames:
        logger.warning("No audio frames captured")
        # Write a silent 1-second file so downstream doesn't crash.
        _write_silent_wav(output_path, sample_rate=sample_rate, duration_sec=1)
        return output_path

    audio_data = np.concatenate(frames, axis=0)
    duration = len(audio_data) / sample_rate
    logger.info("Recorded %.1f seconds of audio", duration)

    _write_wav(output_path, audio_data, sample_rate=sample_rate)
    return output_path


def _write_wav(path: Path, data: np.ndarray, *, sample_rate: int) -> None:
    """Write int16 numpy array to WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def _write_silent_wav(path: Path, *, sample_rate: int, duration_sec: int) -> None:
    """Write a silent WAV file as a fallback."""
    n_frames = sample_rate * duration_sec
    silent = b"\x00\x00" * n_frames
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(silent)


# ----------------------------------------------------------------
# Score parsing
# ----------------------------------------------------------------

# Maps common spoken/written words to integers.
_WORD_TO_INT = {
    # English
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    # Japanese
    "いち": 1, "に": 2, "さん": 3, "し": 4, "よん": 4, "ご": 5,
    "１": 1, "２": 2, "３": 3, "４": 4, "５": 5,
}


def parse_score_from_text(text: str) -> Optional[int]:
    """Extract an integer 1–5 from transcript text.

    Tries digit extraction first, then known word mappings.

    Returns
    -------
    int or None
        Parsed score, or None if unparseable.
    """
    text = text.strip()
    # Try to find a digit 1-5
    match = re.search(r"[1-5１-５]", text)
    if match:
        ch = match.group()
        # Normalize fullwidth digits
        if ch in "１２３４５":
            return "１２３４５".index(ch) + 1
        return int(ch)

    # Try word-based lookup
    lower = text.lower()
    for word, val in _WORD_TO_INT.items():
        if word in lower:
            return val

    return None


# ----------------------------------------------------------------
# Interactive score prompt (voice-first, keyboard fallback)
# ----------------------------------------------------------------

def prompt_score(
    label: str,
    *,
    router: LLMRouter,
    voice_config: VoiceConfig,
    audio_dir: Path,
    domain_id: str,
    dry_run: bool = False,
) -> int:
    """Prompt user for a 1–5 score via voice, with keyboard fallback.

    Parameters
    ----------
    label : str
        Score label for display (e.g. "Importance Score").
    router : LLMRouter
        Router instance for Whisper transcription.
    voice_config : VoiceConfig
        Voice configuration.
    audio_dir : Path
        Directory to save temporary score recordings.
    domain_id : str
        Current domain (for file naming).
    dry_run : bool
        If True, skip mic and return a deterministic score.

    Returns
    -------
    int
        Validated score 1–5.
    """
    from src.notion.alignment_repo import validate_score_range

    if dry_run:
        logger.info("[dry-run] Returning dummy score 3 for %s", label)
        return 3

    print(f"\n  {label} (1–5):")
    print("  Record your score, or type it directly.")

    # Try voice first
    score_path = audio_dir / f"{domain_id}_{label.lower().replace(' ', '_')}.wav"
    try:
        record_audio(score_path)
        transcript = router.transcribe_audio(score_path, voice_config=voice_config)
        logger.info("Score transcript for %s: %r", label, transcript)
        print(f"  [transcript] {transcript}")
        parsed = parse_score_from_text(transcript)
        if parsed is not None:
            validate_score_range(parsed, label)
            print(f"  -> {label}: {parsed}")
            return parsed
        print("  Could not parse score from voice. Falling back to keyboard.")
    except Exception as e:
        logger.warning("Voice score failed for %s: %s", label, e)
        print(f"  Voice input failed ({e}). Falling back to keyboard.")

    # Keyboard fallback
    while True:
        raw = input(f"  Enter {label} (1–5): ").strip()
        try:
            val = int(raw)
            validate_score_range(val, label)
            return val
        except (ValueError, Exception) as e:
            print(f"  Invalid: {e}. Please enter an integer 1–5.")
