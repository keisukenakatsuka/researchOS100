# src/llm/openai_client.py
"""Shared OpenAI client for the weekly research engine (047-052).

Provides:
- ``OpenAIClient`` — retry, caching, fail-loud
- ``build_openai_client_from_env()`` — factory from OPENAI_API_KEY

Pattern follows ``src/llm/claude_client.py``.

Usage::

    from src.llm.openai_client import build_openai_client_from_env

    llm = build_openai_client_from_env()
    result = llm.call_json(
        system="You are a research analyst.",
        user="Classify these papers: ...",
    )
    data = result.parsed  # dict
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OpenAICallError(RuntimeError):
    """Hard failure talking to OpenAI (no silent fallback)."""


class OpenAIParseError(RuntimeError):
    """OpenAI returned non-JSON or malformed JSON."""


@dataclass(frozen=True)
class OpenAIResult:
    """Immutable result of a single OpenAI call."""
    parsed: dict
    raw_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False

    @property
    def parsed_json(self) -> dict:
        """Backward-compatible alias for ``parsed``."""
        return self.parsed


def _is_retryable(exc: Exception) -> bool:
    """Check if an OpenAI exception is transient."""
    name = type(exc).__name__
    # openai library raises RateLimitError, APITimeoutError, InternalServerError, etc.
    if name in ("RateLimitError", "APITimeoutError", "InternalServerError",
                "APIConnectionError", "ServiceUnavailableError"):
        return True
    # Also retry on generic connection errors
    if "timeout" in str(exc).lower() or "connection" in str(exc).lower():
        return True
    return False


def _sleep_backoff(attempt: int, base: float = 1.0, jitter: float = 0.5) -> None:
    t = (base * (2 ** attempt)) + (random.random() * jitter)
    time.sleep(min(t, 30.0))


class OpenAIClient:
    """Shared OpenAI wrapper with retry, caching, fail-loud semantics.

    - ``call_json()`` returns structured JSON output
    - SHA256-based disk cache (deterministic, skip-on-miss)
    - Raises ``OpenAICallError`` on failure (never returns None)
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_retries: int = 3,
        timeout_sec: int = 120,
        cache_dir: Optional[Path] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._cache_dir = cache_dir
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, timeout=timeout_sec)

    @property
    def raw_client(self):
        """Expose the underlying ``openai.OpenAI`` instance for non-chat APIs.

        Used by ``LLMRouter`` for Whisper (audio.transcriptions) and
        TTS (audio.speech) calls.  Business logic must NOT access this
        directly — always go through the router.
        """
        return self._client

    # ---- Cache helpers ----

    def _cache_key(self, model: str, system: str, user: str) -> str:
        blob = json.dumps({"model": model, "system": system, "user": user},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _cache_read(self, key: str) -> Optional[dict]:
        if not self._cache_dir:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _cache_write(self, key: str, data: dict) -> None:
        if not self._cache_dir:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{key}.json"
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # ---- Main call ----

    def call_json(
        self,
        *,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
    ) -> OpenAIResult:
        """Call OpenAI with JSON response format. Fail-loud.

        Parameters
        ----------
        system : str
            System prompt.
        user : str
            User prompt.
        model : str, optional
            Override default model.
        temperature : float, optional
            Override default temperature.
        use_cache : bool
            Whether to check/write disk cache.

        Returns
        -------
        OpenAIResult
            Parsed JSON dict + token usage.

        Raises
        ------
        OpenAICallError
            On any API or network failure.
        OpenAIParseError
            If the response is not valid JSON.
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature

        # Check cache
        cache_key = self._cache_key(model, system, user)
        if use_cache:
            cached = self._cache_read(cache_key)
            if cached is not None:
                logger.debug("OpenAI cache hit: %s", cache_key)
                return OpenAIResult(
                    parsed=cached, raw_text=json.dumps(cached),
                    cached=True,
                )

        # API call with retry
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                break
            except Exception as e:
                last_err = e
                if _is_retryable(e) and attempt < self.max_retries:
                    logger.warning(
                        "OpenAI retryable error (attempt %d/%d): %s",
                        attempt + 1, self.max_retries, e,
                    )
                    _sleep_backoff(attempt)
                    continue
                raise OpenAICallError(
                    f"OpenAI API failed after {attempt + 1} attempt(s): "
                    f"{type(e).__name__}: {e}"
                ) from e
        else:
            raise OpenAICallError(
                f"OpenAI API failed after {self.max_retries + 1} retries: {last_err}"
            )

        # Extract text
        raw_text = resp.choices[0].message.content or ""
        input_tok = resp.usage.prompt_tokens if resp.usage else 0
        output_tok = resp.usage.completion_tokens if resp.usage else 0

        self._total_input_tokens += input_tok
        self._total_output_tokens += output_tok
        self._call_count += 1

        # Parse JSON
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise OpenAIParseError(
                f"OpenAI returned non-JSON: {e}\nRaw: {raw_text[:500]}"
            ) from e

        if not isinstance(parsed, dict):
            raise OpenAIParseError(
                f"Expected JSON object, got {type(parsed).__name__}"
            )

        # Write cache
        if use_cache:
            self._cache_write(cache_key, parsed)

        return OpenAIResult(
            parsed=parsed, raw_text=raw_text,
            input_tokens=input_tok, output_tokens=output_tok,
        )

    # ---- Usage tracking ----

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def call_count(self) -> int:
        return self._call_count

    def usage_summary(self) -> str:
        return (
            f"OpenAI usage: {self._call_count} calls, "
            f"{self._total_input_tokens} input + "
            f"{self._total_output_tokens} output tokens"
        )


def build_openai_client_from_env(
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    cache_dir: Optional[Path] = None,
) -> OpenAIClient:
    """Build an OpenAIClient from OPENAI_API_KEY env var.

    Raises ``OpenAICallError`` immediately if the key is missing.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise OpenAICallError(
            "OPENAI_API_KEY is required but not set. "
            "LLM is mandatory for the weekly research engine."
        )
    if cache_dir is None:
        cache_dir = Path(".cache/llm")
    return OpenAIClient(
        api_key=api_key,
        model=model,
        temperature=temperature,
        cache_dir=cache_dir,
    )
