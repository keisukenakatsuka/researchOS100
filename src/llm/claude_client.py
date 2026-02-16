# src/llm/claude_client.py
from __future__ import annotations

import json
import os
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests


DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_API_VERSION = "2023-06-01"  # Claude Messages API version header
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_MAX_RETRIES = 6


class ClaudeAPIError(RuntimeError):
    """HTTP / protocol error talking to Anthropic."""


class ClaudeStructuredOutputError(RuntimeError):
    """Claude returned non-JSON or refused / truncated when JSON required."""


@dataclass(frozen=True)
class ClaudeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(frozen=True)
class ClaudeResult:
    ok: bool
    stop_reason: Optional[str]
    text: str
    parsed_json: Optional[dict]
    usage: ClaudeUsage
    raw: dict


def _is_retryable_status(code: int) -> bool:
    return code in (429, 500, 502, 503, 504)


def _sleep_backoff(attempt: int, base: float = 0.6, jitter: float = 0.4) -> None:
    # exponential-ish backoff with jitter
    t = (base * (2 ** attempt)) + (random.random() * jitter)
    time.sleep(min(t, 20.0))


def _extract_text_from_message(resp: dict) -> str:
    """
    Anthropic Messages API response includes `content` as a list of blocks.
    For JSON outputs, docs say the valid JSON is in response.content[0].text.
    We'll defensively concatenate text blocks.
    """
    blocks = resp.get("content") or []
    out: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and isinstance(b.get("text"), str):
            out.append(b["text"])
    return "\n".join(out).strip()


def _extract_usage(resp: dict) -> ClaudeUsage:
    u = resp.get("usage") or {}
    # keys may vary; keep safe defaults
    return ClaudeUsage(
        input_tokens=int(u.get("input_tokens") or 0),
        output_tokens=int(u.get("output_tokens") or 0),
        cache_creation_input_tokens=int(u.get("cache_creation_input_tokens") or 0),
        cache_read_input_tokens=int(u.get("cache_read_input_tokens") or 0),
    )

def _coerce_json_schema(schema: dict) -> dict:
    """
    Accept either:
      A) raw JSON Schema: {"type": "object", ...}
      B) wrapped: {"name": "...", "schema": {...raw...}}
    Return raw JSON Schema only.
    """
    if not isinstance(schema, dict):
        raise TypeError("json_schema must be a dict")

    # Wrapped form
    if "schema" in schema and isinstance(schema.get("schema"), dict) and "type" in schema["schema"]:
        return schema["schema"]

    # Raw form
    if "type" in schema:
        return schema

    raise ValueError(f"Invalid json_schema: missing 'type' at root (got keys={list(schema.keys())})")


class ClaudeClient:
    """
    Minimal, dependency-free Anthropic client for:
      - POST /v1/messages
      - JSON schema constrained outputs via output_config.format

    Docs:
      - Messages: POST https://api.anthropic.com/v1/messages
      - Structured outputs: output_config.format = {type:"json_schema", schema:{...}}
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        anthropic_version: str = DEFAULT_API_VERSION,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        max_retries: int = DEFAULT_MAX_RETRIES,
        log_requests: bool = False,
        log_responses: bool = False,
        log_truncate: int = 2000,
        session: Optional[requests.Session] = None,
    ) -> None:
        api_key = (api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "ClaudeClient requires an API key. Set ANTHROPIC_API_KEY (recommended) "
                "or pass api_key=..."
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version.strip()
        self.timeout_sec = int(timeout_sec)
        self.max_retries = int(max_retries)
        self.log_requests = bool(log_requests)
        self.log_responses = bool(log_responses)
        self.log_truncate = int(log_truncate)
        self.session = session or requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }

    def _log(self, label: str, payload: Union[str, dict]) -> None:
        if isinstance(payload, dict):
            s = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            s = payload
        if len(s) > self.log_truncate:
            s = s[: self.log_truncate] + "...(truncated)"
        print(f"[claude_client] {label}:\n{s}")

    def messages_create(self, *, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/messages"
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self.log_requests:
                    self._log("REQUEST body", body)

                resp = self.session.request(
                    method="POST",
                    url=url,
                    headers=self._headers(),
                    data=json.dumps(body),
                    timeout=self.timeout_sec,
                )

                # HTTP errors
                if resp.status_code >= 400:
                    # Attempt to parse error payload
                    try:
                        err_payload = resp.json()
                    except Exception:
                        err_payload = {"error": resp.text}

                    if self.log_responses:
                        self._log(f"RESPONSE {resp.status_code}", err_payload)

                    if _is_retryable_status(resp.status_code) and attempt < self.max_retries:
                        _sleep_backoff(attempt)
                        continue

                    raise ClaudeAPIError(
                        f"Claude API HTTP {resp.status_code}: {err_payload}"
                    )

                data = resp.json()

                if self.log_responses:
                    self._log("RESPONSE 200", data)

                return data

            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                if attempt < self.max_retries:
                    _sleep_backoff(attempt)
                    continue
                raise ClaudeAPIError(f"Claude API network error: {type(e).__name__}: {e}") from e
            except ClaudeAPIError:
                raise
            except Exception as e:
                last_err = e
                raise ClaudeAPIError(f"Claude API unexpected error: {type(e).__name__}: {e}") from e

        raise ClaudeAPIError(f"Claude API failed after retries: {last_err}")

    # -----------------------------
    # JSON (schema-constrained) call
    # -----------------------------
    def call_json(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        json_schema: Dict[str, Any],
        max_tokens: int = 2048,
        system: Optional[Union[str, List[Dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ClaudeResult:
        """
        Enforce JSON via output_config.format type=json_schema.
        """
        raw_schema = _coerce_json_schema(json_schema)

        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": int(max_tokens),
            "messages": messages,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": raw_schema,
                }
            },
        }
        if system is not None:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = float(temperature)
        if extra:
            body.update(extra)

        resp = self.messages_create(body=body)

        stop_reason = resp.get("stop_reason")
        text = _extract_text_from_message(resp)
        usage = _extract_usage(resp)

        if stop_reason in ("refusal", "max_tokens"):
            return ClaudeResult(
                ok=False,
                stop_reason=stop_reason,
                text=text,
                parsed_json=None,
                usage=usage,
                raw=resp,
            )

        try:
            parsed = json.loads(text) if text else None
        except Exception as e:
            raise ClaudeStructuredOutputError(
                f"Expected JSON (structured output), but JSON parsing failed: {type(e).__name__}: {e}\n"
                f"stop_reason={stop_reason}\n"
                f"text_head={text[:200]!r}"
            ) from e

        if not isinstance(parsed, dict):
            raise ClaudeStructuredOutputError(
                f"Expected JSON object at root, got {type(parsed).__name__}."
            )

        return ClaudeResult(
            ok=True,
            stop_reason=stop_reason,
            text=text,
            parsed_json=parsed,
            usage=usage,
            raw=resp,
        )



# -----------------------------
# Convenience helper for your notebooks
# -----------------------------
def build_claude_client_from_env(
    *,
    api_key_env: str = "ANTHROPIC_API_KEY",
    base_url_env: str = "ANTHROPIC_BASE_URL",
    version_env: str = "ANTHROPIC_VERSION",
    default_version: str = DEFAULT_API_VERSION,
    timeout_sec_env: str = "ANTHROPIC_TIMEOUT_SEC",
    max_retries_env: str = "ANTHROPIC_MAX_RETRIES",
    log_requests: bool = False,
    log_responses: bool = False,
) -> ClaudeClient:
    """
    Build ClaudeClient using environment variables.
    - API key: ANTHROPIC_API_KEY (recommended) or CLAUDE_API_KEY as fallback
    - Base URL: ANTHROPIC_BASE_URL (optional)
    - Version: ANTHROPIC_VERSION (optional)
    """
    api_key = os.environ.get(api_key_env, "").strip() or os.environ.get("CLAUDE_API_KEY", "").strip()
    base_url = os.environ.get(base_url_env, DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    version = os.environ.get(version_env, default_version).strip() or default_version

    timeout_sec = int(os.environ.get(timeout_sec_env, str(DEFAULT_TIMEOUT_SEC)))
    max_retries = int(os.environ.get(max_retries_env, str(DEFAULT_MAX_RETRIES)))

    return ClaudeClient(
        api_key=api_key or None,
        base_url=base_url,
        anthropic_version=version,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        log_requests=log_requests,
        log_responses=log_responses,
    )
