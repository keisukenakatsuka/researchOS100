# src/search/google_cse.py
"""Google Custom Search Engine client.

Simple wrapper around the Google CSE REST API.  Retry with backoff,
fail-loud on persistent errors.

Usage::

    from src.search.google_cse import build_google_cse_from_env

    cse = build_google_cse_from_env()
    results = cse.search("Sequoia Capital venture")
    # -> [{"title": "...", "link": "...", "snippet": "..."}, ...]
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class GoogleCSEError(RuntimeError):
    """Hard failure from Google CSE API."""


def _sleep_backoff(attempt: int, base: float = 1.0, jitter: float = 0.5) -> None:
    t = (base * (2 ** attempt)) + (random.random() * jitter)
    time.sleep(min(t, 15.0))


class GoogleCSEClient:
    """Google Custom Search Engine client.

    Parameters
    ----------
    api_key : str
        Google API key (``GOOGLE_API_KEY``).
    cse_cx : str
        Custom Search Engine ID (``GOOGLE_CSE_CX``).
    max_retries : int
        Number of retries on transient errors (default 2).
    """

    ENDPOINT = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, cse_cx: str, *, max_retries: int = 2):
        if not api_key:
            raise GoogleCSEError("GOOGLE_API_KEY is required")
        if not cse_cx:
            raise GoogleCSEError("GOOGLE_CSE_CX is required")
        self.api_key = api_key
        self.cse_cx = cse_cx
        self.max_retries = max_retries
        self._total_queries = 0

    def search(self, query: str, *, num: int = 5) -> List[Dict[str, Any]]:
        """Search Google CSE.

        Returns a list of result dicts with keys: title, link, snippet.
        Returns empty list if no results found.
        """
        params = {
            "key": self.api_key,
            "cx": self.cse_cx,
            "q": query,
            "num": min(num, 10),  # CSE max is 10
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.get(self.ENDPOINT, params=params, timeout=15)
                self._total_queries += 1

                if resp.status_code == 429:
                    logger.warning("Google CSE rate-limited (429), attempt %d", attempt)
                    if attempt < self.max_retries:
                        _sleep_backoff(attempt)
                        continue
                    raise GoogleCSEError(f"Google CSE rate limit exceeded after {self.max_retries + 1} attempts")

                if resp.status_code >= 500:
                    logger.warning("Google CSE server error (%d), attempt %d", resp.status_code, attempt)
                    if attempt < self.max_retries:
                        _sleep_backoff(attempt)
                        continue
                    raise GoogleCSEError(f"Google CSE server error {resp.status_code}")

                resp.raise_for_status()
                data = resp.json()

                items = data.get("items", [])
                results = []
                for item in items[:num]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                    })

                logger.debug("Google CSE: %d results for %r", len(results), query[:60])
                return results

            except requests.exceptions.Timeout:
                last_exc = GoogleCSEError(f"Google CSE timeout on attempt {attempt}")
                if attempt < self.max_retries:
                    _sleep_backoff(attempt)
                    continue
            except requests.exceptions.ConnectionError as e:
                last_exc = GoogleCSEError(f"Google CSE connection error: {e}")
                if attempt < self.max_retries:
                    _sleep_backoff(attempt)
                    continue
            except GoogleCSEError:
                raise
            except Exception as e:
                raise GoogleCSEError(f"Google CSE unexpected error: {e}") from e

        raise last_exc or GoogleCSEError("Google CSE failed after retries")

    def usage_summary(self) -> str:
        return f"Google CSE: {self._total_queries} queries"


def build_google_cse_from_env() -> GoogleCSEClient:
    """Factory: build GoogleCSEClient from environment variables.

    Reads ``GOOGLE_API_KEY`` and ``GOOGLE_CSE_CX``.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    cse_cx = os.environ.get("GOOGLE_CSE_CX", "")
    if not api_key or not cse_cx:
        raise GoogleCSEError(
            "GOOGLE_API_KEY and GOOGLE_CSE_CX must be set in environment"
        )
    return GoogleCSEClient(api_key=api_key, cse_cx=cse_cx)
