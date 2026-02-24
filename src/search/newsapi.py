# src/search/newsapi.py
"""NewsAPI client for searching recent news articles.

Simple wrapper around the NewsAPI ``/v2/everything`` endpoint.
Retry with backoff, fail-loud on persistent errors.

Usage::

    from src.search.newsapi import build_newsapi_from_env

    news = build_newsapi_from_env()
    results = news.search("Sequoia Capital")
    # -> [{"title": "...", "url": "...", "source": "...", ...}, ...]
"""

from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class NewsAPIError(RuntimeError):
    """Hard failure from NewsAPI."""


def _sleep_backoff(attempt: int, base: float = 1.0, jitter: float = 0.5) -> None:
    t = (base * (2 ** attempt)) + (random.random() * jitter)
    time.sleep(min(t, 15.0))


class NewsAPIClient:
    """NewsAPI client.

    Parameters
    ----------
    api_key : str
        NewsAPI key (``NEWSAPI_KEY``).
    max_retries : int
        Number of retries on transient errors (default 2).
    """

    ENDPOINT = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, *, max_retries: int = 2):
        if not api_key:
            raise NewsAPIError("NEWSAPI_KEY is required")
        self.api_key = api_key
        self.max_retries = max_retries
        self._total_queries = 0

    def search(
        self,
        query: str,
        *,
        days_back: int = 30,
        page_size: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search NewsAPI for recent articles.

        Returns a list of result dicts with keys:
        title, url, source, publishedAt, description.
        Returns empty list if no results found.
        """
        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "from": from_date,
            "pageSize": min(page_size, 100),
            "sortBy": "relevancy",
            "apiKey": self.api_key,
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.get(self.ENDPOINT, params=params, timeout=15)
                self._total_queries += 1

                if resp.status_code == 429:
                    logger.warning("NewsAPI rate-limited (429), attempt %d", attempt)
                    if attempt < self.max_retries:
                        _sleep_backoff(attempt)
                        continue
                    raise NewsAPIError(f"NewsAPI rate limit exceeded after {self.max_retries + 1} attempts")

                if resp.status_code >= 500:
                    logger.warning("NewsAPI server error (%d), attempt %d", resp.status_code, attempt)
                    if attempt < self.max_retries:
                        _sleep_backoff(attempt)
                        continue
                    raise NewsAPIError(f"NewsAPI server error {resp.status_code}")

                if resp.status_code == 401:
                    raise NewsAPIError("NewsAPI: invalid API key (401)")

                resp.raise_for_status()
                data = resp.json()

                if data.get("status") != "ok":
                    raise NewsAPIError(f"NewsAPI error: {data.get('message', 'unknown')}")

                articles = data.get("articles", [])
                results = []
                for art in articles[:page_size]:
                    results.append({
                        "title": art.get("title", ""),
                        "url": art.get("url", ""),
                        "source": (art.get("source") or {}).get("name", ""),
                        "publishedAt": art.get("publishedAt", ""),
                        "description": art.get("description", ""),
                    })

                logger.debug("NewsAPI: %d results for %r", len(results), query[:60])
                return results

            except requests.exceptions.Timeout:
                last_exc = NewsAPIError(f"NewsAPI timeout on attempt {attempt}")
                if attempt < self.max_retries:
                    _sleep_backoff(attempt)
                    continue
            except requests.exceptions.ConnectionError as e:
                last_exc = NewsAPIError(f"NewsAPI connection error: {e}")
                if attempt < self.max_retries:
                    _sleep_backoff(attempt)
                    continue
            except NewsAPIError:
                raise
            except Exception as e:
                raise NewsAPIError(f"NewsAPI unexpected error: {e}") from e

        raise last_exc or NewsAPIError("NewsAPI failed after retries")

    def usage_summary(self) -> str:
        return f"NewsAPI: {self._total_queries} queries"


def build_newsapi_from_env() -> NewsAPIClient:
    """Factory: build NewsAPIClient from environment variables.

    Reads ``NEWSAPI_KEY``.
    """
    api_key = os.environ.get("NEWSAPI_KEY", "")
    if not api_key:
        raise NewsAPIError("NEWSAPI_KEY must be set in environment")
    return NewsAPIClient(api_key=api_key)
