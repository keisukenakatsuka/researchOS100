"""Web content fetcher — URL → plaintext.

Fetches web pages and extracts main body text using httpx + trafilatura.
Includes robots.txt checking and per-domain rate limiting.

Usage::

    from src.search.web_fetcher import WebFetcher

    with WebFetcher() as fetcher:
        result = fetcher.fetch("https://example.com/article")
        # result.status  → "success" / "failed" / "skipped"
        # result.text    → extracted plaintext or None
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

logger = logging.getLogger("web_fetcher")

_USER_AGENT = "ResearchOS/1.0"
_TIMEOUT = 10.0
_RATE_LIMIT_SEC = 1.0

# Exported constant — maximum characters kept from fetched text.
MAX_FETCH_CHARS = 10_000


@dataclass
class FetchResult:
    """Result of a single URL fetch attempt."""

    url: str
    status: str  # success / failed / skipped
    text: Optional[str] = None
    error: Optional[str] = None


class WebFetcher:
    """Stateful web fetcher with rate limiting and robots.txt support.

    Parameters
    ----------
    timeout : float
        HTTP request timeout in seconds (default 10).
    max_chars : int
        Maximum characters to keep from extracted text (default 10 000).
    rate_limit : float
        Minimum seconds between requests to the same domain (default 1).
    """

    def __init__(
        self,
        *,
        timeout: float = _TIMEOUT,
        max_chars: int = MAX_FETCH_CHARS,
        rate_limit: float = _RATE_LIMIT_SEC,
    ):
        self.timeout = timeout
        self.max_chars = max_chars
        self.rate_limit = rate_limit
        self._domain_last_req: Dict[str, float] = {}
        self._robots_cache: Dict[str, Optional[RobotFileParser]] = {}
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> WebFetcher:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- internal helpers ------------------------------------------------

    @staticmethod
    def _get_domain(url: str) -> str:
        try:
            return urlparse(url).netloc
        except Exception:
            return ""

    def _rate_limit_wait(self, domain: str) -> None:
        now = time.monotonic()
        last = self._domain_last_req.get(domain, 0.0)
        wait = self.rate_limit - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._domain_last_req[domain] = time.monotonic()

    def _check_robots(self, url: str) -> bool:
        """Return True if the URL is allowed by robots.txt."""
        domain = self._get_domain(url)
        if domain in self._robots_cache:
            rp = self._robots_cache[domain]
            if rp is None:
                return True
            return rp.can_fetch(_USER_AGENT, url)

        robots_url = f"{urlparse(url).scheme}://{domain}/robots.txt"
        try:
            resp = self._client.get(robots_url, timeout=5.0)
            if resp.status_code == 200:
                rp = RobotFileParser()
                rp.parse(resp.text.splitlines())
                self._robots_cache[domain] = rp
                return rp.can_fetch(_USER_AGENT, url)
            self._robots_cache[domain] = None
            return True
        except Exception:
            self._robots_cache[domain] = None
            return True

    # -- public API ------------------------------------------------------

    def fetch(self, url: str) -> FetchResult:
        """Fetch a URL and extract text content.

        Returns a FetchResult with status, text, and any error message.
        Never raises — errors are captured in the result.
        """
        if not url:
            return FetchResult(url=url, status="skipped", error="empty URL")

        domain = self._get_domain(url)

        # robots.txt
        if not self._check_robots(url):
            logger.info("Blocked by robots.txt: %s", url[:100])
            return FetchResult(url=url, status="skipped", error="blocked by robots.txt")

        # rate limit
        self._rate_limit_wait(domain)

        try:
            resp = self._client.get(url)

            if resp.status_code >= 400:
                return FetchResult(
                    url=url, status="failed", error=f"HTTP {resp.status_code}"
                )

            content_type = resp.headers.get("content-type", "")

            # Non-HTML content
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                if content_type.startswith("text/"):
                    text = resp.text[: self.max_chars]
                    return FetchResult(url=url, status="success", text=text)
                if "application/json" in content_type:
                    text = resp.text[: self.max_chars]
                    return FetchResult(url=url, status="success", text=text)
                if "application/pdf" in content_type:
                    return FetchResult(
                        url=url, status="skipped", error="PDF not supported in MVP"
                    )
                return FetchResult(
                    url=url,
                    status="skipped",
                    error=f"unsupported content-type: {content_type.split(';')[0]}",
                )

            # HTML → plaintext
            text = trafilatura.extract(resp.text)
            if text:
                text = text[: self.max_chars]

            return FetchResult(url=url, status="success", text=text)

        except httpx.TimeoutException:
            return FetchResult(url=url, status="failed", error="timeout")
        except httpx.ConnectError as e:
            return FetchResult(url=url, status="failed", error=f"connection error: {e}")
        except Exception as e:
            logger.warning("Unexpected fetch error for %s: %s", url[:100], e)
            return FetchResult(url=url, status="failed", error=str(e))
