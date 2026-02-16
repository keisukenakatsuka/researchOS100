# src/notion/client.py
"""
Notion client wrapper (STRICT CONSTRAINTS COMPLIANT)

Constraints (must follow)
-------------------------
- Assume Notion API version >= 2025-09-03.
- Env IDs are *database IDs* (UUID).
- MUST NOT call POST /v1/databases/{database_id}/query (treat unsupported).
- Use GET /v1/databases/{database_id} for metadata/schema.
- For querying content, use ONLY POST /v1/data_sources/{data_source_id}/query.

data_source_id resolution (EXACT)
---------------------------------
1) Fetch database object via GET /v1/databases/{database_id}
2) Recursively scan returned JSON for UUID-like strings
3) Validate candidates by calling POST /v1/data_sources/{candidate}/query with {"page_size": 1}
4) Select the first candidate that successfully queries

Caching rules (caller responsibility, but supported here)
--------------------------------------------------------
- Resolve data_source_id ONCE in a dedicated setup cell and cache mapping:
    RESOLVED_DB[name] = {"database_id": ..., "data_source_id": ...}
- All subsequent queries MUST use cached data_source_id
- MUST NOT repeat deep scanning after setup cell

Other rules
-----------
- Always normalize 32-hex IDs into hyphenated UUIDs before calling Notion endpoints
- If database.properties is empty, infer schema by sampling one page via data_sources query (page_size=1);
  in inference mode, only trust "type" fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Mapping, Tuple
from datetime import datetime, timezone
import json
import os
import random
import re
import time

import requests


DEFAULT_NOTION_VERSION = "2025-09-03"
DEFAULT_TIMEOUT_SEC = 30


class NotionAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


@dataclass(frozen=True)
class NotionClientConfig:
    token: str
    notion_version: str = DEFAULT_NOTION_VERSION
    base_url: str = "https://api.notion.com/v1"
    timeout_sec: int = DEFAULT_TIMEOUT_SEC

    max_retries: int = 6
    backoff_base_sec: float = 0.6
    backoff_jitter_sec: float = 0.4
    retry_statuses: Tuple[int, ...] = (429, 500, 502, 503, 504)

    log_requests: bool = False
    log_responses: bool = False
    log_truncate: int = 2000


UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
HEX32_RE = re.compile(r"^[0-9a-fA-F]{32}$")


def normalize_uuid(id_like: str) -> str:
    """
    Normalize either:
      - hyphenated UUID
      - 32-hex string
    into hyphenated UUID.

    Raises ValueError if not UUID-like.
    """
    s = (id_like or "").strip()
    if UUID_RE.match(s):
        return s.lower()
    if HEX32_RE.match(s):
        s = s.lower()
        return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"
    raise ValueError(f"Not a UUID-like string: {id_like}")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

class NotionClient:
    def __init__(self, config: "NotionClientConfig"):
        # config is frozen=True. Never mutate it.
        self.config = config

        token = (self.config.token or "").strip()
        if not token:
            raise ValueError("NOTION_TOKEN is not set (config.token is empty)")
        self.token = token

        # requests session (used by _request)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Notion-Version": self.config.notion_version,
                "Content-Type": "application/json",
            }
        )
    # ---- forbidden endpoints (guard rails) ----

    def query_database(self, *args: Any, **kwargs: Any) -> None:
        raise NotionAPIError(
            "Forbidden: POST /v1/databases/{database_id}/query is not allowed by policy. "
            "Use query_data_source(data_source_id=...) instead."
        )




    # ---- allowed endpoints ----

    def get_database(self, *, database_id: str) -> dict:
        database_id = normalize_uuid(database_id)
        return self._request("GET", f"/databases/{database_id}")

    def query_data_source(
        self,
        *,
        data_source_id: str,
        filter: Optional[dict] = None,
        sorts: Optional[List[dict]] = None,
        page_size: int = 100,
        start_cursor: Optional[str] = None,
        fetch_all: bool = True,
    ) -> List[dict]:
        """
        Query content via POST /v1/data_sources/{data_source_id}/query
        (STRICT: the only supported query endpoint)
        """
        # data_source_id is not guaranteed to be UUID-like across API versions.
        # Normalize only if it looks like UUID/32-hex; otherwise keep raw.
        ds_raw = (data_source_id or "").strip()
        try:
            data_source_id = normalize_uuid(ds_raw)
        except ValueError:
            data_source_id = ds_raw
        if not data_source_id:
            raise ValueError("data_source_id is empty")


        body: Dict[str, Any] = {"page_size": page_size}
        if filter is not None:
            body["filter"] = filter
        if sorts is not None:
            body["sorts"] = sorts
        if start_cursor is not None:
            body["start_cursor"] = start_cursor

        results: List[dict] = []
        cursor: Optional[str] = start_cursor

        while True:
            if cursor is not None:
                body["start_cursor"] = cursor
            else:
                body.pop("start_cursor", None)

            data = self._request("POST", f"/data_sources/{data_source_id}/query", json_body=body)
            batch = data.get("results", [])
            results.extend(batch)

            if not fetch_all:
                break

            if data.get("has_more"):
                cursor = data.get("next_cursor")
                if not cursor:
                    break
            else:
                break

        return results

    def retrieve_page(self, *, page_id: str) -> dict:
        page_id = normalize_uuid(page_id)
        return self._request("GET", f"/pages/{page_id}")

    def create_page(
        self,
        *,
        parent_db_id: str,
        properties: dict,
        children: Optional[List[dict]] = None,
    ) -> dict:
        parent_db_id = normalize_uuid(parent_db_id)
        body: Dict[str, Any] = {"parent": {"database_id": parent_db_id}, "properties": properties}
        if children is not None:
            body["children"] = children
        return self._request("POST", "/pages", json_body=body)

    def update_page(
        self,
        *,
        page_id: str,
        properties: dict,
        archived: Optional[bool] = None,
    ) -> dict:
        page_id = normalize_uuid(page_id)
        body: Dict[str, Any] = {"properties": properties}
        if archived is not None:
            body["archived"] = archived
        return self._request("PATCH", f"/pages/{page_id}", json_body=body)

    # ---- internal ----

    def _request(self, method: str, path: str, json_body: Optional[dict] = None) -> dict:
        url = self.config.base_url.rstrip("/") + path
        timeout = self.config.timeout_sec

        for attempt in range(self.config.max_retries + 1):
            try:
                if self.config.log_requests:
                    self._log_request(method, url, json_body)

                resp = self.session.request(
                    method=method,
                    url=url,
                    json=json_body,
                    timeout=timeout,
                )

                if self.config.log_responses:
                    self._log_response(resp)

                if 200 <= resp.status_code < 300:
                    if resp.text:
                        return resp.json()
                    return {}

                if resp.status_code in self.config.retry_statuses and attempt < self.config.max_retries:
                    time.sleep(self._compute_backoff(resp, attempt))
                    continue

                raise self._to_error(resp)

            except requests.RequestException as e:
                if attempt < self.config.max_retries:
                    time.sleep(self._compute_backoff(None, attempt))
                    continue
                raise NotionAPIError(f"Network error calling Notion API: {e}") from e

        raise NotionAPIError("Unexpected NotionClient retry exhaustion")

    def _compute_backoff(self, resp: Optional[requests.Response], attempt: int) -> float:
        if resp is not None:
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    return float(ra)
                except Exception:
                    pass
        base = self.config.backoff_base_sec * (2 ** attempt)
        jitter = random.random() * self.config.backoff_jitter_sec
        return min(30.0, base + jitter)

    def _to_error(self, resp: requests.Response) -> NotionAPIError:
        status = resp.status_code
        text = resp.text or ""
        payload: dict = {}
        msg = f"Notion API error (status={status})"
        try:
            payload = resp.json()
            code = payload.get("code")
            message = payload.get("message")
            if code or message:
                msg = f"Notion API error (status={status}, code={code}): {message}"
        except Exception:
            msg = f"Notion API error (status={status}): {text[:300]}"
        return NotionAPIError(msg, status_code=status, payload=payload)

    def _log_request(self, method: str, url: str, json_body: Optional[dict]) -> None:
        body_str = "" if json_body is None else json.dumps(json_body, ensure_ascii=False)
        if len(body_str) > self.config.log_truncate:
            body_str = body_str[: self.config.log_truncate] + "...(truncated)"
        print(f"[NotionClient] {method} {url}")
        if body_str:
            print(f"[NotionClient] body={body_str}")

    def _log_response(self, resp: requests.Response) -> None:
        text = resp.text or ""
        if len(text) > self.config.log_truncate:
            text = text[: self.config.log_truncate] + "...(truncated)"
        print(f"[NotionClient] status={resp.status_code}")
        if text:
            print(f"[NotionClient] resp={text}")


# -------------------------
# Data source resolver (EXACT resolution approach + cache holder)
# -------------------------

def _collect_uuid_like_strings(obj: Any, out: List[str]) -> None:
    """
    Recursively scan JSON-like structures, collecting UUID-like strings (both
    hyphenated and 32-hex) without validating they are actual Notion IDs yet.
    """
    if obj is None:
        return
    if isinstance(obj, str):
        s = obj.strip()
        # Quick heuristic: accept both forms; normalize later
        if UUID_RE.match(s) or HEX32_RE.match(s):
            out.append(s)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_uuid_like_strings(v, out)
        return
    if isinstance(obj, list):
        for it in obj:
            _collect_uuid_like_strings(it, out)
        return
    # ignore other types


@dataclass
class ResolvedDB:
    name: str
    database_id: str
    data_source_id: str
    resolved_at: str


class NotionDataSourceResolver:
    """
    Resolves data_source_id for a database_id using the exact required approach.

    IMPORTANT: Call resolve_once(...) in a dedicated setup cell and store the result.
    Do NOT call resolve_once repeatedly during normal operation.
    """

    def __init__(self, client: NotionClient):
        self.client = client
        self._cache_by_name: Dict[str, ResolvedDB] = {}

    def resolve_once(self, *, name: str, database_id: str) -> ResolvedDB:
        """
        Resolve and cache data_source_id for a given database_id.

        - Fetch DB object
        - Deep scan for UUID-like strings
        - Validate candidates by querying data_sources
        - Select first successful candidate
        """
        if name in self._cache_by_name:
            cached = self._cache_by_name[name]
            # If the caller passes a different database_id for the same name, treat it as a mismatch.
            try:
                if normalize_uuid(database_id) == normalize_uuid(cached.database_id):
                    return cached
            except Exception:
                pass
            # mismatch -> overwrite by resolving again


        db_id = normalize_uuid(database_id)
        db_meta = self.client.get_database(database_id=db_id)

        candidates_raw: List[str] = []
        _collect_uuid_like_strings(db_meta, candidates_raw)

        # Normalize + de-duplicate while preserving order
        candidates: List[str] = []
        seen: set[str] = set()
        for c in candidates_raw:
            try:
                u = normalize_uuid(c)
            except ValueError:
                continue
            if u not in seen:
                seen.add(u)
                candidates.append(u)

        # Validate candidates by attempting a tiny data_sources query
        for cand in candidates:
            try:
                self.client.query_data_source(data_source_id=cand, page_size=1, fetch_all=False)
                resolved = ResolvedDB(
                    name=name,
                    database_id=db_id,
                    data_source_id=cand,
                    resolved_at=now_iso(),
                )
                self._cache_by_name[name] = resolved
                return resolved
            except NotionAPIError:
                continue
        # Safety cap to avoid too many probe queries
        if len(candidates) > 200:
            candidates = candidates[:200]

        raise NotionAPIError(
            f"Failed to resolve data_source_id for database_id={db_id}. "
            f"Scanned {len(candidates)} UUID-like candidates and none worked."
        )

    def get_cached(self, *, name: str) -> ResolvedDB:
        if name not in self._cache_by_name:
            raise KeyError(f"RESOLVED_DB missing '{name}'. Resolve once in setup cell.")
        return self._cache_by_name[name]

    def export_cache(self) -> Dict[str, dict]:
        """
        Export mapping for your notebook setup cell:
        RESOLVED_DB[name] = {"database_id": ..., "data_source_id": ...}
        """
        out: Dict[str, dict] = {}
        for k, v in self._cache_by_name.items():
            out[k] = {"database_id": v.database_id, "data_source_id": v.data_source_id, "resolved_at": v.resolved_at}
        return out


# -------------------------
# Schema inference helper (only trust "type" fields)
# -------------------------

def infer_schema_types_from_sample_page(sample_page: Mapping[str, Any]) -> Dict[str, str]:
    """
    If database.properties is empty, infer schema by sampling one page
    via data_sources query(page_size=1). In inference mode, only trust "type" fields.

    Returns mapping: {property_name: notion_type}
    """
    props = sample_page.get("properties", {})
    if not isinstance(props, dict):
        return {}

    inferred: Dict[str, str] = {}
    for prop_name, prop_obj in props.items():
        if isinstance(prop_obj, dict):
            t = prop_obj.get("type")
            if isinstance(t, str) and t:
                inferred[prop_name] = t
    return inferred


# -------------------------
# Convenience constructor
# -------------------------

def build_notion_client_from_env(
    *,
    token_env: str = "NOTION_TOKEN",
    version_env: str = "NOTION_VERSION",
    default_version: str = DEFAULT_NOTION_VERSION,
    log_requests: bool = False,
    log_responses: bool = False,
) -> NotionClient:
    token = os.environ.get(token_env, "").strip()
    version = os.environ.get(version_env, default_version).strip()
    cfg = NotionClientConfig(
        token=token,
        notion_version=version,
        log_requests=log_requests,
        log_responses=log_responses,
    )
    return NotionClient(cfg)
