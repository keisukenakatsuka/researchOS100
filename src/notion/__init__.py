# src/notion/__init__.py
"""Notion integration: client, schema, repos, and helpers.

Re-exports the most commonly used symbols so notebooks can write::

    from src.notion import (
        NotionClient, NotionClientConfig, NotionAPIError,
        NotionDataSourceResolver, ResolvedDB,
        build_notion_client_from_env, normalize_uuid,
    )
"""

from src.notion.client import (
    NotionClient,
    NotionClientConfig,
    NotionAPIError,
    NotionDataSourceResolver,
    ResolvedDB,
    build_notion_client_from_env,
    normalize_uuid,
)
from src.notion.properties import extract_property_value, page_to_record

__all__ = [
    "NotionClient",
    "NotionClientConfig",
    "NotionAPIError",
    "NotionDataSourceResolver",
    "ResolvedDB",
    "build_notion_client_from_env",
    "normalize_uuid",
    "extract_property_value",
    "page_to_record",
]
