"""Source — 情報源メタデータ."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Source:
    """Web 記事・論文・レポートなど情報源のレコード.

    URL で一意に特定される。同一 URL の重複登録を防ぐためのキーとして使用。
    """

    source_id: str
    url: str
    title: str
    type: str  # web_article / paper / report / news
    content: str
    retrieved_at: datetime

    # -- serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "url": self.url,
            "title": self.title,
            "type": self.type,
            "content": self.content,
            "retrieved_at": self.retrieved_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Source:
        return cls(
            source_id=d["source_id"],
            url=d["url"],
            title=d["title"],
            type=d["type"],
            content=d["content"],
            retrieved_at=datetime.fromisoformat(d["retrieved_at"]),
        )
