"""Memo — 構造化された研究メモ・成果物."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Memo:
    """Evidence / Claims を引用して構成された調査成果物.

    Phase 1 では type は "evidence_based_memo" 固定。
    """

    memo_id: str
    type: str  # Phase 1: "evidence_based_memo"
    title: str
    body: str
    source_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)

    # -- serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "memo_id": self.memo_id,
            "type": self.type,
            "title": self.title,
            "body": self.body,
            "source_ids": list(self.source_ids),
            "evidence_ids": list(self.evidence_ids),
            "claim_ids": list(self.claim_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Memo:
        return cls(
            memo_id=d["memo_id"],
            type=d["type"],
            title=d["title"],
            body=d["body"],
            source_ids=d.get("source_ids", []),
            evidence_ids=d.get("evidence_ids", []),
            claim_ids=d.get("claim_ids", []),
        )
