"""Claim — Evidence に基づく解釈・主張."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Claim:
    """複数の Evidence を統合して導出された解釈・主張.

    evidence_ids / source_ids で根拠となるエンティティへの
    参照チェーンを保持する。
    """

    claim_id: str
    statement: str
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    confidence: Optional[str] = None
    confidence_reason: Optional[str] = None
    confidence_meta: Optional[Dict[str, Any]] = None
    tags: list[str] = field(default_factory=list)
    key_question_refs: list[int] = field(default_factory=list)
    created_at: Optional[datetime] = None

    # -- serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "evidence_ids": list(self.evidence_ids),
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "confidence_meta": self.confidence_meta,
            "tags": list(self.tags),
            "key_question_refs": list(self.key_question_refs),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        created_at = d.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            claim_id=d["claim_id"],
            statement=d["statement"],
            evidence_ids=d.get("evidence_ids", []),
            source_ids=d.get("source_ids", []),
            confidence=d.get("confidence"),
            confidence_reason=d.get("confidence_reason"),
            confidence_meta=d.get("confidence_meta"),
            tags=d.get("tags", []),
            key_question_refs=d.get("key_question_refs", []),
            created_at=created_at,
        )
