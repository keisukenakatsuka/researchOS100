"""Evidence — 観測された事実・データ・引用."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    """Source から抽出された検証可能な事実・データ.

    confidence / confidence_reason は 069 Extractor では null のまま生成し、
    070 Credibility で付与する。

    confidence_reason は構造化された信号リスト (list[dict]) として保持する。
    各 dict は {"signal": str, "value": str} の形式。
    """

    evidence_id: str
    statement: str
    source_id: str
    source_title: str
    confidence: Optional[str]  # high / medium / low or None
    confidence_reason: Optional[List[Dict[str, str]]]  # list of {signal, value}
    extracted_at: datetime
    tags: list[str] = field(default_factory=list)

    # -- serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "statement": self.statement,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "extracted_at": self.extracted_at.isoformat(),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Evidence:
        return cls(
            evidence_id=d["evidence_id"],
            statement=d["statement"],
            source_id=d["source_id"],
            source_title=d.get("source_title", ""),
            confidence=d.get("confidence"),
            confidence_reason=d.get("confidence_reason"),
            extracted_at=datetime.fromisoformat(d["extracted_at"]),
            tags=d.get("tags", []),
        )
