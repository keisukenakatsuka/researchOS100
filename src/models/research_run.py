"""ResearchRun — 調査実行ログレコード."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ResearchRun:
    """Deep Research pipeline の実行結果を記録するログエンティティ.

    research_target / research_purpose はユーザーの元の request から
    抽出した自然言語の記録であり、ResearchPlan の intent / targets
    (LLM が構造化したフィールド) とは別概念。
    """

    run_id: str
    request: str
    research_target: str
    research_purpose: str
    status: str  # completed / failed / partial
    created_at: datetime
    source_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    memo_ids: list[str] = field(default_factory=list)
    error: Optional[str] = None

    # -- serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "request": self.request,
            "research_target": self.research_target,
            "research_purpose": self.research_purpose,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "source_ids": list(self.source_ids),
            "evidence_ids": list(self.evidence_ids),
            "claim_ids": list(self.claim_ids),
            "memo_ids": list(self.memo_ids),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResearchRun:
        return cls(
            run_id=d["run_id"],
            request=d["request"],
            research_target=d["research_target"],
            research_purpose=d["research_purpose"],
            status=d["status"],
            created_at=datetime.fromisoformat(d["created_at"]),
            source_ids=d.get("source_ids", []),
            evidence_ids=d.get("evidence_ids", []),
            claim_ids=d.get("claim_ids", []),
            memo_ids=d.get("memo_ids", []),
            error=d.get("error"),
        )
