"""ResearchPlan — Deep Research pipeline の共通入力構造."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ResearchPlan:
    """Planner (067) が生成する構造化された調査計画.

    intent / targets は LLM が request を解析して構造化したフィールド。
    ResearchRun の research_target / research_purpose (自然言語の記録) とは
    別概念。
    """

    run_id: str
    request: str
    intent: str  # company_research / interview_prep / tech_review / etc.
    created_at: datetime = field(default_factory=datetime.now)
    targets: list[str] = field(default_factory=list)
    key_questions: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    recalled_evidence_ids: list[str] = field(default_factory=list)
    recalled_claim_ids: list[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)

    # -- serialization ------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "request": self.request,
            "intent": self.intent,
            "created_at": self.created_at.isoformat(),
            "targets": list(self.targets),
            "key_questions": list(self.key_questions),
            "search_queries": list(self.search_queries),
            "deliverables": list(self.deliverables),
            "recalled_evidence_ids": list(self.recalled_evidence_ids),
            "recalled_claim_ids": list(self.recalled_claim_ids),
            "constraints": dict(self.constraints),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResearchPlan:
        created_at = d.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            run_id=d["run_id"],
            request=d["request"],
            intent=d["intent"],
            created_at=created_at,
            targets=d.get("targets", []),
            key_questions=d.get("key_questions", []),
            search_queries=d.get("search_queries", []),
            deliverables=d.get("deliverables", []),
            recalled_evidence_ids=d.get("recalled_evidence_ids", []),
            recalled_claim_ids=d.get("recalled_claim_ids", []),
            constraints=d.get("constraints", {}),
        )
