# src/lit_review/__init__.py
"""Block 3: RQ × Paper Structured Re-reading.

Service logic for literature review pipeline (079–083).

Modules:
    rq_context   — RQContext dataclass, shared across all scripts
    extractor    — Query-focused Evidence extraction (081)
    dimensions   — Research dimension extraction (082/083)
    synthesizer  — Lit Review synthesis (082)
    landscape    — Research Landscape mapping (083)
    matcher      — RQ-Paper relevance scoring (079) [future]
    gap_filler   — External paper supplementation (080) [future]
    prompts      — LLM prompt templates [future]
"""

from src.lit_review.rq_context import RQContext

__all__ = ["RQContext"]
