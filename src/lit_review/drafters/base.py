# src/lit_review/drafters/base.py
"""BaseDrafter — shared base for Block 6 section drafters (095–098).

Each drafter subclass overrides `build_prompt()` to supply the
system/user prompt pair.  Everything else — outline loading, cross-
reference assembly, LLM call, validation, file save, and metadata —
is handled here.

Usage::

    class IntroductionDrafter(BaseDrafter):
        section_id = "introduction"
        output_file = "draft_introduction.md"
        default_max_tokens = 8192

        def build_prompt(self, ctx):
            return ctx.system("..."), ctx.user("...")

    drafter = IntroductionDrafter()
    result = drafter.generate(run_dir, llm_client=client)
"""

from __future__ import annotations

import enum
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Preview length for cross-referencing other drafts (chars).
# Override per-drafter via BaseDrafter.cross_ref_chars if needed.
CROSS_REF_PREVIEW_CHARS = 300

# Draft files that can be cross-referenced
_DRAFT_FILES = [
    "draft_introduction.md",
    "draft_literature_review.md",
    "draft_hypotheses.md",
    "draft_methods.md",
]


# ------------------------------------------------------------------
# Status enum
# ------------------------------------------------------------------

class DraftStatus(str, enum.Enum):
    """Fixed set of statuses for DraftResult."""
    GENERATED = "generated"
    FAILED = "failed"
    SKIPPED = "skipped"

# All upstream input files that drafters may need
INPUT_FILES = [
    "rq_context.json",
    "research_plan.md",
    "lit_review.json",
    "landscape.json",
    "hypotheses.json",
    "assumptions.json",
    "hypothesis_portfolio.json",
    "focused_hypotheses.json",
    "validation_designs.json",
    "data_requirements.json",
    "method_selection.json",
]


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------

@dataclass
class DraftDiagnostics:
    """Quality diagnostics for a generated draft."""
    target_words: int = 0
    actual_words: int = 0
    word_ratio: float = 0.0        # actual / target
    meets_target: bool = False     # within ±20% of target
    has_citations: bool = False    # contains (Author, Year) references
    warnings: List[str] = field(default_factory=list)


@dataclass
class DraftResult:
    """Output of a single drafter run."""
    section_id: str
    status: DraftStatus = DraftStatus.FAILED
    output_path: str = ""          # relative filename
    word_count: int = 0
    diagnostics: DraftDiagnostics = field(default_factory=DraftDiagnostics)
    error: Optional[str] = None
    retryable: bool = True         # False for permanent failures (missing required input)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Enum -> str for JSON serialization
        d["status"] = self.status.value
        return d


# ------------------------------------------------------------------
# Prompt context helper
# ------------------------------------------------------------------

@dataclass
class PromptContext:
    """Pre-loaded data that build_prompt() receives.

    Provides convenient access to outline spec, raw inputs, and
    cross-references without each drafter re-implementing I/O.
    """
    run_dir: Path
    outline: Dict[str, Any]                # full paper_outline.json
    outline_spec: Dict[str, Any]           # this section's spec from outline
    inputs: Dict[str, Any]                 # raw JSON inputs keyed by filename
    cross_refs: str                        # compact cross-reference text
    available: Dict[str, bool]             # filename -> exists?
    deep_lit: Dict[str, Any] = field(default_factory=dict)  # hypothesis_id → deep_lit outputs

    def input(self, filename: str) -> Dict[str, Any]:
        """Get a raw input by filename, empty dict if missing."""
        return self.inputs.get(filename, {})

    def require(self, filename: str) -> Dict[str, Any]:
        """Get a required input; raises if missing."""
        if filename not in self.inputs:
            raise MissingInputError(filename)
        return self.inputs[filename]


class MissingInputError(Exception):
    """Raised when a required input file is not available."""
    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(f"Required input not found: {filename}")


class EmptyInputError(Exception):
    """Raised when a required input file exists but is empty or invalid."""
    def __init__(self, filename: str, reason: str = "empty"):
        self.filename = filename
        super().__init__(f"Required input {reason}: {filename}")


# ------------------------------------------------------------------
# BaseDrafter
# ------------------------------------------------------------------

class BaseDrafter:
    """Base class for section drafters.

    Subclasses MUST set:
        section_id:        str   — e.g., "introduction"
        output_file:       str   — e.g., "draft_introduction.md"
        default_max_tokens: int  — max output tokens for LLM

    Subclasses MUST override:
        build_prompt(ctx: PromptContext) -> Tuple[str, str]
            Returns (system_prompt, user_prompt).

    Subclasses MAY override:
        required_inputs() -> List[str]
            Input filenames that must exist. Default: ["paper_outline.json"]
        validate_content(text, outline_spec) -> DraftDiagnostics
            Additional content validation beyond word count.
    """

    section_id: str = ""
    output_file: str = ""
    default_max_tokens: int = 8192
    model: str = _DEFAULT_MODEL
    cross_ref_chars: int = CROSS_REF_PREVIEW_CHARS

    # ---- public API ------------------------------------------------

    def generate(
        self,
        run_dir: Path,
        *,
        llm_client: Any,
        max_tokens: Optional[int] = None,
    ) -> DraftResult:
        """Generate a draft section. Single entry point for scripts.

        Returns DraftResult with status, output path, diagnostics.
        Handles all errors internally — never raises to caller.
        Diagnostics are always populated, even on failure.
        """
        result = DraftResult(
            section_id=self.section_id,
            output_path=self.output_file,
        )

        try:
            # 1. Load context
            ctx = self._load_context(run_dir)

            # 2. Check required inputs
            self._check_required_inputs(ctx)

            # 3. Build prompt
            system_prompt, user_prompt = self.build_prompt(ctx)

            # 4. Call LLM
            tokens = max_tokens or self.default_max_tokens
            text = self._llm_call(llm_client, system_prompt, user_prompt, tokens, result)

            if not text.strip():
                result.status = DraftStatus.FAILED
                result.error = "LLM returned empty response"
                result.retryable = True
                return result

            # 5. Save output
            out_path = run_dir / self.output_file
            out_path.write_text(text)
            word_count = self._count_words(text)

            # 6. Validate
            diagnostics = self._validate(text, ctx.outline_spec)
            result.status = DraftStatus.GENERATED
            result.word_count = word_count
            result.diagnostics = diagnostics

            logger.info(
                "%s: generated %s (%d words, ratio=%.2f)",
                self.section_id, self.output_file, word_count,
                diagnostics.word_ratio,
            )

        except (MissingInputError, EmptyInputError) as e:
            result.status = DraftStatus.FAILED
            result.error = str(e)
            result.retryable = False  # permanent — upstream must run first
            result.diagnostics.warnings.append(str(e))
            logger.error("%s: %s", self.section_id, e)

        except Exception as e:
            result.status = DraftStatus.FAILED
            result.error = f"{type(e).__name__}: {e}"
            result.retryable = True
            result.diagnostics.warnings.append(f"Unexpected: {e}")
            logger.error("%s: unexpected error: %s", self.section_id, e)

        return result

    # ---- subclass interface ----------------------------------------

    def build_prompt(self, ctx: PromptContext) -> Tuple[str, str]:
        """Return (system_prompt, user_prompt). Subclass must override."""
        raise NotImplementedError

    def required_inputs(self) -> List[str]:
        """Input filenames that must exist. Override to add more."""
        return ["paper_outline.json"]

    def validate_content(self, text: str, outline_spec: Dict[str, Any]) -> List[str]:
        """Return extra warning strings for section-specific checks.

        Override for custom validation.  Base class handles:
        - word count vs target (warning, never fail)
        - citation detection (warning, never fail)

        Subclass checks should also produce **warnings only**.
        Validation never causes a status=failed — it only populates
        diagnostics.warnings for reporting.
        """
        return []

    # ---- internal --------------------------------------------------

    def _load_context(self, run_dir: Path) -> PromptContext:
        """Load outline, upstream inputs, and cross-references."""
        # Outline
        outline_path = run_dir / "paper_outline.json"
        outline: Dict[str, Any] = {}
        if outline_path.exists():
            outline = json.loads(outline_path.read_text())

        # Section spec from outline
        outline_spec = self._extract_section_spec(outline)

        # Raw inputs
        inputs: Dict[str, Any] = {}
        available: Dict[str, bool] = {}
        for fname in INPUT_FILES:
            p = run_dir / fname
            if p.exists():
                if fname.endswith(".json"):
                    inputs[fname] = json.loads(p.read_text())
                else:
                    inputs[fname] = p.read_text()
                available[fname] = True
            else:
                available[fname] = False

        # Cross-references from other drafts
        cross_refs = self._load_cross_references(run_dir)

        # Deep literature outputs (per hypothesis, optional)
        deep_lit: Dict[str, Any] = {}
        try:
            from src.lit_review.deep_lit import load_deep_lit_outputs
            from src.lit_review.focus import load_focused, is_focused
            focused = load_focused(run_dir)
            if is_focused(focused):
                for role in ["primary", "secondary"]:
                    hyp = focused.get(role)
                    if hyp and (role == "primary" or focused.get("has_secondary")):
                        hyp_id = hyp.get("hypothesis_id", "")
                        if hyp_id:
                            dl = load_deep_lit_outputs(run_dir, hyp_id)
                            if dl:
                                deep_lit[hyp_id] = dl
        except Exception as e:
            logger.debug("Deep lit loading skipped: %s", e)

        return PromptContext(
            run_dir=run_dir,
            outline=outline,
            outline_spec=outline_spec,
            inputs=inputs,
            cross_refs=cross_refs,
            available=available,
            deep_lit=deep_lit,
        )

    def _extract_section_spec(self, outline: Dict[str, Any]) -> Dict[str, Any]:
        """Extract this section's spec from paper_outline.json."""
        for section in outline.get("sections", []):
            if section.get("section_id") == self.section_id:
                return section
        return {}

    def _check_required_inputs(self, ctx: PromptContext) -> None:
        """Verify all required inputs exist, are non-empty, and parse correctly.

        Raises MissingInputError (file absent) or EmptyInputError
        (file exists but empty/corrupt).  Both are permanent failures.
        """
        for fname in self.required_inputs():
            if fname == "paper_outline.json":
                if not ctx.outline:
                    raise MissingInputError("paper_outline.json")
                continue

            if not ctx.available.get(fname, False):
                raise MissingInputError(fname)

            data = ctx.inputs.get(fname)
            if fname.endswith(".json"):
                if not isinstance(data, dict) or not data:
                    raise EmptyInputError(fname, "empty or invalid JSON")
            elif isinstance(data, str) and not data.strip():
                raise EmptyInputError(fname, "empty file")

    def _load_cross_references(self, run_dir: Path, max_chars: Optional[int] = None) -> str:
        """Load first N chars of other drafts for narrative coherence."""
        max_chars = max_chars or self.cross_ref_chars
        parts: List[str] = []
        for fname in _DRAFT_FILES:
            if fname == self.output_file:
                continue  # skip own output
            p = run_dir / fname
            if p.exists():
                text = p.read_text()
                preview = text[:max_chars].rstrip()
                if preview:
                    section_name = fname.replace("draft_", "").replace(".md", "")
                    parts.append(f"[{section_name}]\n{preview}")
        if not parts:
            return ""
        return "## Other sections (preview)\n" + "\n\n".join(parts)

    def _llm_call(
        self,
        llm_client: Any,
        system: str,
        user: str,
        max_tokens: int,
        result: DraftResult,
    ) -> str:
        """Call LLM, record metadata, return text."""
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        resp = llm_client.messages_create(body=body)

        # Extract text
        text = ""
        for block in resp.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "")
                break

        # Record metadata
        usage = resp.get("usage", {})
        result.metadata = {
            "model": self.model,
            "max_tokens": max_tokens,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

        logger.info(
            "LLM call: in=%d out=%d tokens",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )

        return text

    @staticmethod
    def _count_words(text: str) -> int:
        """Count words, handling Japanese text.

        For Japanese text (which lacks word-separating spaces), uses
        character count / 1.5 as a rough word equivalent.  For English
        or mixed text, falls back to split().
        """
        # Detect Japanese: if >30% of non-whitespace chars are CJK
        stripped = re.sub(r"\s+", "", text)
        if not stripped:
            return 0
        cjk_count = sum(1 for c in stripped if "\u3000" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff")
        cjk_ratio = cjk_count / len(stripped)

        if cjk_ratio > 0.3:
            # Japanese: ~1.5 chars per "word" equivalent
            return max(1, len(stripped) * 2 // 3)
        return len(text.split())

    def _validate(self, text: str, outline_spec: Dict[str, Any]) -> DraftDiagnostics:
        """Run word count + citation + custom checks."""
        actual_words = self._count_words(text)
        target_words = outline_spec.get("target_words", 0)

        ratio = actual_words / target_words if target_words > 0 else 0.0
        meets_target = 0.8 <= ratio <= 1.2 if target_words > 0 else True

        # Citation check: detect academic references in common formats:
        #   (Author, Year)  (Author et al., Year)  (Author & Author, Year)
        #   （Author, Year）  Author (Year)  Author et al. (Year)
        has_citations = bool(
            # (Author, Year) or （Author, Year）
            re.search(
                r"[(\uff08][A-Z][a-zé]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-zé]+))?[.,]?\s*\d{4}[)\uff09]",
                text,
            )
            # Author (Year) or Author et al. (Year)
            or re.search(
                r"[A-Z][a-zé]+(?:\s+et\s+al\.?)?\s*[(\uff08]\d{4}[)\uff09]",
                text,
            )
        )

        warnings: List[str] = []
        if target_words > 0 and not meets_target:
            warnings.append(
                f"Word count {actual_words} is {'below' if ratio < 0.8 else 'above'} "
                f"target {target_words} (ratio={ratio:.2f})"
            )
        if not has_citations:
            warnings.append("No academic citations detected (Author, Year)")

        # Custom checks from subclass
        warnings.extend(self.validate_content(text, outline_spec))

        return DraftDiagnostics(
            target_words=target_words,
            actual_words=actual_words,
            word_ratio=round(ratio, 3),
            meets_target=meets_target,
            has_citations=has_citations,
            warnings=warnings,
        )
