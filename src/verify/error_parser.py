# src/verify/error_parser.py
"""
Error parser + classifier + next-action suggester.

Role
----
Verifier "brainstem":
- Take notebook execution errors (summary/trace) or quality gate output
- Classify into actionable categories
- Suggest a concise next action for Planner/Implementer
- Extract likely failing symbol/module and provide minimal hints

Principles
----------
- Be deterministic: regex-based, no LLM required.
- Keep suggestions short and operational.
- Never hallucinate fixes; only propose safe next steps.

Inputs supported
----------------
- NotebookRunResult style:
    error_summary (str)
    error_trace_short (str)
- QualityGateResult style:
    cmd_result.stderr/stdout, exit_code

Outputs
-------
- ErrorReport:
    category, severity, likely_root, hints, next_action, extracted_entities
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal, Tuple
import re


# -----------------------------
# Types
# -----------------------------

Category = Literal[
    "NO_ERROR",
    "SYNTAX_ERROR",
    "IMPORT_ERROR",
    "MODULE_NOT_FOUND",
    "NAME_ERROR",
    "ATTRIBUTE_ERROR",
    "TYPE_ERROR",
    "VALUE_ERROR",
    "KEY_ERROR",
    "INDEX_ERROR",
    "FILE_NOT_FOUND",
    "NOTION_API_ERROR",
    "SCHEMA_NOT_FIXED",
    "HTTP_ERROR",
    "TIMEOUT",
    "ASSERTION_FAILED",
    "TEST_FAILED",
    "LINT_FAILED",
    "RUNTIME_ERROR",
    "UNKNOWN",
]

Severity = Literal["INFO", "WARN", "ERROR", "FATAL"]


@dataclass
class ErrorReport:
    category: Category
    severity: Severity
    error_summary: str
    likely_root: str
    extracted: Dict[str, Any]
    hints: List[str]
    next_action: str


# -----------------------------
# Regex patterns
# -----------------------------

RE_SYNTAX = re.compile(r"\bSyntaxError\b|\bIndentationError\b|\bTabError\b")
RE_IMPORT = re.compile(r"\bImportError\b")
RE_MOD_NOT_FOUND = re.compile(r"\bModuleNotFoundError\b")
RE_NAME = re.compile(r"\bNameError\b")
RE_ATTR = re.compile(r"\bAttributeError\b")
RE_TYPE = re.compile(r"\bTypeError\b")
RE_VALUE = re.compile(r"\bValueError\b")
RE_KEY = re.compile(r"\bKeyError\b")
RE_INDEX = re.compile(r"\bIndexError\b")
RE_FILE_NOT_FOUND = re.compile(r"\bFileNotFoundError\b")
RE_TIMEOUT = re.compile(r"\bTimeout\b|\bTimeoutExpired\b|\btime out\b", re.IGNORECASE)

RE_ASSERT = re.compile(r"\bAssertionError\b")
RE_PYTEST_FAIL = re.compile(r"=+ FAILURES =+|FAILED\s+\w+::", re.IGNORECASE)

RE_NOTION = re.compile(
    r"\bNotion\b.*\bAPI\b|\bNotionAPIError\b|\bNOTION_TOKEN\b|\bNOTION_.*_DB_ID\b",
    re.IGNORECASE,
)
RE_HTTP_STATUS = re.compile(r"\bstatus\s*=\s*(\d{3})\b|\bHTTPError\b|\bHTTP\s*(\d{3})\b", re.IGNORECASE)

RE_MOD_QUOTED = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
RE_IMPORT_NAME = re.compile(r"cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]")
RE_NAME_NOT_DEFINED = re.compile(r"name ['\"]([^'\"]+)['\"] is not defined")
RE_ATTR_HAS_NO = re.compile(r"has no attribute ['\"]([^'\"]+)['\"]")
RE_KEYERR_QUOTED = re.compile(r"KeyError:\s*['\"]?([^'\"]+)['\"]?")
RE_FILENOTFOUND = re.compile(r"FileNotFoundError:\s*\[Errno\s*\d+\]\s*([^:]+):\s*['\"]([^'\"]+)['\"]")

# nbclient / notebook-ish patterns
RE_CELL_EXEC_ERROR = re.compile(r"\bCellExecutionError\b")
RE_NBCLIENT_EXEC_BANNER = re.compile(r"An error occurred while executing the following cell", re.IGNORECASE)
RE_CELL_IN = re.compile(r"Cell In\[(\d+)\]", re.MULTILINE)

# generic python exception class names e.g. ZeroDivisionError, RuntimeError, etc.
RE_EXCEPTION_CLASS = re.compile(r"\b([A-Za-z_]+Error)\b")

RE_FILELINE = re.compile(r'File ["\']([^"\']+)["\'], line (\d+)', re.MULTILINE)

# Exception "Class: message" line (prefer tail / final exception line in traceback)
RE_EXCEPTION_LINE = re.compile(r"^\s*([A-Za-z_]+Error)\s*:\s*(.+?)\s*$", re.MULTILINE)

# Notion env var mentions + common "env var not set" patterns
RE_NOTION_ENV = re.compile(r"\b(NOTION_[A-Z0-9_]+)\b")
RE_ENV_NOT_SET = re.compile(
    r"\b([A-Z0-9_]+)\b.*\benvironment variable\b.*\bnot set\b",
    re.IGNORECASE,
)

# notebook-ish cell number variants (nbclient output can vary)
RE_CELL_NUM = re.compile(r"\bcell\s*[#:\[]\s*(\d+)\s*[\]]?\b", re.IGNORECASE)
RE_EXECUTED_UP_TO = re.compile(r"\bexecuted_up_to\s*=\s*(\d+)\b")

RE_SCHEMA_MISSING = re.compile(
    r"\bschema_snapshot\.json\b.*\bnot found\b|\bSCHEMA_SNAPSHOT_MISSING\b|\bSTRUCTURE_JSON\b.*\bempty\b",
    re.IGNORECASE,
)

# -----------------------------
# Public API
# -----------------------------

def parse_error(
    *,
    error_summary: Optional[str] = None,
    error_trace: Optional[str] = None,
    gate_type: Optional[str] = None,
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
    exit_code: Optional[int] = None,
    schema_snapshot_path: Optional[str] = None,
    schema_report_path: Optional[str] = None,
    db_prop_types_path: Optional[str] = None,

) -> ErrorReport:
    """
    Main entry point.

    Provide either:
      - error_summary/error_trace (notebook execution)
      - gate_type + stdout/stderr/exit_code (quality gate)
    """
    summary = (error_summary or "").strip()
    trace = (error_trace or "").strip()

    # If parsing quality gate, synthesize a summary/trace
    if gate_type:
        s2, t2 = _summarize_gate(gate_type=gate_type, stdout=stdout or "", stderr=stderr or "", exit_code=exit_code)
        summary = summary or s2
        trace = trace or t2

    if not summary and not trace:
        return ErrorReport(
            category="NO_ERROR",
            severity="INFO",
            error_summary="",
            likely_root="",
            extracted={},
            hints=[],
            next_action="No error detected.",
        )

    text = "\n".join([summary, trace]).strip()
    # ---- schema artifact evidence (preferred over brittle regex) ----
    schema_artifacts: Dict[str, Any] = {}
    if schema_snapshot_path:
        schema_artifacts["schema_snapshot_path"] = str(schema_snapshot_path)
    if schema_report_path:
        schema_artifacts["schema_report_path"] = str(schema_report_path)
    if db_prop_types_path:
        schema_artifacts["db_prop_types_path"] = str(db_prop_types_path)
        
    category = _classify(text, gate_type=gate_type)
    extracted = _extract_entities(text)
    if schema_artifacts:
        extracted["schema_artifacts"] = schema_artifacts
    # If schema artifacts are expected but missing, prefer SCHEMA_NOT_FIXED
    # (Callers can pass schema_*_path="MISSING" or omit; this is a lightweight hook)
    if "schema_artifacts" in extracted:
        snap = (extracted["schema_artifacts"] or {}).get("schema_snapshot_path") or ""
        if isinstance(snap, str) and snap.strip().upper() == "MISSING":
            category = "SCHEMA_NOT_FIXED"

    severity = _severity_for(category)
    likely_root = _likely_root_for(category, extracted, gate_type=gate_type)
    hints = _hints_for(category, extracted, gate_type=gate_type)
    next_action = _next_action_for(category, extracted, gate_type=gate_type)

    # Keep next_action concise
    next_action = next_action.strip()
    if len(next_action) > 400:
        next_action = next_action[:400].rstrip() + "..."

    # Keep summary bounded
    safe_summary = summary[:500] if summary else (_first_nonempty_line(trace)[:500] if trace else "")

    return ErrorReport(
        category=category,
        severity=severity,
        error_summary=safe_summary,
        likely_root=likely_root,
        extracted=extracted,
        hints=hints,
        next_action=next_action,
    )

def build_llm_error_brief(report: ErrorReport, *, max_points: int = 5) -> List[str]:
    """
    Deterministic 5-point brief for Planner/Implementer.
    Keep it short, stable, and copy-pastable.
    """
    pts: List[str] = []

    pts.append(f"category={report.category} severity={report.severity}")
    if report.error_summary:
        pts.append(f"summary: {report.error_summary[:220]}")
    if report.extracted.get("failing_cell_index") is not None or report.extracted.get("executed_up_to") is not None:
        pts.append(
            f"cells: failing={report.extracted.get('failing_cell_index')} executed_up_to={report.extracted.get('executed_up_to')}"
        )
    if report.likely_root:
        pts.append(f"root: {report.likely_root[:220]}")
    tail = report.extracted.get("trace_tail")
    if isinstance(tail, str) and tail.strip():
        pts.append("trace_tail:\n" + tail)

    # pad with next_action if still short
    if len(pts) < max_points and report.next_action:
        pts.append(f"next: {report.next_action[:220]}")

    return pts[:max_points]

def suggest_next_action_from_notebook_result(nb_result: Any) -> ErrorReport:
    trace = (
        getattr(nb_result, "error_traceback", None)
        or getattr(nb_result, "traceback", None)
        or getattr(nb_result, "error_trace", None)
        or getattr(nb_result, "error_trace_short", "")
        or ""
    )
    return parse_error(
        error_summary=getattr(nb_result, "error_summary", ""),
        error_trace=trace,
        stderr=getattr(nb_result, "stderr", None),
        stdout=getattr(nb_result, "stdout", None),
    )


def suggest_next_action_from_quality_gate(qg_result: Any) -> ErrorReport:
    """
    Convenience for QualityGateResult.
    Expects:
      - gate_type
      - cmd_result.stdout
      - cmd_result.stderr
      - cmd_result.exit_code
    """
    cmd = getattr(qg_result, "cmd_result", None)
    return parse_error(
        gate_type=getattr(qg_result, "gate_type", "CMD"),
        stdout=getattr(cmd, "stdout", "") if cmd else "",
        stderr=getattr(cmd, "stderr", "") if cmd else "",
        exit_code=getattr(cmd, "exit_code", None) if cmd else None,
    )


# -----------------------------
# Internals: classify
# -----------------------------

def _classify(text: str, *, gate_type: Optional[str]) -> Category:
    t = text

    # Gate-specific first
    if gate_type:
        gt = gate_type.upper()
        if gt == "PYTEST":
            if RE_PYTEST_FAIL.search(t) or RE_ASSERT.search(t):
                return "TEST_FAILED"
        if gt == "RUFF":
            return "LINT_FAILED"

    # Notebook/runtime errors
    if RE_TIMEOUT.search(t):
        return "TIMEOUT"
    if RE_SYNTAX.search(t):
        return "SYNTAX_ERROR"
    # Schema precondition missing (Cell02 facts not fixed)
    if RE_SCHEMA_MISSING.search(t):
        return "SCHEMA_NOT_FIXED"

    # ✅ Notion & HTTP should be classified before generic ValueError/KeyError
    # (e.g., missing NOTION_* env vars often raise ValueError in notebooks)
    if RE_NOTION.search(t):
        return "NOTION_API_ERROR"
    if RE_HTTP_STATUS.search(t):
        return "HTTP_ERROR"

    if RE_MOD_NOT_FOUND.search(t):
        return "MODULE_NOT_FOUND"
    if RE_IMPORT.search(t):
        return "IMPORT_ERROR"
    if RE_NAME.search(t):
        return "NAME_ERROR"
    if RE_ATTR.search(t):
        return "ATTRIBUTE_ERROR"
    if RE_TYPE.search(t):
        return "TYPE_ERROR"
    if RE_VALUE.search(t):
        return "VALUE_ERROR"
    if RE_KEY.search(t):
        return "KEY_ERROR"
    if RE_INDEX.search(t):
        return "INDEX_ERROR"
    if RE_FILE_NOT_FOUND.search(t):
        return "FILE_NOT_FOUND"
    if RE_ASSERT.search(t):
        return "ASSERTION_FAILED"

    # nbclient signature -> runtime error (even if we don't know which)
    if RE_CELL_EXEC_ERROR.search(t) or RE_NBCLIENT_EXEC_BANNER.search(t):
        return "RUNTIME_ERROR"

    # Any "*Error" class present -> runtime error (avoid UNKNOWN spam)
    if RE_EXCEPTION_CLASS.search(t):
        return "RUNTIME_ERROR"

    return "UNKNOWN"


def _severity_for(category: Category) -> Severity:
    if category in ("NO_ERROR",):
        return "INFO"
    if category in ("LINT_FAILED", "TEST_FAILED", "ASSERTION_FAILED"):
        return "WARN"
    if category in ("SYNTAX_ERROR", "MODULE_NOT_FOUND", "IMPORT_ERROR", "NOTION_API_ERROR", "HTTP_ERROR", "TIMEOUT"):
        return "ERROR"
    if category in ("RUNTIME_ERROR", "UNKNOWN"):
        return "ERROR"
    return "ERROR"


# -----------------------------
# Internals: extraction
# -----------------------------

def _extract_entities(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    m = RE_MOD_QUOTED.search(text)
    if m:
        out["missing_module"] = m.group(1)

    m = RE_IMPORT_NAME.search(text)
    if m:
        out["import_name"] = m.group(1)
        out["import_from"] = m.group(2)

    m = RE_NAME_NOT_DEFINED.search(text)
    if m:
        out["undefined_name"] = m.group(1)

    m = RE_ATTR_HAS_NO.search(text)
    if m:
        out["missing_attribute"] = m.group(1)

    m = RE_KEYERR_QUOTED.search(text)
    if m:
        out["missing_key"] = m.group(1)

    m = RE_FILENOTFOUND.search(text)
    if m:
        out["file_error_kind"] = m.group(1).strip()
        out["missing_file"] = m.group(2)

    m = RE_HTTP_STATUS.search(text)
    if m:
        code = next((g for g in m.groups() if g), None)
        if code:
            try:
                out["http_status"] = int(code)
            except Exception:
                pass

    # notebook cell number (often appears in nbclient output)
    m = RE_CELL_IN.search(text)
    if m:
        try:
            n = int(m.group(1))
            out["cell_in_number"] = n          # keep backward-compat
            out["cell_index"] = n              # ✅ preferred canonical key for orchestrator
            out["failing_cell_index"] = n      # ✅ extra alias (helps callers)
        except Exception:
            pass

    # executed_up_to appears in orchestrator/runner summaries sometimes
    m = RE_EXECUTED_UP_TO.search(text)
    if m:
        try:
            out["executed_up_to"] = int(m.group(1))
        except Exception:
            pass

    # broader cell number variants (only if not already set)
    if out.get("cell_index") is None:
        m = RE_CELL_NUM.search(text)
        if m:
            try:
                n = int(m.group(1))
                out["cell_index"] = n
                out["failing_cell_index"] = n
            except Exception:
                pass

    # exception class heuristic
    m = RE_EXCEPTION_CLASS.search(text)
    if m:
        out["exception_class"] = m.group(1)

    # Prefer explicit "XError: message" line; take the LAST occurrence if multiple.
    exc_lines = RE_EXCEPTION_LINE.findall(text)
    if exc_lines:
        cls, msg = exc_lines[-1]
        out["exception_class"] = out.get("exception_class") or cls
        out["exception_message"] = msg.strip()

    # Heuristic: find last "File "...", line N"
    file_line = _extract_last_file_line(text)
    if file_line:
        out.update(file_line)

    # Notion env mentions
    envs = sorted(set(RE_NOTION_ENV.findall(text)))
    if envs:
        out["notion_env_mentions"] = envs

    # Missing env var (common pattern in notebook guards)
    m = RE_ENV_NOT_SET.search(text)
    if m:
        out["missing_env_var"] = m.group(1)

    # Keep a small tail of trace text (helps LLM implement minimal fixes)
    lines = (text or "").splitlines()
    if lines:
        out["trace_tail"] = "\n".join(lines[-20:])

    return out


def _extract_last_file_line(text: str) -> Optional[Dict[str, Any]]:
    matches = list(RE_FILELINE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    try:
        return {"trace_file": m.group(1), "trace_line": int(m.group(2))}
    except Exception:
        return {"trace_file": m.group(1)}


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s
    return ""


# -----------------------------
# Internals: roots/hints/actions
# -----------------------------

def _likely_root_for(category: Category, extracted: Dict[str, Any], *, gate_type: Optional[str]) -> str:
    if category == "MODULE_NOT_FOUND":
        return f"Missing dependency/module: {extracted.get('missing_module', '')}".strip()
    if category == "IMPORT_ERROR":
        if extracted.get("import_name") and extracted.get("import_from"):
            return f"Import mismatch: {extracted['import_name']} from {extracted['import_from']}"
        return "ImportError in code"
    if category == "NAME_ERROR":
        return f"Undefined name: {extracted.get('undefined_name', '')}".strip()
    if category == "ATTRIBUTE_ERROR":
        return f"Missing attribute: {extracted.get('missing_attribute', '')}".strip()
    if category == "KEY_ERROR":
        return f"Missing dict key: {extracted.get('missing_key', '')}".strip()
    if category == "FILE_NOT_FOUND":
        mf = extracted.get("missing_file", "")
        return f"Missing file/path: {mf}".strip()
    if category == "NOTION_API_ERROR":
        return "Notion API failure (auth/schema/data_source_id/query/filter)"
    if category == "HTTP_ERROR":
        return f"HTTP error (status={extracted.get('http_status')})".strip()
    if category == "SYNTAX_ERROR":
        return "Syntax/indentation error in patched cell"
    if category == "TIMEOUT":
        return "Execution timeout"
    if category in ("TEST_FAILED", "LINT_FAILED"):
        return f"{category} in quality gate"
    if extracted.get("trace_file"):
        return f"Trace points to {extracted['trace_file']}:{extracted.get('trace_line')}"
    if extracted.get("exception_class"):
        return f"Runtime exception: {extracted['exception_class']}"
    if extracted.get("cell_index") is not None:
        return f"Notebook runtime failure near Cell In[{extracted['cell_index']}]"
    if extracted.get("cell_in_number") is not None:  # fallback
        return f"Notebook runtime failure near Cell In[{extracted['cell_in_number']}]"
    return "Unknown root cause"


def _hints_for(category: Category, extracted: Dict[str, Any], *, gate_type: Optional[str]) -> List[str]:
    hints: List[str] = []

    if extracted.get("trace_file"):
        hints.append(f"Check: {extracted['trace_file']}:{extracted.get('trace_line')}")

    if extracted.get("cell_in_number") is not None:
        hints.append(f"Notebook shows error near: Cell In[{extracted['cell_in_number']}] (nbclient message)")

    if extracted.get("exception_class"):
        hints.append(f"Exception class: {extracted['exception_class']}")

    if extracted.get("exception_message"):
        hints.append(f"Exception message: {extracted['exception_message']}")

    if category == "MODULE_NOT_FOUND":
        mod = extracted.get("missing_module")
        if mod:
            hints.append(f"Ensure dependency installed/importable: {mod}")
            hints.append("If local module: verify src/ is on PYTHONPATH or package is installed (pip -e .).")

    if category == "IMPORT_ERROR":
        hints.append("Verify import path/name; check refactor moved symbols or changed __init__.py exports.")

    if category == "NAME_ERROR":
        nm = extracted.get("undefined_name")
        if nm:
            hints.append(f"Define '{nm}' earlier in the notebook or import it; check variable naming.")
        hints.append("If cell-wise loop: confirm earlier cells executed in prefix run.")

    if category == "ATTRIBUTE_ERROR":
        hints.append("Check object type; attribute may have been renamed or returned None.")
        hints.append("If using dict-like config: ensure you aren't treating dict as object attributes.")

    if category == "KEY_ERROR":
        k = extracted.get("missing_key")
        if k:
            hints.append(f"Key '{k}' missing: inspect keys() and upstream schema/data contract.")
        hints.append("For Notion: confirm property names and select option names match schema.py.")

    if category == "VALUE_ERROR":
        hints.append("ValueError often indicates missing config/env, invalid parameter, or violated precondition.")

    if category == "NOTION_API_ERROR":
        miss = extracted.get("missing_env_var")
        if miss:
            hints.append(f"Missing env var: {miss}")
        envs = extracted.get("notion_env_mentions")
        if envs:
            hints.append(f"Notion env mentions: {', '.join(envs[:6])}" + ("" if len(envs) <= 6 else f" (+{len(envs)-6})"))
        hints.append("Confirm NOTION_TOKEN and database IDs; verify data_source_id resolution & caching.")
        hints.append("If query fails: verify RESOLVED_DB mapping and filter schema.")
        hints.append("If database.properties empty: infer only 'type' from sample page; don't trust values.")

    if category == "TIMEOUT":
        hints.append("Increase timeout_sec or reduce scope (execute prefix fewer cells).")
        hints.append("If stuck cell: isolate heavy loops / network calls / large data ops.")

    if category == "TEST_FAILED":
        hints.append("Open the pytest failure section; fix assertion or update expected output.")
    if category == "LINT_FAILED":
        hints.append("Run `ruff check .` locally; apply `ruff check --fix .` if allowed.")
    if category == "SCHEMA_NOT_FIXED":
        hints.append("Cell02 must run to fix facts (schema snapshot/report).")
        hints.append("Run PREFIX with min_prefix_cell_index=2 (or run up to Cell02) and ensure schema files are written.")
        hints.append("If schema extraction depends on Notion: verify RESOLVED_DB mapping and property types output.")

    
    return hints


def _next_action_for(category: Category, extracted: Dict[str, Any], *, gate_type: Optional[str]) -> str:
    """
    One-liner operational next action intended for Proposal/Run DB fields.
    """
    if category == "SCHEMA_NOT_FIXED":
        return "Run PREFIX up to Cell02 to generate schema_snapshot/report (facts fixed), then re-run the failing step."

    if category == "NO_ERROR":
        return "Proceed."

    if category == "SYNTAX_ERROR":
        return "Revert the last patch and re-apply with corrected syntax/indentation; rerun prefix to failing cell."

    if category == "MODULE_NOT_FOUND":
        mod = extracted.get("missing_module")
        if mod:
            return f"Add/install dependency '{mod}' (or fix import path / CWD+sys.path), then rerun."
        return "Add/install missing dependency (or fix import path / CWD+sys.path), then rerun."

    if category == "IMPORT_ERROR":
        if extracted.get("import_name") and extracted.get("import_from"):
            return (
                f"Fix import: ensure '{extracted['import_name']}' is exported from '{extracted['import_from']}' "
                "(and CWD+sys.path OK), then rerun."
            )
        return "Fix failing import (module path / symbol name / CWD+sys.path), then rerun."

    if category == "NAME_ERROR":
        nm = extracted.get("undefined_name")
        if nm:
            return f"Define/import '{nm}' earlier (or correct variable name), then rerun prefix."
        return "Define/import the missing name earlier (or correct variable name), then rerun prefix."

    if category == "ATTRIBUTE_ERROR":
        ma = extracted.get("missing_attribute")
        if ma:
            return f"Check object type and rename/use correct attribute '{ma}' (or guard None), then rerun."
        return "Check object type and correct attribute access (or guard None), then rerun."

    if category == "KEY_ERROR":
        k = extracted.get("missing_key")
        if k:
            return f"Handle missing key '{k}' (fix upstream schema/data or add default), then rerun."
        return "Handle missing key (fix upstream schema/data or add default), then rerun."

    if category == "VALUE_ERROR":
        msg = extracted.get("exception_message")
        if msg:
            return (
                f"ValueError: {msg} — inspect failing line; apply minimal fix (prefer optional fallback); rerun prefix."
            )
        return "Inspect traceback for ValueError; apply minimal fix (prefer optional fallback); rerun prefix."

    if category == "FILE_NOT_FOUND":
        mf = extracted.get("missing_file")
        if mf:
            return f"Fix path or ensure file exists: {mf}, then rerun."
        return "Fix file path or ensure required file exists, then rerun."

    if category == "NOTION_API_ERROR":
        miss = extracted.get("missing_env_var")
        if miss:
            return f"Set {miss} (or make it optional with fallback) and rerun the failing cell."
        envs = extracted.get("notion_env_mentions")
        if envs:
            shown = ", ".join(envs[:4])
            more = "" if len(envs) <= 4 else f" (+{len(envs)-4} more)"
            return f"Verify Notion env vars {shown}{more} and schema mapping; rerun the failing query step."
        return "Verify Notion env vars + schema mapping; rerun the failing query step."

    if category == "HTTP_ERROR":
        code = extracted.get("http_status")
        if code:
            return f"Handle HTTP {code}: check endpoint/auth/rate-limit; retry with backoff; rerun."
        return "Handle HTTP error: check endpoint/auth/rate-limit; rerun."

    if category == "TIMEOUT":
        return "Increase timeout or split work: rerun prefix with fewer cells; isolate heavy cell."

    if category == "TEST_FAILED":
        return "Fix failing test (or update expected output) then rerun pytest gate."

    if category == "LINT_FAILED":
        return "Fix lint violations (optionally run ruff --fix) then rerun ruff gate."

    if category == "ASSERTION_FAILED":
        return "Investigate failing assertion and update logic/expected outputs; rerun."

    if category == "RUNTIME_ERROR":
        ec = extracted.get("exception_class")
        if ec:
            return f"Inspect traceback for {ec}; apply minimal fix; rerun prefix."
        return "Inspect traceback; apply minimal fix; rerun prefix."
    if category == "SYNTAX_ERROR":
        return "[IMPLEMENTER] Revert the last patch and re-apply with corrected syntax/indentation; rerun prefix to failing cell."
    
    return "Inspect traceback location and apply minimal fix; rerun prefix."


def _summarize_gate(*, gate_type: str, stdout: str, stderr: str, exit_code: Optional[int]) -> Tuple[str, str]:
    gt = gate_type.upper()
    code = "" if exit_code is None else f" exit={exit_code}"
    head = (stderr.strip().splitlines()[:1] or stdout.strip().splitlines()[:1] or [""])[0]
    summary = f"{gt}{code}: {head}".strip()
    trace = "\n".join((stderr or stdout).splitlines()[:80])
    return summary, trace
