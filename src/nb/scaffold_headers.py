# src/nb/scaffold_headers.py
from __future__ import annotations

"""
Deterministic header scaffolding for notebook cells.

Goal
----
Generate:
  - Cell 00: notebook-level header (comment-only; English)
  - Cell 01: deterministic setup/config code (project-root CWD + env + Notion repos + LLM client)
  - Cell XX: per-cell header blocks (English, consistent format)

Design
------
- Deterministic (no LLM).
- Idempotent:
    * Each generated header includes a signature line + range markers.
    * Utilities help detect existing headers and avoid duplicates.
- Cell00 is immutable after first write:
    * Locked by CELL0_LOCK markers.
    * Stores structure JSON between STRUCTURE_JSON markers.
- Cell01 is deterministic bootstrap:
    * Marked by AUTOSETUP_CELL01 marker.
"""

from hashlib import sha256
from typing import Any, Dict, List, Sequence
import re

# ---------------------------------------------------------------------
# Structure normalization helper
# ---------------------------------------------------------------------
def _coerce_structure_list(structure: Any) -> List[Dict[str, Any]]:
    """
    Accept a few common shapes and coerce into list[dict].
    - list[dict]                             (expected)
    - {"structure": [...]}                   (LLM seed schema output)
    - {"items": [...]}                       (context_pack notebook_structure payload)
    Anything else -> []
    """
    if isinstance(structure, list):
        return structure
    if isinstance(structure, dict):
        if isinstance(structure.get("structure"), list):
            return structure["structure"]
        if isinstance(structure.get("items"), list):
            return structure["items"]
    return []

# ---------------------------------------------------------------------
# Signatures / markers (used for idempotency)
# ---------------------------------------------------------------------

AUTOHEADER_VERSION = "v1"

AUTOHEADER_BEGIN = "# === AUTOHEADER:BEGIN ==="
AUTOHEADER_END = "# === AUTOHEADER:END ==="

CELLHEADER_BEGIN = "# === AUTOHEADER_CELL:BEGIN ==="
CELLHEADER_END = "# === AUTOHEADER_CELL:END ==="

AUTOHEADER_MARK = f"# === AUTOHEADER:{AUTOHEADER_VERSION} ==="
CELL_HEADER_MARK = f"# === AUTOHEADER_CELL:{AUTOHEADER_VERSION} ==="

# ---------------------------------------------------------------------
# Cell00 lock + embedded structure (Cell00 must never be modified)
# ---------------------------------------------------------------------

CELL0_LOCK_BEGIN = "# === CELL0_LOCK:BEGIN ==="
CELL0_LOCK_END = "# === CELL0_LOCK:END ==="

STRUCTURE_JSON_BEGIN = "# === STRUCTURE_JSON:BEGIN ==="
STRUCTURE_JSON_END = "# === STRUCTURE_JSON:END ==="

# ---------------------------------------------------------------------
# Cell01 deterministic bootstrap marker
# ---------------------------------------------------------------------

CELL01_SETUP_MARK = "# === AUTOSETUP_CELL01:v1 ==="


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def replace_structure_json_in_cell00(cell00_source: str, structure: Any, *, min_cell_index: int = 1) -> str:
    """
    Replace only the STRUCTURE_JSON block inside an existing locked Cell00.
    Keeps everything else untouched.
    """
    s = cell00_source or ""
    if (STRUCTURE_JSON_BEGIN not in s) or (STRUCTURE_JSON_END not in s):
        # If markers are missing, do nothing (caller can decide to rebuild Cell00)
        return s.rstrip() + "\n"

    payload = sanitize_structure(structure, min_cell_index=int(min_cell_index))
    import json
    raw_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)

    new_block_lines = [STRUCTURE_JSON_BEGIN]
    for ln in raw_json.splitlines():
        new_block_lines.append(f"# {ln}".rstrip())
    new_block_lines.append(STRUCTURE_JSON_END)
    new_block = "\n".join(new_block_lines)

    a = s.find(STRUCTURE_JSON_BEGIN)
    b = s.find(STRUCTURE_JSON_END, a + 1)
    if a == -1 or b == -1 or b <= a:
        return s.rstrip() + "\n"

    b2 = b + len(STRUCTURE_JSON_END)
    replaced = s[:a].rstrip() + "\n" + new_block + "\n" + s[b2:].lstrip("\n")
    return replaced.rstrip() + "\n"

def sanitize_structure(structure: Any, *, min_cell_index: int = 1) -> List[Dict[str, Any]]:
    # Accept dict wrappers too (e.g., {"structure":[...]} or {"items":[...]})
    structure_list = _coerce_structure_list(structure)
    if not structure_list:
        return []

    items: List[Dict[str, Any]] = []
    for it in structure_list:
        if not isinstance(it, dict):
            continue

        raw_idx = it.get("cell_index")
        try:
            idx = int(raw_idx)
        except Exception:
            idx = None
            if isinstance(raw_idx, str):
                m = re.search(r"\d+", raw_idx)
                if m:
                    idx = int(m.group(0))

        if idx is None or idx < int(min_cell_index):
            continue

        title = _s(it.get("title"))
        if not title:
            continue

        items.append(
            {
                "cell_index": idx,
                "title": title,
                "overview": _s(it.get("overview", "")),
                "io": _s(it.get("io", "")),
                "notes": _s(it.get("notes", "")),
            }
        )

    dedup: Dict[int, Dict[str, Any]] = {}
    for it in items:
        dedup[int(it["cell_index"])] = it
    out = list(dedup.values())
    out.sort(key=lambda x: int(x["cell_index"]))
    return out



def build_cell00(task_fields: Dict[str, Any], policy: Dict[str, Any], structure: List[Dict[str, Any]]) -> str:
    """
    Build the Cell 00 notebook header block (English, comment-only).

    Notes
    -----
    - Comment-only output (intended for a code cell; safe to execute).
    - Best-effort: missing fields are OK.
    - policy supports either:
        - top-level keys: required_env / writes_allowed / writes_forbidden / notion_access_layer / safety / structure
        - or a nested shape: policy["writes"] = {"allowed": [...], "forbidden": [...]} (we normalize)
    """
    def _pick(*keys: str) -> str:
        for k in keys:
            if k in task_fields and task_fields.get(k):
                return _s(task_fields.get(k))
        return ""

    title = _pick("title", "Title", "Name") or "Notebook"
    domain = _pick("domain", "Domain")
    entry_point = _pick("entry_point", "Entry Point", "EntryPoint")
    objective = _pick("objective", "Objective", "目的", "目標")
    acceptance = _pick("acceptance_criteria", "Acceptance Criteria", "Acceptance", "受け入れ条件")
    constraints = _pick("constraints", "Constraints", "制約")
    run_policy = _pick("run_policy", "Run Policy", "RunPolicy")
    scope = _pick("scope", "Scope")

    required_env_raw = policy.get("required_env") or []
    required_env: List[str] = []
    if isinstance(required_env_raw, (list, tuple)):
        for x in required_env_raw:
            s = _s(x)
            if s:
                required_env.append(s)

    # --- normalize writes shape ---
    writes_allowed = policy.get("writes_allowed")
    writes_forbidden = policy.get("writes_forbidden")
    writes = policy.get("writes") if isinstance(policy.get("writes"), dict) else {}

    if writes_allowed is None:
        writes_allowed = writes.get("allowed") if isinstance(writes, dict) else []
    if writes_forbidden is None:
        writes_forbidden = writes.get("forbidden") if isinstance(writes, dict) else []

    writes_allowed = writes_allowed or []
    writes_forbidden = writes_forbidden or []

    notion_access_layer = policy.get("notion_access_layer") or []
    data_sources = policy.get("data_sources") or {}
    safety = policy.get("safety") or {}

    structure = sanitize_structure(structure, min_cell_index=1)
    structure_lines = _render_structure_lines(structure)
    structure_json = _dump_structure_json(structure, min_cell_index=1)

    lines: List[str] = []

    # =========================
    # Cell00 lock range (NEVER MODIFY AFTER FIRST WRITE)
    # =========================
    lines.append(CELL0_LOCK_BEGIN)

    # Existing autoheader range (kept for idempotency signature)
    lines.append(AUTOHEADER_BEGIN)
    lines.append(AUTOHEADER_MARK)
    lines.append("# ============================================================")
    lines.append(f"# {title}")
    lines.append("# ============================================================")
    lines.append("#")
    lines.append("# Overview")
    lines.append("# ----------------")
    if objective:
        lines.extend(_wrap_comment_lines(objective))
    else:
        lines.append("# (Objective not provided.)")

    if domain or entry_point:
        meta = " | ".join(
            [x for x in [f"Domain={domain}" if domain else "", f"Entry={entry_point}" if entry_point else ""] if x]
        )
        if meta:
            lines.append("#")
            lines.append(f"# {meta}")

    lines.append("#")
    lines.append("# Inputs / Outputs")
    lines.append("# ----------------")
    lines.append("# Inputs:")
    if domain:
        lines.append(f"#   - Domain: {domain}")
    if entry_point:
        lines.append(f"#   - Entry Point (source): {entry_point}")
    if required_env:
        lines.append("#   - Required env vars:")
        for k in required_env:
            lines.append(f"#       - {_s(k)}")
    if data_sources:
        lines.append("#   - Data sources (read scope) [env var names]:")
        for k, v in data_sources.items():
            lines.append(f"#       - {_s(k)}: {_s(v)}")

    lines.append("#")
    lines.append("# Outputs:")
    if writes_allowed:
        lines.append("#   - Writes are restricted to (append-only proposals):")
        for k in writes_allowed:
            lines.append(f"#       - {_s(k)}")
    else:
        lines.append("#   - Writes: (not specified)")

    lines.append("#")
    lines.append("# Structure")
    lines.append("# ----------------")
    if structure_lines:
        lines.extend(structure_lines)
    else:
        lines.append("# (Structure not provided. Expected Cell 01+ entries.)")

    lines.append("#")
    lines.append("# Notes")
    lines.append("# ----------------")
    if acceptance:
        lines.append("# Acceptance Criteria:")
        lines.extend(_wrap_comment_lines(acceptance, indent="#   "))
        lines.append("#")

    if scope:
        lines.append("# Allowable Scope:")
        lines.extend(_wrap_comment_lines(scope, indent="#   "))
        lines.append("#")

    if constraints:
        lines.append("# Constraints:")
        lines.extend(_wrap_comment_lines(constraints, indent="#   "))
        lines.append("#")

    if run_policy:
        lines.append("# Run Policy:")
        lines.extend(_wrap_comment_lines(run_policy, indent="#   "))
        lines.append("#")

    if notion_access_layer:
        lines.append("# Notion access rule:")
        lines.append("#   - All Notion access MUST go through existing infrastructure:")
        for p in notion_access_layer:
            lines.append(f"#       - {_s(p)}")
        lines.append("#   - Direct Notion API calls outside the repository layer are not allowed.")
        lines.append("#")

    if writes_forbidden:
        lines.append("# Write prohibitions:")
        for k in writes_forbidden:
            lines.append(f"#   - {_s(k)} is read-only in this notebook (no writes).")
        lines.append("#")

    if safety:
        lines.append("# Execution & Safety Constraints (additional):")
        for k, v in safety.items():
            if isinstance(v, (list, tuple)):
                lines.append(f"#   - {_s(k)}:")
                for item in v:
                    lines.append(f"#       - {_s(item)}")
            else:
                lines.append(f"#   - {_s(k)}: {_s(v)}")
        lines.append("#")

    # -------------------------
    # Embedded structure JSON (source of truth)
    # -------------------------
    lines.append(STRUCTURE_JSON_BEGIN)
    for ln in structure_json.splitlines():
        lines.append(f"# {ln}".rstrip())
    lines.append(STRUCTURE_JSON_END)

    # --- footer mark (range) ---
    lines.append(AUTOHEADER_END)

    # Close Cell00 lock
    lines.append(CELL0_LOCK_END)

    return "\n".join(lines).rstrip() + "\n"


def build_cell01_setup(*, required_env: Sequence[str] | None = None) -> str:
    """
    Build deterministic Cell01 bootstrap code.

    Includes:
      - Project-root CWD enforcement (verbatim-ish)
      - load notebooks/env.txt
      - required env checks (policy-aligned + NOTION_TOKEN)
      - Notion client + BaseRepo registry from resolved DBs
      - LLM client (ClaudeClient) creation

    This is code (not comment-only) and is intended to RUN.
    """
    req = [str(x).strip() for x in (required_env or []) if str(x).strip()]
    # We always require NOTION_TOKEN here.
    if "NOTION_TOKEN" not in req:
        req = ["NOTION_TOKEN"] + [x for x in req if x != "NOTION_TOKEN"]

    lines: List[str] = []
    lines.append(CELL01_SETUP_MARK)
    lines.append("# ============================================================")
    lines.append("# Cell 01 — Environment bootstrap & Repo contract")
    lines.append("# ============================================================")
    lines.append("# Goal:")
    lines.append("#   - Resolve project root and enforce deterministic import path")
    lines.append("#   - Load notebooks/env.txt and validate required env vars")
    lines.append("#   - Build Notion client and BaseRepo registry (single access layer)")
    lines.append("#   - Expose canonical repo_query() for ALL downstream DB access")
    lines.append("#   - Initialize Claude client")
    lines.append("#")
    lines.append("# Contract (do NOT violate in Cell02+):")
    lines.append("#   1. Notion access MUST go through BaseRepo (never call notion_client directly).")
    lines.append("#   2. Use repo_query() instead of repo.query_pages().")
    lines.append("#   3. Monitoring Targets DB is READ-ONLY.")
    lines.append("#   4. Weekly Target Update DB is APPEND-ONLY (create_page only).")
    lines.append("#   5. Do NOT redefine repos, RESOLVED_DB, or claude in later cells.")
    lines.append("#")
    lines.append("# Failure policy:")
    lines.append("#   - Missing env → hard fail.")
    lines.append("#   - DB resolution failure → hard fail.")
    lines.append("#   - LLM client failure → hard fail.")
    lines.append("#")
    lines.append("# This cell is deterministic and should rarely change.")
    lines.append("# ============================================================")
    lines.append("")
    lines.append("from pathlib import Path")
    lines.append("import os")
    lines.append("import sys")
    lines.append("from dotenv import load_dotenv")
    lines.append("")
    lines.append("# --- Discover project root (directory containing \"src/\") ---")
    lines.append("cwd = Path.cwd()")
    lines.append("project_root = None")
    lines.append("")
    lines.append("cursor = cwd")
    lines.append("for _ in range(8):  # safety limit to avoid infinite loop")
    lines.append("    if (cursor / \"src\").is_dir():")
    lines.append("        project_root = cursor")
    lines.append("        break")
    lines.append("    if cursor.parent == cursor:")
    lines.append("        break")
    lines.append("    cursor = cursor.parent")
    lines.append("")
    lines.append("if project_root is None:")
    lines.append("    raise RuntimeError(")
    lines.append("        \"Project root not found. \"")
    lines.append("        \"Expected a directory containing 'src/'. \"")
    lines.append("        \"Please launch Jupyter from the project root.\"")
    lines.append("    )")
    lines.append("")
    lines.append("# --- Enforce project-root as CWD ---")
    lines.append("os.chdir(project_root)")
    lines.append("")
    lines.append("# --- Ensure project-root is first on sys.path ---")
    lines.append("root_str = str(project_root.resolve())")
    lines.append("if sys.path[0] != root_str:")
    lines.append("    if root_str in sys.path:")
    lines.append("        sys.path.remove(root_str)")
    lines.append("    sys.path.insert(0, root_str)")
    lines.append("")
    lines.append("# --- Locate notebooks/env.txt explicitly ---")
    lines.append('NOTEBOOKS_DIR = Path.cwd() / "notebooks"')
    lines.append('ENV_PATH = NOTEBOOKS_DIR / "env.txt"')
    lines.append("load_dotenv(dotenv_path=ENV_PATH)")
    lines.append("")
    lines.append("# --- Sanity checks ---")
    lines.append("print(\"Project root resolved to:\", project_root)")
    lines.append("print(\"CWD:\", Path.cwd())")
    lines.append("print(\"sys.path[0]:\", sys.path[0])")
    lines.append("print(\"src/ exists:\", (project_root / \"src\").exists())")
    lines.append("print(\"notebooks/ exists:\", (project_root / \"notebooks\").exists())")
    lines.append("")

    # Preflight env
    lines.append("# ============================================================")
    lines.append("# Preflight (env + repos + LLM client)")
    lines.append("# ============================================================")
    lines.append("# --- Required environment variables (policy-aligned) ---")
    lines.append(f"_required_env = {repr(list(req))}")
    lines.append("missing = [k for k in _required_env if not os.getenv(k)]")
    lines.append("if missing:")
    lines.append("    raise RuntimeError(")
    lines.append("        \"Missing required environment variables in notebooks/env.txt:\\n\"")
    lines.append("        + \"\\n\".join(f\"  - {k}\" for k in missing)")
    lines.append("    )")
    lines.append("print(\"[ok] required env present ✅\")")
    lines.append("for k in _required_env:")
    lines.append("    print(f\"  - {k}: [OK]\")")
    lines.append("")

    # Notion client + BaseRepo
    lines.append("# --- Build Notion repos (single access layer) ---")
    lines.append("from src.notion.client import build_notion_client_from_env")
    lines.append("from src.notion.repos import ResolvedDBRegistry, NotionSchema, BaseRepo")
    lines.append("")
    lines.append("notion_client = build_notion_client_from_env()")
    lines.append("schema = NotionSchema()")
    lines.append("")
    lines.append("# --- Resolve RESOLVED_DB (database_id + data_source_id) ---")
    lines.append("from src.notion.client import NotionDataSourceResolver")
    lines.append("")
    lines.append("# DB env keys (token is NOT a database id)")
    lines.append("DB_ENV_KEYS = [")
    lines.append("    \"NOTION_LIT_DB_ID\",")
    lines.append("    \"NOTION_EVENTS_DB_ID\",")
    lines.append("    \"NOTION_MONITORING_TARGETS_DB_ID\",")
    lines.append("    \"NOTION_RQ_DB_ID\",")
    lines.append("    \"NOTION_WEEKLY_TARGET_UPDATE_DB_ID\",")
    lines.append("]")
    lines.append("")
    lines.append("missing_db = [k for k in DB_ENV_KEYS if not os.getenv(k)]")
    lines.append("if missing_db:")
    lines.append("    raise RuntimeError(")
    lines.append("        \"Missing required database env vars in notebooks/env.txt:\\n\"")
    lines.append("        + \"\\n\".join(f\"  - {k}\" for k in missing_db)")
    lines.append("    )")
    lines.append("")
    lines.append("resolver = NotionDataSourceResolver(notion_client)")
    lines.append("RESOLVED_DB: dict[str, dict] = {}")
    lines.append("")
    lines.append("for env_key in DB_ENV_KEYS:")
    lines.append("    db_id = os.getenv(env_key)")
    lines.append("    if not db_id:")
    lines.append("        continue")
    lines.append("    resolved = resolver.resolve_once(name=env_key, database_id=db_id)")
    lines.append("    RESOLVED_DB[env_key] = resolved.__dict__")
    lines.append("    print(f\"  ✓ resolved: {env_key}\")")
    lines.append("")
    lines.append("if not RESOLVED_DB:")
    lines.append("    raise RuntimeError(")
    lines.append("        \"Failed to resolve any database_id → data_source_id.\\n\"")
    lines.append("        \"Check: (1) Notion integration access, (2) DB IDs are correct, (3) resolver implementation.\"")
    lines.append("    )")
    lines.append("")
    lines.append("print(\"RESOLVED_DB (cached for this notebook run):\")")
    lines.append("for k, v in RESOLVED_DB.items():")
    lines.append("    print(f\"  - {k}: database_id={v['database_id']}  data_source_id={v['data_source_id']}\")")
    lines.append("")
    lines.append("reg = ResolvedDBRegistry(RESOLVED_DB)")
    lines.append("")
    lines.append("repos_by_env: dict[str, BaseRepo] = {}")
    lines.append("for env_key in RESOLVED_DB.keys():")
    lines.append("    repos_by_env[env_key] = BaseRepo(")
    lines.append("        notion_client=notion_client,")
    lines.append("        db=reg.get(env_key),")
    lines.append("        schema=schema,")
    lines.append("    )")
    lines.append("")
    lines.append("def _need_repo(key: str) -> BaseRepo:")
    lines.append("    if key not in repos_by_env:")
    lines.append("        known = \", \".join(sorted(repos_by_env.keys()))")
    lines.append("        raise KeyError(f\"Missing repo key: {key}. Available keys: {known}\")")
    lines.append("    return repos_by_env[key]")
    lines.append("")
    lines.append('weekly_target_update_repo = _need_repo("NOTION_WEEKLY_TARGET_UPDATE_DB_ID")')
    lines.append('events_repo = _need_repo("NOTION_EVENTS_DB_ID")')
    lines.append('papers_repo = _need_repo("NOTION_LIT_DB_ID")')
    lines.append('rq_repo = _need_repo("NOTION_RQ_DB_ID")')
    lines.append("print(\"[info] Repo ready ✅\")")
    lines.append("")
    lines.append("def repo_query(repo, *, filter=None, sorts=None, page_size=None):")
    lines.append("    \"\"\"")
    lines.append("    Canonical query function for BaseRepo.")
    lines.append("    Use THIS in Cell02+ (do not call repo.query_pages directly).")
    lines.append("    \"\"\"")
    lines.append("    fn = getattr(repo, 'query_pages', None)")
    lines.append("    if not callable(fn):")
    lines.append("        raise AttributeError(f\"{type(repo).__name__} missing query_pages(). dir={dir(repo)}\")")
    lines.append("")
    lines.append("    kwargs = {}")
    lines.append("    if filter is not None:")
    lines.append("        kwargs['filter'] = filter")
    lines.append("    if sorts is not None:")
    lines.append("        kwargs['sorts'] = sorts")
    lines.append("    if page_size is not None:")
    lines.append("        kwargs['page_size'] = page_size")
    lines.append("")
    lines.append("    return fn(**kwargs)")
    lines.append("")
    lines.append("# --- Build LLM client ---")
    lines.append("from src.llm.claude_client import ClaudeClient")
    lines.append("claude = ClaudeClient()")
    lines.append("print(\"[info] claude client ready ✅\")")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


CELL02_SCHEMA_MARK = "# [AUTO] CELL02_SCHEMA_INTROSPECTION"

def build_cell02_schema_introspection_repo_first() -> str:
    """
    Cell02: Schema introspection (repo-first).
    - Prefer BaseRepo metadata getter if available.
    - Fallback: infer property types from a sample page via repo_query(page_size=1).
    - Print snapshots to prevent hallucinated property names downstream.
    """
    lines: List[str] = []
    lines.append(CELL02_SCHEMA_MARK)
    lines.append("")
    lines.append("# ============================================================")
    lines.append("# Cell 02 — Schema introspection (repo-first) + optional validation")
    lines.append("# ============================================================")
    lines.append("# Goal: make DB property names/types explicit to avoid hallucination (e.g., 'Created').")
    lines.append("# Outputs (globals): DB_META, DB_PROPERTIES, DB_PROP_TYPES, SCHEMA_REPORT")
    lines.append("")
    lines.append("from typing import Any, Dict, Optional")
    lines.append("")
    lines.append("# --- Preflight: require objects from Cell01 ---")
    lines.append("for _k in ['RESOLVED_DB', 'repos_by_env', '_need_repo', 'repo_query']:")
    lines.append("    if _k not in globals():")
    lines.append("        raise RuntimeError(f\"Cell01 must define '{_k}' before running Cell02\")")
    lines.append("")
    lines.append("DB_META: Dict[str, Dict[str, Any]] = {}")
    lines.append("DB_PROPERTIES: Dict[str, Dict[str, Any]] = {}")
    lines.append("DB_PROP_TYPES: Dict[str, Dict[str, str]] = {}")
    lines.append("SCHEMA_REPORT: Dict[str, Any] = {}")
    lines.append("")
    lines.append("def _infer_property_types_from_sample_page(page_obj: Dict[str, Any]) -> Dict[str, str]:")
    lines.append("    props = (page_obj or {}).get('properties', {}) or {}")
    lines.append("    inferred: Dict[str, str] = {}")
    lines.append("    for prop_name, prop_obj in props.items():")
    lines.append("        if isinstance(prop_obj, dict):")
    lines.append("            t = prop_obj.get('type')")
    lines.append("            if isinstance(t, str) and t:")
    lines.append("                inferred[prop_name] = t")
    lines.append("    return inferred")
    lines.append("")
    lines.append("def _extract_types_from_metadata(properties: Dict[str, Any]) -> Dict[str, str]:")
    lines.append("    types: Dict[str, str] = {}")
    lines.append("    for pn, po in (properties or {}).items():")
    lines.append("        if isinstance(po, dict):")
    lines.append("            t = po.get('type')")
    lines.append("            if isinstance(t, str) and t:")
    lines.append("                types[pn] = t")
    lines.append("    return types")
    lines.append("")
    lines.append("def _repo_get_db_meta(repo) -> Optional[Dict[str, Any]]:")
    lines.append("    # Prefer a first-class metadata getter if BaseRepo exposes one.")
    lines.append("    for name in ('get_database_meta', 'retrieve_database', 'get_database', 'fetch_database'):")
    lines.append("        fn = getattr(repo, name, None)")
    lines.append("        if callable(fn):")
    lines.append("            try:")
    lines.append("                return fn()")
    lines.append("            except TypeError:")
    lines.append("                # Some implementations require database_id; try to supply from RESOLVED_DB via env key later.")
    lines.append("                return None")
    lines.append("    return None")
    lines.append("")
    lines.append("def introspect_env_key(env_key: str) -> None:")
    lines.append("    repo = globals()['_need_repo'](env_key)")
    lines.append("")
    lines.append("    # 1) Try metadata-first via repo")
    lines.append("    meta = _repo_get_db_meta(repo)")
    lines.append("    if isinstance(meta, dict) and meta:")
    lines.append("        DB_META[env_key] = meta")
    lines.append("        props = meta.get('properties', {}) or {}")
    lines.append("        DB_PROPERTIES[env_key] = props")
    lines.append("        if props:")
    lines.append("            types = _extract_types_from_metadata(props)")
    lines.append("            DB_PROP_TYPES[env_key] = types")
    lines.append("            print(f\"✓ {env_key}: metadata properties(types)={len(types)}\")")
    lines.append("            return")
    lines.append("        print(f\"⚠ {env_key}: meta.properties empty; falling back to sample inference...\")")
    lines.append("")
    lines.append("    # 2) Fallback: infer type-only from one sample page via repo_query(page_size=1)")
    lines.append("    try:")
    lines.append("        res = repo_query(repo, page_size=1)")
    lines.append("    except Exception as e:")
    lines.append("        DB_PROP_TYPES[env_key] = {}")
    lines.append("        print(f\"✗ {env_key}: repo_query(page_size=1) failed: {type(e).__name__}: {e}\")")
    lines.append("        return")
    lines.append("")
    lines.append("    # BaseRepo.query_pages return shape may differ; handle a few common ones.")
    lines.append("    results = None")
    lines.append("    if isinstance(res, dict):")
    lines.append("        results = res.get('results') or res.get('pages') or res.get('data') or res.get('items')")
    lines.append("    elif isinstance(res, list):")
    lines.append("        results = res")
    lines.append("")
    lines.append("    results = results or []")
    lines.append("    if not results:")
    lines.append("        DB_PROP_TYPES[env_key] = {}")
    lines.append("        print(f\"  ✗ {env_key}: no pages found to infer property types (empty DB?)\")")
    lines.append("        return")
    lines.append("")
    lines.append("    inferred = _infer_property_types_from_sample_page(results[0])")
    lines.append("    DB_PROP_TYPES[env_key] = inferred")
    lines.append("    print(f\"  ✓ {env_key}: inferred types from sample page = {len(inferred)} (type-only trust)\")")
    lines.append("")
    lines.append("def validate_against_declared_schema(env_key: str, declared_schema: Dict[str, Any]) -> Dict[str, Any]:")
    lines.append("    required = (declared_schema or {}).get('required_properties', {}) or {}")
    lines.append("    actual_types = DB_PROP_TYPES.get(env_key, {}) or {}")
    lines.append("    missing = []")
    lines.append("    mismatches = []")
    lines.append("    for prop, expected_type in required.items():")
    lines.append("        if prop not in actual_types:")
    lines.append("            missing.append(prop)")
    lines.append("        else:")
    lines.append("            actual = actual_types[prop]")
    lines.append("            if expected_type != actual:")
    lines.append("                mismatches.append({'property': prop, 'expected': expected_type, 'actual': actual})")
    lines.append("    return {")
    lines.append("        'missing_required_properties': missing,")
    lines.append("        'type_mismatches': mismatches,")
    lines.append("        'required_count': len(required),")
    lines.append("        'actual_count': len(actual_types),")
    lines.append("    }")
    lines.append("")
    lines.append("def _print_type_snapshot(env_key: str, types_map: Dict[str, str], limit: int = 60) -> None:")
    lines.append("    items = sorted((types_map or {}).items(), key=lambda x: x[0])")
    lines.append("    print(f\"  properties(types) snapshot ({min(len(items), limit)}/{len(items)}):\")")
    lines.append("    for k, v in items[:limit]:")
    lines.append("        print(f\"    - {k}: {v}\")")
    lines.append("")
    lines.append("def _print_report(env_key: str, report: Dict[str, Any]) -> None:")
    lines.append("    print(f\"--- {env_key} ---\")")
    lines.append("    if 'note' in report:")
    lines.append("        print(f\"  {report['note']}\")")
    lines.append("        _print_type_snapshot(env_key, DB_PROP_TYPES.get(env_key, {}) or {})")
    lines.append("        return")
    lines.append("    print(f\"  required: {report['required_count']} | actual(types): {report['actual_count']}\")")
    lines.append("    if report.get('missing_required_properties'):")
    lines.append("        print(f\"  ⚠ Missing required: {report['missing_required_properties']}\")")
    lines.append("    if report.get('type_mismatches'):")
    lines.append("        print(\"  ⚠ Type mismatches:\")")
    lines.append("        for m in report['type_mismatches']:")
    lines.append("            print(f\"    - {m['property']}: expected={m['expected']} actual={m['actual']}\")")
    lines.append("    if (not report.get('missing_required_properties')) and (not report.get('type_mismatches')):")
    lines.append("        print(\"  ✓ Schema looks compatible\")")
    lines.append("    _print_type_snapshot(env_key, DB_PROP_TYPES.get(env_key, {}) or {})")
    lines.append("")
    lines.append("print(\"Introspecting database schemas (repo-first)...\\n\")")
    lines.append("for env_key in sorted((globals()['RESOLVED_DB'] or {}).keys()):")
    lines.append("    try:")
    lines.append("        introspect_env_key(env_key)")
    lines.append("    except Exception as e:")
    lines.append("        print(f\"✗ {env_key}: introspection failed: {type(e).__name__}: {e}\")")
    lines.append("")
    lines.append("print(\"\\nValidating against declared schemas (if provided)...\\n\")")
    lines.append("DECLARED_SCHEMA_BY_ENV_KEY = {")
    lines.append("    'NOTION_MONITORING_TARGETS_DB_ID': globals().get('MONITORING_TARGETS_SCHEMA'),")
    lines.append("    'NOTION_EVENTS_DB_ID': globals().get('EVENTS_SCHEMA'),")
    lines.append("    'NOTION_LIT_DB_ID': globals().get('PAPERS_SCHEMA') or globals().get('LIT_SCHEMA'),")
    lines.append("    'NOTION_RQ_DB_ID': globals().get('RQ_SCHEMA'),")
    lines.append("    'NOTION_WEEKLY_TARGET_UPDATE_DB_ID': globals().get('WEEKLY_TARGET_UPDATE_SCHEMA'),")
    lines.append("}")
    lines.append("")
    lines.append("for env_key in sorted((globals()['RESOLVED_DB'] or {}).keys()):")
    lines.append("    declared = DECLARED_SCHEMA_BY_ENV_KEY.get(env_key)")
    lines.append("    if declared is not None:")
    lines.append("        SCHEMA_REPORT[env_key] = validate_against_declared_schema(env_key, declared)")
    lines.append("    else:")
    lines.append("        SCHEMA_REPORT[env_key] = {")
    lines.append("            'note': 'No declared schema dict found in globals(); validation skipped.',")
    lines.append("            'actual_count': len(DB_PROP_TYPES.get(env_key, {}) or {}),")
    lines.append("        }")
    lines.append("")
    lines.append("for env_key, rep in SCHEMA_REPORT.items():")
    lines.append("    _print_report(env_key, rep)")
    lines.append("")
    lines.append("print(\"\\n✓ Schema introspection + (optional) validation complete\")")
    lines.append("# --- Persist schema snapshot for orchestrator (best-effort) ---")
    lines.append("try:")
    lines.append("    import json")
    lines.append("    from pathlib import Path")
    lines.append("    base = None")
    lines.append("    # nb_runner may inject `artifacts` (SimpleNamespace with base_dir)")
    lines.append("    if 'artifacts' in globals() and hasattr(globals()['artifacts'], 'base_dir'):")
    lines.append("        base = Path(str(globals()['artifacts'].base_dir))")
    lines.append("    if base is None:")
    lines.append("        base = Path.cwd() / 'artifacts'")
    lines.append("    out_dir = base / 'schema'")
    lines.append("    out_dir.mkdir(parents=True, exist_ok=True)")
    lines.append("    out_path = out_dir / 'schema_snapshot.json'")
    lines.append("    payload = {")
    lines.append("        'DB_PROP_TYPES': DB_PROP_TYPES,")
    lines.append("        'SCHEMA_REPORT': SCHEMA_REPORT,")
    lines.append("        'RESOLVED_DB': globals().get('RESOLVED_DB', {}),")
    lines.append("        'generated_at': __import__('datetime').datetime.now().isoformat(timespec='seconds'),")
    lines.append("    }")
    lines.append("    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')")
    lines.append("    print('[info] schema snapshot saved:', str(out_path))")
    lines.append("except Exception as e:")
    lines.append("    print('[warn] failed to persist schema snapshot:', type(e).__name__, str(e))")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"

def build_cell_header(
    cell_index: int,
    title: str,
    overview: str = "",
    io: str = "",
    notes: str = "",
) -> str:
    """
    Build a per-cell header block that sits at the top of a cell's code.

    Example desired format:
      # ============================================================
      # Cell 01 — Imports and environment setup
      # ============================================================
      # Overview:
      # ...
      # Inputs / Outputs:
      # ...
      # Notes:
      # ...
    """
    idx = int(cell_index)
    num = f"{idx:02d}" if idx < 100 else str(idx)
    t = _s(title)

    lines: List[str] = []
    lines.append(CELLHEADER_BEGIN)
    lines.append(CELL_HEADER_MARK)
    lines.append("# ============================================================")
    lines.append(f"# Cell {num} — {t}")
    lines.append("# ============================================================")
    lines.append("# Overview:")
    if overview:
        lines.extend(_wrap_comment_lines(overview, indent="#   "))
    else:
        lines.append("#   (fill in)")
    lines.append("#")
    lines.append("# Inputs / Outputs:")
    if io:
        lines.extend(_wrap_comment_lines(io, indent="#   "))
    else:
        lines.append("#   Inputs: (fill in)")
        lines.append("#   Outputs: (fill in)")
    lines.append("#")
    lines.append("# Notes:")
    if notes:
        lines.extend(_wrap_comment_lines(notes, indent="#   "))
    else:
        lines.append("#   (fill in)")
    lines.append("#")
    lines.append(CELLHEADER_END)
    lines.append("#")
    return "\n".join(lines).rstrip() + "\n"


def build_scaffold_digest(
    *,
    task_page_id: str,
    proposal_page_id: str,
    notebook_path: str,
    structure: Sequence[Dict[str, Any]],
    policy: Dict[str, Any],
) -> str:
    """
    Create a stable digest for debounce.

    Store this in state.json (e.g., state["scaffold"]["digest"]).
    If unchanged, you can skip re-scaffolding.

    The digest is intentionally stable across runs if inputs don't change.
    """
    payload = {
        "task_page_id": _s(task_page_id),
        "proposal_page_id": _s(proposal_page_id),
        "notebook_path": _s(notebook_path),
        "structure": _normalize_structure_for_digest(structure),
        "policy": _normalize_policy_for_digest(policy),
        "autoheader_version": AUTOHEADER_VERSION,
    }

    raw = _stable_json(payload)
    return sha256(raw.encode("utf-8")).hexdigest()[:16]


def looks_like_autoheader(source: str) -> bool:
    s = source or ""
    if (AUTOHEADER_BEGIN in s and AUTOHEADER_END in s) or (CELLHEADER_BEGIN in s and CELLHEADER_END in s):
        return True
    return (AUTOHEADER_MARK in s) or (CELL_HEADER_MARK in s) or (CELL01_SETUP_MARK in s)


def strip_existing_autoheader(source: str) -> str:
    """
    Remove existing autoheader blocks (best-effort, idempotent).

    IMPORTANT:
    - Cell00 lock range (CELL0_LOCK_BEGIN..CELL0_LOCK_END) must NEVER be removed.
      Once Cell00 is created, it is immutable in the orchestrator loop.
    - For Cell01 setup, we do NOT strip it here (it is not a header range).
      Use explicit logic if you want to replace Cell01 body.
    """
    if not source:
        return ""

    s = source

    # ✅ If this is Cell00 (locked), return as-is.
    if (CELL0_LOCK_BEGIN in s) and (CELL0_LOCK_END in s):
        return s.rstrip() + "\n"

    def _strip_all_ranges(text: str, begin: str, end: str) -> str:
        while True:
            a = text.find(begin)
            if a == -1:
                return text
            b = text.find(end, a)
            if b == -1:
                return text[:a].rstrip() + "\n"
            b2 = b + len(end)
            text = (text[:a] + text[b2:]).lstrip("\n")

    s2 = _strip_all_ranges(s, AUTOHEADER_BEGIN, AUTOHEADER_END)
    s2 = _strip_all_ranges(s2, CELLHEADER_BEGIN, CELLHEADER_END)

    if s2 != s:
        return s2.rstrip() + "\n"

    if not any(m in s for m in (AUTOHEADER_MARK, CELL_HEADER_MARK)):
        return s.rstrip() + "\n"

    # Legacy heuristic
    lines = s.splitlines()
    out: List[str] = []
    skipping = False

    for ln in lines:
        if (AUTOHEADER_MARK in ln) or (CELL_HEADER_MARK in ln):
            skipping = True
            continue

        if skipping:
            if (not ln.strip()) or ln.lstrip().startswith("#"):
                continue
            skipping = False
            out.append(ln)
            continue

        out.append(ln)

    return "\n".join(out).rstrip() + "\n"


def extract_structure_from_cell00(source: str) -> List[Dict[str, Any]]:
    """
    Read embedded structure JSON from Cell00 source.
    Expected:
      STRUCTURE_JSON_BEGIN
      # { ...json... }   (commented lines)
      STRUCTURE_JSON_END
    Returns [] if not found / invalid.
    """
    if not source:
        return []

    s = source
    a = s.find(STRUCTURE_JSON_BEGIN)
    b = s.find(STRUCTURE_JSON_END, a + 1)
    if a == -1 or b == -1 or b <= a:
        return []

    block = s[a + len(STRUCTURE_JSON_BEGIN) : b]
    lines = []
    for ln in block.splitlines():
        t = ln.lstrip()
        if t.startswith("#"):
            t = t[1:]
            if t.startswith(" "):
                t = t[1:]
        lines.append(t.rstrip("\n"))

    raw = "\n".join(lines).strip()
    if not raw:
        return []

    import json
    try:
        obj = json.loads(raw)
    except Exception:
        return []

    if not isinstance(obj, list):
        return []

    # Cell00 structure is the source of truth for Cell01+ (including 1-3).
    return sanitize_structure(obj, min_cell_index=1)


def has_cell01_setup_marker(source: str) -> bool:
    """True if the source contains the deterministic Cell01 marker."""
    return CELL01_SETUP_MARK in (source or "")

def normalize_cell_source_for_header_insertion(source: str) -> str:
    s = source or ""
    for _ in range(5):
        if not looks_like_autoheader(s):
            break
        s2 = strip_existing_autoheader(s)
        if s2 == s:
            break
        s = s2
    return s

# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------

def _s(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _wrap_comment_lines(text: str, indent: str = "# ") -> List[str]:
    """
    Render possibly multi-line text as comment lines.
    We intentionally do not hard-wrap by width to keep deterministic behavior.
    """
    t = _s(text)
    if not t:
        return [f"{indent}(empty)"]
    return [f"{indent}{ln}".rstrip() for ln in t.splitlines()]


def _render_structure_lines(structure: Any) -> List[str]:
    """
    structure: list of dicts like:
      {"cell_index": 1, "title": "...", "overview": "...", "io": "...", "notes": "..."}
    """
    if not isinstance(structure, list) or not structure:
        return []
    out: List[str] = []
    for item in structure:
        if not isinstance(item, dict):
            continue
        idx = item.get("cell_index")
        ttl = item.get("title")
        if idx is None or not ttl:
            continue
        try:
            i = int(idx)
        except Exception:
            continue
        num = f"{i:02d}" if i < 100 else str(i)
        out.append(f"# Cell {num}: {_s(ttl)}")
    return out


def _stable_json(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dump_structure_json(structure: Any, *, min_cell_index: int = 1) -> str:
    """
    Dump structure as stable JSON (no trailing spaces).
    Stored inside Cell00 as comment lines between STRUCTURE_JSON markers.
    """
    import json
    payload = sanitize_structure(structure, min_cell_index=int(min_cell_index))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def _normalize_structure_for_digest(structure: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Include Cell01+ in digest so early-structure changes affect debounce.
    return sanitize_structure(list(structure or []), min_cell_index=1)

def _normalize_policy_for_digest(policy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only policy keys that should affect scaffold output.
    Also normalize nested "writes" shape into top-level writes_allowed/writes_forbidden.
    """
    if not isinstance(policy, dict):
        return {}

    writes = policy.get("writes") if isinstance(policy.get("writes"), dict) else {}

    normalized: Dict[str, Any] = {
        "required_env": policy.get("required_env") or [],
        "data_sources": policy.get("data_sources") or {},
        "notion_access_layer": policy.get("notion_access_layer") or [],
        "safety": policy.get("safety") or {},
    }

    wa = policy.get("writes_allowed")
    wf = policy.get("writes_forbidden")
    if wa is None:
        wa = writes.get("allowed") if isinstance(writes, dict) else []
    if wf is None:
        wf = writes.get("forbidden") if isinstance(writes, dict) else []

    normalized["writes_allowed"] = wa or []
    normalized["writes_forbidden"] = wf or []
    return normalized
