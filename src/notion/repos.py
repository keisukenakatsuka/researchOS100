# src/notion/repos.py
"""
Notion repository layer (STRICT data_sources query compliant)

Must-follow constraints (from user rules)
-----------------------------------------
- Notion API version >= 2025-09-03
- Env IDs are *database IDs* (UUID)
- MUST NOT call POST /v1/databases/{database_id}/query
- For querying content, use ONLY POST /v1/data_sources/{data_source_id}/query
- Resolve data_source_id once in setup and cache:
    RESOLVED_DB[name] = {"database_id": ..., "data_source_id": ...}
- After setup, repos MUST ONLY use cached data_source_id (no deep scan here)

This repo layer:
- uses database_id for create_page(parent=database_id)
- uses data_source_id for query content
- centralizes property mapping & common update operations

New (generic schema mapping)
----------------------------
- Keep NotionSchema for your "core" DBs (tasks/proposals/runs/decisions).
- Add a DB-specific "schema_map" (logical_field -> actual property name) to BaseRepo.
  This enables DB-by-DB property-name differences without creating per-DB repo classes.

Example schema_map (per DB key):
  schema_map["NOTION_EVENTS_DB_ID"] = {
      "title": "Name",
      "status": "Status",
      "summary": "Summary",
      "source_url": "Source URL",
  }

Notebook can build schema_map from introspection (DB_PROP_TYPES/DB_PROPERTIES)
and pass it into build_repos(..., schema_map=...).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Literal, Sequence

from src.notion.client import normalize_uuid


# ============================================================
# Schema container (property names) — Core DBs only
# ============================================================

@dataclass(frozen=True)
class NotionSchema:
    # TASKS_DB
    TASK_TITLE: str = "Title"
    TASK_STATUS: str = "Status"
    TASK_PRIORITY: str = "Priority"
    TASK_DOMAIN: str = "Domain"
    TASK_OBJECTIVE: str = "Objective"
    TASK_AC: str = "Acceptance Criteria"
    TASK_CONSTRAINTS: str = "Constraints"
    TASK_SCOPE: str = "Scope"
    TASK_ENTRY: str = "Entry Point"
    TASK_RUN_POLICY: str = "Run Policy"
    TASK_OWNER: str = "Owner"
    TASK_RUNS_REL: str = "Runs"
    TASK_PROPOSALS_REL: str = "Proposals"
    TASK_LATEST_RUN_REL: str = "Latest Run"               # optional
    TASK_LATEST_PROPOSAL_REL: str = "Latest Proposal"     # optional

    # PROPOSALS_DB
    PROP_TITLE: str = "Title"
    PROP_STATUS: str = "Status"
    PROP_TASK_REL: str = "Task"
    PROP_OWNER: str = "Owner"
    PROP_NOTEBOOK_PATH: str = "Notebook Path"
    PROP_CELL_INDEX: str = "Cell Index"
    PROP_INTENT: str = "Intent"
    PROP_ACCEPTANCE: str = "Acceptance"
    PROP_DIFF_SUMMARY: str = "Diff Summary"
    PROP_PATCH_PATH: str = "Patch Path"
    PROP_ARTIFACTS_PATH: str = "Artifacts Path"
    PROP_COMMIT: str = "Commit"
    PROP_RISK: str = "Risk"
    PROP_ROLLBACK: str = "Rollback"
    PROP_RUNS_REL: str = "Runs"
    PROP_LAST_RUN_REL: str = "Last Run"                   # optional
    PROP_FAILURE_REASON: str = "Failure Reason"
    PROP_NEXT_ACTION: str = "Next Action"

    # RUNS_DB
    RUN_TITLE: str = "Run ID"
    RUN_STATUS: str = "Status"
    RUN_TYPE: str = "Run Type"
    RUN_PHASE: str = "Phase"                              # optional
    RUN_TASK_REL: str = "Task"
    RUN_PROPOSAL_REL: str = "Proposal"
    RUN_NOTEBOOK_PATH: str = "Notebook Path"
    RUN_UP_TO_CELL: str = "Up To Cell"
    RUN_FAILING_CELL: str = "Failing Cell"
    RUN_ERROR_SUMMARY: str = "Error Summary"
    RUN_ERROR_TRACE: str = "Error Trace (Short)"
    RUN_ARTIFACTS_PATH: str = "Artifacts Path"
    RUN_LOGS_LINK: str = "Logs Link"                      # optional
    RUN_BRANCH_REF: str = "Branch / Ref"                  # optional
    RUN_COMMIT_BEFORE: str = "Commit Before"              # optional
    RUN_COMMIT_AFTER: str = "Commit After"                # optional
    RUN_STARTED_AT: str = "Started At"
    RUN_FINISHED_AT: str = "Finished At"
    RUN_DURATION_SEC: str = "Duration Sec"
    RUN_EVIDENCE: str = "Result Evidence"
    RUN_NEXT_ACTION: str = "Next Action"

    # DECISIONS_DB
    DEC_TITLE: str = "Decision"
    DEC_STATUS: str = "Status"
    DEC_TASK_REL: str = "Task"
    DEC_RUN_REL: str = "Run"                              # optional
    DEC_PROPOSAL_REL: str = "Proposal"                    # optional
    DEC_RATIONALE: str = "Rationale"
    DEC_ALTERNATIVES: str = "Alternatives"                # optional
    DEC_IMPACT_RISK: str = "Impact / Risk"                # optional
    DEC_OWNER: str = "Owner"


# ============================================================
# Resolved DB mapping (cached from setup cell)
# ============================================================

@dataclass(frozen=True)
class ResolvedDBRef:
    """
    A resolved pair:
      - database_id: used for create_page(parent.database_id=...)
      - data_source_id: used for querying content (POST /data_sources/{id}/query)
    """
    name: str
    database_id: str
    data_source_id: str

    @staticmethod
    def from_cached(name: str, cached: Mapping[str, Any]) -> "ResolvedDBRef":
        # database_id is a UUID (normalize)
        db_id = normalize_uuid(str(cached["database_id"]))

        # data_source_id is NOT guaranteed to be UUID-looking across API versions.
        # Keep as raw string (but still strip/validate non-empty).
        ds_raw = str(cached["data_source_id"]).strip()
        if not ds_raw:
            raise ValueError(f"RESOLVED_DB['{name}'] has empty data_source_id")
        ds_id = ds_raw

        return ResolvedDBRef(name=name, database_id=db_id, data_source_id=ds_id)



@dataclass(frozen=True)
class ResolvedDBRegistry:
    """
    Thin wrapper around RESOLVED_DB dict created in your setup cell.

    Example cached mapping (your rule):
      RESOLVED_DB[name] = {"database_id": "...", "data_source_id": "..."}
    """
    refs: Mapping[str, Mapping[str, Any]]

    def get(self, name: str) -> ResolvedDBRef:
        if name not in self.refs:
            raise KeyError(f"RESOLVED_DB missing '{name}'. Resolve in setup cell and cache it.")
        return ResolvedDBRef.from_cached(name, self.refs[name])


# ============================================================
# Property builder helpers (Notion payload)
# ============================================================

def _rt(text: str) -> dict:
    return {"rich_text": [{"type": "text", "text": {"content": text}}]}

def _title(text: str) -> dict:
    return {"title": [{"type": "text", "text": {"content": text}}]}

def _select(name: str) -> dict:
    return {"select": {"name": name}}

def _number(n: Optional[float | int]) -> dict:
    return {"number": None if n is None else float(n)}

def _date_iso(dt: Optional[datetime]) -> dict:
    if dt is None:
        return {"date": None}
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return {"date": {"start": dt.isoformat()}}

def _url(u: Optional[str]) -> dict:
    return {"url": u}

def _relation(page_ids: Sequence[str]) -> dict:
    return {"relation": [{"id": normalize_uuid(pid)} for pid in page_ids]}

def _relation_one(page_id: Optional[str]) -> dict:
    return {"relation": []} if not page_id else _relation([page_id])

def _maybe(props: Dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    props[key] = value


# ============================================================
# Generic schema mapping (logical -> actual property name)
# ============================================================

LogicalField = str
SchemaMap = Mapping[str, Mapping[LogicalField, str]]

DEFAULT_LOGICAL_BUILDERS: Dict[LogicalField, Any] = {
    "title": _title,
    "rich_text": _rt,
    "select": _select,
    "number": _number,
    "date": _date_iso,
    "url": _url,
    "relation": _relation,
    "relation_one": _relation_one,
}


# ============================================================
# Base Repo (extended)
# ============================================================

class BaseRepo:
    """
    Each repo is bound to:
      - database_id (for create_page parent)
      - data_source_id (for querying content)

    Extensions for generic use:
      - repo_key: key used in RESOLVED_DB / schema_map
      - schema_map: mapping from logical fields to actual property names (per DB)
      - helper methods to build properties by logical keys
    """

    def __init__(
        self,
        *,
        notion_client: Any,
        db: ResolvedDBRef,
        schema: NotionSchema,
        repo_key: Optional[str] = None,
        schema_map: Optional[SchemaMap] = None,
    ):
        self.client = notion_client
        self.db = db
        self.schema = schema

        # key used to lookup schema_map (defaults to resolved db.name)
        self.repo_key = str(repo_key or db.name)

        # schema_map[repo_key] = {"title": "Name", "status": "Status", ...}
        self._schema_map: Dict[str, Dict[LogicalField, str]] = {}
        if schema_map:
            # copy to avoid accidental mutation from caller side
            self._schema_map = {k: dict(v) for k, v in schema_map.items()}

    # ---------- low-level CRUD ----------

    def create_page(self, *, properties: Dict[str, Any], children: Optional[List[dict]] = None) -> dict:
        return self.client.create_page(parent_db_id=self.db.database_id, properties=properties, children=children)

    def update_page(self, *, page_id: str, properties: Dict[str, Any], archived: Optional[bool] = None) -> dict:
        return self.client.update_page(page_id=page_id, properties=properties, archived=archived)

    def retrieve_page(self, *, page_id: str) -> dict:
        return self.client.retrieve_page(page_id=page_id)

    def get_database_meta(self) -> dict:
        return self.client.get_database(database_id=self.db.database_id)

    # STRICT: query must go through data_sources
    def query_pages(
        self,
        *,
        filter: Optional[dict] = None,
        sorts: Optional[List[dict]] = None,
        page_size: int = 100,
        start_cursor: Optional[str] = None,
        fetch_all: bool = True,
    ) -> List[dict]:
        # Guard: enforce strict data_sources query usage (never database query)
        if not hasattr(self.client, "query_data_source"):
            raise AttributeError("Notion client must implement query_data_source(data_source_id=...).")
        if hasattr(self.client, "query_database"):
            # Not a hard error by itself, but useful to catch accidental use elsewhere.
            pass

        return self.client.query_data_source(
            data_source_id=self.db.data_source_id,
            filter=filter,
            sorts=sorts,
            page_size=page_size,
            start_cursor=start_cursor,
            fetch_all=fetch_all,
        )

    # ---------- schema_map helpers ----------

    def set_schema_map(self, schema_map: SchemaMap) -> None:
        """Replace schema_map (caller can set after introspection)."""
        self._schema_map = {k: dict(v) for k, v in (schema_map or {}).items()}
    def _schema_map_candidates(self) -> List[str]:
        """
        schema_map lookup keys in order.
        This prevents silent failures when caller used env-style keys
        (e.g., NOTION_EVENTS_DB_ID) while repo_key is resolved DB name (e.g., EVENTS_DB).
        """
        cands = [str(self.repo_key), str(self.db.name)]
        # also allow common lowercase aliases
        cands.extend([c.lower() for c in list(cands)])
        # de-dup while preserving order
        out: List[str] = []
        for k in cands:
            if k and k not in out:
                out.append(k)
        return out

    def _schema_map_for_this_repo(self) -> Dict[LogicalField, str]:
        for k in self._schema_map_candidates():
            m = self._schema_map.get(k)
            if m:
                return dict(m)
        return {}

    def schema_for_db(self) -> Dict[LogicalField, str]:
        """Return the db-specific mapping (logical_field -> property_name)."""
        return self._schema_map_for_this_repo()

    def prop_name(self, logical_field: LogicalField, *, default: Optional[str] = None) -> Optional[str]:
        """Resolve logical_field -> actual Notion property name for this DB."""
        m = self._schema_map_for_this_repo()
        return str(m.get(logical_field)) if logical_field in m else default

    def build_properties_from_logical(self, logical_values: Mapping[LogicalField, Any]) -> Dict[str, Any]:
        """
        Convert:
          {"title": "...", "status": "OPEN", "summary": "..."}
        into Notion properties dict using schema_map for this repo.

        Convention (you can extend freely):
          - title     -> title
          - status    -> select
          - summary   -> rich_text
          - rationale -> rich_text
          - source_url -> url
          - confidence -> number
          - date      -> date
          - target_relation -> relation_one or relation (caller provides list/str)
        """
        m = self._schema_map_for_this_repo()
        props: Dict[str, Any] = {}


        def _put(logical_key: str, payload: Any) -> None:
            pn = m.get(logical_key)
            if not pn:
                return
            props[str(pn)] = payload

        # common fields
        if "title" in logical_values:
            _put("title", _title(str(logical_values["title"])))

        if "status" in logical_values and logical_values["status"] is not None:
            _put("status", _select(str(logical_values["status"])))

        if "summary" in logical_values and logical_values["summary"] is not None:
            _put("summary", _rt(str(logical_values["summary"])))

        if "rationale" in logical_values and logical_values["rationale"] is not None:
            _put("rationale", _rt(str(logical_values["rationale"])))

        if "current_value" in logical_values and logical_values["current_value"] is not None:
            _put("current_value", _rt(str(logical_values["current_value"])))

        if "proposed_value" in logical_values and logical_values["proposed_value"] is not None:
            _put("proposed_value", _rt(str(logical_values["proposed_value"])))

        if "source_url" in logical_values and logical_values["source_url"] is not None:
            _put("source_url", _url(str(logical_values["source_url"])))

        if "confidence" in logical_values and logical_values["confidence"] is not None:
            _put("confidence", _number(logical_values["confidence"]))

        if "date" in logical_values:
            _put("date", _date_iso(logical_values["date"]))

        # select-like domain fields
        if "field" in logical_values and logical_values["field"] is not None:
            _put("field", _select(str(logical_values["field"])))

        if "proposal_type" in logical_values and logical_values["proposal_type"] is not None:
            _put("proposal_type", _select(str(logical_values["proposal_type"])))

        # relations
        if "target_relation" in logical_values and logical_values["target_relation"] is not None:
            v = logical_values["target_relation"]
            if isinstance(v, (list, tuple)):
                _put("target_relation", _relation([str(x) for x in v]))
            else:
                _put("target_relation", _relation_one(str(v)))

        if "week_relation" in logical_values and logical_values["week_relation"] is not None:
            v = logical_values["week_relation"]
            if isinstance(v, (list, tuple)):
                _put("week_relation", _relation([str(x) for x in v]))
            else:
                _put("week_relation", _relation_one(str(v)))

        return props


# ============================================================
# TASKS Repo (core)
# ============================================================

TaskStatus = Literal["READY", "RUNNING", "REVIEW", "DONE", "BLOCKED", "ARCHIVED"]
TaskPriority = Literal["P0", "P1", "P2", "P3"]

class TasksRepo(BaseRepo):
    def create_task(
        self,
        *,
        title: str,
        status: TaskStatus = "READY",
        priority: TaskPriority = "P2",
        domain: Optional[str] = None,
        objective: Optional[str] = None,
        acceptance_criteria: Optional[str] = None,
        constraints: Optional[str] = None,
        scope: Optional[str] = None,
        entry_point: Optional[str] = None,
        run_policy: Optional[str] = None,
        owner_text: Optional[str] = None,
    ) -> dict:
        s = self.schema
        props: Dict[str, Any] = {
            s.TASK_TITLE: _title(title),
            s.TASK_STATUS: _select(status),
            s.TASK_PRIORITY: _select(priority),
        }
        _maybe(props, s.TASK_DOMAIN, _select(domain) if domain else None)
        _maybe(props, s.TASK_OBJECTIVE, _rt(objective) if objective else None)
        _maybe(props, s.TASK_AC, _rt(acceptance_criteria) if acceptance_criteria else None)
        _maybe(props, s.TASK_CONSTRAINTS, _rt(constraints) if constraints else None)
        _maybe(props, s.TASK_SCOPE, _rt(scope) if scope else None)
        _maybe(props, s.TASK_ENTRY, _rt(entry_point) if entry_point else None)
        _maybe(props, s.TASK_RUN_POLICY, _rt(run_policy) if run_policy else None)
        _maybe(props, s.TASK_OWNER, _rt(owner_text) if owner_text else None)
        return self.create_page(properties=props)

    def set_status(self, *, task_page_id: str, status: TaskStatus) -> dict:
        s = self.schema
        return self.update_page(page_id=task_page_id, properties={s.TASK_STATUS: _select(status)})

    def link_run(self, *, task_page_id: str, run_page_id: str, set_latest: bool = True) -> dict:
        s = self.schema
        props: Dict[str, Any] = {s.TASK_RUNS_REL: _relation([run_page_id])}
        if set_latest:
            props[s.TASK_LATEST_RUN_REL] = _relation_one(run_page_id)
        return self.update_page(page_id=task_page_id, properties=props)

    def link_proposal(self, *, task_page_id: str, proposal_page_id: str, set_latest: bool = False) -> dict:
        s = self.schema
        props: Dict[str, Any] = {s.TASK_PROPOSALS_REL: _relation([proposal_page_id])}
        if set_latest:
            props[s.TASK_LATEST_PROPOSAL_REL] = _relation_one(proposal_page_id)
        return self.update_page(page_id=task_page_id, properties=props)

    def query_tasks(
        self,
        *,
        statuses: List[TaskStatus],
        domain: Optional[str] = None,
        page_size: int = 50,
        fetch_all: bool = True,
        sorts: Optional[List[dict]] = None,
    ) -> List[Dict[str, Any]]:
        if not statuses:
            raise ValueError("statuses must be non-empty")

        s = self.schema
        status_or = [{"property": s.TASK_STATUS, "select": {"equals": st}} for st in statuses]
        and_filters: List[Dict[str, Any]] = [{"or": status_or}]
        if domain:
            and_filters.append({"property": s.TASK_DOMAIN, "select": {"equals": domain}})

        flt: Dict[str, Any] = {"and": and_filters}
        return self.query_pages(
            filter=flt,
            sorts=sorts,
            page_size=min(int(page_size), 100),
            fetch_all=fetch_all,
        )


# ============================================================
# PROPOSALS Repo (core)
# ============================================================

ProposalStatus = Literal["DRAFT", "APPLIED", "VERIFIED", "FAILED", "REVERTED", "MERGED"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]

class ProposalsRepo(BaseRepo):
    def create_changeset(
        self,
        *,
        title: str,
        task_page_id: str,
        status: ProposalStatus = "DRAFT",
        notebook_path: Optional[str] = None,
        cell_index: Optional[int] = None,
        intent: Optional[str] = None,
        acceptance: Optional[str] = None,
        diff_summary: Optional[str] = None,
        patch_path: Optional[str] = None,
        artifacts_path: Optional[str] = None,
        commit: Optional[str] = None,
        risk: Optional[RiskLevel] = None,
        rollback: Optional[str] = None,
        owner_text: Optional[str] = None,
    ) -> dict:
        s = self.schema
        props: Dict[str, Any] = {
            s.PROP_TITLE: _title(title),
            s.PROP_STATUS: _select(status),
            s.PROP_TASK_REL: _relation_one(task_page_id),
        }
        _maybe(props, s.PROP_NOTEBOOK_PATH, _rt(notebook_path) if notebook_path else None)
        _maybe(props, s.PROP_CELL_INDEX, _number(cell_index) if cell_index is not None else None)
        _maybe(props, s.PROP_INTENT, _rt(intent) if intent else None)
        _maybe(props, s.PROP_ACCEPTANCE, _rt(acceptance) if acceptance else None)
        _maybe(props, s.PROP_DIFF_SUMMARY, _rt(diff_summary) if diff_summary else None)
        _maybe(props, s.PROP_PATCH_PATH, _rt(patch_path) if patch_path else None)
        _maybe(props, s.PROP_ARTIFACTS_PATH, _rt(artifacts_path) if artifacts_path else None)
        _maybe(props, s.PROP_COMMIT, _rt(commit) if commit else None)
        _maybe(props, s.PROP_RISK, _select(risk) if risk else None)
        _maybe(props, s.PROP_ROLLBACK, _rt(rollback) if rollback else None)
        _maybe(props, s.PROP_OWNER, _rt(owner_text) if owner_text else None)
        return self.create_page(properties=props)

    def link_run(self, *, proposal_page_id: str, run_page_id: str, set_last_run: bool = True) -> dict:
        s = self.schema
        props: Dict[str, Any] = {s.PROP_RUNS_REL: _relation([run_page_id])}
        if set_last_run:
            props[s.PROP_LAST_RUN_REL] = _relation_one(run_page_id)
        return self.update_page(page_id=proposal_page_id, properties=props)

    def mark_verified(
        self,
        *,
        proposal_page_id: str,
        run_page_id: str,
        artifacts_path: Optional[str] = None,
        next_action: Optional[str] = None,
    ) -> dict:
        s = self.schema
        props: Dict[str, Any] = {
            s.PROP_STATUS: _select("VERIFIED"),
            s.PROP_RUNS_REL: _relation([run_page_id]),
        }
        props[s.PROP_LAST_RUN_REL] = _relation_one(run_page_id)
        _maybe(props, s.PROP_ARTIFACTS_PATH, _rt(artifacts_path) if artifacts_path else None)
        _maybe(props, s.PROP_NEXT_ACTION, _rt(next_action) if next_action else None)
        return self.update_page(page_id=proposal_page_id, properties=props)

    def mark_failed(
        self,
        *,
        proposal_page_id: str,
        run_page_id: str,
        failure_reason: str,
        next_action: Optional[str] = None,
        artifacts_path: Optional[str] = None,
    ) -> dict:
        s = self.schema
        props: Dict[str, Any] = {
            s.PROP_STATUS: _select("FAILED"),
            s.PROP_RUNS_REL: _relation([run_page_id]),
            s.PROP_FAILURE_REASON: _rt(failure_reason),
        }
        props[s.PROP_LAST_RUN_REL] = _relation_one(run_page_id)
        _maybe(props, s.PROP_NEXT_ACTION, _rt(next_action) if next_action else None)
        _maybe(props, s.PROP_ARTIFACTS_PATH, _rt(artifacts_path) if artifacts_path else None)
        return self.update_page(page_id=proposal_page_id, properties=props)

    def set_status(self, *, proposal_page_id: str, status: ProposalStatus) -> dict:
        s = self.schema
        return self.update_page(page_id=proposal_page_id, properties={s.PROP_STATUS: _select(status)})


# ============================================================
# RUNS Repo (core)
# ============================================================

RunStatus = Literal["PASS", "FAIL", "ABORTED", "TIMEOUT"]
RunType = Literal["EXECUTE_PREFIX", "EXECUTE_FULL", "RUN_TESTS", "LINT"]

class RunsRepo(BaseRepo):
    def create_run(
        self,
        *,
        run_id: str,
        status: RunStatus,
        run_type: RunType,
        task_page_id: str,
        proposal_page_ids: Optional[List[str]] = None,
        notebook_path: Optional[str] = None,
        up_to_cell: Optional[int] = None,
        failing_cell: Optional[int] = None,
        error_summary: Optional[str] = None,
        error_trace_short: Optional[str] = None,
        artifacts_path: Optional[str] = None,
        logs_link: Optional[str] = None,
        branch_ref: Optional[str] = None,
        commit_before: Optional[str] = None,
        commit_after: Optional[str] = None,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
        duration_sec: Optional[float] = None,
        evidence: Optional[str] = None,
        next_action: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> dict:
        s = self.schema
        props: Dict[str, Any] = {
            s.RUN_TITLE: _title(run_id),
            s.RUN_STATUS: _select(status),
            s.RUN_TYPE: _select(run_type),
            s.RUN_TASK_REL: _relation_one(task_page_id),
        }
        if phase:
            props[s.RUN_PHASE] = _select(phase)
        if proposal_page_ids:
            props[s.RUN_PROPOSAL_REL] = _relation(proposal_page_ids)

        _maybe(props, s.RUN_NOTEBOOK_PATH, _rt(notebook_path) if notebook_path else None)
        _maybe(props, s.RUN_UP_TO_CELL, _number(up_to_cell) if up_to_cell is not None else None)
        _maybe(props, s.RUN_FAILING_CELL, _number(failing_cell) if failing_cell is not None else None)
        _maybe(props, s.RUN_ERROR_SUMMARY, _rt(error_summary) if error_summary else None)
        _maybe(props, s.RUN_ERROR_TRACE, _rt(error_trace_short) if error_trace_short else None)
        _maybe(props, s.RUN_ARTIFACTS_PATH, _rt(artifacts_path) if artifacts_path else None)
        _maybe(props, s.RUN_LOGS_LINK, _url(logs_link) if logs_link else None)
        _maybe(props, s.RUN_BRANCH_REF, _rt(branch_ref) if branch_ref else None)
        _maybe(props, s.RUN_COMMIT_BEFORE, _rt(commit_before) if commit_before else None)
        _maybe(props, s.RUN_COMMIT_AFTER, _rt(commit_after) if commit_after else None)
        _maybe(props, s.RUN_STARTED_AT, _date_iso(started_at) if started_at else None)
        _maybe(props, s.RUN_FINISHED_AT, _date_iso(finished_at) if finished_at else None)
        _maybe(props, s.RUN_DURATION_SEC, _number(duration_sec) if duration_sec is not None else None)
        _maybe(props, s.RUN_EVIDENCE, _rt(evidence) if evidence else None)
        _maybe(props, s.RUN_NEXT_ACTION, _rt(next_action) if next_action else None)

        return self.create_page(properties=props)


# ============================================================
# DECISIONS Repo (core)
# ============================================================

DecisionStatus = Literal["PROPOSED", "ACCEPTED", "REJECTED", "REVERTED"]

class DecisionsRepo(BaseRepo):
    def create_decision(
        self,
        *,
        title: str,
        status: DecisionStatus,
        task_page_id: str,
        rationale: str,
        alternatives: Optional[str] = None,
        impact_risk: Optional[str] = None,
        owner_text: Optional[str] = None,
        run_page_id: Optional[str] = None,
        proposal_page_id: Optional[str] = None,
    ) -> dict:
        s = self.schema
        props: Dict[str, Any] = {
            s.DEC_TITLE: _title(title),
            s.DEC_STATUS: _select(status),
            s.DEC_TASK_REL: _relation_one(task_page_id),
            s.DEC_RATIONALE: _rt(rationale),
        }
        _maybe(props, s.DEC_ALTERNATIVES, _rt(alternatives) if alternatives else None)
        _maybe(props, s.DEC_IMPACT_RISK, _rt(impact_risk) if impact_risk else None)
        _maybe(props, s.DEC_OWNER, _rt(owner_text) if owner_text else None)
        _maybe(props, s.DEC_RUN_REL, _relation_one(run_page_id) if run_page_id else None)
        _maybe(props, s.DEC_PROPOSAL_REL, _relation_one(proposal_page_id) if proposal_page_id else None)
        return self.create_page(properties=props)


# ============================================================
# Repo bundle
# ============================================================

@dataclass(frozen=True)
class NotionRepos:
    # core (required)
    tasks: TasksRepo
    proposals: ProposalsRepo
    runs: RunsRepo
    decisions: DecisionsRepo

    # generic db access (optional): key -> BaseRepo
    by_key: Optional[Dict[str, BaseRepo]] = None


# ============================================================
# Repo factory
# ============================================================

def build_repos(
    *,
    notion_client: Any,
    resolved_registry: ResolvedDBRegistry,
    schema: Optional[NotionSchema] = None,
    schema_map: Optional[SchemaMap] = None,

    # Keys in RESOLVED_DB for the core DBs (support uppercase/lowercase)
    tasks_name: str = "TASKS_DB",
    proposals_name: str = "PROPOSALS_DB",
    runs_name: str = "RUNS_DB",
    decisions_name: str = "DECISIONS_DB",

    # If provided, build a generic BaseRepo for each of these keys and expose as repos.by_key[key]
    extra_repo_keys: Optional[List[str]] = None,
) -> NotionRepos:
    """
    Build NotionRepos from an already-resolved DB registry.

    - Core repos (tasks/proposals/runs/decisions) are always created.
    - Optional: create generic BaseRepo instances for arbitrary DBs (extra_repo_keys),
      which is the recommended path for domain DBs with different property names.
    - schema_map can be attached to every repo for logical-field property resolution.
    """

    if notion_client is None:
        raise ValueError(
            "build_repos(): notion_client is required.\n"
            "Example:\n"
            "  repos = build_repos(notion_client=client, resolved_registry=ResolvedDBRegistry(RESOLVED_DB))"
        )
    if resolved_registry is None:
        raise ValueError("build_repos(): resolved_registry is required.")
    if isinstance(resolved_registry, dict):
        raise TypeError("build_repos(): resolved_registry must be ResolvedDBRegistry, not dict.")
    if not hasattr(resolved_registry, "get"):
        raise TypeError(f"build_repos(): resolved_registry must implement .get(name). Got type={type(resolved_registry)}")

    schema = schema or NotionSchema()

    def _get_any(*names: str) -> ResolvedDBRef:
        last_exc: Optional[Exception] = None
        for n in names:
            try:
                return resolved_registry.get(n)
            except Exception as e:
                last_exc = e
        raise RuntimeError(
            "build_repos(): failed to resolve required DB ref.\n"
            f"Tried keys: {list(names)}\n"
            "Fix: ensure your setup cell caches RESOLVED_DB with one of these keys."
        ) from last_exc

    tasks_ref = _get_any(tasks_name, "tasks", "TASKS_DB")
    proposals_ref = _get_any(proposals_name, "proposals", "PROPOSALS_DB")
    runs_ref = _get_any(runs_name, "runs", "RUNS_DB")
    decisions_ref = _get_any(decisions_name, "decisions", "DECISIONS_DB")

    tasks_repo = TasksRepo(
        notion_client=notion_client,
        db=tasks_ref,
        schema=schema,
        repo_key=tasks_ref.name,
        schema_map=schema_map,
    )
    proposals_repo = ProposalsRepo(
        notion_client=notion_client,
        db=proposals_ref,
        schema=schema,
        repo_key=proposals_ref.name,
        schema_map=schema_map,
    )
    runs_repo = RunsRepo(
        notion_client=notion_client,
        db=runs_ref,
        schema=schema,
        repo_key=runs_ref.name,
        schema_map=schema_map,
    )
    decisions_repo = DecisionsRepo(
        notion_client=notion_client,
        db=decisions_ref,
        schema=schema,
        repo_key=decisions_ref.name,
        schema_map=schema_map,
    )

    by_key: Dict[str, BaseRepo] = {}

    # Put core repos into by_key as well (handy for generic flows)
    by_key[tasks_ref.name] = tasks_repo
    by_key[proposals_ref.name] = proposals_repo
    by_key[runs_ref.name] = runs_repo
    by_key[decisions_ref.name] = decisions_repo

    # Generic repos for arbitrary DB keys (recommended for domain DBs)
    if extra_repo_keys:
        for k in extra_repo_keys:
            if not k:
                continue
            try:
                ref = resolved_registry.get(k)
            except Exception:
                # keep it non-fatal; caller might pass optional DB keys
                continue
            by_key[k] = BaseRepo(
                notion_client=notion_client,
                db=ref,
                schema=schema,
                repo_key=ref.name,   # align with resolved db name
                schema_map=schema_map,

            )

    return NotionRepos(
        tasks=tasks_repo,
        proposals=proposals_repo,
        runs=runs_repo,
        decisions=decisions_repo,
        by_key=by_key if by_key else None,
    )


def build_repos_from_resolved_db(
    *,
    notion_client: Any,
    resolved_db: dict,
    schema: Optional[NotionSchema] = None,
    schema_map: Optional[SchemaMap] = None,
    **kwargs: Any,
) -> NotionRepos:
    if not isinstance(resolved_db, dict) or not resolved_db:
        raise ValueError("build_repos_from_resolved_db(): resolved_db (dict) is required and must be non-empty.")
    return build_repos(
        notion_client=notion_client,
        resolved_registry=ResolvedDBRegistry(resolved_db),
        schema=schema,
        schema_map=schema_map,
        **kwargs,
    )


# ============================================================
# Convenience: create Run payload from NotebookRunResult
# ============================================================

def run_payload_from_notebook_result(
    *,
    run_id: str,
    task_page_id: str,
    proposal_page_ids: Optional[List[str]],
    run_type: RunType,
    notebook_path: str,
    nb_result: Any,  # NotebookRunResult from src/exec/nb_runner.py
    artifacts_path: Optional[str] = None,
    branch_ref: Optional[str] = None,
    commit_before: Optional[str] = None,
    commit_after: Optional[str] = None,
    phase: Optional[str] = "VERIFY",
) -> dict:
    def _clip(s: Any, limit: int) -> Optional[str]:
        if s is None:
            return None
        t = str(s).strip()
        if not t:
            return None
        if len(t) <= limit:
            return t
        return t[:limit].rstrip() + "…(truncated)"

    NOTION_RT_LIMIT = 1900
    NOTION_SUMMARY_LIMIT = 800

    # Best-effort: keep FAIL vs TIMEOUT if runner surfaced it
    if getattr(nb_result, "ok"):
        status: RunStatus = "PASS"
    else:
        # If upstream provided a hint (e.g., error_summary contains Timeout)
        es = str(getattr(nb_result, "error_summary", "") or "")
        status = "TIMEOUT" if "Timeout" in es or "time out" in es.lower() else "FAIL"

    started_at = datetime.fromtimestamp(getattr(nb_result, "started_at_unix"), tz=timezone.utc)
    finished_at = datetime.fromtimestamp(getattr(nb_result, "finished_at_unix"), tz=timezone.utc)

    evidence = nb_result.to_evidence_text() if hasattr(nb_result, "to_evidence_text") else None

    err_sum = _clip(getattr(nb_result, "error_summary", None), NOTION_SUMMARY_LIMIT)
    err_short = _clip(getattr(nb_result, "error_trace_short", None), NOTION_RT_LIMIT)
    ev_txt = _clip(evidence, NOTION_RT_LIMIT)

    return dict(
        run_id=run_id,
        status=status,
        run_type=run_type,
        phase=phase,
        task_page_id=task_page_id,
        proposal_page_ids=proposal_page_ids,
        notebook_path=notebook_path,
        up_to_cell=getattr(nb_result, "executed_up_to_cell", None),
        failing_cell=getattr(nb_result, "failing_cell_index", None),
        error_summary=err_sum,
        error_trace_short=err_short,
        artifacts_path=artifacts_path,
        branch_ref=branch_ref,
        commit_before=commit_before,
        commit_after=commit_after,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=getattr(nb_result, "duration_sec", None),
        evidence=ev_txt,
        next_action=None,
    )
