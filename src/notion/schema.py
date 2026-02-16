# src/notion/schema.py
"""
Notion schema definitions + lightweight validation helpers.

Goal
----
Centralize:
- Property names used in TASKS_DB / PROPOSALS_DB / RUNS_DB / DECISIONS_DB
- Enum values (Status, Priority, etc.)
- Optional runtime checks to catch "property name mismatch" early

This module intentionally does NOT call Notion API.
Validation here is against "expected property names/enums" that you configured
in Notion, and (optionally) the database property metadata you fetched elsewhere.

Usage
-----
from src.notion.schema import (
    NotionSchema, NotionEnums, NotionDBIds, validate_db_schema
)

schema = NotionSchema()
enums = NotionEnums()

# Later, after you fetch database metadata via client.retrieve_database(...)
# validate_db_schema(db_meta, expected_props=set(schema.tasks_props().values()), db_label="TASKS_DB")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Mapping, Any, Literal


# -----------------------------
# DB IDs container
# -----------------------------

@dataclass(frozen=True)
class NotionDBIds:
    tasks_db_id: str
    proposals_db_id: str
    runs_db_id: str
    decisions_db_id: str


# -----------------------------
# Property name schema (single source of truth)
# -----------------------------

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
    TASK_RISK_LEVEL: str = "Risk Level"                   # optional
    TASK_BUDGET: str = "Budget"                           # optional

    # PROPOSALS_DB (ChangeSet)
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

    # ---- convenience ----

    def tasks_props(self) -> Dict[str, str]:
        return {
            "TITLE": self.TASK_TITLE,
            "STATUS": self.TASK_STATUS,
            "PRIORITY": self.TASK_PRIORITY,
            "DOMAIN": self.TASK_DOMAIN,
            "OBJECTIVE": self.TASK_OBJECTIVE,
            "AC": self.TASK_AC,
            "CONSTRAINTS": self.TASK_CONSTRAINTS,
            "SCOPE": self.TASK_SCOPE,
            "ENTRY": self.TASK_ENTRY,
            "RUN_POLICY": self.TASK_RUN_POLICY,
            "OWNER": self.TASK_OWNER,
            "RUNS_REL": self.TASK_RUNS_REL,
            "PROPOSALS_REL": self.TASK_PROPOSALS_REL,
            "LATEST_RUN_REL": self.TASK_LATEST_RUN_REL,
            "LATEST_PROPOSAL_REL": self.TASK_LATEST_PROPOSAL_REL,
            "RISK_LEVEL": self.TASK_RISK_LEVEL,
            "BUDGET": self.TASK_BUDGET,
        }

    def proposals_props(self) -> Dict[str, str]:
        return {
            "TITLE": self.PROP_TITLE,
            "STATUS": self.PROP_STATUS,
            "TASK_REL": self.PROP_TASK_REL,
            "OWNER": self.PROP_OWNER,
            "NOTEBOOK_PATH": self.PROP_NOTEBOOK_PATH,
            "CELL_INDEX": self.PROP_CELL_INDEX,
            "INTENT": self.PROP_INTENT,
            "ACCEPTANCE": self.PROP_ACCEPTANCE,
            "DIFF_SUMMARY": self.PROP_DIFF_SUMMARY,
            "PATCH_PATH": self.PROP_PATCH_PATH,
            "ARTIFACTS_PATH": self.PROP_ARTIFACTS_PATH,
            "COMMIT": self.PROP_COMMIT,
            "RISK": self.PROP_RISK,
            "ROLLBACK": self.PROP_ROLLBACK,
            "RUNS_REL": self.PROP_RUNS_REL,
            "LAST_RUN_REL": self.PROP_LAST_RUN_REL,
            "FAILURE_REASON": self.PROP_FAILURE_REASON,
            "NEXT_ACTION": self.PROP_NEXT_ACTION,
        }

    def runs_props(self) -> Dict[str, str]:
        return {
            "TITLE": self.RUN_TITLE,
            "STATUS": self.RUN_STATUS,
            "TYPE": self.RUN_TYPE,
            "PHASE": self.RUN_PHASE,
            "TASK_REL": self.RUN_TASK_REL,
            "PROPOSAL_REL": self.RUN_PROPOSAL_REL,
            "NOTEBOOK_PATH": self.RUN_NOTEBOOK_PATH,
            "UP_TO_CELL": self.RUN_UP_TO_CELL,
            "FAILING_CELL": self.RUN_FAILING_CELL,
            "ERROR_SUMMARY": self.RUN_ERROR_SUMMARY,
            "ERROR_TRACE": self.RUN_ERROR_TRACE,
            "ARTIFACTS_PATH": self.RUN_ARTIFACTS_PATH,
            "LOGS_LINK": self.RUN_LOGS_LINK,
            "BRANCH_REF": self.RUN_BRANCH_REF,
            "COMMIT_BEFORE": self.RUN_COMMIT_BEFORE,
            "COMMIT_AFTER": self.RUN_COMMIT_AFTER,
            "STARTED_AT": self.RUN_STARTED_AT,
            "FINISHED_AT": self.RUN_FINISHED_AT,
            "DURATION_SEC": self.RUN_DURATION_SEC,
            "EVIDENCE": self.RUN_EVIDENCE,
            "NEXT_ACTION": self.RUN_NEXT_ACTION,
        }

    def decisions_props(self) -> Dict[str, str]:
        return {
            "TITLE": self.DEC_TITLE,
            "STATUS": self.DEC_STATUS,
            "TASK_REL": self.DEC_TASK_REL,
            "RUN_REL": self.DEC_RUN_REL,
            "PROPOSAL_REL": self.DEC_PROPOSAL_REL,
            "RATIONALE": self.DEC_RATIONALE,
            "ALTERNATIVES": self.DEC_ALTERNATIVES,
            "IMPACT_RISK": self.DEC_IMPACT_RISK,
            "OWNER": self.DEC_OWNER,
        }

    def all_expected_props(self) -> Dict[str, Set[str]]:
        return {
            "TASKS_DB": set(self.tasks_props().values()),
            "PROPOSALS_DB": set(self.proposals_props().values()),
            "RUNS_DB": set(self.runs_props().values()),
            "DECISIONS_DB": set(self.decisions_props().values()),
        }


# -----------------------------
# Enum values (select options)
# -----------------------------

TaskStatus = Literal["READY", "RUNNING", "REVIEW", "DONE", "BLOCKED", "ARCHIVED"]
TaskPriority = Literal["P0", "P1", "P2", "P3"]
TaskDomain = Literal["Daily", "Weekly", "Targets", "Papers", "Events", "RQ", "Infra"]

ProposalStatus = Literal["DRAFT", "APPLIED", "VERIFIED", "FAILED", "REVERTED", "MERGED"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]

RunStatus = Literal["PASS", "FAIL", "ABORTED", "TIMEOUT"]
RunType = Literal["EXECUTE_PREFIX", "EXECUTE_FULL", "RUN_TESTS", "LINT"]
RunPhase = Literal["PLAN", "PATCH", "EXECUTE", "VERIFY", "REPORT"]

DecisionStatus = Literal["PROPOSED", "ACCEPTED", "REJECTED", "REVERTED"]


@dataclass(frozen=True)
class NotionEnums:
    task_status: List[str] = ("READY RUNNING REVIEW DONE BLOCKED ARCHIVED").split()
    task_priority: List[str] = ("P0 P1 P2 P3").split()
    task_domain: List[str] = ("Daily Weekly Targets Papers Events RQ Infra").split()

    proposal_status: List[str] = ("DRAFT APPLIED VERIFIED FAILED REVERTED MERGED").split()
    risk_level: List[str] = ("LOW MEDIUM HIGH").split()

    run_status: List[str] = ("PASS FAIL ABORTED TIMEOUT").split()
    run_type: List[str] = ("EXECUTE_PREFIX EXECUTE_FULL RUN_TESTS LINT").split()
    run_phase: List[str] = ("PLAN PATCH EXECUTE VERIFY REPORT").split()

    decision_status: List[str] = ("PROPOSED ACCEPTED REJECTED REVERTED").split()


# -----------------------------
# Lightweight schema validation helpers
# -----------------------------

class SchemaValidationError(ValueError):
    pass


def validate_db_schema(
    db_meta: Mapping[str, Any],
    *,
    expected_props: Set[str],
    db_label: str,
    allow_missing: Optional[Set[str]] = None,
) -> None:
    """
    Validate that a Notion database has the required properties.

    Parameters
    ----------
    db_meta:
        Output of Notion API retrieve_database(database_id).
        Must include `properties` mapping.
    expected_props:
        Set of property names we expect to exist in the DB.
    db_label:
        Label for error messages.
    allow_missing:
        Properties that are optional in your deployment (e.g. Latest Run).
    """
    allow_missing = allow_missing or set()
    props = db_meta.get("properties", {})
    if not isinstance(props, dict):
        raise SchemaValidationError(f"{db_label}: db_meta.properties missing or invalid")

    existing = set(props.keys())
    required = expected_props - allow_missing
    missing = sorted(required - existing)
    if missing:
        raise SchemaValidationError(
            f"{db_label}: missing properties: {missing}. "
            f"Existing: {sorted(existing)}"
        )


def validate_select_options(
    db_meta: Mapping[str, Any],
    *,
    prop_name: str,
    expected_options: List[str],
    db_label: str,
) -> None:
    """
    Validate that a select property includes expected option names.

    Note:
    - Notion allows extra options; we only check that expected ones exist.
    """
    props = db_meta.get("properties", {})
    p = props.get(prop_name)
    if not p:
        raise SchemaValidationError(f"{db_label}: property not found: {prop_name}")

    ptype = p.get("type")
    if ptype != "select":
        raise SchemaValidationError(f"{db_label}: {prop_name} is not select (type={ptype})")

    options = p.get("select", {}).get("options", [])
    existing = set(o.get("name") for o in options if isinstance(o, dict))
    missing = [o for o in expected_options if o not in existing]
    if missing:
        raise SchemaValidationError(
            f"{db_label}: select {prop_name} missing options: {missing}. "
            f"Existing: {sorted(existing)}"
        )


def validate_multi_select_options(
    db_meta: Mapping[str, Any],
    *,
    prop_name: str,
    expected_options: List[str],
    db_label: str,
) -> None:
    props = db_meta.get("properties", {})
    p = props.get(prop_name)
    if not p:
        raise SchemaValidationError(f"{db_label}: property not found: {prop_name}")

    ptype = p.get("type")
    if ptype != "multi_select":
        raise SchemaValidationError(f"{db_label}: {prop_name} is not multi_select (type={ptype})")

    options = p.get("multi_select", {}).get("options", [])
    existing = set(o.get("name") for o in options if isinstance(o, dict))
    missing = [o for o in expected_options if o not in existing]
    if missing:
        raise SchemaValidationError(
            f"{db_label}: multi_select {prop_name} missing options: {missing}. "
            f"Existing: {sorted(existing)}"
        )


def validate_relation_property(
    db_meta: Mapping[str, Any],
    *,
    prop_name: str,
    db_label: str,
) -> None:
    props = db_meta.get("properties", {})
    p = props.get(prop_name)
    if not p:
        raise SchemaValidationError(f"{db_label}: property not found: {prop_name}")

    ptype = p.get("type")
    if ptype != "relation":
        raise SchemaValidationError(f"{db_label}: {prop_name} is not relation (type={ptype})")


def validate_full_schema(
    *,
    tasks_db_meta: Mapping[str, Any],
    proposals_db_meta: Mapping[str, Any],
    runs_db_meta: Mapping[str, Any],
    decisions_db_meta: Mapping[str, Any],
    schema: Optional[NotionSchema] = None,
    enums: Optional[NotionEnums] = None,
    optional_task_props: Optional[Set[str]] = None,
    optional_proposal_props: Optional[Set[str]] = None,
    optional_run_props: Optional[Set[str]] = None,
    optional_decision_props: Optional[Set[str]] = None,
) -> None:
    """
    Validate all DB schemas + core enum options.

    Use this once at startup to fail fast if Notion DB columns don't match.
    """
    schema = schema or NotionSchema()
    enums = enums or NotionEnums()

    optional_task_props = optional_task_props or {
        schema.TASK_LATEST_RUN_REL,
        schema.TASK_LATEST_PROPOSAL_REL,
        schema.TASK_RISK_LEVEL,
        schema.TASK_BUDGET,
    }
    optional_proposal_props = optional_proposal_props or {schema.PROP_LAST_RUN_REL}
    optional_run_props = optional_run_props or {
        schema.RUN_PHASE,
        schema.RUN_LOGS_LINK,
        schema.RUN_BRANCH_REF,
        schema.RUN_COMMIT_BEFORE,
        schema.RUN_COMMIT_AFTER,
    }
    optional_decision_props = optional_decision_props or {
        schema.DEC_RUN_REL,
        schema.DEC_PROPOSAL_REL,
        schema.DEC_ALTERNATIVES,
        schema.DEC_IMPACT_RISK,
    }

    # Property existence checks
    validate_db_schema(tasks_db_meta, expected_props=schema.all_expected_props()["TASKS_DB"], db_label="TASKS_DB", allow_missing=optional_task_props)
    validate_db_schema(proposals_db_meta, expected_props=schema.all_expected_props()["PROPOSALS_DB"], db_label="PROPOSALS_DB", allow_missing=optional_proposal_props)
    validate_db_schema(runs_db_meta, expected_props=schema.all_expected_props()["RUNS_DB"], db_label="RUNS_DB", allow_missing=optional_run_props)
    validate_db_schema(decisions_db_meta, expected_props=schema.all_expected_props()["DECISIONS_DB"], db_label="DECISIONS_DB", allow_missing=optional_decision_props)

    # Core select options checks (only the most important ones)
    validate_select_options(tasks_db_meta, prop_name=schema.TASK_STATUS, expected_options=enums.task_status, db_label="TASKS_DB")
    validate_select_options(tasks_db_meta, prop_name=schema.TASK_PRIORITY, expected_options=enums.task_priority, db_label="TASKS_DB")
    validate_select_options(tasks_db_meta, prop_name=schema.TASK_DOMAIN, expected_options=enums.task_domain, db_label="TASKS_DB")

    validate_select_options(proposals_db_meta, prop_name=schema.PROP_STATUS, expected_options=enums.proposal_status, db_label="PROPOSALS_DB")
    validate_select_options(proposals_db_meta, prop_name=schema.PROP_RISK, expected_options=enums.risk_level, db_label="PROPOSALS_DB")

    validate_select_options(runs_db_meta, prop_name=schema.RUN_STATUS, expected_options=enums.run_status, db_label="RUNS_DB")
    validate_select_options(runs_db_meta, prop_name=schema.RUN_TYPE, expected_options=enums.run_type, db_label="RUNS_DB")

    # Decisions status
    validate_select_options(decisions_db_meta, prop_name=schema.DEC_STATUS, expected_options=enums.decision_status, db_label="DECISIONS_DB")
