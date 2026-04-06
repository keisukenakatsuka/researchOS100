#!/usr/bin/env python3
"""Paper management CLI — create, track, and review research papers.

Usage:
    python -m src.scripts.140_paper_manager <subcommand> [options]

Subcommands:
    init        Create a new paper
    ls          List all papers
    show        Show paper details
    edit        Edit paper metadata
    stage       Transition to a new stage
    set-status  Change status within current stage
    stages      Show stage history
    link        Link a run_id to a paper
    decide      Record a research decision
    log         Show decision history
    task        Manage tasks (add / update / done)
    tasks       List open tasks
    overview    Regenerate current_status.md
    history     Show unified timeline
    dashboard   Show all-papers dashboard
"""

from __future__ import annotations

import argparse
import sys

from src.paper import registry, stage as stage_mod, decision as decision_mod, task as task_mod
from src.paper import overview as overview_mod, history as history_mod, dashboard as dashboard_mod
from src.paper.models import (
    DECISION_TYPES,
    PAPER_TYPES,
    STAGES,
    STATUSES,
    TASK_OWNERS,
    TASK_PRIORITIES,
    TASK_SOURCES,
)


def cmd_init(args: argparse.Namespace) -> None:
    paper = registry.create_paper(
        paper_id=args.paper_id,
        title=args.title,
        paper_type=args.type,
        rq=args.rq or "",
        target_journal=args.journal or "",
        authors=[a.strip() for a in args.authors.split(",")] if args.authors else [],
        stage=args.stage,
        status=args.status,
        data_dir=args.data_dir or "",
        tags=[t.strip() for t in args.tags.split(",")] if args.tags else [],
        notes=args.notes or "",
    )

    # If initial stage is not "idea", record a stage transition
    if args.stage != "idea":
        stage_mod.transition_stage(
            paper.paper_id,
            args.stage,
            entry_reason=f"Initialized at stage '{args.stage}'",
            source="migration",
        )

    print(f"Created paper '{paper.paper_id}'")
    print(f"  Stage: {paper.current_stage} | Status: {paper.current_status}")
    print(f"  Dir:   data/papers/{paper.paper_id}/")


def cmd_ls(args: argparse.Namespace) -> None:
    entries = registry.list_papers()
    if not entries:
        print("No papers found.")
        return

    # Header
    print()
    print(f"  {'Paper':<30s} {'Type':<12s} {'Stage':<20s} {'Status':<10s} {'Tasks':<8s} {'Updated':<12s}")
    print(f"  {'─' * 92}")
    for e in entries:
        updated = e.updated_at[:10] if e.updated_at else ""
        tasks_str = f"{e.open_tasks} open" if e.open_tasks else "—"
        print(
            f"  {e.paper_id:<30s} {e.paper_type:<12s} {e.current_stage:<20s} "
            f"{e.current_status:<10s} {tasks_str:<8s} {updated:<12s}"
        )
    print()


def cmd_show(args: argparse.Namespace) -> None:
    paper = registry.get_paper(args.paper_id)
    print()
    print(f"  Paper:   {paper.paper_id}")
    print(f"  Title:   {paper.title}")
    print(f"  Type:    {paper.paper_type}")
    print(f"  RQ:      {paper.rq}")
    print(f"  Journal: {paper.target_journal}")
    print(f"  Authors: {', '.join(paper.authors) if paper.authors else '—'}")
    print(f"  Stage:   {paper.current_stage} | Status: {paper.current_status}")
    print(f"  Created: {paper.created_at}")
    print(f"  Updated: {paper.updated_at}")
    if paper.run_ids:
        print(f"  Runs:    {', '.join(paper.run_ids)}")
    if paper.data_dir:
        print(f"  Data:    {paper.data_dir}")
    if paper.tags:
        print(f"  Tags:    {', '.join(paper.tags)}")
    if paper.notes:
        print(f"  Notes:   {paper.notes}")
    print()


def cmd_edit(args: argparse.Namespace) -> None:
    fields = {}
    if args.title is not None:
        fields["title"] = args.title
    if args.rq is not None:
        fields["rq"] = args.rq
    if args.journal is not None:
        fields["target_journal"] = args.journal
    if args.notes is not None:
        fields["notes"] = args.notes
    if args.type is not None:
        if args.type not in PAPER_TYPES:
            print(f"Error: Invalid type '{args.type}'. Must be one of {PAPER_TYPES}")
            sys.exit(1)
        fields["paper_type"] = args.type
    if args.data_dir is not None:
        fields["data_dir"] = args.data_dir

    if not fields:
        print("No fields to update. Use --title, --rq, --journal, --notes, --type, --data-dir.")
        return

    paper = registry.update_paper(args.paper_id, **fields)
    print(f"Updated paper '{paper.paper_id}': {', '.join(fields.keys())}")


def cmd_stage(args: argparse.Namespace) -> None:
    source = "migration" if args.at else "cli"
    record, warnings = stage_mod.transition_stage(
        paper_id=args.paper_id,
        to_stage=args.new_stage,
        entry_reason=args.entry_reason,
        exit_reason=args.exit_reason,
        effective_at=args.at,
        source=source,
    )

    for w in warnings:
        print(w)

    src_label = f" (source={source})" if source == "migration" else ""
    print(
        f"  {record.from_stage or '—'} → {record.to_stage}{src_label}"
    )
    if source == "migration":
        print(f"  effective_at: {record.effective_at}")
        print(f"  recorded_at:  {record.recorded_at}")


def cmd_set_status(args: argparse.Namespace) -> None:
    record = stage_mod.change_status(
        paper_id=args.paper_id,
        to_status=args.new_status,
        reason=args.reason,
    )
    print(
        f"  [{record.stage}] {record.from_status} → {record.to_status}"
    )


def cmd_stages(args: argparse.Namespace) -> None:
    records = stage_mod.get_stage_history(args.paper_id)
    if not records:
        print("No stage history.")
        return

    print()
    for r in records:
        d = r.to_dict()
        ts = d.get("effective_at", "")[:19]
        source_tag = f" [{d.get('source', '')}]" if d.get("source") == "migration" else ""

        if d["type"] == "stage_transition":
            from_s = d.get("from_stage") or "—"
            to_s = d["to_stage"]
            print(f"  {ts}  [stage] {from_s} → {to_s} ({d['to_status']}){source_tag}")
            if d.get("entry_reason"):
                print(f"              Entry: {d['entry_reason']}")
            if d.get("exit_reason"):
                print(f"              Exit:  {d['exit_reason']}")
            if d.get("gate_result"):
                gate = d["gate_result"]
                status = "✅" if gate.get("passed") else "❌"
                print(f"              Gate {status} {gate.get('phase', '')}")
        elif d["type"] == "status_change":
            print(
                f"  {ts}  [status] {d['stage']}: "
                f"{d['from_status']} → {d['to_status']}{source_tag}"
            )
            if d.get("reason"):
                print(f"              Reason: {d['reason']}")
    print()


def cmd_link(args: argparse.Namespace) -> None:
    paper = registry.link_run(args.paper_id, args.run_id)
    print(f"Linked run '{args.run_id}' to paper '{paper.paper_id}'")
    print(f"  Runs: {', '.join(paper.run_ids)}")


# ---------------------------------------------------------------------------
# Phase B: Decision commands
# ---------------------------------------------------------------------------


def _parse_rejected(raw_list: list[str] | None) -> list[dict[str, str]]:
    """Parse --rejected "option:reason" into dicts."""
    if not raw_list:
        return []
    result = []
    for item in raw_list:
        if ":" not in item:
            result.append({"option": item, "rejection_reason": ""})
        else:
            opt, reason = item.split(":", 1)
            result.append({"option": opt.strip(), "rejection_reason": reason.strip()})
    return result


def cmd_decide(args: argparse.Namespace) -> None:
    source = "migration" if args.at else "cli"
    rejected = _parse_rejected(args.rejected)
    refs = [r.strip() for r in args.refs.split(",")] if args.refs else []

    record = decision_mod.add_decision(
        paper_id=args.paper_id,
        decision=args.decision,
        reason=args.reason,
        decision_type=args.type,
        rejected_alternatives=rejected,
        refs=refs,
        effective_at=args.at,
        source=source,
    )

    print(f"  [{record.id}] {record.decision_type}: {record.decision}")
    if record.rejected_alternatives:
        for ra in record.rejected_alternatives:
            print(f"    ❌ {ra.option} — {ra.rejection_reason}")
    if source == "migration":
        print(f"  effective_at: {record.effective_at}")


def cmd_log(args: argparse.Namespace) -> None:
    decisions = decision_mod.get_decisions(args.paper_id)
    if not decisions:
        print("No decisions recorded.")
        return

    print()
    for d in decisions:
        ts = d.effective_at[:10] if d.effective_at else "?"
        source_tag = " [migration]" if d.source == "migration" else ""
        type_tag = f" ({d.decision_type})" if d.decision_type != "other" else ""
        print(f"  {ts}  [{d.id}]{type_tag}{source_tag}")
        print(f"          ✅ {d.decision}")
        print(f"          Reason: {d.reason}")
        for ra in d.rejected_alternatives:
            reason_str = f" — {ra.rejection_reason}" if ra.rejection_reason else ""
            print(f"          ❌ {ra.option}{reason_str}")
        if d.refs:
            print(f"          Refs: {', '.join(d.refs)}")
        print()


# ---------------------------------------------------------------------------
# Phase B: Task commands
# ---------------------------------------------------------------------------


def cmd_task(args: argparse.Namespace) -> None:
    """Dispatch task subcommands: add / update / done."""
    if args.task_action == "add":
        _cmd_task_add(args)
    elif args.task_action == "update":
        _cmd_task_update(args)
    elif args.task_action == "done":
        _cmd_task_done(args)
    else:
        print("Unknown task action. Use: add / update / done")
        sys.exit(1)


def _cmd_task_add(args: argparse.Namespace) -> None:
    deps = [d.strip() for d in args.depends_on.split(",")] if args.depends_on else []
    task = task_mod.add_task(
        paper_id=args.paper_id,
        content=args.content,
        source=args.source,
        owner=args.owner,
        priority=args.priority,
        linked_stage=args.linked_stage or "",
        linked_decision=args.linked_decision,
        next_action=args.next_action or "",
        depends_on=deps,
        due=args.due,
        blocked_reason=args.blocked_reason or "",
    )
    blocked_str = ""
    if task.status == "blocked":
        blocked_str = f" (blocked: {task.blocked_reason})" if task.blocked_reason else " (blocked)"
    print(f"  [{task.id}] {task.content}")
    print(f"    owner={task.owner} priority={task.priority} status={task.status}{blocked_str}")


def _cmd_task_update(args: argparse.Namespace) -> None:
    fields = {}
    if args.status is not None:
        fields["status"] = args.status
    if args.priority is not None:
        fields["priority"] = args.priority
    if args.next_action is not None:
        fields["next_action"] = args.next_action
    if args.owner is not None:
        fields["owner"] = args.owner
    if args.blocked_reason is not None:
        fields["blocked_reason"] = args.blocked_reason

    if not fields:
        print("No fields to update. Use --status, --priority, --next-action, --owner, --blocked-reason.")
        return

    task = task_mod.update_task(args.paper_id, args.task_id, **fields)
    print(f"  [{task.id}] status={task.status} priority={task.priority}")


def _cmd_task_done(args: argparse.Namespace) -> None:
    task = task_mod.complete_task(args.paper_id, args.task_id)
    print(f"  [{task.id}] ✅ done")


def cmd_tasks(args: argparse.Namespace) -> None:
    """List open tasks, optionally filtered by paper."""
    if args.paper_id:
        tasks = task_mod.get_open_tasks(args.paper_id)
        label = args.paper_id
    else:
        tasks = task_mod.get_all_open_tasks()
        label = "All papers"

    if not tasks:
        print(f"No open tasks ({label}).")
        return

    # Sort: critical > high > medium > low, then by id
    prio_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    tasks.sort(key=lambda t: (prio_order.get(t.priority, 9), t.id))

    print()
    print(f"  Open tasks — {label}")
    print()
    print(f"  {'#':<6s} {'Task':<50s} {'Owner':<8s} {'Priority':<9s} {'Status':<9s} {'Blocked'}")
    print(f"  {'─' * 100}")
    for t in tasks:
        prefix = f"{t.paper_id}/" if not args.paper_id else ""
        blocked_str = ""
        if t.status == "blocked":
            if t.blocked_reason:
                blocked_str = t.blocked_reason[:40]
            elif t.depends_on:
                blocked_str = f"← {', '.join(t.depends_on)}"
        content = t.content[:48] + ".." if len(t.content) > 50 else t.content
        print(
            f"  {prefix}{t.id:<6s} {content:<50s} {t.owner:<8s} "
            f"{t.priority:<9s} {t.status:<9s} {blocked_str}"
        )
    print()


# ---------------------------------------------------------------------------
# Phase C: Overview, History, Dashboard commands
# ---------------------------------------------------------------------------


def cmd_overview(args: argparse.Namespace) -> None:
    path = overview_mod.regenerate_status(args.paper_id)
    print(f"  Generated: {path}")
    print()
    print(path.read_text())


def cmd_history(args: argparse.Namespace) -> None:
    entries = history_mod.get_timeline(args.paper_id)
    if not entries:
        print("No history.")
        return
    print(history_mod.format_timeline(args.paper_id, entries))


def cmd_dashboard(args: argparse.Namespace) -> None:
    print(dashboard_mod.show_dashboard())


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="140_paper_manager",
        description="Research paper management CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    # --- init ---
    p_init = sub.add_parser("init", help="Create a new paper")
    p_init.add_argument("paper_id", help="Unique paper ID (slug)")
    p_init.add_argument("--title", required=True, help="Paper title")
    p_init.add_argument("--type", default="empirical", choices=sorted(PAPER_TYPES), help="Paper type")
    p_init.add_argument("--rq", help="Research question")
    p_init.add_argument("--journal", help="Target journal")
    p_init.add_argument("--authors", help="Comma-separated author names")
    p_init.add_argument("--stage", default="idea", choices=STAGES, help="Initial stage")
    p_init.add_argument("--status", default="active", choices=sorted(STATUSES), help="Initial status")
    p_init.add_argument("--data-dir", help="Path to empirical data directory")
    p_init.add_argument("--tags", help="Comma-separated tags")
    p_init.add_argument("--notes", help="Free-form notes")

    # --- ls ---
    sub.add_parser("ls", help="List all papers")

    # --- show ---
    p_show = sub.add_parser("show", help="Show paper details")
    p_show.add_argument("paper_id", help="Paper ID")

    # --- edit ---
    p_edit = sub.add_parser("edit", help="Edit paper metadata")
    p_edit.add_argument("paper_id", help="Paper ID")
    p_edit.add_argument("--title", help="New title")
    p_edit.add_argument("--rq", help="New RQ")
    p_edit.add_argument("--journal", help="New target journal")
    p_edit.add_argument("--notes", help="New notes")
    p_edit.add_argument("--type", choices=sorted(PAPER_TYPES), help="New paper type")
    p_edit.add_argument("--data-dir", help="New data directory path")

    # --- stage ---
    p_stage = sub.add_parser("stage", help="Transition to a new stage")
    p_stage.add_argument("paper_id", help="Paper ID")
    p_stage.add_argument("new_stage", choices=STAGES, help="Target stage")
    p_stage.add_argument("--entry-reason", required=True, help="Why entering this stage")
    p_stage.add_argument("--exit-reason", help="Why leaving the previous stage")
    p_stage.add_argument(
        "--at",
        help="Effective timestamp (ISO 8601). Migration/bootstrap only.",
    )

    # --- set-status ---
    p_status = sub.add_parser("set-status", help="Change status within current stage")
    p_status.add_argument("paper_id", help="Paper ID")
    p_status.add_argument("new_status", choices=sorted(STATUSES), help="New status")
    p_status.add_argument("--reason", required=True, help="Reason for status change")

    # --- stages ---
    p_stages = sub.add_parser("stages", help="Show stage history")
    p_stages.add_argument("paper_id", help="Paper ID")

    # --- link ---
    p_link = sub.add_parser("link", help="Link a run_id to a paper")
    p_link.add_argument("paper_id", help="Paper ID")
    p_link.add_argument("run_id", help="Run ID (must exist in data/lit_review/)")

    # --- decide ---
    p_decide = sub.add_parser("decide", help="Record a research decision")
    p_decide.add_argument("paper_id", help="Paper ID")
    p_decide.add_argument("--decision", required=True, help="What was decided")
    p_decide.add_argument("--reason", required=True, help="Why this was decided")
    p_decide.add_argument("--type", default="other", choices=sorted(DECISION_TYPES), help="Decision type")
    p_decide.add_argument("--rejected", action="append", help="Rejected alternative (option:reason). Repeatable.")
    p_decide.add_argument("--refs", help="Comma-separated file references")
    p_decide.add_argument("--at", help="Effective timestamp (ISO 8601). Migration only.")

    # --- log ---
    p_log = sub.add_parser("log", help="Show decision history")
    p_log.add_argument("paper_id", help="Paper ID")

    # --- task ---
    p_task = sub.add_parser("task", help="Manage tasks (add / update / done)")
    task_sub = p_task.add_subparsers(dest="task_action", help="Task action")

    # task add
    p_task_add = task_sub.add_parser("add", help="Add a new task")
    p_task_add.add_argument("paper_id", help="Paper ID")
    p_task_add.add_argument("--content", required=True, help="Task description")
    p_task_add.add_argument("--source", required=True, choices=sorted(TASK_SOURCES), help="Task source")
    p_task_add.add_argument("--owner", required=True, choices=sorted(TASK_OWNERS), help="Who does this")
    p_task_add.add_argument("--priority", required=True, choices=sorted(TASK_PRIORITIES), help="Priority")
    p_task_add.add_argument("--next-action", help="Concrete next step")
    p_task_add.add_argument("--depends-on", help="Comma-separated task IDs")
    p_task_add.add_argument("--linked-stage", help="Related stage")
    p_task_add.add_argument("--linked-decision", help="Related decision ID")
    p_task_add.add_argument("--due", help="Due date (ISO)")
    p_task_add.add_argument("--blocked-reason", help="Why this task is blocked (manual block)")

    # task update
    p_task_upd = task_sub.add_parser("update", help="Update task fields")
    p_task_upd.add_argument("paper_id", help="Paper ID")
    p_task_upd.add_argument("task_id", help="Task ID (e.g. t001)")
    p_task_upd.add_argument("--status", choices=sorted({"open", "in_progress", "done", "blocked", "wontfix"}))
    p_task_upd.add_argument("--priority", choices=sorted(TASK_PRIORITIES))
    p_task_upd.add_argument("--next-action")
    p_task_upd.add_argument("--owner", choices=sorted(TASK_OWNERS))
    p_task_upd.add_argument("--blocked-reason", help="Why blocked (set with --status blocked)")

    # task done
    p_task_done = task_sub.add_parser("done", help="Mark task as done")
    p_task_done.add_argument("paper_id", help="Paper ID")
    p_task_done.add_argument("task_id", help="Task ID")

    # --- tasks ---
    p_tasks = sub.add_parser("tasks", help="List open tasks")
    p_tasks.add_argument("paper_id", nargs="?", default=None, help="Paper ID (omit for all papers)")

    # --- overview ---
    p_overview = sub.add_parser("overview", help="Regenerate current_status.md")
    p_overview.add_argument("paper_id", help="Paper ID")

    # --- history ---
    p_history = sub.add_parser("history", help="Show unified timeline")
    p_history.add_argument("paper_id", help="Paper ID")

    # --- dashboard ---
    sub.add_parser("dashboard", help="Show all-papers dashboard")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "init": cmd_init,
        "ls": cmd_ls,
        "show": cmd_show,
        "edit": cmd_edit,
        "stage": cmd_stage,
        "set-status": cmd_set_status,
        "stages": cmd_stages,
        "link": cmd_link,
        "decide": cmd_decide,
        "log": cmd_log,
        "task": cmd_task,
        "tasks": cmd_tasks,
        "overview": cmd_overview,
        "history": cmd_history,
        "dashboard": cmd_dashboard,
    }

    try:
        dispatch[args.command](args)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
