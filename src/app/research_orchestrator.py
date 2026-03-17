#!/usr/bin/env python
"""researchOS Web UI Orchestrator (Streamlit v0).

Provides Browser-based access to the Research Cycle pipeline.

Screens:
  1. Dashboard — run list with progress
  2. Run Detail — pipeline step view
  3. Artifact Viewer — markdown/json renderer

Usage::

    streamlit run src/app/research_orchestrator.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Ensure project root is on path
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.lit_review.run_manifest import (
    load_manifest,
    list_runs,
    get_artifacts,
    SCRIPT_LABELS,
    SCRIPT_BLOCKS,
    SCRIPT_DEPS,
    SCRIPT_OUTPUTS,
    STATUS_COMPLETED,
)

_DATA_DIR = _PROJECT_ROOT / "data" / "lit_review"


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="researchOS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------

def sidebar():
    st.sidebar.title("🔬 researchOS")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Run Detail", "Artifact Viewer", "Writeback", "New Run"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Data: `{_DATA_DIR}`")

    return page


# ------------------------------------------------------------------
# Screen 1: Dashboard
# ------------------------------------------------------------------

def render_dashboard():
    st.title("📊 Dashboard")

    runs = list_runs(_DATA_DIR)

    if not runs:
        st.info("No runs found. Create a run with `python -m src.scripts.079_rq_paper_matcher --full-pipeline`")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", len(runs))
    completed = sum(1 for r in runs if r["status"] == "completed")
    col2.metric("Completed", completed)
    in_progress = sum(1 for r in runs if r["status"] == "in_progress")
    col3.metric("In Progress", in_progress)

    st.markdown("---")

    # Header
    hdr = st.columns([2, 3, 2, 2, 1, 1])
    hdr[0].caption("Run ID")
    hdr[1].caption("Research Question")
    hdr[2].caption("Progress")
    hdr[3].caption("Latest Step")
    hdr[4].caption("Updated")
    hdr[5].caption("Status")

    # Run table
    for run in runs:
        with st.container():
            cols = st.columns([2, 3, 2, 2, 1, 1])

            # Run ID (truncated)
            run_id_short = run["run_id"][:20] + "..."
            cols[0].markdown(f"**`{run_id_short}`**")

            # RQ title
            rq_short = run["rq_title"][:50] + ("..." if len(run["rq_title"]) > 50 else "")
            cols[1].markdown(rq_short)

            # Progress
            progress = run["completed_steps"] / max(run["total_steps"], 1)
            cols[2].progress(progress, text=f"{run['completed_steps']}/{run['total_steps']}")

            # Latest step
            latest = run.get("latest_step", "")
            latest_label = SCRIPT_LABELS.get(latest, latest[:15] if latest else "—")
            cols[3].markdown(f"📍 {latest_label}")

            # Last updated
            last_upd = run.get("last_updated", "")[:16].replace("T", " ")
            cols[4].caption(last_upd if last_upd else "—")

            # Status
            status_icons = {
                "completed": "✅", "in_progress": "🔄",
                "running": "⏳", "partial": "⚠️", "pending": "⬜",
            }
            icon = status_icons.get(run["status"], "❓")
            cols[5].markdown(f"{icon}")

            st.markdown("---")


# ------------------------------------------------------------------
# Screen 2: Run Detail
# ------------------------------------------------------------------

def render_run_detail():
    st.title("🔍 Run Detail")

    runs = list_runs(_DATA_DIR)
    if not runs:
        st.info("No runs found.")
        return

    # Run selector
    run_options = {f"{r['run_id'][:20]}... — {r['rq_title'][:40]}": r["run_id"] for r in runs}
    selected_label = st.selectbox("Select Run", list(run_options.keys()))
    if not selected_label:
        return

    run_id = run_options[selected_label]
    run_dir = _DATA_DIR / run_id
    manifest = load_manifest(run_dir)

    # RQ info
    st.markdown(f"### RQ: {manifest.get('rq_title', '')}")
    st.caption(f"Run ID: `{run_id}`")

    st.markdown("---")

    # Pipeline view
    st.markdown("### Pipeline")

    steps = manifest.get("steps", {})
    runnable = set(manifest.get("runnable_steps", []))
    optional = set(manifest.get("optional_steps", []))
    blocked = set(manifest.get("blocked_steps", []))

    # Display steps in dependency order
    ordered_scripts = list(SCRIPT_DEPS.keys())

    for script_id in ordered_scripts:
        step_info = steps.get(script_id, {})
        status = step_info.get("status", "")
        label = SCRIPT_LABELS.get(script_id, script_id)
        block = SCRIPT_BLOCKS.get(script_id.split("_")[0], "")
        outputs = step_info.get("outputs", [])
        can_run = script_id in runnable or script_id in optional

        # Status icon
        if status == STATUS_COMPLETED:
            icon = "✅"
        elif status == "running":
            icon = "⏳"
        elif status == "failed":
            icon = "❌"
        elif script_id in runnable:
            icon = "▶️"
        elif script_id in optional:
            icon = "⭕"
        elif script_id in blocked:
            icon = "🔒"
        else:
            icon = "⬜"

        col1, col2, col3, col4, col5 = st.columns([1, 3, 1.5, 2, 1.5])

        col1.markdown(f"{icon}")
        col2.markdown(f"**{label}** `{script_id.split('_')[0]}`")
        col3.markdown(f"_{block}_" if block else "")

        # Status / info
        if status == STATUS_COMPLETED and outputs:
            md_count = sum(1 for f in outputs if f.endswith(".md"))
            col4.markdown(f"📄 {md_count} artifacts" if md_count else f"✓ {len(outputs)} files")
        elif status == "failed":
            col4.markdown(f"❌ Failed")
        elif script_id in blocked:
            deps = SCRIPT_DEPS.get(script_id, [])
            col4.caption(f"🔒 Needs: {', '.join(deps)}")
        elif can_run:
            col4.markdown("🟢 Ready" if script_id in runnable else "⚪ Optional")

        # [Run] button for runnable steps (with multi-run guard)
        is_executing = st.session_state.get("executing", False)
        if can_run and status != STATUS_COMPLETED and not is_executing:
            btn_key = f"run_{script_id}_{run_id}"
            if col5.button("▶ Run", key=btn_key):
                st.session_state["executing"] = True
                _execute_step(run_id, script_id, run_dir)
                st.session_state["executing"] = False
                st.rerun()
        elif status == STATUS_COMPLETED:
            completed_at = step_info.get("completed_at", "")
            ts = completed_at[:16].replace("T", " ") if completed_at else ""
            col5.caption(f"✓ {ts}" if ts else "done")
        elif is_executing:
            col5.caption("⏳")

    # Sequential runner
    st.markdown("---")
    runnable_list = manifest.get("runnable_steps", [])
    is_executing = st.session_state.get("executing", False)
    if runnable_list and not is_executing:
        st.markdown(f"### Run Pipeline ({len(runnable_list)} steps ready)")
        col_a, col_b = st.columns(2)
        next_label = SCRIPT_LABELS.get(runnable_list[0], runnable_list[0])
        if col_a.button(f"▶ Run Next: {next_label}", key=f"run_next_{run_id}"):
            st.session_state["executing"] = True
            _execute_step(run_id, runnable_list[0], run_dir)
            st.session_state["executing"] = False
            st.rerun()
        if len(runnable_list) > 1:
            if col_b.button(f"▶▶ Run All ({len(runnable_list)} steps)", key=f"run_all_{run_id}"):
                st.session_state["executing"] = True
                _execute_sequential(run_id, runnable_list, run_dir)
                st.session_state["executing"] = False
                st.rerun()
    elif is_executing:
        st.info("⏳ Execution in progress...")

    # Execution log (last results)
    exec_log = st.session_state.get("execution_log", [])
    if exec_log:
        st.markdown("---")
        st.markdown("### Execution Log")
        for entry in reversed(exec_log[-5:]):
            exit_code = entry["exit_code"]
            label = SCRIPT_LABELS.get(entry["script_id"], entry["script_id"])
            duration = entry.get("duration", "?")

            if exit_code == 0:
                st.success(f"✅ **{label}** — completed in {duration}s")
            elif exit_code == 2:
                st.warning(f"⚠️ **{label}** — partial (exit {exit_code}, {duration}s)")
            else:
                st.error(f"❌ **{label}** — failed (exit {exit_code}, {duration}s)")

            with st.expander(f"Details: {entry['script_id']}"):
                st.caption(f"Command: `{entry.get('command', '')}`")
                st.caption(f"Started: {entry.get('started_at', '')} | Ended: {entry.get('ended_at', '')}")
                if entry.get("stdout"):
                    st.code(entry["stdout"][-2000:], language="text")
                if entry.get("stderr"):
                    st.code(entry["stderr"][-1000:], language="text")

    # Writeback status
    st.markdown("---")
    st.markdown("### Writeback Status")

    wb_step = steps.get("084_lit_review_writeback", {})
    if wb_step.get("status") == STATUS_COMPLETED:
        st.success("✅ Writeback completed")
    else:
        st.warning("⚠️ Writeback not executed. Run `084 --writeback` to persist to Notion.")


# ------------------------------------------------------------------
# Step Executor
# ------------------------------------------------------------------

def _build_command(run_id: str, script_id: str) -> list[str]:
    """Build subprocess command for a script."""
    module_name = f"src.scripts.{script_id}"
    cmd = [sys.executable, "-m", module_name, "--run-id", run_id]

    # Special args
    if script_id == "079_rq_paper_matcher":
        cmd.append("--full-pipeline")

    return cmd


def _execute_step(run_id: str, script_id: str, run_dir: Path):
    """Execute a single pipeline step via subprocess."""
    from datetime import datetime, timezone

    cmd = _build_command(run_id, script_id)
    cmd_str = " ".join(cmd)
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_PROJECT_ROOT),
        )
        ended_at = datetime.now(timezone.utc).isoformat()

        # Calculate duration
        from datetime import datetime as dt
        start = dt.fromisoformat(started_at)
        end = dt.fromisoformat(ended_at)
        duration = round((end - start).total_seconds(), 1)

        entry = {
            "script_id": script_id,
            "command": cmd_str,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration": duration,
        }

    except subprocess.TimeoutExpired as te:
        partial_out = te.stdout or "" if hasattr(te, "stdout") else ""
        entry = {
            "script_id": script_id,
            "command": cmd_str,
            "exit_code": -1,
            "stdout": partial_out,
            "stderr": f"⏱ Timeout: execution exceeded 600 seconds. The script may still be producing partial results.",
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration": 600,
        }
    except FileNotFoundError:
        entry = {
            "script_id": script_id,
            "command": cmd_str,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"🔍 Script not found: src.scripts.{script_id}. Check that the script exists.",
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration": 0,
        }
    except Exception as e:
        entry = {
            "script_id": script_id,
            "command": cmd_str,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"💥 Unexpected error: {type(e).__name__}: {e}",
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration": 0,
        }

    # Append to execution log
    if "execution_log" not in st.session_state:
        st.session_state["execution_log"] = []
    st.session_state["execution_log"].append(entry)

    return entry


def _execute_sequential(run_id: str, steps_to_run: list[str], run_dir: Path):
    """Execute multiple steps sequentially, stopping on failure."""
    results = []
    for script_id in steps_to_run:
        entry = _execute_step(run_id, script_id, run_dir)
        results.append(entry)

        # Stop on non-zero exit (but continue on partial=2)
        if entry["exit_code"] not in (0, 2):
            break

    return results


# ------------------------------------------------------------------
# Screen 3: Artifact Viewer
# ------------------------------------------------------------------

def render_artifact_viewer():
    st.title("📄 Artifact Viewer")

    runs = list_runs(_DATA_DIR)
    if not runs:
        st.info("No runs found.")
        return

    # Run selector
    run_options = {f"{r['run_id'][:20]}... — {r['rq_title'][:40]}": r["run_id"] for r in runs}
    selected_label = st.selectbox("Select Run", list(run_options.keys()), key="artifact_run")
    if not selected_label:
        return

    run_id = run_options[selected_label]
    run_dir = _DATA_DIR / run_id

    artifacts = get_artifacts(run_dir)
    if not artifacts:
        st.info("No artifacts found in this run.")
        return

    # Artifact selector
    md_artifacts = [a for a in artifacts if a["type"] == "markdown"]
    json_artifacts = [a for a in artifacts if a["type"] == "json"]
    graph_artifacts = [a for a in artifacts if a["type"] == "graph"]

    tabs = ["📝 Markdown", "📊 JSON"]
    if graph_artifacts:
        tabs.append("🔗 Graph")

    tab_objs = st.tabs(tabs)

    with tab_objs[0]:
        if md_artifacts:
            options = [f"{a['name']} ({a['size'] // 1024}KB)" for a in md_artifacts]
            selected_idx = st.selectbox(
                "Select file",
                range(len(options)),
                format_func=lambda i: options[i],
                key="md_select",
            )
            if selected_idx is not None:
                artifact = md_artifacts[selected_idx]
                content = Path(artifact["path"]).read_text()
                word_count = len(content.split())
                st.caption(f"📄 {artifact['name']} — {word_count} words, {artifact['size'] // 1024}KB")
                st.markdown(content)
        else:
            st.info("No Markdown artifacts.")

    with tab_objs[1]:
        if json_artifacts:
            options_j = [f"{a['name']} ({a['size'] // 1024}KB)" for a in json_artifacts]
            selected_j_idx = st.selectbox(
                "Select file",
                range(len(options_j)),
                format_func=lambda i: options_j[i],
                key="json_select",
            )
            if selected_j_idx is not None:
                artifact = json_artifacts[selected_j_idx]
                st.caption(f"📊 {artifact['name']} — {artifact['size'] // 1024}KB")
                content = json.loads(Path(artifact["path"]).read_text())
                st.json(content)
        else:
            st.info("No JSON artifacts.")

    if graph_artifacts and len(tab_objs) > 2:
        with tab_objs[2]:
            options_g = [f"{a['name']} ({a['size'] // 1024}KB)" for a in graph_artifacts]
            selected_g_idx = st.selectbox(
                "Select graph",
                range(len(options_g)),
                format_func=lambda i: options_g[i],
                key="graph_select",
            )
            if selected_g_idx is not None:
                artifact = graph_artifacts[selected_g_idx]
                st.caption(f"🔗 {artifact['name']} — {artifact['size'] // 1024}KB")
                import streamlit.components.v1 as components
                html_content = Path(artifact["path"]).read_text()
                components.html(html_content, height=850, scrolling=True)


# ------------------------------------------------------------------
# Screen 4: Writeback Control
# ------------------------------------------------------------------

# Scripts that support --writeback
WRITEBACK_SCRIPTS = {
    "084_lit_review_writeback": {
        "label": "Evidence + Claims + Memos + Research Run",
        "description": "Block 3 outputs → KML (Evidence DB, Claims DB, Memos DB, Research Runs DB)",
        "entities": ["Evidence", "Claims (run-local)", "Memos (Lit Review + Landscape)", "Research Run"],
    },
    "086_claim_canonicalization": {
        "label": "Canonical Claims",
        "description": "Cross-run Claims → Canonical Claims in Claims DB",
        "entities": ["Canonical Claims"],
        "extra_args": [],  # needs --run-ids, handled specially
    },
    "087_hypothesis_generator": {
        "label": "Hypotheses",
        "description": "Generated hypotheses → Claims DB (type=hypothesis)",
        "entities": ["Hypotheses"],
    },
    "088_assumption_analyzer": {
        "label": "Assumptions",
        "description": "Critical + significant assumptions → Claims DB (type=assumption)",
        "entities": ["Assumptions (critical + significant)"],
    },
}


def render_writeback():
    st.title("📤 Writeback Control")

    runs = list_runs(_DATA_DIR)
    if not runs:
        st.info("No runs found.")
        return

    run_options = {f"{r['run_id'][:20]}... — {r['rq_title'][:40]}": r["run_id"] for r in runs}
    selected_label = st.selectbox("Select Run", list(run_options.keys()), key="wb_run")
    if not selected_label:
        return

    run_id = run_options[selected_label]
    run_dir = _DATA_DIR / run_id
    manifest = load_manifest(run_dir)
    steps = manifest.get("steps", {})

    st.markdown("---")

    # Check env
    import os
    wb_enabled = os.environ.get("ENABLE_NOTION_WRITEBACK", "").lower() == "true"
    if wb_enabled:
        st.success("🟢 ENABLE_NOTION_WRITEBACK = true")
    else:
        st.warning("🔴 ENABLE_NOTION_WRITEBACK is not 'true'. Writeback will be blocked by scripts.")
        st.caption("Set `ENABLE_NOTION_WRITEBACK=true` in `notebooks/env.txt` to enable.")

    st.markdown("---")

    # Writeback status per script
    for script_id, wb_info in WRITEBACK_SCRIPTS.items():
        step_status = steps.get(script_id, {}).get("status", "")
        completed_at = steps.get(script_id, {}).get("completed_at", "")

        # Check if prerequisite outputs exist
        has_prereqs = True
        if script_id == "084_lit_review_writeback":
            has_prereqs = (run_dir / "evidence.json").exists()
        elif script_id == "087_hypothesis_generator":
            has_prereqs = (run_dir / "hypotheses.json").exists()
        elif script_id == "088_assumption_analyzer":
            has_prereqs = (run_dir / "assumptions.json").exists()

        col1, col2, col3, col4 = st.columns([3, 3, 2, 2])

        # Label
        col1.markdown(f"**{wb_info['label']}**")

        # Entities
        col2.caption(", ".join(wb_info["entities"]))

        # Status
        if step_status == STATUS_COMPLETED:
            ts = completed_at[:16].replace("T", " ") if completed_at else ""
            col3.markdown(f"✅ {ts}")
        elif not has_prereqs:
            col3.caption("⬜ No data")
        else:
            col3.caption("⬜ Not run")

        # Action
        if script_id == "086_claim_canonicalization":
            col4.caption("CLI only")  # needs --run-ids
        elif has_prereqs and wb_enabled:
            btn_key = f"wb_{script_id}_{run_id}"
            if col4.button("📤 Write", key=btn_key):
                _execute_writeback(run_id, script_id, run_dir)
                st.rerun()
        elif has_prereqs and not wb_enabled:
            col4.caption("🔒 Disabled")
        else:
            col4.caption("—")

    # Writeback execution result
    if "wb_result" in st.session_state:
        result = st.session_state.pop("wb_result")
        st.markdown("---")
        if result["exit_code"] == 0:
            st.success(f"✅ Writeback `{result['script_id']}` completed ({result['duration']}s)")
        else:
            st.error(f"❌ Writeback `{result['script_id']}` failed")
        with st.expander("Details"):
            st.code(result.get("stdout", "")[-2000:])


def _execute_writeback(run_id: str, script_id: str, run_dir: Path):
    """Execute a writeback step."""
    from datetime import datetime, timezone

    cmd = [sys.executable, "-m", f"src.scripts.{script_id}", "--run-id", run_id, "--writeback"]
    started = datetime.now(timezone.utc)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(_PROJECT_ROOT))
        duration = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
        st.session_state["wb_result"] = {
            "script_id": script_id, "exit_code": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr, "duration": duration,
        }
    except Exception as e:
        st.session_state["wb_result"] = {
            "script_id": script_id, "exit_code": -1,
            "stdout": "", "stderr": str(e), "duration": 0,
        }


# ------------------------------------------------------------------
# Screen 5: New Run
# ------------------------------------------------------------------

def render_new_run():
    st.title("➕ New Run")

    st.markdown("Create a new research run by specifying an RQ.")

    tab_text, tab_notion = st.tabs(["📝 Text Input", "📋 From Notion RQ DB"])

    with tab_text:
        rq_text = st.text_area(
            "Research Question (free text)",
            height=100,
            key="new_rq_text",
            placeholder="例: LP投資者として行動する政府系ファンド（SWF）は、国内スタートアップ・エコシステムの形成にどのような影響を与えるか？",
        )
        if st.button("Create Run", key="create_text"):
            text = rq_text.strip()
            if not text:
                st.warning("⚠️ Please enter a Research Question.")
            elif len(text) < 10:
                st.warning("⚠️ RQ is too short. Please enter a more specific question.")
            else:
                run_info = _create_run(rq_text=text)
                st.success(f"✅ Run created: `{run_info['run_id']}`")
                st.caption(f"Directory: `{run_info['run_dir']}`")
                st.info("Go to **Run Detail** → select this run → **▶ Run Next** to start the pipeline.")

    with tab_notion:
        st.caption("Load RQs from Notion RQ DB")
        if st.button("Load RQs", key="load_rqs"):
            try:
                rqs = _load_notion_rqs()
                st.session_state["available_rqs"] = rqs
            except Exception as e:
                st.error(f"Failed to load RQs: {e}")

        available_rqs = st.session_state.get("available_rqs", [])
        if available_rqs:
            rq_options = {f"[{r['priority']}] {r['title'][:60]}": r for r in available_rqs}
            selected = st.selectbox("Select RQ", list(rq_options.keys()), key="notion_rq_select")
            if selected and st.button("Create Run from Selected RQ", key="create_notion"):
                rq = rq_options[selected]
                run_info = _create_run(rq_dict=rq)
                st.success(f"✅ Run created: `{run_info['run_id']}`")
                st.caption(f"RQ: {rq['title']}")
                st.info("Go to **Run Detail** to start the pipeline.")


def _create_run(*, rq_text: str = "", rq_dict: dict = None) -> dict:
    """Create a new run directory with rq_context.json and run_manifest.json."""
    import uuid
    from datetime import datetime, timezone

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = _DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build RQ context
    if rq_dict:
        from src.lit_review.rq_context import RQContext
        ctx = RQContext.from_notion_rq(rq_dict)
        rq_data = ctx.to_dict()
    else:
        from src.lit_review.rq_context import RQContext
        ctx = RQContext.from_text(rq_text)
        rq_data = ctx.to_dict()

    # Save rq_context.json
    (run_dir / "rq_context.json").write_text(json.dumps(rq_data, ensure_ascii=False, indent=2))

    # Initialize run_manifest.json
    from src.lit_review.run_manifest import load_manifest, _save_manifest
    manifest = load_manifest(run_dir)
    _save_manifest(run_dir, manifest)

    return {"run_id": run_id, "run_dir": str(run_dir)}


def _load_notion_rqs() -> list[dict]:
    """Load RQs from Notion RQ DB."""
    from src.config import load_env, get_db_id
    from src.notion import build_notion_client_from_env, NotionDataSourceResolver
    from src.notion.rq_normalize import normalize_rqs

    load_env()
    nc = build_notion_client_from_env()
    resolver = NotionDataSourceResolver(nc)
    db_id = get_db_id("NOTION_RQ_DB_ID")
    resolved = resolver.resolve_once(name="RQ_DB", database_id=db_id)
    pages = nc.query_data_source(data_source_id=resolved.data_source_id, fetch_all=True)
    rqs = normalize_rqs(pages)

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    rqs.sort(key=lambda r: (priority_order.get(r.get("priority", ""), 9), r.get("title", "")))
    return rqs


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    page = sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Run Detail":
        render_run_detail()
    elif page == "Artifact Viewer":
        render_artifact_viewer()
    elif page == "Writeback":
        render_writeback()
    elif page == "New Run":
        render_new_run()


if __name__ == "__main__":
    main()
