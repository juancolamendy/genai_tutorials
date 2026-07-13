"""Handlers for the triage-and-fix harness.

Gate: AWAIT_APPROVAL (wait_kind=either). No LangGraph interrupt().
Handlers return partial deltas; current_state/status stamped by the engine.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Literal, Optional

from src.engine.handler_registry import handler

from .artifacts import ArtifactStore
from .clients import gather_context
from .session_state import TriageState
from .state_transitions import State
from .worker import FakeWorker, Worker, WorkerTask
from . import worktree

log = logging.getLogger(__name__)

# Injected by build_graph / tests
_WORKER: Worker = FakeWorker()


def set_worker(worker: Worker) -> None:
    global _WORKER
    _WORKER = worker


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _store(state: TriageState) -> ArtifactStore:
    return ArtifactStore(state.get("artifacts_dir") or ".triage_artifacts")


def _log_cmd(cmd: str, detail: str = "") -> list[dict[str, Any]]:
    return [{"ts": _now(), "cmd": cmd, "detail": detail}]


def _resolve_decision(state: TriageState) -> Optional[Literal["approve", "reject"]]:
    """Option C: system event_type approve|reject, or human text/payload."""
    event_type = (state.get("current_event_type") or "").lower()
    if event_type in ("approve", "reject"):
        return event_type  # type: ignore[return-value]

    # Human message path (event_type=message) or either-path text
    text = (state.get("input_message") or "").strip().lower()
    if text.startswith("reject") or text == "no":
        return "reject"
    if text.startswith("approve") or text == "yes":
        return "approve"

    # Structured fields if a system payload merged them
    if state.get("approved") is False:
        return "reject"
    if state.get("approved") is True:
        return "approve"
    return None


@handler(state=State.INTAKE.value, waits_for_input=False, description="Normalize trigger into a run")
def handle_intake(state: TriageState) -> dict[str, Any]:
    run_id = state.get("run_id") or state.get("session_id") or "run-unknown"
    trigger = state.get("trigger") or {"source": "sentry", "issue": state.get("inputs", {}).get("issue")}
    log.info("[HANDLER] intake  run_id=%s  trigger=%s", run_id, trigger)
    return {
        "run_id": run_id,
        "run_status": "running",
        "trigger": trigger,
        "artifacts": {},
        "command_log": _log_cmd("intake", f"trigger={trigger}"),
        "output_messages": [f"Started triage run {run_id}"],
    }


@handler(
    state=State.GATHER_CONTEXT.value,
    waits_for_input=False,
    description="Gather Sentry/GitHub/Linear context into artifacts",
)
def handle_gather_context(state: TriageState) -> dict[str, Any]:
    run_id = state["run_id"]
    store = _store(state)
    paths = gather_context(run_id, state.get("inputs") or {}, store)
    log.info("[HANDLER] gather_context  artifacts=%s", list(paths))
    return {
        "artifacts": {**(state.get("artifacts") or {}), **paths},
        "command_log": _log_cmd("gather_context", f"clients={list(paths)}"),
    }


@handler(
    state=State.EXECUTE_PATCH.value,
    waits_for_input=False,
    description="Run worker inside isolated worktree",
)
def handle_execute_patch(state: TriageState) -> dict[str, Any]:
    run_id = state["run_id"]
    repo = (state.get("inputs") or {}).get("repo") or "."
    wt = worktree.create_worktree(repo, run_id)
    worker = _WORKER
    result = worker.run(
        WorkerTask(
            run_id=run_id,
            worktree=wt,
            artifact_paths=state.get("artifacts") or {},
            instruction="Diagnose the Sentry issue and fix the root cause.",
        )
    )
    store = _store(state)
    summary_path = store.write(run_id, "worker_summary.md", result.summary)
    log.info("[HANDLER] execute_patch  worker=%s  worktree=%s", worker.name, wt)
    return {
        "worktree_path": wt,
        "artifacts": {**(state.get("artifacts") or {}), "worker_summary": summary_path},
        "command_log": _log_cmd(f"worker:{worker.name}", result.summary),
    }


@handler(
    state=State.COLLECT_DIFF.value,
    waits_for_input=False,
    description="Collect diff artifact and park for approval",
)
def handle_collect_diff(state: TriageState) -> dict[str, Any]:
    wt = state.get("worktree_path") or ""
    diff = worktree.collect_diff(wt)
    path = _store(state).write(state["run_id"], "changes.patch", diff)
    log.info("[HANDLER] collect_diff  path=%s  bytes=%d", path, len(diff))
    return {
        "diff_path": path,
        "run_status": "awaiting_approval",
        "artifacts": {**(state.get("artifacts") or {}), "diff": path},
        "command_log": _log_cmd("collect_diff", f"{len(diff)} bytes"),
        "output_messages": [
            f"Awaiting approval for run {state['run_id']}. Diff: {path}"
        ],
    }


@handler(
    state=State.AWAIT_APPROVAL.value,
    waits_for_input=True,
    wait_kind="either",
    expected_events=["approve", "reject"],
    description="Human or system approval gate (no interrupt())",
)
def handle_await_approval(state: TriageState) -> dict[str, Any]:
    decision = _resolve_decision(state)
    by = state.get("approval_by") or state.get("by") or "unknown"
    note = state.get("approval_note") or state.get("note") or ""
    if decision is None:
        # Shouldn't normally resume without a decision; treat as reject-safe no-op
        # by asking again via output — stay decision unset so guardrail won't publish.
        log.warning("[HANDLER] await_approval  no decision resolved — defaulting unset")
        return {
            "output_messages": ["Send approve or reject to continue."],
            "command_log": _log_cmd("await_approval", "no decision"),
        }

    log.info("[HANDLER] await_approval  decision=%s  by=%s", decision, by)
    return {
        "approval_decision": decision,
        "approval_by": by,
        "approval_note": note,
        "approvals": [
            {
                "ts": _now(),
                "by": by,
                "approved": decision == "approve",
                "note": note,
            }
        ],
        "run_status": "running" if decision == "approve" else "rejected",
        "command_log": _log_cmd("await_approval", f"decision={decision} by={by}"),
    }


@handler(state=State.PUBLISH.value, waits_for_input=False, description="Publish approved patch")
def handle_publish(state: TriageState) -> dict[str, Any]:
    if state.get("published"):
        log.info("[HANDLER] publish  skip — already published (idempotent)")
        return {
            "run_status": "published",
            "command_log": _log_cmd("publish", "already published"),
        }
    wt = state.get("worktree_path") or ""
    sha = worktree.publish(
        wt,
        state["run_id"],
        message=f"harness({state['run_id']}): automated fix (approved)",
    )
    log.info("[HANDLER] publish  commit=%s", sha)
    return {
        "published": True,
        "run_status": "published",
        "command_log": _log_cmd("publish", f"commit={sha}"),
        "output_messages": [f"Published fix commit={sha}"],
    }


@handler(state=State.REJECT.value, waits_for_input=False, description="Record rejection")
def handle_reject(state: TriageState) -> dict[str, Any]:
    log.info("[HANDLER] reject  discarding changes")
    return {
        "run_status": "rejected",
        "command_log": _log_cmd("reject", "changes discarded"),
        "output_messages": ["Changes rejected — nothing published."],
    }


@handler(
    state=State.WRITE_REPORT.value,
    waits_for_input=False,
    description="Write HTML report artifact",
)
def handle_write_report(state: TriageState) -> dict[str, Any]:
    rows = "".join(
        f"<tr><td>{c.get('ts')}</td><td>{c.get('cmd')}</td><td>{c.get('detail')}</td></tr>"
        for c in state.get("command_log") or []
    )
    arts = "".join(
        f"<li><code>{n}</code>: {p}</li>" for n, p in (state.get("artifacts") or {}).items()
    )
    html = (
        f"<h1>Run {state.get('run_id')} — {state.get('run_status')}</h1>"
        f"<h2>Artifacts</h2><ul>{arts}</ul>"
        f"<h2>Command log</h2><table>{rows}</table>"
    )
    path = _store(state).write(state["run_id"], "report.html", html)
    log.info("[HANDLER] write_report  path=%s", path)
    return {
        "report_path": path,
        "artifacts": {**(state.get("artifacts") or {}), "report": path},
        "command_log": _log_cmd("write_report", path),
    }


@handler(state=State.COMPLETE.value, waits_for_input=False, description="Terminal success")
def handle_complete(state: TriageState) -> dict[str, Any]:
    return {
        "command_log": _log_cmd("complete", state.get("run_status") or "published"),
        "output_messages": [f"Run {state.get('run_id')} complete ({state.get('run_status')})."],
    }


@handler(state=State.REJECTED.value, waits_for_input=False, description="Terminal rejection")
def handle_rejected(state: TriageState) -> dict[str, Any]:
    return {
        "command_log": _log_cmd("rejected", state.get("approval_note") or ""),
        "output_messages": [f"Run {state.get('run_id')} rejected."],
    }


@handler(state=State.ERROR.value, waits_for_input=False, description="Terminal error")
def handle_error(state: TriageState) -> dict[str, Any]:
    log.error("[HANDLER] error  reason=%s", state.get("error_message"))
    return {
        "run_status": "failed",
        "command_log": _log_cmd("error", state.get("error_message") or "unknown"),
    }


handler_map = {
    State.INTAKE: handle_intake,
    State.GATHER_CONTEXT: handle_gather_context,
    State.EXECUTE_PATCH: handle_execute_patch,
    State.COLLECT_DIFF: handle_collect_diff,
    State.AWAIT_APPROVAL: handle_await_approval,
    State.PUBLISH: handle_publish,
    State.REJECT: handle_reject,
    State.WRITE_REPORT: handle_write_report,
    State.COMPLETE: handle_complete,
    State.REJECTED: handle_rejected,
    State.ERROR: handle_error,
}

__all__ = [
    "handler_map",
    "set_worker",
    "handle_intake",
    "handle_gather_context",
    "handle_execute_patch",
    "handle_collect_diff",
    "handle_await_approval",
    "handle_publish",
    "handle_reject",
    "handle_write_report",
    "handle_complete",
    "handle_rejected",
    "handle_error",
]
