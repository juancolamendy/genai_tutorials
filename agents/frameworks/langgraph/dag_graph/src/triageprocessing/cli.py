"""CLI for the triage-and-fix harness (automation review surface).

  python -m src.triageprocessing.cli run --repo /path --issue PROJ-142 --thread-id thread-1
  python -m src.triageprocessing.cli pending              # uses last thread_id
  python -m src.triageprocessing.cli approve --by alice
  python -m src.triageprocessing.cli reject --by alice --note "..."
  python -m src.triageprocessing.cli report
  python -m src.triageprocessing.cli chat "approve"       # human either-path

Gate resume uses aemit_event (Option C: human or system), never interrupt().
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from src.triageprocessing.graph import Graph, build_graph
from src.triageprocessing.state_transitions import State

log = logging.getLogger(__name__)

DEFAULT_SESSIONS_DIR = ".triage_sessions"
DEFAULT_ARTIFACTS_DIR = ".triage_artifacts"
CURRENT_THREAD_FILENAME = ".current_thread_id"
# Backward-compatible alias from the earlier .current_run_id name
_LEGACY_CURRENT_RUN_FILENAME = ".current_run_id"


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if not verbose:
        for name in ("httpx", "httpcore", "anthropic", "openai"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _current_thread_path(sessions_dir: str) -> Path:
    return Path(sessions_dir) / CURRENT_THREAD_FILENAME


def save_current_thread_id(sessions_dir: str, thread_id: str) -> None:
    path = _current_thread_path(sessions_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(thread_id.strip() + "\n", encoding="utf-8")
    log.info("[CLI] saved current thread_id=%s → %s", thread_id, path)


def load_current_thread_id(sessions_dir: str) -> Optional[str]:
    path = _current_thread_path(sessions_dir)
    if path.is_file():
        thread_id = path.read_text(encoding="utf-8").strip()
        if thread_id:
            return thread_id
    legacy = Path(sessions_dir) / _LEGACY_CURRENT_RUN_FILENAME
    if legacy.is_file():
        thread_id = legacy.read_text(encoding="utf-8").strip()
        return thread_id or None
    return None


def resolve_thread_id(sessions_dir: str, thread_id: Optional[str]) -> str:
    """Explicit thread_id wins; otherwise use the last id saved by `run`."""
    if thread_id:
        return thread_id
    current = load_current_thread_id(sessions_dir)
    if current:
        return current
    raise SystemExit(
        "No thread_id given and no current thread saved. "
        "Run `… cli run --repo …` first, or pass thread_id explicitly."
    )


# Back-compat aliases for tests / imports
save_current_run_id = save_current_thread_id
load_current_run_id = load_current_thread_id
resolve_run_id = resolve_thread_id


def _format_result(result: dict[str, Any]) -> str:
    return (
        f"emit_status={result.get('emit_status')} "
        f"current_state={result.get('current_state')} "
        f"run_status={result.get('run_status')} "
        f"report={result.get('report_path')}"
    )


async def _cmd_run(
    graph: Graph, sessions_dir: str, repo: str, issue: str, thread_id: str | None
) -> str:
    thread_id = thread_id or f"run-{uuid.uuid4().hex[:8]}"
    log.info("[CLI] run  thread_id=%s  repo=%s  issue=%s", thread_id, repo, issue)
    result = await graph.arun_to_completion(
        user_id="",
        session_id=thread_id,
        initial_state_delta={
            "run_id": thread_id,
            "trigger": {"source": "sentry", "issue": issue},
            "inputs": {"repo": repo, "issue": issue},
            "artifacts_dir": graph.artifacts_dir,
        },
    )
    save_current_thread_id(sessions_dir, thread_id)
    line = f"thread_id={thread_id}\n{_format_result(result)}"
    if result.get("emit_status") == "blocked_needs_input":
        line += f"\nawaiting approval — diff: {result.get('diff_path')}"
    return line


def _cmd_pending(graph: Graph, thread_id: str) -> str:
    state = graph._get_or_init_state(session_id=thread_id, user_id="")
    current = state.get("current_state")
    if current != State.AWAIT_APPROVAL.value:
        return f"{thread_id}: not awaiting approval (current_state={current})"
    return (
        f"kind=publish_approval\n"
        f"thread_id={thread_id}\n"
        f"summary_artifact={state.get('artifacts', {}).get('worker_summary')}\n"
        f"diff_artifact={state.get('diff_path')}\n"
        f"options=approve,reject\n"
        f"wait_kind=either"
    )


async def _cmd_decide(
    graph: Graph,
    thread_id: str,
    decision: str,
    by: str,
    note: str,
    *,
    as_system: bool,
) -> str:
    event_id = f"triage:{thread_id}:{decision}"
    if as_system:
        result = await graph.aemit_event(
            thread_id=thread_id,
            event_source="system",
            event_type=decision,
            event_id=event_id,
            payload={"by": by, "note": note, "approved": decision == "approve"},
        )
    else:
        result = await graph.aemit_event(
            thread_id=thread_id,
            event_source="human",
            event_type="message",
            input_message=decision,
        )
    log.info("[CLI] %s  result=%s", decision, _format_result(result))
    return f"{thread_id}: {_format_result(result)}"


async def _cmd_chat(graph: Graph, thread_id: str, message: str) -> str:
    result = await graph.aemit_event(
        thread_id=thread_id,
        event_source="human",
        event_type="message",
        input_message=message,
    )
    return f"{thread_id}: {_format_result(result)}"


def _cmd_report(graph: Graph, thread_id: str) -> str:
    state = graph._get_or_init_state(session_id=thread_id, user_id="")
    return state.get("report_path") or "no report yet"


def _cmd_status(graph: Graph, thread_id: str) -> str:
    state = graph._get_or_init_state(session_id=thread_id, user_id="")
    return (
        f"thread={thread_id} current_state={state.get('current_state')} "
        f"run_status={state.get('run_status')} session_status={state.get('session_status')}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="triageprocessing-cli")
    p.add_argument("--sessions-dir", default=DEFAULT_SESSIONS_DIR)
    p.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run")
    r.add_argument("--repo", required=True)
    r.add_argument("--issue", default="PROJ-142")
    r.add_argument(
        "--thread-id",
        "--run-id",
        dest="thread_id",
        default=None,
        help="EngineGraph thread/session id (default: generate run-<hex>)",
    )

    for name in ("pending", "report", "status"):
        sp = sub.add_parser(name)
        sp.add_argument(
            "thread_id",
            nargs="?",
            default=None,
            help="Defaults to the thread id saved by the last `run` command",
        )

    a = sub.add_parser("approve")
    a.add_argument("thread_id", nargs="?", default=None)
    a.add_argument("--by", required=True)
    a.add_argument("--note", default="")
    a.add_argument(
        "--human",
        action="store_true",
        help="Resume via human message instead of system event (Option C)",
    )

    j = sub.add_parser("reject")
    j.add_argument("thread_id", nargs="?", default=None)
    j.add_argument("--by", required=True)
    j.add_argument("--note", default="")
    j.add_argument("--human", action="store_true")

    ch = sub.add_parser("chat")
    ch.add_argument(
        "thread_id_or_message",
        help="Message text, or thread_id when followed by message",
    )
    ch.add_argument(
        "message",
        nargs="?",
        default=None,
        help="If omitted, thread_id_or_message is the message and current thread is used",
    )

    return p


async def _run(args: argparse.Namespace) -> None:
    graph = build_graph(sessions_dir=args.sessions_dir, artifacts_dir=args.artifacts_dir)
    log.info("[CLI] sessions_dir=%s  command=%s", args.sessions_dir, args.command)

    if args.command == "run":
        print(
            await _cmd_run(
                graph, args.sessions_dir, args.repo, args.issue, args.thread_id
            )
        )
        return

    if args.command == "chat":
        if args.message is None:
            thread_id = resolve_thread_id(args.sessions_dir, None)
            message = args.thread_id_or_message
        else:
            thread_id = resolve_thread_id(args.sessions_dir, args.thread_id_or_message)
            message = args.message
        print(await _cmd_chat(graph, thread_id, message))
        return

    thread_id = resolve_thread_id(args.sessions_dir, getattr(args, "thread_id", None))
    log.info("[CLI] using thread_id=%s", thread_id)

    if args.command == "pending":
        print(_cmd_pending(graph, thread_id))
    elif args.command == "approve":
        print(
            await _cmd_decide(
                graph,
                thread_id,
                "approve",
                args.by,
                args.note,
                as_system=not args.human,
            )
        )
    elif args.command == "reject":
        print(
            await _cmd_decide(
                graph,
                thread_id,
                "reject",
                args.by,
                args.note,
                as_system=not args.human,
            )
        )
    elif args.command == "report":
        print(_cmd_report(graph, thread_id))
    elif args.command == "status":
        print(_cmd_status(graph, thread_id))

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(verbose=args.verbose)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
