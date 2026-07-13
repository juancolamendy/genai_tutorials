"""CLI for exercising the onboarding domain's event-extension design.

Subcommands (design spec §1.2/§2.2):
  chat <thread> <msg>                          — human-sourced turn
  event <thread> <type> --event-id ID [--payload k=v ...]  — system-sourced turn
  sweep                                         — run the timeout sweep once
  status <thread>                               — show a thread's current state
  serve                                         — long-lived asyncio loop reading
                                                   simulated events from stdin

serve is the primary, hardened path (confirmed with the user, design §2.2):
one-shot subcommands are separate processes and never exercise the
per-thread asyncio.Lock or concurrent-arrival behavior for real — serve is
the only mode where the lock and event ledger actually mean something.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from src.onboarding.graph import Graph, build_graph
from src.onboarding.sweep import sweep as run_sweep

DEFAULT_SESSIONS_DIR = ".onboarding_sessions"

# How long a thread may sit at a park state before a sweep escalates it.
DEFAULT_THRESHOLDS = {
    "await_documents_signed": 7 * 24 * 3600.0,  # 7 days
    "await_hardware_delivered": 3 * 24 * 3600.0,  # 3 days
}


def _parse_payload(pairs: list[str] | None) -> dict[str, str]:
    if not pairs:
        return {}
    return dict(pair.split("=", 1) for pair in pairs)


def _format_result(result: dict[str, Any]) -> str:
    status = result.get("status")
    current_state = result.get("current_state")
    return f"status={status} current_state={current_state}"


async def _cmd_chat(graph: Graph, thread_id: str, message: str) -> str:
    result = await graph.aemit_event(
        thread_id=thread_id, source="human", event_type="message", payload={"text": message}
    )
    return _format_result(result)


async def _cmd_event(
    graph: Graph, thread_id: str, event_type: str, event_id: str, payload: dict[str, str]
) -> str:
    result = await graph.aemit_event(
        thread_id=thread_id,
        source="system",
        event_type=event_type,
        event_id=event_id,
        payload=payload,
    )
    return _format_result(result)


async def _cmd_sweep(graph: Graph) -> str:
    results = await run_sweep(graph, DEFAULT_THRESHOLDS)
    lines = [f"Swept {len(results)} stale thread(s)"]
    lines.extend(_format_result(r) for r in results)
    return "\n".join(lines)


def _cmd_status(graph: Graph, thread_id: str) -> str:
    state = graph._get_or_init_state(session_id=thread_id, user_id="")
    current_state = state.get("current_state")
    status = state.get("status")
    return f"thread={thread_id} current_state={current_state} status={status}"


async def _serve(graph: Graph) -> None:
    """Long-lived asyncio loop reading simulated events from stdin, one
    command per line, until "quit" or EOF:
      chat <thread> <message...>
      event <thread> <type> [key=val ...]
      sweep
      status <thread>
    """
    print("serve mode — reading commands from stdin (one per line), 'quit' to exit")
    loop = asyncio.get_event_loop()
    event_counter = 0

    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        if line == "quit":
            break

        parts = line.split()
        cmd, rest = parts[0], parts[1:]
        try:
            if cmd == "chat" and len(rest) >= 2:
                print(await _cmd_chat(graph, rest[0], " ".join(rest[1:])))
            elif cmd == "event" and len(rest) >= 2:
                event_counter += 1
                event_id = f"serve:{rest[0]}:{rest[1]}:{event_counter}"
                print(await _cmd_event(graph, rest[0], rest[1], event_id, _parse_payload(rest[2:])))
            elif cmd == "sweep":
                print(await _cmd_sweep(graph))
            elif cmd == "status" and len(rest) >= 1:
                print(_cmd_status(graph, rest[0]))
            else:
                print(f"unrecognized command: {line!r}")
        except Exception as e:
            print(f"error: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onboarding-cli")
    parser.add_argument("--sessions-dir", default=DEFAULT_SESSIONS_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("thread_id")
    chat_parser.add_argument("message")

    event_parser = subparsers.add_parser("event")
    event_parser.add_argument("thread_id")
    event_parser.add_argument("event_type")
    event_parser.add_argument("--event-id", required=True)
    event_parser.add_argument("--payload", action="append", default=[])

    subparsers.add_parser("sweep")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("thread_id")

    subparsers.add_parser("serve")

    return parser


async def _run(args: argparse.Namespace) -> None:
    graph = build_graph(sessions_dir=args.sessions_dir)

    if args.command == "chat":
        print(await _cmd_chat(graph, args.thread_id, args.message))
    elif args.command == "event":
        payload = _parse_payload(args.payload)
        print(await _cmd_event(graph, args.thread_id, args.event_type, args.event_id, payload))
    elif args.command == "sweep":
        print(await _cmd_sweep(graph))
    elif args.command == "status":
        print(_cmd_status(graph, args.thread_id))
    elif args.command == "serve":
        await _serve(graph)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
