"""CLI for the HR helpdesk chatbot."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from src.hrhelpdesk.graph import Graph, build_graph
from src.hrhelpdesk.sweep import sweep as run_sweep

log = logging.getLogger(__name__)

DEFAULT_SESSIONS_DIR = ".hrhelpdesk_sessions"

DEFAULT_THRESHOLDS = {
    "topic_booking": 48 * 3600.0,
}


def _parse_payload(pairs: list[str] | None) -> dict[str, str]:
    if not pairs:
        return {}
    return dict(pair.split("=", 1) for pair in pairs)


def _format_result(result: dict[str, Any]) -> str:
    emit_status = result.get("emit_status")
    current_state = result.get("current_state")
    active_topic = result.get("active_topic")
    return (
        f"emit_status={emit_status} current_state={current_state} "
        f"active_topic={active_topic}"
    )


def _format_output_messages(result: dict[str, Any]) -> str:
    """Render handler ``output_messages`` for CLI display (NOTIFY_USER, etc.)."""
    outs = result.get("output_messages") or []
    if not outs:
        return ""
    lines: list[str] = []
    for msg in outs:
        if isinstance(msg, list):
            parts: list[str] = []
            for part in msg:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif part:
                    parts.append(str(part))
            lines.append("".join(parts) if parts else str(msg))
        elif msg:
            lines.append(str(msg))
    return "\n".join(lines)


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


async def _stream_human_turn(graph: Graph, thread_id: str, message: str) -> dict[str, Any]:
    """Run a human turn via aemit_event_stream; print tokens as they arrive."""
    log.info("[CLI] chat(stream)  thread=%s  message=%r", thread_id, message[:80])
    final: dict[str, Any] = {"emit_status": "error"}
    async for chunk in graph.aemit_event_stream(
        thread_id=thread_id,
        source="human",
        event_type="message",
        input_message=message,
    ):
        if chunk.get("type") == "token":
            print(chunk.get("text", ""), end="", flush=True)
        elif chunk.get("type") == "result":
            final = chunk
        elif chunk.get("type") == "status":
            final = chunk

    if final.get("type") == "result":
        state = final.get("state") or {}
        print()
        print(_format_result({**state, "emit_status": final.get("emit_status")}))
        return {**state, "emit_status": final.get("emit_status")}

    print()
    print(_format_result(final))
    return final


async def _cmd_chat(graph: Graph, thread_id: str, message: str) -> str:
    result = await _stream_human_turn(graph, thread_id, message)
    return _format_result(result)


async def _cmd_event(
    graph: Graph, thread_id: str, event_type: str, event_id: str, payload: dict[str, str]
) -> str:
    log.info(
        "[CLI] event  thread=%s  type=%s  event_id=%s  payload=%s",
        thread_id,
        event_type,
        event_id,
        payload,
    )
    result = await graph.aemit_event(
        thread_id=thread_id,
        source="system",
        event_type=event_type,
        event_id=event_id,
        payload=payload,
    )
    status_line = _format_result(result)
    notify_text = _format_output_messages(result)
    log.info("[CLI] event  result=%s", status_line)
    if notify_text:
        return f"{notify_text}\n{status_line}"
    return status_line


async def _cmd_sweep(graph: Graph) -> str:
    log.info("[CLI] sweep  thresholds=%s", DEFAULT_THRESHOLDS)
    results = await run_sweep(graph, DEFAULT_THRESHOLDS)
    lines = [f"Swept {len(results)} stale thread(s)"]
    lines.extend(_format_result(r) for r in results)
    return "\n".join(lines)


def _cmd_status(graph: Graph, thread_id: str) -> str:
    state = graph._get_or_init_state(session_id=thread_id, user_id="")
    return (
        f"thread={thread_id} current_state={state.get('current_state')} "
        f"active_topic={state.get('active_topic')} session_status={state.get('session_status')}"
    )


async def _serve(graph: Graph) -> None:
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
                await _stream_human_turn(graph, rest[0], " ".join(rest[1:]))
            elif cmd == "event" and len(rest) >= 2:
                event_counter += 1
                event_id = f"serve:{rest[0]}:{rest[1]}:{event_counter}"
                print(
                    await _cmd_event(
                        graph, rest[0], rest[1], event_id, _parse_payload(rest[2:])
                    )
                )
            elif cmd == "sweep":
                print(await _cmd_sweep(graph))
            elif cmd == "status" and len(rest) >= 1:
                print(_cmd_status(graph, rest[0]))
            else:
                print(f"unrecognized command: {line!r}")
        except Exception as exc:
            print(f"error: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hrhelpdesk-cli")
    parser.add_argument("--sessions-dir", default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
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
    log.info("[CLI] sessions_dir=%s  command=%s", args.sessions_dir, args.command)

    if args.command == "chat":
        await _cmd_chat(graph, args.thread_id, args.message)
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
    _configure_logging(verbose=args.verbose)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
