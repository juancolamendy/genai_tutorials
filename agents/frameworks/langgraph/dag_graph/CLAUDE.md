# Project Standards — LangGraph DAG Graph (State Machine Engine)

## Architecture

- **Engine layer** (`src/engine/`): Reusable, domain-agnostic state-machine infrastructure —
  `EngineGraph` (router → guardrail → handler dispatch), checkpointing, event ledger,
  message hygiene, chains/model factories. No domain vocabulary belongs here.
- **Domain layers** (`src/onboarding/`, `src/docprocessing/`, `src/triageprocessing/`,
  `src/hrhelpdesk/`, …): Business state, handlers, guardrails, and chains for one workflow.
  Domains override `EngineGraph` hooks (`_build_routing_table`, `_resolve_proposed_next`,
  `_is_system_event_legal`, `_get_guardrails`, …) rather than forking engine code.
- Chatbot (hub + sticky-topic) domains subclass `ChatEngineGraph` and use
  `ChatEngineSessionState` / `new_chat_session_state()` (see `chat_engine_graph.py` /
  `chat_engine_session_state.py`), not linear `EngineGraph` + `EngineSessionState`.

## Running tests / type checks

- Tests import as `from src.<domain>...`, which requires the **project root** (not `src/`) on
  `PYTHONPATH`. Run as `PYTHONPATH=. uv run pytest` — a bare `uv run pytest` fails every test
  module with `ModuleNotFoundError: No module named 'src'`.
- Before trusting a red/green result, check `git status` for uncommitted changes on top of the
  commit(s) under review — an in-progress rename or edit in the working tree produces failures
  that have nothing to do with the commits being tested. Stash (`git stash push -u`), verify
  against the clean committed tree, then pop.

## State machines with sticky (multi-turn) topic states

- Any state registered `waits_for_input=True` that is **not** the hub/IDLE park state (e.g. a
  sticky `TOPIC_*` state) can receive a **system-sourced** event directly, because
  `EngineGraph.invoke/ainvoke/astream` dispatches whatever handler is parked at
  `current_state` *before* the router or `_is_system_event_legal` ever run (the
  "resume-at-blocking-state" step). A sticky topic's handler must check
  `state.get("current_event_source") == "system"` and short-circuit *before* doing any LLM/tool
  work — mirroring the hub's own `handle_idle` — so a system event never triggers a live model
  call with empty input.
  - **Why:** confirmed by reproduction — a `topic_timeout` event delivered while a booking was
    genuinely sticky (`current_state == "topic_booking"`) invoked the booking LLM agent with an
    empty message, hit a real API error, and drove the session into the terminal `ERROR` state
    with the sticky topic never cleared. The thread was then permanently stuck
    (`already_terminal` on every future call) since `ERROR` is a `terminal_states` member.
- Every domain `allowed_transitions[state]` entry that a system event can legally route through
  (typically `NOTIFY_USER`) must include that target for **every** state the event can arrive
  at while parked — not just the hub state. Missing entries fail silently as a guardrail
  fallback to `ERROR`, not as an obvious error at authoring time.
  - **How to apply:** whenever adding a sticky (multi-turn) topic state, add a test that starts
    from the state the sticky topic actually parks at between turns (verify via
    `graph._get_or_init_state()` after a real turn, not a hand-constructed state dict) — see
    below.

## Test quality for multi-turn/system-event workflows

- Never hand-construct `current_state`/`active_topic` combinations for a test
  (`compiled_graph.update_state(...)` with an arbitrary dict) without first confirming that
  combination is actually reachable by driving the real turns that would produce it. An
  unreachable combination can make a "golden test" pass while the real code path it's meant to
  cover is broken.
- A streaming API test that asserts membership in a large set including `"error"` (e.g.
  `result["state"]["current_state"] in {"a", "b", ..., "error"}`) does not actually verify
  success — tighten streaming/result assertions to the specific non-error states expected for
  that scenario.

## Untrusted content in prompts

- Retrieved/external content (KB snippets, tool results, RAG chunks) injected into an LLM
  prompt must be wrapped in a clearly delimited untrusted block (see
  `src/engine/chains.py::render_as_xml`) rather than concatenated into the prompt as plain text,
  even when the current data source is a hardcoded fake — the delimiting is what stays safe
  when the fake is later swapped for a real source.

## Ruff / mypy hygiene

- Run `ruff check` and fix `F401` (unused imports) and import-sort (`I001`) before committing —
  both are auto-fixable (`ruff check --fix`).
- New code should not add mypy errors of shapes not already prevalent in the codebase (e.g.
  passing `Optional[str]`/`Any` where a helper expects `str`). Existing `TypedDict`
  return-value / guardrail-callable mypy errors are known pre-existing debt shared by every
  domain package (`onboarding`, `docprocessing`, `triageprocessing`) — matching that existing
  (imperfect) pattern is acceptable; introducing a *new* class of mypy error is not.
