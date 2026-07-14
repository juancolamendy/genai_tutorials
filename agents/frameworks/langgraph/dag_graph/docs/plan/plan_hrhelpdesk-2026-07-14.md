# HR Helpdesk + Chat Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `src/engine` with generic chatbot primitives, then ship `src/hrhelpdesk/` (hub + sticky topics) with CLI streaming generations via graph `.astream()`.

**Architecture:** Outer `EngineGraph` (router → guardrail → handler) supervises topic specialists built with `make_llm_agent`. Sticky lanes use `ChatEngineSessionState`. Effects/delivery share a namespaced keyed ledger. No FastAPI.

**Tech Stack:** LangGraph, LangChain `create_agent`, JsonCheckpointer, pytest-asyncio.

**Spec:** `docs/design/design_hrhelpdesk-2026-07-14.md`

---

## File map

### Engine (modify / create)
- `src/engine/engine_session_state.py` — add `ChatEngineSessionState`, `new_chat_session_state()`
- `src/engine/message_hygiene.py` — **create** trim + segment reset helpers
- `src/engine/event_ledger.py` — namespace helpers (`event_key`, `effect_key`); keep `EventLedger` API
- `src/engine/chains.py` — `get_model(role)`, optional stream helper for agents
- `src/engine/engine_graph.py` — `_resolve_proposed_next`, `_is_system_event_legal`, `astream`, `aemit_event_stream`
- `src/engine/__init__.py` — exports

### Domain (create)
- `src/hrhelpdesk/` — full package per design §5
- `tests/test_chat_session_state.py`, `test_message_hygiene.py`, `test_event_ledger_namespaces.py`, `test_engine_astream.py`, `test_hrhelpdesk_*.py`

---

## Phase 1 — Chat session state + message hygiene + ledger namespaces

**Files:** `engine_session_state.py`, `message_hygiene.py`, `event_ledger.py`, tests

- [ ] Add `ChatEngineSessionState` + `new_chat_session_state()` (schema_version=1, active_topic=None, topic_started_at=None, topic_data={})
- [ ] `trim_messages(messages, max_n=12)`, `segment_reset_messages(messages, summary)` returning RemoveMessage ops + summary AIMessage
- [ ] `event_key(event_id)`, `effect_key(thread_id, kind, *parts)` helpers; existing EventLedger callers unchanged (they pass opaque ids; helpdesk prefixes keys)
- [ ] Tests green; commit

## Phase 2 — get_model + routing/legality hooks + astream

**Files:** `chains.py`, `engine_graph.py`, tests

- [ ] `get_model(role: str) -> str` mapping router/topic → model ids (env-overridable)
- [ ] `_resolve_proposed_next(state, config)` default = current semantic/code router body; domains override for hub logic
- [ ] `_is_system_event_legal(state, event_type, payload)` default = today's `expected_events` check; `aemit_event` uses hook
- [ ] `astream(...)` / `aemit_event_stream(...)` — same contracts as ainvoke/aemit_event; yield `{"type":"token"|"result"|"status", ...}`; use `compiled_graph.astream`; final state matches non-stream path
- [ ] Tests; commit

## Phase 3 — hrhelpdesk skeleton (state machine + graph shell)

**Files:** `src/hrhelpdesk/{state_transitions,session_state,guardrails,graph,__init__}.py`

- [ ] States: ROUTE, HUB_CLARIFY, TOPIC_FAQ, TOPIC_ESCALATE, TOPIC_BOOKING, NOTIFY_USER, IDLE, ERROR
- [ ] HelpdeskState, build_graph, legality override, routing override stubs
- [ ] Commit

## Phase 4 — providers, chains, handlers, guardrails

**Files:** `providers.py`, `chains.py`, `handlers.py`, `guardrails.py`

- [ ] Fake RAG / ticket / desk providers
- [ ] Router + escape `make_chain`; FAQ/escalate/booking `make_llm_agent` with scoped tools
- [ ] Handlers: route logic in ROUTE/overrides; topic handlers; notify; close topic via message hygiene
- [ ] Guardrails: booking self-loop; clarify → idle; etc.
- [ ] Tests with mocked chains/agents; commit

## Phase 5 — sweep, CLI streaming, golden tests, README

**Files:** `sweep.py`, `cli.py`, `README.md`, `tests/test_hrhelpdesk_*.py`

- [ ] Sweep topic_timeout 48h; ticket reminder stub
- [ ] CLI chat/serve use `aemit_event_stream`, print token generations
- [ ] Golden tests per design §5
- [ ] README; commit

---

## Success criteria

- `pytest tests/test_chat_session_state.py tests/test_message_hygiene.py tests/test_event_ledger.py tests/test_engine_astream.py tests/test_hrhelpdesk*.py` green
- Existing onboarding/triage/docprocessing tests still pass
- CLI can stream a mocked FAQ reply to stdout
