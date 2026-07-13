# Implementation Plan: engine-event-extension
_Date: 2026-07-12_
_Input: docs/design/design_spec_engine-event-extension-2026-07-12.md_

## Summary
Ship the async, unified event-gate extension to `src/engine/*` (`ainvoke`, `aemit_event`, `arun_to_completion`, `wait_kind`/`expected_events` metadata) plus a new `onboarding/` domain and CLI that exercise it — starting with the three execution-confirmed bugs the adversarial design review found (thread-ID divergence, empty-input validation rejection, missing async checkpointer methods), since every later phase depends on them being fixed first.

## High-Risk Items (Scheduled in Phases 2–4)
- **Phase 2 — thread-ID divergence** (`invoke()`'s inline `f"{user_id}:{session_id}"` vs. `_get_or_init_state()`'s guarded version): confirmed by execution to silently create a new session on every turn instead of resuming, once `user_id=""`. No rollback path once real sessions are corrupted this way — must be fixed before any caller uses the new `user_id=""` convention.
- **Phase 2 — empty-input validation rejection** (`validate_turn_input("")` unconditionally raises): confirmed by execution to reject the *only* input every system-sourced and bg-run call ever supplies. Bundled with the thread-ID fix since both live in the same method and are fixed via the same two extracted helpers.
- **Phase 3 — async checkpointer support**: `compiled_graph.ainvoke()` confirmed (by execution) to raise `NotImplementedError` against the existing `JsonCheckpointer`/`SqliteCheckpointer`. This is a hard blocker for every subsequent async method — nothing past this phase can be exercised end-to-end without it. Also addresses the `SqliteCheckpointer` shared-connection concurrency race identified in review.
- **Phase 4 — idempotency/concurrency (event ledger + per-thread lock)**: the double-delivery / duplicate-webhook analog of "double-booking prevention" — a system event (e.g. a timeout sweep or a retried webhook) delivered twice must not double-apply a state transition. Must land before `aemit_event` (Phase 5) is built on top of it, and is file-backed (not in-memory) specifically because the CLI's one-shot subcommands are separate processes.

## Phases

### Phase 1 — Data model additions ✅ COMPLETE (commit `b0a0f4a`)
**Goal:** Add `wait_kind`/`expected_events` to `HandlerMetadata` (and the `handler()` decorator factory that constructs it), add `current_event_source`/`current_event_type`/`output_messages` to `EngineSessionState`, and add the new `errors.py` exception types — the shared vocabulary every later phase builds on.
**Size:** M
**Requirements satisfied:** §4 (Data model additions), §2.2 (`handler_registry.py`, `engine_session_state.py`, `errors.py` components)
**Files affected:**
- `src/engine/handler_registry.py`
- `src/engine/engine_session_state.py`
- `src/engine/errors.py` (new)
- `tests/test_handler_registry.py`
- `tests/test_engine_session_state.py` (new)
**Tasks:**
- [x] Write tests: `handler()` decorator forwards `wait_kind`/`expected_events` into `HandlerMetadata`; existing no-kwarg call sites (mirroring `docprocessing/handlers.py`'s usage) still default to `wait_kind="either"`, `expected_events=None`; `does_state_wait_for_input` behavior unchanged for existing states
- [x] Write tests: `new_engine_session_state()` includes `current_event_source` defaulting to absent/`"human"`-on-read, `current_event_type` defaulting to absent/`"message"`-on-read, `output_messages` defaulting to `[]`
- [x] Write tests: each new error class (`GraphIncompleteError`, `GraphAlreadyInteractiveError`, `GraphAlreadyCompleteError`, `GraphRunError`) is importable, raisable, and carries a message
- [x] Implement `HandlerMetadata.wait_kind`/`expected_events` fields and extend the `handler(...)` factory to accept and forward them (design spec §4)
- [x] Implement `EngineSessionState` field additions (`current_event_source`, `current_event_type`, `output_messages` with `operator.add` reducer)
- [x] Implement `src/engine/errors.py` with the four exception classes
- [x] Verify tests pass; verify `docprocessing`'s existing 12 test files still pass unmodified
**Success criteria:** New fields/decorator kwargs exist with documented defaults; all new unit tests green; `pytest tests/test_doc_pipeline_router.py tests/test_graph_methods.py tests/test_handler_integration.py tests/test_handler_registry.py tests/test_input_validation.py tests/test_integration.py tests/test_main_examples.py tests/test_multi_turn.py tests/test_multiturn_workflow.py tests/test_router_integration.py tests/test_semantic_router_integration.py tests/test_semantic_router.py tests/test_session_checkpointer.py` still passes.
**Commit message:** `feat(engine): add wait_kind/expected_events metadata and event-source session fields`
**Git worktree:** `git worktree add ../worktrees/engine-event-extension-phase-1 -b phase/engine-event-extension-1`

---

### Phase 2 — [HIGH RISK] Fix `invoke()`'s thread-ID divergence and empty-input rejection ✅ COMPLETE (commit `e1a1215`)
**Goal:** Extract and fix the two execution-confirmed bugs in `EngineGraph.invoke()` (§5.1 of the design spec) via two shared helpers, `_thread_id()` and `_prepare_input()`, used by both `invoke()` and the `ainvoke()` built in Phase 4. Also adds the `pytest-asyncio` tooling every async phase from here on depends on.
**Size:** M
**Requirements satisfied:** §5.1 (both confirmed bugs), §16 item 1 (test tooling prerequisite)
**Files affected:**
- `src/engine/engine_graph.py`
- `pyproject.toml`
- `pytest.ini`
- `tests/test_graph_methods.py`
**Tasks:**
- [x] Write a test that reproduces the thread-ID bug against current `invoke()`: two calls with `user_id=""` and the same `session_id` must resume (turn 2 sees turn 1's state), not create a fresh session — this test must FAIL against today's code before any fix lands
- [x] Write a test that reproduces the empty-input bug: `invoke(user_id="x", session_id="y", input_message="", ...)` must not raise `InputValidationError` — must FAIL against today's code before any fix lands
- [x] Write a test asserting existing non-empty-`user_id`/non-empty-`input_message` behavior is byte-for-byte unchanged (regression guard for the "strict narrowing" claim)
- [x] Add `pytest-asyncio` to `[project.optional-dependencies].dev` in `pyproject.toml`; set `asyncio_mode = "auto"` in `pytest.ini`
- [x] Implement `EngineGraph._thread_id(user_id, session_id)` and `EngineGraph._prepare_input(input_message)`; replace the inline formula in `invoke()` and the duplicate guarded formula in `_get_or_init_state()` with calls to `_thread_id()`; replace `invoke()`'s unconditional `validate_turn_input`/`escape_for_llm` calls with `_prepare_input()`
- [x] Verify both previously-failing tests now pass; verify `docprocessing`'s 12 tests still pass unmodified
**Success criteria:** The two reproduction tests pass; `docprocessing` regression suite green; a trivial `async def test_asyncio_smoke(): assert True` runs under pytest without manual event-loop wiring (proves the tooling change took effect).
**Commit message:** `fix(engine): resolve thread-ID divergence and empty-input rejection in invoke(); add async test tooling`
**Git worktree:** `git worktree add ../worktrees/engine-event-extension-phase-2 -b phase/engine-event-extension-2`

---

### Phase 3 — [HIGH RISK] Async checkpointer support ✅ COMPLETE (commit `188fcc0`)
**Goal:** Add working `aget_tuple`/`aget`/`aput` async methods to `JsonCheckpointer` and `SqliteCheckpointer` (currently inherited `NotImplementedError` stubs — confirmed by execution to crash `compiled_graph.ainvoke()` immediately), and serialize `SqliteCheckpointer`'s shared `:memory:` connection against concurrent access from the thread pool.
**Size:** M
**Requirements satisfied:** §5.2, §14 (async checkpoint compatibility + SQLite concurrency)
**Files affected:**
- `src/engine/json_checkpointer.py`
- `src/engine/sqlite_checkpointing.py`
- `tests/test_session_checkpointer.py`
- `tests/test_checkpointer_async.py` (new)
**Tasks:**
- [x] Write a test that reproduces the crash: build a `docprocessing` graph with the existing `JsonCheckpointer`, call `await graph.compiled_graph.ainvoke(state, config=config)` — must FAIL with `NotImplementedError` against today's code
- [x] Write an equivalent reproduction test for `SqliteCheckpointer`
- [x] Write a concurrency test: two `asyncio.to_thread`-wrapped calls against the same in-memory `SqliteCheckpointer` for two different `thread_id`s, run concurrently, must not raise `sqlite3.OperationalError` and must both persist correctly
- [x] Implement `aget_tuple`/`aget`/`aput` on `JsonCheckpointer` as `asyncio.to_thread`-wrapped calls to the existing sync methods (safe as-is: each `thread_id` is its own file)
- [x] Implement the same three async methods on `SqliteCheckpointer`, guarded by one shared lock (serializing access to `_memory_conn` / the file connection) per the design's fix
- [x] Verify the two crash-reproduction tests now pass (assert no exception, correct round-tripped state); verify the concurrency test passes; verify `docprocessing`'s 12 tests still pass unmodified (sync path untouched)
**Success criteria:** `await graph.compiled_graph.ainvoke(...)` succeeds against both checkpointer types with a real state round-trip; concurrency test passes reliably across 3 consecutive runs (no flaky lock contention).
**Commit message:** `feat(engine): implement async checkpointer methods with SQLite concurrency guard`
**Git worktree:** `git worktree add ../worktrees/engine-event-extension-phase-3 -b phase/engine-event-extension-3`

---

## Checkpoint: After Phases 1–3
- [x] All tests pass (`docprocessing` regression suite + all new unit tests)
- [x] Build succeeds (`ruff check src/`, `mypy src/`)
- [x] Foundation is working: metadata/session-state fields exist, `invoke()`'s two confirmed bugs are fixed, `compiled_graph.ainvoke()` runs without crashing against both checkpointers
- [x] Review with human before proceeding

---

### Phase 4 — [HIGH RISK] `ainvoke()` + event ledger + per-thread lock ✅ COMPLETE (commit `bb8ae0a`)
**Goal:** Add the async twin of `invoke()` (`EngineGraph.ainvoke()`, using the Phase 2 helpers and Phase 3 checkpointer support), the file-backed `event_ledger.py` dedupe store, and the per-thread `asyncio.Lock` registry — the idempotency/concurrency mechanism `aemit_event` (Phase 5) is built on top of.
**Size:** M
**Requirements satisfied:** §5 (`ainvoke`), §2.2 (`event_ledger.py`), §14 (concurrency — per-thread lock, file-backed ledger)
**Files affected:**
- `src/engine/engine_graph.py`
- `src/engine/event_ledger.py` (new)
- `tests/test_event_ledger.py` (new)
- `tests/test_engine_async.py` (new)
**Tasks:**
- [x] Write tests for `event_ledger.is_processed`/`mark_processed`: dedupe within one instance; **dedupe persists across two separate `EventLedger` instances pointed at the same directory** (the scenario an in-memory ledger would fail — simulates two separate CLI process invocations)
- [x] Write a test for `ainvoke()`: mirrors `invoke()`'s existing multi-turn resume test (`tests/test_multi_turn.py`'s shape) but async, using `user_id=""`, asserting turn 2 correctly resumes turn 1's state (regression guard tied to Phase 2's fix)
- [x] Write a test asserting `self._locks` is a `defaultdict(asyncio.Lock)` keyed by thread_id and two concurrent calls for the *same* thread_id serialize (second doesn't start until first's lock releases)
- [x] Implement `src/engine/event_ledger.py`: file-backed `is_processed(event_id)`/`mark_processed(event_id)`, same storage-family pattern as `JsonCheckpointer` (atomic writes via temp-file + rename)
- [x] Implement `EngineGraph.ainvoke(user_id, session_id, input_message, state_delta, timeout_sec)`: same algorithm as `invoke()`, using `await compiled_graph.ainvoke(...)`, sharing `_thread_id()`/`_prepare_input()`
- [x] Implement the `self._locks` `defaultdict(asyncio.Lock)` registry on `EngineGraph.__init__`
- [x] Verify all new tests pass; verify `docprocessing`'s 12 tests still pass unmodified
**Success criteria:** `ainvoke()` correctly resumes a multi-turn session with `user_id=""` (the exact scenario that was broken in the round-1 design draft); ledger dedupe survives a simulated process restart (new instance, same directory).
**Commit message:** `feat(engine): add ainvoke(), file-backed event ledger, and per-thread lock registry`
**Git worktree:** `git worktree add ../worktrees/engine-event-extension-phase-4 -b phase/engine-event-extension-4`

---

### Phase 5 — `aemit_event()` unified gate ✅ COMPLETE (commit `8ce0921`)
**Goal:** Implement the single unified event gate (§6) — human/system branches, `wait_kind`/`expected_events` legality checks (including the symmetric `wait_kind == "human"` guard the design review added), the full status vocabulary, and `_describe_wait()`.
**Size:** M
**Requirements satisfied:** §6 (`aemit_event` — corrected algorithm, including the round-2 symmetric-guard fix)
**Files affected:**
- `src/engine/engine_graph.py`
- `tests/test_aemit_event.py` (new)
**Tasks:**
- [x] Write one test per status outcome: `ok` (human), `ok` (system), `duplicate`, `ignored` (event not in `expected_events`), `ignored` (symmetric guard: system event against a `wait_kind="human"` state, even with `expected_events` populated), `not_waiting`, `already_terminal`
- [x] Write a test proving `user_id=""` is always used regardless of `source` (thread identity never splits by event source)
- [x] Write a smoke-test against the *existing* `docprocessing` graph: a human-sourced `aemit_event` call against its `UPLOAD_DOCUMENTS` park state (which defaults to `wait_kind="either"`) succeeds and resumes correctly — validates the gate against real, already-working state before the `onboarding` domain exists
- [x] Implement `EngineGraph.aemit_event(thread_id, source, event_type, payload, event_id)` per the corrected §6 algorithm, including the symmetric `wait_kind == "human"` guard in the system branch
- [x] Implement `EngineGraph._describe_wait(state, meta)` returning `{"status": "waiting", "current_state": ..., "wait_kind": ..., "expected_events": ...}`
- [x] Verify all new tests pass; verify `docprocessing`'s 12 tests still pass unmodified
**Success criteria:** All 7 status outcomes are individually testable and pass; the `docprocessing` smoke test proves the gate works against a real, pre-existing graph without any `onboarding`-specific code.
**Commit message:** `feat(engine): add aemit_event unified event gate with symmetric wait_kind guard`
**Git worktree:** `git worktree add ../worktrees/engine-event-extension-phase-5 -b phase/engine-event-extension-5`

---

### Phase 6 — Router bypass, message convention, session enumeration ✅ COMPLETE (commit `9cf7cd1`)
**Goal:** Three small, independent additions that round out the engine core: the one-line semantic-router bypass for system-sourced turns (§9), the `output_messages` message-convention rewrite (§11), and the `get_active_sessions()` helper (§13) that hides the checkpointer's internal file shape from future callers like `sweep.py`.
**Size:** M
**Requirements satisfied:** §9 (router bypass), §11 (message/output convention), §13 (`get_active_sessions()` — resolves the `sweep.py` shape bug found in review)
**Files affected:**
- `src/engine/engine_graph.py`
- `tests/test_router_integration.py`
- `tests/test_engine_async.py`
**Tasks:**
- [x] Write a test: with a `semantic_router` attached, a state with `current_event_source == "system"` still routes via the code table, never the LLM (mock the semantic router to assert it's never called)
- [x] Write tests for the message convention: human turn with text → `HumanMessage` + fallback/`output_messages` `AIMessage`(s); system turn with `output_messages` set → only those `AIMessage`(s), no fallback; system turn with nothing to say → `messages` completely untouched (no empty turn recorded)
- [x] Write a test for `get_active_sessions()`: build a graph, run one `invoke()`, call `get_active_sessions()`, assert the returned shape is `[{"thread_id": ..., "state": {...}}]` with a real, unwrapped state dict (regression guard for the `sweep.py` shape bug found in review)
- [x] Implement the one-line `_router_node` bypass: `if self.semantic_router is not None and state.get("current_event_source") != "system"`
- [x] Implement the `output_messages`-driven message convention, replacing the current unconditional `[HumanMessage(...), AIMessage(f"Transitioned to {state}")]` append
- [x] Implement `EngineGraph.get_active_sessions()` per §13, reusing `_extract_state_from_checkpoint()`
- [x] Verify all new tests pass; verify `docprocessing`'s 12 tests still pass unmodified (fallback branch reproduces today's exact message)
**Success criteria:** Router bypass test proves the LLM path is structurally unreachable for system events; message-convention tests cover all three branches; `get_active_sessions()` returns real state dicts, not raw checkpointer internals.
**Commit message:** `feat(engine): add router bypass for system events, output_messages convention, get_active_sessions helper`
**Git worktree:** `git worktree add ../worktrees/engine-event-extension-phase-6 -b phase/engine-event-extension-6`

---

## Checkpoint: After Phases 4–6
- [x] All tests pass
- [x] High-risk NFRs addressed and verified (idempotent event ledger, per-thread locking, resumable `ainvoke`)
- [x] End-to-end flow works for the core feature: a human-sourced and a system-sourced `aemit_event` call both succeed against the existing `docprocessing` graph (no `onboarding` domain needed yet to prove this)
- [x] Review with human before proceeding

---

### Phase 7 — `arun_to_completion()` ✅ COMPLETE (commit `a4238f5`)
**Goal:** Implement the bg/batch run-to-completion entry point (§12), including the round-2 fix (reject reuse of an already-terminal thread, not just a mid-flow one) and the user-confirmed graceful-degradation outcome for the needs-input case.
**Size:** S
**Requirements satisfied:** §12 (`arun_to_completion` — corrected twice; resolves Open Question 1)
**Files affected:**
- `src/engine/engine_graph.py`
- `tests/test_arun_to_completion.py` (new)
**Tasks:**
- [x] Write a test: calling `arun_to_completion` against a thread already parked mid-flow raises `GraphAlreadyInteractiveError`
- [x] Write a test: calling `arun_to_completion` twice with the same `session_id` after the first run reached a terminal state raises `GraphAlreadyCompleteError` on the second call (regression guard for the round-2 bug — this must FAIL against a naive `current not in terminal_states` check before the fix)
- [x] Write a test: a run that ends parked at a `waits_for_input` state returns `{"status": "blocked_needs_input", ...}` rather than raising
- [x] Write a test: a run that ends at a genuine terminal state returns the result normally; a run that errors raises `GraphRunError`
- [x] Implement `EngineGraph.arun_to_completion(user_id, session_id, initial_state_delta, timeout_sec, max_auto_iters)` per the corrected §12 algorithm
- [x] Verify all four outcome tests pass; verify `docprocessing`'s 12 tests still pass unmodified
**Success criteria:** All four `arun_to_completion` outcomes (`GraphAlreadyInteractiveError`, `GraphAlreadyCompleteError`, `blocked_needs_input`, normal completion/`GraphRunError`) are independently tested and pass.
**Commit message:** `feat(engine): add arun_to_completion with terminal-reuse guard and graceful degradation`
**Git worktree:** `git worktree add ../worktrees/engine-event-extension-phase-7 -b phase/engine-event-extension-7`

---

### Phase 8 — `onboarding/` state machine skeleton ✅ COMPLETE (commit `4ada424`)
**Goal:** Build the `onboarding/` domain's static pieces — state enum/transitions, session state, guardrails (including the round-1 fix excluding `check_fallback_depth` from `COLLECT`), and LLM chains — mirroring `docprocessing/*` file-for-file per §1.2. This phase only depends on Phase 1 (the `wait_kind`/`expected_events` metadata support) and can start as soon as that lands, in parallel with Phases 4–7 (see Parallelization).
**Size:** M
**Requirements satisfied:** §3 (state chart), §4 (`OnboardingState`), §8 (guardrails, `COLLECT` cascade-detection exclusion), §10 (collect node's tool-calling agent, IT_PROVISIONED's LCEL chain)
**Files affected:**
- `src/onboarding/state_transitions.py` (new)
- `src/onboarding/session_state.py` (new)
- `src/onboarding/guardrails.py` (new)
- `src/onboarding/chains.py` (new)
- `tests/test_onboarding_state_machine.py` (new)
**Tasks:**
- [x] Write tests for `state_transitions.py`: `State` enum has all 8 states (`COLLECT`, `WELCOME_SENT`, `AWAIT_DOCUMENTS_SIGNED`, `IT_PROVISIONED`, `AWAIT_HARDWARE_DELIVERED`, `SCHEDULE_SENT`, `COMPLETE`, `ESCALATED`); `happy_path` maps each non-terminal state forward per §3's chart; `terminal_states = {COMPLETE, ESCALATED}`
- [x] Write a test: `guardrails[State.COLLECT]` does NOT include `check_fallback_depth` (regression guard for the round-1 finding — three ordinary clarifying turns must not trip the cascade cap)
- [x] Write a test: `guardrails[State.IT_PROVISIONED]` and the equivalent guardrail for `SCHEDULE_SENT` both divert to `ESCALATED` when `current_event_type == "timeout_escalation"`, via `check_not_timeout_escalation`
- [x] Implement `state_transitions.py` (`State`, `happy_path`, `terminal_states`, `allowed_transitions`, `is_transition_allowed`) mirroring `docprocessing/state_transitions.py`'s shape
- [x] Implement `session_state.py::OnboardingState` (adds `new_hire_details`, guard flags `welcome_sent`/`it_provisioned`/`schedule_sent`/`hr_notified`, `username_prefix`, `hardware_tracking_id`) and `new_onboarding_session_state()`
- [x] Implement `guardrails.py`: `check_not_timeout_escalation`, the guardrail registry for all 8 states — `COLLECT` excludes `check_fallback_depth`; both `IT_PROVISIONED` and `SCHEDULE_SENT` include `check_not_timeout_escalation` first (short-circuit ordering)
- [x] Implement `chains.py`: the `submit_new_hire` tool-calling agent chain for `COLLECT`, the deterministic LCEL chain for `IT_PROVISIONED`'s username selection
- [x] Verify all new tests pass
**Success criteria:** State machine skeleton importable with no handler wiring yet; guardrail unit tests prove both round-1 findings (COLLECT cascade exclusion, dual timeout-diversion guardrails) are correctly implemented.
**Commit message:** `feat(onboarding): add state machine skeleton (transitions, session state, guardrails, chains)`
**Git worktree:** `git worktree add ../worktrees/engine-event-extension-phase-8 -b phase/engine-event-extension-8`

---

### Phase 9 — `onboarding/` handlers and graph wiring ✅ COMPLETE (commit `44dc0e5`)
**Goal:** Implement all 8 state handlers (including the `current_event_type`-branching required on both `AWAIT_*` handlers per §3) and wire them into an `EngineGraph` subclass, mirroring `docprocessing/graph.py`.
**Size:** M
**Requirements satisfied:** §3 (AWAIT handlers branch on `current_event_type`), §2.2 (`onboarding/*` component)
**Files affected:**
- `src/onboarding/handlers.py` (new)
- `src/onboarding/graph.py` (new)
- `tests/test_onboarding_handlers.py` (new)
**Tasks:**
- [x] Write tests for both `AWAIT_*` handlers: a `timeout_escalation`-typed resume records the timeout audit entry and does NOT set the "happy" guard flag/business field; a legal-event-typed resume does the opposite (regression guard for §3's finding)
- [x] Write a test for `handle_collect`: given complete `new_hire_details`, transitions forward with `welcome_sent` unset until `WELCOME_SENT`'s own handler runs; given incomplete details, guardrail fallback loops without tripping cascade detection (ties to Phase 8's guardrail fix)
- [x] Write a test for `handle_it_provisioned`: sets `it_provisioned=True` guard flag exactly once even if dispatched twice (idempotent side effect per §2.3)
- [x] Implement all 8 handlers (`handle_collect`, `handle_welcome_sent`, `handle_await_documents_signed`, `handle_it_provisioned`, `handle_await_hardware_delivered`, `handle_schedule_sent`, `handle_complete`, `handle_escalated`) with `@handler(...)` metadata (`waits_for_input`, `wait_kind`, `expected_events` on the two `AWAIT_*` states and `COLLECT`)
- [x] Implement `graph.py::Graph(EngineGraph)` and `build_graph()`, wiring `state_enum`, `terminal_states`, `handler_map`, routing table, guardrails — mirroring `docprocessing/graph.py`'s structure
- [x] Verify all new tests pass
**Success criteria:** Both `AWAIT_*` handlers are proven to branch correctly on `current_event_type`; `build_graph()` produces a working compiled graph that can be driven with plain `invoke()` end-to-end from `COLLECT` to `COMPLETE` using only synthetic state deltas (no `aemit_event` yet — that's Phase 10's integration test).
**Commit message:** `feat(onboarding): implement state handlers and graph wiring`
**Git worktree:** `git worktree add ../worktrees/engine-event-extension-phase-9 -b phase/engine-event-extension-9`

---

### Phase 10 — CLI, sweep, and end-to-end integration ✅ COMPLETE (commit `bc166b4`)
**Goal:** Implement the CLI (`chat`/`event`/`sweep`/`status`/`serve` subcommands), `sweep.py`, and the full end-to-end integration test driving the onboarding state chart entirely through `aemit_event`/`arun_to_completion` — the final proof that every phase's pieces compose correctly.
**Size:** M
**Requirements satisfied:** §1.2 (CLI), §13 (`sweep.py`, using Phase 6's `get_active_sessions()`), §16 items 2–4 (integration test, docprocessing regression)
**Files affected:**
- `src/onboarding/cli.py` (new)
- `src/onboarding/sweep.py` (new)
- `tests/test_onboarding_cli.py` (new)
- `tests/test_onboarding_flow.py` (new)
**Tasks:**
- [x] Write the end-to-end integration test first: full happy path `COLLECT` (human) → `document_signed` (system event) → `hardware_delivered` (system event) → `COMPLETE`, driven entirely through `aemit_event`; a second scenario exercising `timeout_escalation` diversion to `ESCALATED` from each `AWAIT_*` state; a duplicate-`event_id` delivery asserting `"duplicate"` status and no double-processing (`audit_trail` not appended twice)
- [x] Write CLI tests: `chat`/`event`/`status` subcommands call through to `aemit_event`/checkpointer correctly (can mock the graph); `sweep` calls `sweep.py::sweep()` correctly
- [x] Write a test for `sweep.py`: a stale `AWAIT_*` thread past its threshold gets a `timeout_escalation` event emitted via `aemit_event`; a fresh thread does not
- [x] Implement `sweep.py::sweep(graph, thresholds)` per the corrected §13 algorithm, using `graph.get_active_sessions()` (Phase 6)
- [x] Implement `cli.py` with `argparse` + `asyncio`: `chat <thread> <msg>`, `event <thread> <type> --event-id ID [--payload k=v]`, `sweep`, `status <thread>`, and `serve` (long-lived asyncio loop reading simulated events from stdin — confirmed with the user as the primary, hardened path)
- [x] Run the full existing `docprocessing` regression suite one final time (§16 item 4) to confirm zero behavioral drift across all 10 phases
- [x] Verify all new tests pass
**Success criteria:** The end-to-end integration test drives the full onboarding state chart through `aemit_event` alone, including both the happy path and the timeout-diversion path, with duplicate-delivery correctly deduped; `docprocessing`'s 12 tests still pass unmodified; `python -m src.onboarding.cli chat <thread> "hello"` works manually against a real `.onboarding_sessions` directory.
**Commit message:** `feat(onboarding): add CLI, timeout sweep, and end-to-end integration test`
**Git worktree:** `git worktree add ../worktrees/engine-event-extension-phase-10 -b phase/engine-event-extension-10`

---

## Checkpoint: After Phases 7–10
- [x] All tests pass
- [x] Full feature complete end-to-end: `onboarding/` domain fully exercises `aemit_event`, `arun_to_completion`, timeout diversion, and duplicate-event dedup
- [x] `docprocessing`'s 12 tests still pass unmodified (final confirmation of §1.4's backward-compatibility claim)
- [x] Ready for `cg-review`

## Parallelization

| Category | Phases | Rule |
|----------|--------|------|
| Safe to parallelize | 1, 2, 3 | No shared files between them (`handler_registry.py`/`engine_session_state.py`/`errors.py` vs. `engine_graph.py` vs. the two checkpointer files) — all three can start immediately and merge independently, but Phase 4 must wait for both 2 and 3 to land. |
| Safe to parallelize | 8 (and, once 8 lands, 9) alongside 4, 5, 6, 7 | `onboarding/*` only depends on Phase 1's metadata support, not on `ainvoke`/`aemit_event`/`arun_to_completion` existing yet — the state machine skeleton (8) and handlers/graph (9) can be built and unit-tested on a separate track from the engine-core chain (4→5→6→7), joining only at Phase 10. |
| Must be sequential | 2, 3 → 4 → 5 → 6 → 7 | Engine-core dependency chain: `ainvoke` (4) needs both the thread-ID/validation fix (2) and working async checkpointers (3); `aemit_event` (5) needs `ainvoke`+ledger+lock (4); router/message/session-enumeration (6) and `arun_to_completion` (7) both touch `engine_graph.py` and build on 5. |
| Must be sequential | 8 → 9 | Handlers (9) import the state enum, session state, guardrails, and chains built in 8. |
| Needs coordination | 6, 7 | Both edit `src/engine/engine_graph.py` in the same region of the class; sequence them (6 before 7) rather than running as literal parallel worktrees, even though the logic is otherwise independent, to avoid a merge conflict on every commit. |
| Needs coordination | 10 | Final join point — depends on the full engine-core chain (through 7) AND the full onboarding chain (through 9) both being complete; do not start until both checkpoints (4–6 and the onboarding half of 7–10) are green. |

## TODO List (ordered)
- [x] Phase 1: Data model additions
- [x] Phase 2: [HIGH RISK] Fix `invoke()`'s thread-ID divergence and empty-input rejection
- [x] Phase 3: [HIGH RISK] Async checkpointer support
- [x] Phase 4: [HIGH RISK] `ainvoke()` + event ledger + per-thread lock
- [x] Phase 5: `aemit_event()` unified gate
- [x] Phase 6: Router bypass, message convention, session enumeration
- [x] Phase 7: `arun_to_completion()`
- [x] Phase 8: `onboarding/` state machine skeleton
- [x] Phase 9: `onboarding/` handlers and graph wiring
- [x] Phase 10: CLI, sweep, and end-to-end integration
