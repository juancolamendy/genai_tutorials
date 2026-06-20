# Implementation Plan: LangGraph State Machine Workflow
_Date: 2026-06-20_
_Input: docs/design/design_spec_langgraph-state-machine-workflow-2026-06-20.md_

## Summary

Port the proven agno Router-Dispatcher state machine pattern to LangGraph. Deliver a production-ready, reusable engine that supports one-turn document processing with safety guards (timeouts, cascade detection, input validation), comprehensive error handling (transient vs permanent classification), and session persistence via LangGraph checkpoints. The design prioritizes safety via early guardrail and timeout enforcement, followed by handler architecture and graph assembly.

---

## High-Risk Items (Scheduled in Phases 2–4)

These non-functional requirements are critical for reliability and security. Scheduled early so downstream work depends on a solid foundation.

1. **Timeout Enforcement (Phase 4)**
   - Prevents indefinite hangs (missing timeout = DoS vulnerability)
   - Must be checked at every guardrail iteration
   - Tests verify abort at correct time

2. **Guardrail Composition & Cascade Detection (Phase 3)**
   - Core safety mechanism; guards must work correctly or all recovery paths fail
   - Cascade detection prevents infinite fallback loops (missing → infinite loop)
   - Tests verify short-circuit evaluation and cascade bounds

3. **Exception Handling Pattern (Phase 5)**
   - All handlers MUST catch ALL exceptions; missing catch = graph crash
   - Error type classification (transient vs permanent) affects retry logic
   - Tests verify all exception types caught and error_type set correctly

4. **Input Validation (Phase 8)**
   - Prevents DoS attacks (oversized docs, invalid document_id, etc.)
   - Must happen at entry point before graph invoked
   - Tests verify validation catches all invalid inputs

5. **Checkpoint Mechanism (Phase 8)**
   - Foundation for session persistence and resume
   - Must work before other features depend on it
   - Tests verify checkpoint save/load cycle

---

## Task Sizing Summary

| Component | Size | Files |
|-----------|------|-------|
| Data Models | S | 1 (state_machine.py) |
| Guardrail Framework | M | 2 (guardrail.py + test_guardrails.py) |
| Router & Timeouts | S | 1 (router.py, integrated into workflow) |
| Handler Architecture | M | 2 (handlers.py + test_handlers.py) |
| Specific Handlers | M | 2 (handlers.py + test_handlers_integration.py) |
| Graph Assembly | S | 1 (workflow.py) |
| Session & Checkpoint | M | 2 (workflow.py + test_session.py) |
| Observability | S | 1 (Added to handlers/guardrails) |
| E2E Integration | M | 1 (test_integration.py) |
| Documentation | S | 1 (README/examples) |

**Total blast radius per phase: ≤ 5 files**

---

## Phases

### Phase 1 — Project Structure & Dependencies
**Goal:** Set up pyproject.toml, uv environment, directory structure, and test framework.

**Size:** XS

**Requirements satisfied:**
- Infrastructure for development (Section 1: Scope — production-ready implementation)

**Files affected:**
- `pyproject.toml` (update dependencies)
- `pytest.ini` (test config)
- `src/engine/` (directory)
- `src/pipeline/` (directory)
- `tests/` (directory)

**Tasks:**
- [ ] Add `langgraph>=0.0.1` and `pytest>=7.0` to pyproject.toml
- [ ] Create directory structure: `src/engine/`, `src/pipeline/`, `tests/`
- [ ] Create `pytest.ini` with coverage settings (80% minimum)
- [ ] Create `tests/__init__.py` and helper fixtures (MockLLMAgent, sample_state)
- [ ] Verify imports work: `python -c "import langgraph; import pytest"`

**Success criteria:**
- All imports resolve
- `pytest tests/` runs with no errors (even if no tests yet)
- Directory structure matches design spec (engine vs pipeline separation)

**Commit message:**
```
chore: initial project structure and dependencies for langgraph implementation
```

**Git worktree:** `git worktree add ../worktrees/phase-1-setup -b phase/setup-1`

---

### Phase 2 — State Machine & Core Data Models
**Goal:** Define State enum, ALLOWED_TRANSITIONS, PipelineState TypedDict, and GuardrailResult dataclass. These are the foundation for all subsequent work.

**Size:** S

**Requirements satisfied:**
- Section 4: Data Models (State, PipelineState, GuardrailResult)
- Section 9: Always Do — State Machine Integrity

**Files affected:**
- `src/engine/state_machine.py` (new)
- `src/engine/__init__.py` (export types)
- `tests/test_state_machine.py` (new)

**Tasks:**
- [ ] Write test: `State` enum has 9 values (INIT, FETCH, VALIDATE, ENRICH, STORE, COMPLETE, RETRY, HUMAN_REVIEW, ERROR)
- [ ] Write test: `ALLOWED_TRANSITIONS[State.FETCH]` contains {VALIDATE, RETRY, ERROR}
- [ ] Implement `State` enum and `ALLOWED_TRANSITIONS` dict
- [ ] Write test: `PipelineState` TypedDict has all 17 fields (Section 4.1)
- [ ] Implement `PipelineState` TypedDict with type hints
- [ ] Write test: `GuardrailResult` has passed, reason, fallback fields
- [ ] Implement `GuardrailResult` dataclass
- [ ] Write test: `is_transition_allowed(State.FETCH, State.VALIDATE)` returns True; `is_transition_allowed(State.FETCH, State.COMPLETE)` returns False
- [ ] Implement `is_transition_allowed()` function

**Success criteria:**
- All 9 State enum values defined correctly
- `ALLOWED_TRANSITIONS` covers all states (9 keys)
- `PipelineState` TypedDict includes all 17 fields: current_state, proposed_next, retry_count, error_message, error_type, audit_trail, fallback_depth, started_at, node_timeout_seconds, document_id, raw_data, validated_data, enriched_data
- `GuardrailResult` dataclass instantiates and serializes
- `is_transition_allowed()` passes all unit tests

**Commit message:**
```
feat: define state machine, PipelineState TypedDict, and GuardrailResult

- State enum with 9 values (INIT through ERROR)
- ALLOWED_TRANSITIONS adjacency list for valid state paths
- PipelineState TypedDict with 17 control, execution, and business payload fields
- GuardrailResult dataclass with passed, reason, fallback
- is_transition_allowed() validator function
- Unit tests for all types and transitions
```

**Git worktree:** `git worktree add ../worktrees/phase-2-datatypes -b phase/datatypes-2`

---

### Phase 3 — Guardrail Framework (High-Risk Safety)
**Goal:** Implement composable guardrail checks, short-circuit evaluation, and the GUARDRAILS registry. This is a critical safety mechanism.

**Size:** M

**Requirements satisfied:**
- Section 4: Guardrails Registry (individual checks + composition)
- Section 8: Guardrail Composition Algorithm (short-circuit evaluation)
- Section 9: Always Do — Guardrail Execution (composition, short-circuit, fallback)

**Files affected:**
- `src/engine/guardrail.py` (new)
- `src/engine/__init__.py` (export GuardrailFn, make_guardrail)
- `tests/test_guardrails.py` (new)

**Tasks:**
- [ ] Write test: `make_guardrail(check1, check2)` returns GuardrailFn
- [ ] Write test: Composed guardrail short-circuits on first failure (returns immediately, doesn't call check3)
- [ ] Implement `GuardrailFn` type alias
- [ ] Implement `make_guardrail(*checks)` composition function with short-circuit logic
- [ ] Write test: `check_transition_allowed(state_fetch_to_validate)` returns GuardrailResult(passed=True)
- [ ] Write test: `check_transition_allowed(state_fetch_to_store)` returns GuardrailResult(passed=False, fallback=State.ERROR)
- [ ] Implement `check_transition_allowed(state)` using `is_transition_allowed()`
- [ ] Write test: `check_retry_budget_with_error_type` passes if retry_count ≤ 3, fails if > 3, and rejects if error_type="permanent" immediately
- [ ] Implement `check_retry_budget_with_error_type(state)` with MAX_RETRIES=3
- [ ] Implement `check_raw_data_present()`, `check_validated_data_present()`, `check_enriched_data_present()` (simple None checks)
- [ ] Implement `check_document_size(state)` with MAX_SIZE_BYTES=10MB
- [ ] Implement `check_fallback_depth(state)` with max depth=2
- [ ] Implement `check_pipeline_timeout(state)` using time.time() and state["started_at"]
- [ ] Write test: Each individual check returns GuardrailResult with correct passed, reason, fallback
- [ ] Build `GUARDRAILS` registry dict mapping State → composed GuardrailFn (per Section 4.6)
- [ ] Write test: GUARDRAILS[State.FETCH] composition runs 3 checks in order
- [ ] Write test: GUARDRAILS[State.ERROR] always passes (error is always reachable)

**Success criteria:**
- `make_guardrail()` short-circuits on first failure (verified by count of check invocations)
- All 7 individual checks implemented and tested in isolation
- GUARDRAILS registry has entries for all 8 states
- Composition tests verify that short-circuit prevents downstream checks from running
- check_fallback_depth correctly limits to depth ≤ 2
- check_pipeline_timeout correctly reads started_at and node_timeout_seconds

**Commit message:**
```
feat: implement guardrail framework with composable checks (high-risk safety)

- GuardrailFn type alias for reusable check functions
- make_guardrail(*checks) composition with short-circuit evaluation
- 7 individual guardrail checks:
  - check_transition_allowed: ALLOWED_TRANSITIONS validation
  - check_retry_budget_with_error_type: MAX_RETRIES with error_type distinction
  - check_raw_data_present, check_validated_data_present, check_enriched_data_present: data presence
  - check_document_size: DoS protection (max 10MB)
  - check_fallback_depth: cascade loop detection (max 2 redirects)
  - check_pipeline_timeout: hang prevention (default 60s)
- GUARDRAILS registry mapping State → composed checks
- Comprehensive unit tests for composition and each check
```

**Git worktree:** `git worktree add ../worktrees/phase-3-guardrails -b phase/guardrails-3`

---

### Phase 4 — Router & Timeout Guards (High-Risk)
**Goal:** Implement the Router node (routing table lookup) and ensure timeout checks are integrated. Router is simple but critical; timeout guard prevents hangs.

**Size:** S

**Requirements satisfied:**
- Section 2: Components — Router Node
- Section 8: Router Algorithm
- Section 6: Latency — timeout enforcement
- Section 9: Always Do — timeout checks

**Files affected:**
- `src/engine/router.py` (new)
- `src/engine/__init__.py` (export router, HAPPY_PATH)
- `tests/test_router.py` (new)

**Tasks:**
- [ ] Write test: `router()` with State.INIT returns state with proposed_next=State.FETCH
- [ ] Write test: `router()` with State.FETCH returns proposed_next=State.VALIDATE
- [ ] Write test: `router()` with State.RETRY returns proposed_next=State.FETCH (back to start)
- [ ] Write test: `router()` with State.HUMAN_REVIEW returns proposed_next=State.ENRICH
- [ ] Implement `HAPPY_PATH` dict (routing table per Section 7.2)
- [ ] Implement `router(state: PipelineState) → PipelineState` that reads current_state, looks up proposed_next, appends to audit_trail
- [ ] Write test: `router()` appends "router: A → B" to audit_trail
- [ ] Write test: `router()` returns unchanged state dict except for proposed_next and audit_trail
- [ ] Write test: Timeout check at router (call check_pipeline_timeout) returns error if elapsed > 60s
- [ ] Implement timeout check in router (or ensure guardrail_node calls it)

**Success criteria:**
- Router output proposed_next matches HAPPY_PATH[current_state] for all states
- Router appends correct audit trail entry
- Router does not modify other state fields
- Timeout guard correctly detects and rejects exceeded timeouts

**Commit message:**
```
feat: implement router and timeout guards (high-risk safety)

- HAPPY_PATH routing table (INIT→FETCH, FETCH→VALIDATE, etc.)
- router(state) function: pure code routing with audit trail
- Timeout enforcement: check_pipeline_timeout integrated
- Unit tests for all routing paths
- Tests verify timeout detection and abort
```

**Git worktree:** `git worktree add ../worktrees/phase-4-router -b phase/router-4`

---

### Phase 5 — Handler Architecture & Exception Handling Pattern (High-Risk)
**Goal:** Establish the handler execution contract, exception handling pattern (catch ALL, log exc_info, set error_type), and error_type classification. This prevents graph crashes and enables smart recovery.

**Size:** M

**Requirements satisfied:**
- Section 4: Handler Signature (enhanced with error_type and exception handling)
- Section 5: Handler Execution (contract and error handling)
- Section 9: Always Do — Handler Execution (exception handling, error_type)
- Section 9: Never Do — Handler Violations (catch ALL exceptions, don't raise)

**Files affected:**
- `src/pipeline/handlers.py` (new; template and utilities)
- `src/pipeline/__init__.py` (export handler decorator/base)
- `tests/test_handlers_exceptions.py` (new)

**Tasks:**
- [ ] Write test: Handler that raises TimeoutError is caught; state returned with error_type="transient", error_message set, current_state set
- [ ] Write test: Handler that raises DocumentNotFound is caught; state returned with error_type="permanent", current_state set
- [ ] Write test: Handler that raises unexpected exception (e.g., TypeError) is caught; state returned with error_type="permanent" (assume permanent unless classified)
- [ ] Write test: Handler logs with exc_info=True (verified by checking log records)
- [ ] Implement handler template with try/except ALL pattern
- [ ] Implement error_type classification in handlers (transient for network/timeout; permanent for logic/not-found)
- [ ] Implement logging pattern: log.error(..., exc_info=True) for exceptions
- [ ] Write test: Handler does NOT raise exceptions (all caught and returned as state)
- [ ] Write test: Handler MUST set current_state on return (verified by state["current_state"] == expected)
- [ ] Write test: Handler with no exception returns error_type=None

**Success criteria:**
- All exception types (TimeoutError, DocumentNotFound, ValueError, etc.) are caught
- error_type is set correctly ("transient", "permanent", or None)
- error_message is populated on exception
- current_state is always set (even on exception)
- exc_info=True is present on all exception logs
- No exceptions propagate out of handler (all caught)

**Commit message:**
```
feat: establish handler architecture and exception handling pattern (high-risk safety)

- Handler contract: (state) → state with current_state always set
- Exception handling: catch ALL exceptions, never raise
- Error type classification: transient (retry), permanent (escalate), None (success)
- Logging: exc_info=True for debugging
- Handler template with try/except/except pattern
- Unit tests for all exception types and error classifications
```

**Git worktree:** `git worktree add ../worktrees/phase-5-handlers-arch -b phase/handlers-arch-5`

---

### Phase 6 — Implement Specific Handlers
**Goal:** Implement all 8 handler functions (FETCH, VALIDATE, ENRICH, STORE, RETRY, HUMAN_REVIEW, COMPLETE, ERROR) following the pattern from Phase 5. These execute the business logic.

**Size:** M

**Requirements satisfied:**
- Section 2: Components — Handler Nodes (all 8 handlers)
- Section 6: Use Cases (happy path, retry, human review, error paths)
- Section 9: Always Do — Handler Execution (clear stale data on RETRY, set current_state)

**Files affected:**
- `src/pipeline/handlers.py` (add 8 handler functions)
- `src/pipeline/agents.py` (new; mock LLM agents for testing)
- `tests/test_handlers_integration.py` (new; happy path, retry, error paths)

**Tasks:**
- [ ] Write test: Happy path FETCH → VALIDATE → ENRICH → STORE → COMPLETE produces correct audit trail
- [ ] Implement `handle_fetch()` with simulated success case
- [ ] Implement mock `VALIDATE_AGENT.run()` (stub that returns fixed result)
- [ ] Implement `handle_validate()` using VALIDATE_AGENT
- [ ] Implement mock `ENRICH_AGENT.run()`
- [ ] Implement `handle_enrich()` using ENRICH_AGENT
- [ ] Implement `handle_store()` (simple state mutation)
- [ ] Implement `handle_complete()` (terminal success)
- [ ] Implement `handle_retry()` with retry_count increment and raw_data clear
- [ ] Write test: `handle_retry()` increments retry_count by exactly 1 (with assertion)
- [ ] Implement mock `REVIEW_AGENT.run()`
- [ ] Implement `handle_human_review()` using REVIEW_AGENT
- [ ] Implement `handle_error()` (terminal error)
- [ ] Create HANDLER_MAP dict mapping State → handler function
- [ ] Write test: Each handler sets current_state correctly
- [ ] Write test: RETRY clears raw_data; next FETCH re-fetches
- [ ] Write test: RETRY increments retry_count (assertion catches if off-by-one)
- [ ] Write test: Happy path produces 16 audit entries (per Section 3.1)

**Success criteria:**
- All 8 handlers implemented and pass tests
- HANDLER_MAP covers all non-ERROR states (7 handlers)
- Happy path test verifies correct state transitions and audit trail
- RETRY increments by exactly 1 (assertion guards)
- RETRY clears stale raw_data
- Each handler sets current_state to its own state value
- Mock agents return consistent results for testing

**Commit message:**
```
feat: implement 8 state handlers for document processing pipeline

- handle_fetch: retrieve document (simulated)
- handle_validate: LLM validation (mocked VALIDATE_AGENT)
- handle_enrich: LLM enrichment (mocked ENRICH_AGENT)
- handle_store: persist record (simulated)
- handle_retry: increment counter, clear stale data, loop back
- handle_human_review: manual approval (mocked REVIEW_AGENT)
- handle_complete: terminal success
- handle_error: terminal error
- HANDLER_MAP registry mapping State → handler function
- Mock LLM agents for testing (no real API calls)
- Integration tests: happy path, retry flow, error handling
```

**Git worktree:** `git worktree add ../worktrees/phase-6-handlers -b phase/handlers-6`

---

### Phase 7 — LangGraph StateGraph Assembly
**Goal:** Build the LangGraph StateGraph with nodes, edges, conditional routing, and compilation. This is the graph execution engine.

**Size:** M

**Requirements satisfied:**
- Section 2: High-Level Architecture (complete graph structure)
- Section 2.1: Execution Flow (all steps from entry to END)
- Section 5: Internal — build_graph()

**Files affected:**
- `src/engine/workflow.py` (new; build_graph function)
- `src/engine/__init__.py` (export build_graph)
- `tests/test_graph_structure.py` (new)

**Tasks:**
- [ ] Write test: `build_graph()` returns compiled StateGraph
- [ ] Implement `build_graph()` function that creates StateGraph
- [ ] Add router node: `g.add_node("router", router)`
- [ ] Add guardrail node: `g.add_node("guardrail", guardrail_node)`
- [ ] Implement `guardrail_node(state)` function that runs GUARDRAILS[proposed_next] and updates state
- [ ] Implement `guardrail_router(state)` that returns state["proposed_next"] (for conditional edge)
- [ ] Add 8 handler nodes: `g.add_node(State.FETCH.value, handle_fetch)`, etc.
- [ ] Set entry point: `g.set_entry_point("router")`
- [ ] Add edges: router → guardrail (always)
- [ ] Add conditional edge: guardrail → {fetch, validate, enrich, store, complete, retry, human_review, error} based on guardrail_router
- [ ] Add loop-back edges: non-terminal handlers → router
- [ ] Add terminal edges: COMPLETE → END, ERROR → END
- [ ] Compile graph: `return g.compile()`
- [ ] Write test: Graph invocation with initial_state returns final_state
- [ ] Write test: Happy path (INIT → COMPLETE) produces 16 audit entries
- [ ] Write test: Graph respects TERMINAL_STATES and exits to END

**Success criteria:**
- Graph compiles without errors
- Graph has 10 nodes (router, guardrail, 8 handlers)
- Graph has all edges (entry, router→guardrail, conditional, loop-back, terminal)
- Happy path invocation produces correct sequence of states
- Audit trail grows through each iteration
- Terminal states cause graph to exit to END

**Commit message:**
```
feat: assemble LangGraph StateGraph with nodes, edges, and conditional routing

- build_graph() creates StateGraph with 10 nodes and proper edges
- Node: router (propose next state)
- Node: guardrail (validate and fallback if needed)
- Nodes: 8 handlers (FETCH, VALIDATE, ENRICH, STORE, RETRY, HUMAN_REVIEW, COMPLETE, ERROR)
- Edges: entry → router, router → guardrail, guardrail → (conditional) handlers
- Loop-back: non-terminal handlers → router
- Terminal: COMPLETE → END, ERROR → END
- Conditional edge uses guardrail_router() to read proposed_next
- Unit tests verify graph structure, compilation, and execution
```

**Git worktree:** `git worktree add ../worktrees/phase-7-graph -b phase/graph-7`

---

### Phase 8 — Session Management & Checkpoint Mechanism (High-Risk)
**Goal:** Implement `run_pipeline()` entry point with input validation, checkpoint loading, and graph invocation. This is the public API.

**Size:** M

**Requirements satisfied:**
- Section 5: API Endpoints — run_pipeline() and run_pipeline_with_checkpoint()
- Section 5: Input validation (document_id, size, timeout)
- Section 9: Always Do — validate input, initialize state, save checkpoints

**Files affected:**
- `src/pipeline/workflow.py` (new; run_pipeline, checkpoint functions)
- `src/pipeline/__init__.py` (export run_pipeline)
- `tests/test_session_checkpoint.py` (new)

**Tasks:**
- [ ] Write test: `run_pipeline("DOC-001")` returns final PipelineState
- [ ] Write test: `run_pipeline("")` raises ValueError ("document_id cannot be empty")
- [ ] Write test: `run_pipeline("x" * 300)` raises ValueError (exceeds 256 chars)
- [ ] Implement `run_pipeline(document_id, session_id, initial_state, timeout_seconds)` function
- [ ] Implement input validation: non-empty document_id, ≤ 256 chars
- [ ] Implement state initialization: set started_at = time.time(), node_timeout_seconds = timeout_seconds, fallback_depth = 0, audit_trail = ["init"]
- [ ] Implement checkpoint loading: if session_id provided, load checkpoint (stub for now; full implementation in Phase 9)
- [ ] Invoke graph: `graph.invoke(initial_state)` and return final_state
- [ ] Write test: run_pipeline returns state with current_state, audit_trail, error_message (if error)
- [ ] Write test: Timeout parameter is passed and enforced (default 60s)
- [ ] Implement `run_pipeline_with_checkpoint(session_id, checkpoint_key)` stub (basic version)
- [ ] Write test: run_pipeline happy path produces COMPLETE state
- [ ] Write test: run_pipeline error path produces ERROR state with error_message

**Success criteria:**
- `run_pipeline()` accepts document_id and optional parameters
- Input validation catches empty, too-long, and invalid document_ids
- Initial state is created with all required fields (started_at, timeout, audit_trail)
- Graph invocation works and returns final state
- timeout_seconds parameter flows through to guardrails
- Happy and error paths tested end-to-end

**Commit message:**
```
feat: implement session management and run_pipeline entry point

- run_pipeline(document_id, session_id=None, initial_state=None, timeout_seconds=60)
- Input validation: non-empty, ≤256 chars document_id
- Initial state creation with started_at, timeout, audit_trail
- Checkpoint loading stub (full implementation in Phase 9)
- Graph invocation and return of final PipelineState
- run_pipeline_with_checkpoint() stub
- Tests: happy path, error path, validation, timeout passing
```

**Git worktree:** `git worktree add ../worktrees/phase-8-session -b phase/session-8`

---

### Phase 9 — Observability, Logging, & Audit Trail Enforcement
**Goal:** Ensure audit trail capping, structured logging, metrics hooks, and verification that all safety boundaries are enforced.

**Size:** S

**Requirements satisfied:**
- Section 6: Monitoring, Observability, Metrics & Logging
- Section 9: Always Do — audit trail, logging
- Section 9: Never Do — audit trail overflow, missing exc_info

**Files affected:**
- `src/engine/guardrail.py` (update to log guardrail decisions)
- `src/pipeline/handlers.py` (update logging)
- `src/engine/workflow.py` (audit trail capping logic)
- `tests/test_observability.py` (new)

**Tasks:**
- [ ] Write test: Audit trail is capped at 1000 entries (oldest trimmed)
- [ ] Implement audit trail capping: `audit_trail[-1000:]` in guardrail_node
- [ ] Write test: Router logs "[Router] X → proposes Y"
- [ ] Write test: Guardrail logs "[Guardrail] ✅ X passed" on pass, "[Guardrail] ❌ X failed → fallback Y" on fail
- [ ] Add logging to guardrail_node (use log.info for pass, log.warning for fail)
- [ ] Write test: Handler logs entry and exception with exc_info=True
- [ ] Verify logging present in all handlers (check log fixtures)
- [ ] Write test: Timeout detection logs "[Guardrail] ❌ Pipeline timeout (Xs > 60s)"
- [ ] Write test: Cascade detection logs "[Guardrail] ❌ Fallback cascade detected"
- [ ] Write test: Audit trail never contains stack traces (only step names and reasons)
- [ ] Create metrics hooks: places where metrics collection can attach (e.g., after guardrail check)

**Success criteria:**
- Audit trail capped at 1000 entries (verified by overflow test)
- All logging statements present and use correct log levels
- Exception logs include exc_info=True
- No secrets (API keys, PII) in logs or audit trail
- All safety boundaries enforced and observable via logs

**Commit message:**
```
feat: implement observability, logging, and audit trail enforcement

- Audit trail capping: max 1000 entries (trim oldest on overflow)
- Structured logging in router, guardrail, handlers
- Exception logging with exc_info=True for debugging
- Metrics collection hooks for monitoring
- Tests verify logging, capping, and no secrets exposed
```

**Git worktree:** `git worktree add ../worktrees/phase-9-observability -b phase/observability-9`

---

### Phase 10 — Integration Testing, Documentation & Reference Implementation
**Goal:** Full end-to-end test scenarios covering all paths (happy, retry, human review, timeout, cascade), plus API docs and reference examples.

**Size:** M

**Requirements satisfied:**
- Section 3: Workflow / Use Cases (all 5 use cases from 3.5)
- Section 6: Reliability — edge cases (9 tested)
- Reference implementation for teams building on this

**Files affected:**
- `tests/test_integration_e2e.py` (new; 5+ integration test scenarios)
- `docs/IMPLEMENTATION_GUIDE.md` (new; how to use the API)
- `examples/basic_pipeline_demo.py` (new; working example)

**Tasks:**
- [ ] Write integration test: Happy path (INIT → COMPLETE) with correct audit trail
- [ ] Write integration test: Retry path (FETCH fails once → RETRY → FETCH succeeds)
- [ ] Write integration test: Human review path (VALIDATE fails → HUMAN_REVIEW → ENRICH → COMPLETE)
- [ ] Write integration test: Permanent error path (handler sets error_type="permanent" → ERROR)
- [ ] Write integration test: Timeout exceeded → ERROR
- [ ] Write integration test: Cascade detection (VALIDATE fallback → HUMAN_REVIEW → ENRICH fails → fallback > 2 → ERROR)
- [ ] Write integration test: Oversized document rejected at entry
- [ ] Write integration test: Retry count assertion (RETRY increments by 1, not 2)
- [ ] Write IMPLEMENTATION_GUIDE.md: how to implement a new pipeline (extend handlers, routes, guardrails)
- [ ] Create examples/basic_pipeline_demo.py with sample run_pipeline() calls
- [ ] Run full test suite: `pytest tests/ -v --cov=src --cov-fail-under=80`
- [ ] Verify all 10 phases' tests pass

**Success criteria:**
- All 5 use case paths tested end-to-end and pass
- All 9 edge cases tested and pass
- Test coverage ≥ 80% (guardrails, handlers, router, graph)
- IMPLEMENTATION_GUIDE explains how to build a new pipeline on this engine
- Example code runs without errors
- All prior phases' tests still pass (regression test)

**Commit message:**
```
feat: add integration tests, documentation, and reference implementation

- 5 end-to-end integration tests covering all use cases:
  1. Happy path (INIT → COMPLETE)
  2. Retry path (transient failure + recovery)
  3. Human review path (validation failure + manual approval)
  4. Permanent error path (unrecoverable failure)
  5. Timeout and cascade detection paths
- 9 edge case tests (oversized doc, retry count, assertions, etc.)
- IMPLEMENTATION_GUIDE.md: how to build a new pipeline
- examples/basic_pipeline_demo.py: working reference code
- Full test coverage ≥ 80% with all phases tested
- Ready for production and team reuse
```

**Git worktree:** `git worktree add ../worktrees/phase-10-integration -b phase/integration-10`

---

## Checkpoints

### Checkpoint 1: After Phases 1–3
**Status:** Foundation + Core Safety (Guardrails)

**Pre-merge checklist:**
- [ ] All tests pass: `pytest tests/test_state_machine.py tests/test_guardrails.py -v`
- [ ] Coverage ≥ 80%: `pytest --cov=src/engine --cov-fail-under=80`
- [ ] No linting issues: `ruff check src/`
- [ ] Guard composition short-circuits verified
- [ ] All 8 states in ALLOWED_TRANSITIONS
- [ ] Review: Are guardrails logic correct? (fallbacks sensible?)

**Decision:** 
- ✅ Proceed to Phase 4 if checkpoint passes
- 🚫 Stop and fix if guardrail logic flawed (impacts all downstream phases)

---

### Checkpoint 2: After Phases 4–6
**Status:** High-Risk Safety + Handler Logic

**Pre-merge checklist:**
- [ ] All tests pass: `pytest tests/test_router.py tests/test_handlers_exceptions.py tests/test_handlers_integration.py -v`
- [ ] Coverage ≥ 80%
- [ ] Timeout enforcement verified (test shows timeout abort)
- [ ] Error type classification tested (transient vs permanent behavior different)
- [ ] RETRY increment assertion in code
- [ ] Happy path test produces correct audit trail (16 entries, Section 3.1)
- [ ] Review: Are handlers idempotent? Are error types classified correctly?

**Decision:**
- ✅ Proceed to Phase 7 if all safety checks pass
- 🚫 Stop and fix if handler exception handling incomplete (impacts graph reliability)

---

### Checkpoint 3: After Phases 7–10
**Status:** Full Implementation + Integration

**Pre-merge checklist:**
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage ≥ 80% across entire src/
- [ ] No linting issues: `ruff check src/`
- [ ] All 5 use case integration tests pass
- [ ] All 9 edge case tests pass
- [ ] Example code runs without errors
- [ ] IMPLEMENTATION_GUIDE complete and reviewed
- [ ] Ready for `cg-review` and production deployment

**Decision:**
- ✅ Proceed to code review and merge if all tests pass and doc complete
- 🚫 Stop and fix if integration test failures (indicates architectural issue)

---

## Parallelization

### Safe to Parallelize
These phases can run in parallel (independent features, no shared code):

| Phase | Phase | Reason |
|-------|-------|--------|
| Phase 2 (State Models) | Phase 1 (Setup) | Sequential: setup must complete first |
| Phase 3 (Guardrails) | Phase 2 (State Models) | Sequential: guardrails depend on State enum |
| Phase 4 (Router) | Phase 3 (Guardrails) | Sequential: router defined by guardrails |

**Phases 1–4 are sequential chain** (each depends on prior).

| Phase | Phase | Reason |
|-------|-------|--------|
| Phase 5 (Handler Arch) | Phase 4 (Router) | Sequential: handlers follow from State/Router |
| Phase 6 (Handlers) | Phase 5 (Handler Arch) | Sequential: implementation follows architecture |
| Phase 7 (Graph) | Phase 6 (Handlers) | Sequential: graph assembles handlers |

**Phases 5–7 are sequential chain** (each depends on prior).

| Phase | Phase | Reason |
|-------|-------|--------|
| Phase 8 (Session) | Phase 7 (Graph) | Sequential: session API wraps graph |
| Phase 9 (Observability) | Phase 8 (Session) | Mostly independent: can add logging to prior phases, but safer after core works |
| Phase 10 (Integration) | Phase 9 (Observability) | Sequential: integration tests depend on all prior |

**Phases 8–10 are sequential chain** (each depends on prior, or enhances prior).

### Must Be Sequential
**All phases must be sequential** (Phase N depends on Phase N-1).

The three "chains" are:
1. **Chain 1 (Phases 1–4):** Setup → Data Models → Guardrails → Router
2. **Chain 2 (Phases 5–7):** Handler Arch → Handlers → Graph
3. **Chain 3 (Phases 8–10):** Session → Observability → Integration

**Minimal path to completion:** 10 sequential phases (~5 hours agent work, ~1–2 days wall-clock including review).

**Note:** If running with multiple agents, Chains 2 and 3 can start in parallel with Chain 1, but only **after** Phase 4 completes. (Phase 5 doesn't depend on Phase 4's implementation, only on phases 1–3's data models and guarantees.)

Actually, more carefully:
- **Chain 1 (Setup + Safety):** Phases 1–4 (guardrails, router, timeout) — must be sequential and complete first
- **Chain 2 (Handlers + Graph):** Phases 5–7 — can start after Phase 3 (guardrails defined), but safer to wait for Phase 4 (router)
- **Chain 3 (Session + Integration):** Phases 8–10 — can start after Phase 7 (graph complete)

**Safe parallel arrangement:**
- **Worker 1:** Phases 1–4 (sequential; ~2 hours)
- **Worker 2:** Starts Phase 5 after Phase 3 completes (Phases 5–7; ~1.5 hours; parallel with Phase 4)
- **Worker 3:** Starts Phase 8 after Phase 7 completes (Phases 8–10; ~1.5 hours; parallel with Phases 1–7)

**Wall-clock with 3 workers:** ~2 hours + max(1.5, 1.5) = ~3.5 hours total (vs 5 hours sequential).

---

## TODO List (Ordered)

- [ ] **Phase 1** — Project Structure & Dependencies
- [ ] **Checkpoint 1:** Tests pass, coverage ≥ 80%, guardrails logic reviewed
- [ ] **Phase 2** — State Machine & Core Data Models
- [ ] **Phase 3** — Guardrail Framework (High-Risk Safety)
- [ ] **Phase 4** — Router & Timeout Guards (High-Risk)
- [ ] **Phase 5** — Handler Architecture & Exception Handling Pattern (High-Risk)
- [ ] **Phase 6** — Implement Specific Handlers
- [ ] **Phase 7** — LangGraph StateGraph Assembly
- [ ] **Checkpoint 2:** All tests pass, timeout verified, happy path audit trail correct
- [ ] **Phase 8** — Session Management & Checkpoint Mechanism (High-Risk)
- [ ] **Phase 9** — Observability, Logging, & Audit Trail Enforcement
- [ ] **Phase 10** — Integration Testing, Documentation & Reference Implementation
- [ ] **Checkpoint 3:** Full test coverage, integration tests pass, ready for review

---

## Implementation Notes

### File Structure (Target)
```
src/
├── engine/
│   ├── __init__.py
│   ├── state_machine.py         # Phase 2
│   ├── guardrail.py             # Phase 3
│   ├── router.py                # Phase 4
│   └── workflow.py              # Phase 7
├── pipeline/
│   ├── __init__.py
│   ├── handlers.py              # Phases 5–6
│   ├── agents.py                # Phase 6 (mock agents)
│   └── workflow.py              # Phase 8

tests/
├── test_state_machine.py         # Phase 2
├── test_guardrails.py            # Phase 3
├── test_router.py                # Phase 4
├── test_handlers_exceptions.py   # Phase 5
├── test_handlers_integration.py  # Phase 6
├── test_graph_structure.py       # Phase 7
├── test_session_checkpoint.py    # Phase 8
├── test_observability.py         # Phase 9
├── test_integration_e2e.py       # Phase 10
└── conftest.py                   # Fixtures (Phase 1)

docs/
├── IMPLEMENTATION_GUIDE.md       # Phase 10

examples/
└── basic_pipeline_demo.py        # Phase 10
```

### Key Imports & Dependencies
- `from langgraph.graph import END, StateGraph` — core LangGraph
- `from typing_extensions import TypedDict` — Python 3.10+ compatibility
- `import logging` — structured logging
- `import time` — timeout tracking
- `import json` — document size checks
- `from dataclasses import dataclass` — GuardrailResult
- `from enum import Enum` — State enum

### Testing Strategy
- **Unit tests:** Each guardrail check, router, handler in isolation
- **Integration tests:** Graph execution with mocked agents
- **E2E tests:** Full scenarios (happy path, retry, error, timeout, cascade)
- **Mocking:** All external APIs (VALIDATE_AGENT, ENRICH_AGENT, REVIEW_AGENT) mocked
- **Coverage:** Minimum 80% required; higher for safety-critical code (guardrails, exception handling)

### Commit Strategy
- **One atomic commit per phase** — all work for a phase in one commit
- **Descriptive message:** What was implemented, why, what it enables
- **Worktree per phase:** Isolated branches prevent accidental mixing

### Code Quality Gates
- **Linting:** `ruff check src/` must pass (no unused imports, f-string errors, etc.)
- **Type hints:** All functions annotated; `mypy` optional but encouraged
- **Tests:** `pytest tests/ -v --cov=src --cov-fail-under=80`
- **Audit:** Each phase reviewed at checkpoint before proceeding

---

## Risk Mitigation

| Risk | Phase | Mitigation |
|------|-------|-----------|
| Guardrail logic bug | 3 | Unit test each check; short-circuit tested |
| Exception escapes handler | 5 | `except Exception` (catch-all) pattern; tests for all exception types |
| Timeout not enforced | 4, 8 | check_pipeline_timeout in guardrails; test with slow handler |
| Cascade loop infinite | 3 | check_fallback_depth capped at 2; test with design that triggers loop |
| Audit trail exhausts memory | 9 | Cap at 1000 entries; test overflow |
| Graph structure broken | 7 | Unit tests for graph structure, edges, conditional routing |
| Input validation missing | 8 | Validate at entry point; test invalid inputs |
| Retry count off-by-one | 6 | Assertion in RETRY handler; test catches increment error |

---

## Success Criteria (Overall)

By end of Phase 10, the implementation must satisfy:

1. **Functional:** All 5 use cases tested and working (happy path, retry, human review, permanent error, timeout)
2. **Safety:** All 5 high-risk items (guardrails, timeouts, exception handling, cascade detection, input validation) verified
3. **Quality:** ≥80% test coverage, 0 linting errors, no regressions
4. **Usability:** IMPLEMENTATION_GUIDE complete; reference example runs
5. **Production-ready:** No unhandled exceptions; timeouts enforced; DoS protections in place; audit trail complete and capped

---

_End of Implementation Plan_
