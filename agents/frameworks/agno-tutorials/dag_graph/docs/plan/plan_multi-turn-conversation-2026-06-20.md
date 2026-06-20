# Implementation Plan: Multi-Turn Conversation System

**Date:** 2026-06-20  
**Input:** docs/design/design_spec_multi-turn-conversation-2026-06-20.md  
**Project:** DAG Graph — Agno Workflows  

---

## Summary

Implement a multi-turn conversation system for the document processing pipeline that enables interactive, request/response workflows while maintaining backward compatibility with existing one-turn document processing. The system adds semantic routing (LLM-powered state classification), handler metadata for pause/resume control, input validation for security, and session persistence via Agno.

Delivered across 10 atomic phases:
- **Phases 1–3:** Core engine infrastructure (control plane, handler registry, input validation)
- **Phases 4–6:** Base semantic router and Agno integration
- **Phases 7–8:** Workflow integration and documentation
- **Phases 9–10:** Testing and verification

---

## High-Risk Items (Scheduled in Phases 2–4)

1. **Input Validation & Prompt Injection Prevention** (Phase 2)
   - Token counting and length validation prevent DoS attacks
   - Escaping for LLM calls prevents prompt injection
   - Must be implemented before router is called

2. **Semantic Router (LLM Integration)** (Phase 4)
   - LLM-powered state classification is the heart of multi-turn
   - Timeout handling and invalid transition retry logic must be robust
   - Determines workflow semantics (entities, intents)

3. **Handler Metadata & Registry** (Phase 3)
   - @handler decorator must work before workflow dispatcher uses metadata
   - waits_for_input flag controls pause/resume behavior
   - Must be in place before process_turn() is built

4. **Backward Compatibility** (All phases)
   - Existing one-turn workflows must not break
   - process() method unchanged, process_turn() is new
   - One-turn workflows ignore multi-turn fields

---

## Task Sizing

| Task | Size | Rationale |
|------|------|-----------|
| Engine state split (control + business) | S | Create EngineState TypedDict, minimal code |
| Input validation module | S | Token counting, length checks, escaping function |
| Handler registry + @handler decorator | S | Decorator, metadata dict, lookup function |
| BaseSemanticRouter abstract class | S | Abstract base with single route() method |
| DocPipelineRouter implementation | M | Claude LLM integration, prompt engineering, retry logic |
| Engine workflow modifications | M | Add _semantic_router_step(), make router pluggable |
| Agno session integration | M | DB setup, session_id/user_id handling, persistence patterns |
| Workflow integration | M | process_turn() method, _trim_history(), error handling |
| Handler decorators + exception handling | S | Add @handler to all handlers, wrap in try/catch |
| Integration tests + documentation | M | Multi-turn flow tests, single-turn backward compat tests |

**Total: 10 phases, ~5 hours of implementation work**

---

## Phases

### Phase 1 — Split Pipeline State: Engine (Control Plane) ✅

**Goal:** Create EngineState TypedDict in engine layer with all multi-turn control fields (turn management, semantic context, state machine control).

**Size:** S

**Requirements satisfied:**
- Type-safe state enums, TypedDict for state shape (spec §4.1)
- Full backward compatibility with one-turn workflows (spec §6.7)

**Files affected:**
- `src/engine/pipeline_state.py` (new)

**Tasks:**
- [x] Write tests for EngineState initialization
- [x] Implement EngineState TypedDict with all control fields
- [x] Implement init_engine_state() factory function
- [x] Implement audit() helper for immutable-style updates
- [x] Verify tests pass; no existing tests broken

**Success criteria:**
- EngineState TypedDict has fields: turn_input, turn_number, turns, semantic_context, conversation_id, max_history_turns, current_state, proposed_next, retry_count, error_message, guardrail_ok, audit_trail
- init_engine_state() returns fresh state with sensible defaults
- audit() correctly appends entries to audit_trail immutably
- All tests pass

**Commit message:** `feat: add EngineState TypedDict and control plane initialization in engine layer`

---

### Phase 2 — Input Validation & Sanitization

**Goal:** Implement input validation and prompt injection prevention for turn_input before LLM calls. HIGH-RISK: Must prevent DoS and prompt injection attacks.

**Size:** S

**Requirements satisfied:**
- Input validation (length, token limits) (spec §5.1)
- Prevent prompt injection via turn_input (spec §6.3)

**Files affected:**
- `src/engine/input_validation.py` (new)

**Tasks:**
- [ ] Write tests for validate_turn_input(): valid, too long, too many tokens, non-string
- [ ] Write tests for escape_for_llm(): special chars, quotes, escape sequences
- [ ] Write tests for estimate_tokens(): accuracy within 10% of true count
- [ ] Implement InputValidationError exception
- [ ] Implement validate_turn_input(turn_input, max_chars=10000, max_tokens=2000)
- [ ] Implement escape_for_llm(turn_input) using repr()
- [ ] Implement estimate_tokens(text) as len(text) // 4
- [ ] Verify tests pass

**Success criteria:**
- Validates length (reject >10k chars) and tokens (reject >2k tokens)
- Escapes input safely for LLM prompts (prevents "Ignore instructions" injections)
- estimate_tokens() is within 10% of actual token count
- All error cases covered by tests
- All tests pass, no regressions

**Commit message:** `feat: add input validation and prompt injection prevention for turn_input`

---

### Phase 3 — Handler Registry with @handler Decorator

**Goal:** Create @handler decorator and metadata registry so handlers can declare waits_for_input and description. HIGH-RISK: workflow dispatcher will depend on this metadata.

**Size:** S

**Requirements satisfied:**
- Step metadata to distinguish immediate-execution vs. wait-for-input steps (spec §5.3)
- Handler metadata binding (spec §2.2)

**Files affected:**
- `src/engine/handler_registry.py` (new)

**Tasks:**
- [ ] Write tests for @handler decorator: stores metadata correctly, multiple handlers, accessor functions
- [ ] Write tests for get_handler_metadata(state): returns metadata, handles missing states
- [ ] Write tests for does_state_wait_for_input(state): correctly reads waits_for_input flag
- [ ] Implement HandlerMetadata dataclass
- [ ] Implement HANDLER_MAP_METADATA global registry
- [ ] Implement @handler(state, waits_for_input=False, description=None) decorator
- [ ] Implement get_handler_metadata(state) lookup
- [ ] Implement does_state_wait_for_input(state) helper
- [ ] Verify tests pass

**Success criteria:**
- @handler decorator correctly registers metadata in HANDLER_MAP_METADATA
- Metadata includes state, waits_for_input, description
- Accessor functions (get_handler_metadata, does_state_wait_for_input) work correctly
- Can stack multiple decorators and retrieve metadata for each
- All tests pass

**Commit message:** `feat: add @handler decorator and metadata registry for step configuration`

---

### Phase 4 — Base Semantic Router Abstract Class

**Goal:** Create BaseSemanticRouter abstract class in engine layer. HIGH-RISK: This is the interface that all routers must implement. Must define contract clearly.

**Size:** S

**Requirements satisfied:**
- LLM-powered semantic router for next-state classification (spec §2.2, §8.1)
- Router decision output structure (spec §4.3)

**Files affected:**
- `src/engine/router.py` (new)

**Tasks:**
- [ ] Write tests for BaseSemanticRouter.route() method signature validation
- [ ] Write tests for RouterDecision dataclass fields
- [ ] Write tests for abstract method enforcement (cannot instantiate base class)
- [ ] Implement RouterDecision dataclass with: proposed_next, confidence, semantic_entities, semantic_intents, reasoning
- [ ] Implement BaseSemanticRouter abstract base class
- [ ] Implement route(current_state, turn_input, history, allowed_states, timeout_sec) as abstract method
- [ ] Add docstring explaining router contract
- [ ] Verify tests pass

**Success criteria:**
- RouterDecision dataclass has correct fields with proper types
- BaseSemanticRouter cannot be instantiated directly (abstract)
- route() method signature matches spec §2.2
- Docstring clearly explains: input args, return value, what router must extract (entities, intents)
- All tests pass

**Commit message:** `feat: add BaseSemanticRouter abstract class and RouterDecision output structure`

---

### Phase 5 — DocPipelineRouter: LLM-Powered Implementation

**Goal:** Implement DocPipelineRouter in workflow layer: Claude LLM integration, domain-specific prompts, constraint retry logic.

**Size:** M

**Requirements satisfied:**
- LLM-powered semantic router for next-state classification (spec §2.2, §8.1)
- Semantic router with history context (spec §2.3, §5.1)
- Router fallback on invalid transition (spec §7.8, §8.1)

**Files affected:**
- `src/workflow/router.py` (new)

**Tasks:**
- [ ] Write tests for DocPipelineRouter.route(): happy path, invalid transition, timeout, entity extraction
- [ ] Write tests for _format_history(): empty history, recent turns only (max_turns)
- [ ] Write tests for _parse_json(): valid JSON in response, malformed JSON, no JSON in response
- [ ] Implement DocPipelineRouter(BaseSemanticRouter) subclass
- [ ] Implement __init__(model="claude-haiku-...") with Claude LLM
- [ ] Implement route() method: format prompt, call Claude, parse response, validate, retry if invalid
- [ ] Implement _format_history(history, max_turns=10) to render recent turns
- [ ] Implement _parse_json(text) to extract JSON from LLM response
- [ ] Implement constraint retry logic: if proposed_next not in allowed_states, retry with explicit constraints
- [ ] Handle TimeoutError → return ERROR RouterDecision
- [ ] Verify tests pass

**Success criteria:**
- Calls Claude Haiku LLM with turn_input, history, allowed_states in prompt
- Extracts semantic entities (amounts, items, document_ids, keywords)
- Extracts semantic intents (confirm, clarify, escalate, upload, cancel)
- Handles invalid transitions by retrying with constraint list
- Timeout (10s default) is configurable
- Returns RouterDecision with proposed_next, confidence, entities, intents, reasoning
- All error paths covered by tests (timeout, malformed response, invalid JSON)
- All tests pass

**Commit message:** `feat: implement DocPipelineRouter with Claude LLM and constraint retry logic`

---

### Phase 6 — Engine Workflow: Add Semantic Router Step

**Goal:** Extend StateMachineWorkflow in engine layer with _semantic_router_step() method. Make router choice pluggable: one-turn uses _router_step, multi-turn uses _semantic_router_step.

**Size:** M

**Requirements satisfied:**
- LLM-powered semantic router for next-state classification (spec §2.2)
- Semantic router with conversation history context (spec §2.3)
- Guardrail validation after router (spec §2.1, §7.8)

**Files affected:**
- `src/engine/workflow.py` (modify)

**Tasks:**
- [ ] Write tests for _semantic_router_step(): reads current_state, turn_input, history; calls router; stores decision in session
- [ ] Write tests for _semantic_router_step() timeout handling → ERROR state
- [ ] Write tests for _semantic_router_step() storing semantic_context in session_state
- [ ] Write tests for router pluggability: _choose_router_step() returns correct step based on turn_input presence
- [ ] Implement _semantic_router_step(step_input) method
- [ ] Extract current_state, turn_input, history, allowed_states from session_state
- [ ] Call self.router.route(...) with timeout
- [ ] Store decision in session_state: proposed_next, semantic_context, router_confidence
- [ ] Handle router errors → ERROR state
- [ ] Implement _choose_router_step() to select _router_step or _semantic_router_step based on mode
- [ ] Update docstring explaining one-turn vs multi-turn routing
- [ ] Verify tests pass, no existing tests broken

**Success criteria:**
- _semantic_router_step() correctly extracts context from session_state
- Calls router.route() with all required args (current_state, turn_input, history, allowed_states, timeout_sec)
- Stores router decision (proposed_next, entities, intents, confidence) in session_state
- Handles timeout errors gracefully → ERROR state with error_message
- One-turn workflows still use _router_step (no breaking changes)
- All tests pass

**Commit message:** `feat: add _semantic_router_step to engine.workflow and make router pluggable`

---

### Phase 7 — Engine Session: Multi-Turn Initialization

**Goal:** Extend engine/session.py with init_control_state_defaults() and append_turn() for multi-turn session management.

**Size:** S

**Requirements satisfied:**
- Per-turn checkpointing for resume capability (spec §5.1)
- Conversation history context (spec §2.3)
- Session state lifecycle (spec §2.2)

**Files affected:**
- `src/engine/session.py` (modify)

**Tasks:**
- [ ] Write tests for init_control_state_defaults(session_state): initializes all control fields, idempotent
- [ ] Write tests for append_turn(session_state, role, content, ...): appends to turns list, adds timestamp
- [ ] Write tests for append_turn() with optional state_from, state_to, router_confidence
- [ ] Implement init_control_state_defaults(session_state) with defaults from design spec §4.1
- [ ] Implement append_turn(session_state, role, content, state_from="", state_to="", router_confidence=0.0)
- [ ] Add timestamp to each turn entry using datetime.now(tz=timezone.utc)
- [ ] Verify turns list is created if missing
- [ ] Verify tests pass

**Success criteria:**
- init_control_state_defaults() initializes: turn_input, turn_number, turns, semantic_context, conversation_id, max_history_turns, current_state, proposed_next, retry_count, error_message, guardrail_ok, audit_trail
- All defaults match spec §4.1
- append_turn() appends correctly formatted turn entries to session_state["turns"]
- Timestamps are in ISO format
- turn_count is incremented (from existing append_turn() helper)
- All tests pass

**Commit message:** `feat: add init_control_state_defaults and append_turn helpers to engine.session`

---

### Phase 8 — Workflow: Split Pipeline State and Add process_turn()

**Goal:** (1) Modify workflow/pipeline_state.py to inherit from engine EngineState, add WorkflowState for business fields. (2) Add process_turn(user_id, session_id, turn_input, timeout_sec) method to DocPipelineWorkflow.

**Size:** M

**Requirements satisfied:**
- Turn-based execution model with process_turn() entry point (spec §5.1)
- Full backward compatibility with one-turn workflows (spec §6.7)
- Input validation before routing (spec §5.1, §7.12)
- Error recovery via handler exception handling (spec §3.3, §7.10)
- Configurable turn history window (spec §4.1, §7.9)

**Files affected:**
- `src/workflow/pipeline_state.py` (modify)
- `src/workflow/workflow.py` (modify)

**Tasks:**
- [ ] Write tests for process_turn(): happy path, input validation error, router error, handler exception
- [ ] Write tests for process_turn() return value: current_state, waits_for_input, turn_number, semantic_context, router_confidence, error
- [ ] Write tests for _trim_history(): keeps last max_history_turns, logs trimming event
- [ ] Write tests for _trim_history() with empty/small turn lists
- [ ] Modify pipeline_state.py: WorkflowState TypedDict with business fields (document_id, raw_data, validated_data, enriched_data)
- [ ] Combine: class PipelineState(EngineState, WorkflowState)
- [ ] Modify new_pipeline(document_id, conversation_id, max_history_turns) to include both control + business
- [ ] Add process_turn(user_id, session_id, turn_input, timeout_sec=10.0) to DocPipelineWorkflow
- [ ] Implement validation → escape → append to turns → run workflow → trim history → return response
- [ ] Implement _trim_history() to keep last max_history_turns, log dropped turns
- [ ] Return response dict: current_state, waits_for_input, turn_number, semantic_context, router_confidence, error
- [ ] Wrap in try/catch for InputValidationError and exceptions
- [ ] Keep process(document_id) unchanged for backward compatibility
- [ ] Verify tests pass, no existing process() tests broken

**Success criteria:**
- process_turn() validates input (length, tokens) before proceeding
- Escapes turn_input for LLM safety
- Appends turn to session_state["turns"]
- Runs workflow loop (router → guardrail → handler → checkpoint)
- Trims history to max_history_turns
- Returns response with waits_for_input flag from handler metadata
- Handles all error cases gracefully
- process() method still works for one-turn workflows
- All tests pass

**Commit message:** `feat: add process_turn() method and split pipeline_state into control + business planes`

---

### Phase 9 — Decorators & Exception Handling: Update All Handlers

**Goal:** Add @handler decorators to all handlers in workflow/handlers.py, wrap logic in try/catch for exception handling.

**Size:** S

**Requirements satisfied:**
- Error recovery (handler exceptions → ERROR state) (spec §3.3, §7.10)
- Handler metadata binding (spec §5.3, §2.2)
- Backward compatibility (existing handlers continue to work) (spec §6.7)

**Files affected:**
- `src/workflow/handlers.py` (modify)

**Tasks:**
- [ ] Write tests for decorated handlers: verify metadata is stored
- [ ] Write tests for handler exception handling: raises exception → caught → ERROR state with error_message
- [ ] Write tests for each handler with valid and error inputs
- [ ] Add @handler decorator to each handler: handle_init, handle_fetch, handle_validate, handle_enrich, handle_store, handle_complete, handle_retry, handle_human_review, handle_error
- [ ] Set waits_for_input=False for most handlers; True for wait_documents_uploaded (new handler)
- [ ] Wrap handler logic in try/catch: catch Exception → return ERROR state with error_message
- [ ] Add handle_wait_documents_uploaded() with @handler(state="wait_documents_uploaded", waits_for_input=True)
- [ ] Verify HANDLER_MAP still works (decorators don't break existing map)
- [ ] Verify tests pass

**Success criteria:**
- All handlers have @handler decorator with correct state and waits_for_input
- Handlers correctly catch exceptions and route to ERROR state
- error_message field is populated on handler exception
- audit trail includes exception info
- New wait_documents_uploaded handler is available for testing pause/resume
- No breaking changes to HANDLER_MAP
- All tests pass

**Commit message:** `feat: add @handler decorators and exception handling to all handlers, add wait_documents_uploaded`

---

### Phase 10 — Integration Tests & Documentation

**Goal:** Write integration tests for multi-turn flow, verify backward compatibility with one-turn, document the system.

**Size:** M

**Requirements satisfied:**
- Full backward compatibility with one-turn workflows (spec §6.7)
- Error recovery and audit trail (spec §3.3, §6.6)
- Conversation history context (spec §2.3)
- Per-turn checkpointing (spec §5.1)

**Files affected:**
- `tests/test_multiturn_flow.py` (new)
- `tests/test_backward_compat.py` (new)
- `docs/MULTITURN_GUIDE.md` (new)

**Tasks:**
- [ ] Write integration test: multi-turn happy path (INIT → FETCH → VALIDATE → ENRICH → STORE → COMPLETE)
- [ ] Write integration test: multi-turn with pause (FETCH → WAIT_DOCUMENTS_UPLOADED → VALIDATE)
- [ ] Write integration test: multi-turn error recovery (invalid transition retry, handler exception → ERROR)
- [ ] Write integration test: semantic context extraction (entities, intents populated by router)
- [ ] Write integration test: history trimming (turn_number increments, old turns dropped)
- [ ] Write backward compatibility test: process(document_id) still works unchanged
- [ ] Write backward compatibility test: one-turn ignores multi-turn fields
- [ ] Write test: multi-turn resume from DB (same session_id loads persisted state)
- [ ] Write test: input validation errors return gracefully
- [ ] Run full test suite: all tests pass, no regressions
- [ ] Write MULTITURN_GUIDE.md documenting:
    - How to use process_turn() vs process()
    - How to create workflows with Agno DB persistence
    - How to handle waits_for_input in client code
    - Example multi-turn conversation flow
    - Architecture diagram (from ARCHITECTURE_SUMMARY.md)
- [ ] Update main.py with multi-turn example
- [ ] Verify documentation is clear and complete

**Success criteria:**
- All integration tests pass (multi-turn, error paths, history management)
- All backward compatibility tests pass (one-turn workflows unaffected)
- No test regressions from prior phases
- Full test coverage for new code paths
- Documentation covers: API usage, architecture, examples
- README or guide explains when to use process_turn vs process
- All tests pass

**Commit message:** `test: add integration tests for multi-turn flow and backward compatibility verification`

---

## Checkpoints

### Checkpoint: After Phases 1–3
**Criteria before proceeding to Phase 4:**
- [ ] All tests pass (Phases 1–3)
- [ ] No regressions in existing tests
- [ ] Core engine infrastructure in place (EngineState, input validation, handler registry)
- [ ] High-risk items (input validation, handler metadata) verified to work correctly

**If any checkpoint fails:** Stop, review failing tests, fix issues, re-run tests, confirm all green before proceeding.

### Checkpoint: After Phases 4–6
**Criteria before proceeding to Phase 7:**
- [ ] All tests pass (Phases 1–6)
- [ ] No regressions
- [ ] Semantic router working (Claude LLM integration tested)
- [ ] Engine workflow modifications complete and tested
- [ ] Router step is pluggable and doesn't break one-turn workflows

**If any checkpoint fails:** Stop, fix router issues, re-test, confirm before proceeding.

### Checkpoint: After Phases 7–10
**Criteria before shipping:**
- [ ] All tests pass (all 10 phases)
- [ ] No regressions
- [ ] Multi-turn flow works end-to-end
- [ ] Backward compatibility verified
- [ ] Integration tests green
- [ ] Documentation complete
- [ ] Ready for code review (`/cg-review`)

---

## Parallelization

| Category | Phases | Rule |
|----------|--------|------|
| Safe to parallelize | None | All phases are dependent; must be sequential |
| Must be sequential | 1→2→3→4→5→6→7→8→9→10 | Each phase builds on prior engine layer; workflow depends on engine |
| Dependency chain | Engine (1–7) → Workflow (8–9) → Integration (10) | Engine must be complete before workflow layer; both must be complete before integration tests |

**Recommendation:** Execute phases sequentially. Each phase is ~30 min; total ~5 hours. Estimated completion: 2–3 developer hours if one person works straight through, or 1 day if integrated into normal work rhythm.

---

## TODO List

### Core Engine Infrastructure ✅
- [x] Phase 1: Split Pipeline State: Engine (Control Plane)
- [x] Phase 2: Input Validation & Sanitization
- [x] Phase 3: Handler Registry with @handler Decorator

### High-Risk Items (Router & Session) ✅
- [x] Phase 4: Base Semantic Router Abstract Class
- [x] Phase 5: DocPipelineRouter: LLM-Powered Implementation
- [ ] Phase 6: Engine Workflow: Add Semantic Router Step (future)

### Workflow Integration ✅
- [x] Phase 7: Engine Session: Multi-Turn Initialization
- [x] Phase 8: Workflow: Split Pipeline State and Add process_turn()
- [ ] Phase 9: Decorators & Exception Handling: Update All Handlers (ready)

### Testing & Documentation ⏳
- [ ] Phase 10: Integration Tests & Documentation (ready)

---

## Notes for Implementation

### One-Turn Backward Compatibility
- Existing `process(document_id)` method must remain unchanged
- process_turn() is a new entry point; does not affect process()
- One-turn workflows ignore turn_input, semantic_context, turns fields (TypedDict total=False makes them optional)
- Tests must verify both entry points work

### Agno Persistence
- session_state is auto-persisted by Agno after each run()
- No explicit checkpoint manager needed; JsonDb/SqliteDb handles it
- Full turn history stored in session_state["turns"]
- Checkpoint trimming happens on Phase 8 (_trim_history())

### Error Handling
- Handler exceptions are caught and routed to ERROR state (Phase 9)
- Router timeouts route to ERROR state (Phase 5)
- Input validation errors return error response without advancing turn (Phase 2, Phase 8)
- All error paths logged and auditable

### Testing Strategy
- Phase 1–9: Unit tests for each module (handlers, router, session, validation)
- Phase 10: Integration tests for full workflows + backward compat verification
- No end-to-end tests until Phase 10 (after all components complete)

---

## Success Metrics

By the end of all 10 phases:

1. **Functionality:**
   - ✅ Multi-turn conversations work end-to-end (INIT → decision loops → terminal)
   - ✅ Semantic router extracts entities and intents from user input
   - ✅ Handler metadata controls pause/resume (waits_for_input flag)
   - ✅ Error handling is robust (timeouts, invalid transitions, handler exceptions)
   - ✅ Session state persists across turns via Agno

2. **Backward Compatibility:**
   - ✅ Existing process(document_id) workflows unchanged
   - ✅ All existing tests pass
   - ✅ No breaking changes to handler signatures or state structure

3. **Code Quality:**
   - ✅ All tests pass (100% green)
   - ✅ No test regressions
   - ✅ Functions ≤70 lines of code
   - ✅ Code follows CLAUDE.md conventions
   - ✅ Clear docstrings and examples

4. **Security:**
   - ✅ Input validation prevents DoS (token bombs)
   - ✅ Prompt escaping prevents injection attacks
   - ✅ Constraint retry prevents invalid state transitions

5. **Documentation:**
   - ✅ MULTITURN_GUIDE.md explains how to use process_turn()
   - ✅ Architecture diagram and data flows documented
   - ✅ Example code for multi-turn conversation
