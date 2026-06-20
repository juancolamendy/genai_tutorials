# Requirements Analysis: Multi-Turn Conversation Workflows
_Date: 2026-06-20_

## Executive Summary

The multi-turn conversation system extends the state machine workflow with user interaction support. Key difference: router becomes LLM-powered semantic classifier, and execution model changes from "loop until terminal" to "execute one turn, checkpoint, wait for user input."

**Core Innovation**: Turn-based execution with semantic router enables flexible user interaction while preserving deterministic state machine architecture.

---

## Functional Requirements

1. **Multi-Turn Conversation Support** — Handle multiple user interaction rounds
2. **Turn-Based Input Processing** — Accept and incorporate user input text per turn
3. **Checkpoint Persistence** — Save and restore conversation state between turns
4. **Smart Router Classification** — LLM-powered router considers state + turn input + history
5. **Continuation vs. Waiting Steps** — Some steps execute immediately; others wait for user input
6. **Input-Aware State Transitions** — Next state depends on semantic meaning of input text
7. **Turn History Tracking** — Keep record of all turns (input/output pairs) for context

## Non-Functional Requirements

1. **Conversational Latency** — Sub-1s per-turn response time (vs. sub-100ms for one-turn)
2. **Context Window Management** — Manage conversation history within LLM token limits
3. **Checkpoint Efficiency** — Minimal overhead; incremental checkpoint storage
4. **Semantic Routing** — LLM-powered router (vs. pure-code in current system)
5. **State Resumption** — Seamless resume after network interruption

---

## Architectural Implications

| Requirement | Type | Implication |
|---|---|---|
| **Multi-Turn Conversation** | Functional | Replace Loop(until_terminal) with Turn-based iteration; each `process(turn_input)` executes ONE turn, not full pipeline |
| **Turn-Based Input Processing** | Functional | Add `turn_input: str` parameter to state dicts; pass user text through Router for classification |
| **Checkpoint Persistence** | Functional | Checkpoint AFTER each turn (not just at completion); persist checkpoints to enable resume |
| **Smart Router Classification** | Functional | Replace pure-code routing table with LLM Router agent; Router input: {current_state, turn_input, checkpoint_context} → proposed_next_state |
| **Continuation vs. Waiting Steps** | Functional | Add step metadata: `@handler(waits_for_input=False)` vs. `@handler(waits_for_input=True)`; non-waiting steps execute immediately; waiting steps pause |
| **Input-Aware Transitions** | Functional | Router must be LLM agent (semantic classifier); route based on meaning of input text ("yes" vs. "no" → different states) |
| **Turn History Tracking** | Functional | Store turn_input/turn_output pairs in session_state["turns"]; build conversation history prompt for Router context |
| **Conversational Latency** | Non-functional | Latency budget ~500ms-1s per turn; Router LLM call dominates; use prompt caching to reuse context |
| **Context Window Management** | Non-functional | Implement conversation history trimming; keep last N turns in context; summarize older turns if needed; monitor total tokens |
| **Checkpoint Efficiency** | Non-functional | Checkpoint only changed fields (delta), not full state; or compress checkpoint (last N turns vs. all turns) |

---

## Execution Model Comparison

### Current: One-Turn Document Processing
```
process(document_id) {
  loop until terminal {
    router(current_state) → proposed_next          # Pure code
    guardrail(proposed_next) → override or pass
    dispatch(handler) → new_state
  }
  return final_state
}
```

### Required: Multi-Turn Conversation
```
process(turn_input) {
  router(current_state, turn_input, checkpoint_context) → proposed_next  # LLM-powered
  
  if proposed_next.waits_for_input {
    checkpoint()
    return checkpoint_state  # Wait for next turn
  } else {
    dispatch(handler) → new_state
    checkpoint()
    if new_state.waits_for_input {
      return new_state
    } else {
      recurse process(turn_input) with new_state  # Continue in same turn
    }
  }
}
```

---

## Key Architectural Decisions

### 1. Router Becomes LLM-Powered

**What Changes:**
- Pure-code routing table → LLM semantic classifier
- Input: `{current_state, turn_input, checkpoint_context}`
- Output: `{proposed_next_state, confidence_score}`

**Example:**
```
State: "confirm_payment"
Input: "Yes, proceed with $99.99"
History: [... 5 turns ...]

LLM Router → {proposed_next: "process_payment", confidence: 0.95}
```

### 2. Turn-Based Execution (Not Loop-Until-Terminal)

**What Changes:**
- From: Loop continuously until COMPLETE/ERROR
- To: Execute one turn, checkpoint, return control to user

**Why:** Enables user interaction; supports request/response pattern.

### 3. Step Metadata: `waits_for_input`

**New Capability:**
```python
@handler("confirm_payment", waits_for_input=True)
def handle_confirm_payment(state):
    # This step waits for next user input
    return state

@handler("process_payment", waits_for_input=False)
def handle_process_payment(state):
    # This step executes immediately, no user input needed
    return state
```

### 4. Checkpoint Per Turn (Not Just at Completion)

**What Changes:**
- From: Save state only at workflow completion
- To: Save checkpoint after each turn to enable resume

**Checkpoint Structure:**
```python
checkpoint = {
    "current_state": "waiting_for_confirmation",
    "turn_number": 3,
    "turns": [
        {"input": "I want to buy...", "output": "Showing options"},
        {"input": "That one", "output": "Confirming details"},
        {"input": "Yes", "output": "Ready to process"},
    ],
    "semantic_context": {...}  # For LLM router
}
```

### 5. Conversation History as Router Context

**New Requirement:**
- Build conversation history prompt from `turns` list
- Pass to LLM router for semantic understanding
- Handle history trimming when it exceeds context window

---

## Integration with One-Turn System

**Option A: Extend Current System**
- Add `turn_input` parameter to PipelineState
- Replace pure-code router with LLM router
- Add `waits_for_input` metadata to handlers
- Keep guardrails, error recovery paths

**Option B: Create Separate Multi-Turn Subsystem**
- Keep one-turn system as-is for document processing
- Build new `ConversationalWorkflow` alongside `StateMachineWorkflow`
- Reuse base infrastructure (persistence, audit trail, error recovery)

**Recommended: Option A** (extend current system with backwards compatibility)
- Existing one-turn workflows add `turn_input=None` and skip semantic routing
- New multi-turn workflows provide `turn_input` and use LLM router
- Single codebase, dual execution modes

---

## Open Questions

1. **Immediate vs. Deferred Execution**
   - When non-waiting step completes, auto-continue in same `process()` call?
   - Or return and let caller decide?

2. **Router Prompting**
   - How does Router know valid next states?
   - Pass full state machine diagram? Allowed transitions? All states?

3. **Conversation History Truncation**
   - Keep last N turns? Summarize old turns? Sliding window?
   - What's max conversation length?

4. **Semantic Context Storage**
   - Maintain separate extracted entities/intents separate from turn history?
   - Or rely entirely on turn history for context?

5. **Step Metadata Declaration**
   - Decorator syntax: `@handler(waits_for_input=True)`?
   - Or separate dict: `STEP_METADATA = {}`?

6. **Error Recovery in Multi-Turn**
   - Return error checkpoint and wait for user input?
   - Retry automatically? Escalate to human review?

7. **Guardrails in Multi-Turn**
   - Do guardrails need semantic understanding?
   - E.g., "can't proceed until user confirms amount"

---

## Testing Strategy

### Unit Tests (Missing)
- Test LLM Router output parsing
- Test continuation vs. waiting logic
- Test checkpoint serialization/deserialization
- Test history trimming logic

### Integration Tests (Missing)
- Test multi-turn conversation flow
- Test checkpoint resume after interruption
- Test semantic routing decisions
- Test context window limits

### E2E Tests (Missing)
- Full conversation from greeting to completion
- Resume after network interruption
- Long conversations with history trimming

---

## Performance Characteristics

| Aspect | Current | Multi-Turn | Trade-off |
|--------|---------|-----------|----------|
| **Execution** | Loop until terminal | One turn per request | More requests, but user-interactive |
| **Router** | Pure code (< 1ms) | LLM call (500-1000ms) | Much slower, but semantic |
| **Latency Budget** | < 100ms total | ~500ms-1s per turn | User-acceptable wait |
| **Checkpoint** | End-of-pipeline | After each turn | More I/O, but enables resume |
| **Context** | Audit trail only | Full conversation history | More storage, but richer context |

---

## Security & Compliance

| Concern | Mitigation | Status |
|---------|-----------|--------|
| **Prompt Injection** | Sanitize turn_input; don't concat directly into Router prompt | ⚠️ Needs implementation |
| **Token Limit DoS** | Cap conversation history; reject turns if history exceeds limit | ⚠️ Needs implementation |
| **Checkpoint Tampering** | Audit trail in checkpoint; encrypt at rest if needed | ⚠️ Needs implementation |
| **User Impersonation** | Session ID is UUID; requires valid checkpoint to resume | ✅ By design |

---

## References

### Input Documents
- `docs/requirement/requirements.md` — Updated requirements (includes multi-turn)

### Architecture References
- `docs/analysis/requirements_analysis_state-machine-workflow-2026-06-20.md` — One-turn system analysis

### Related Files
- `engine/workflow.py` — Base class (needs LLM router extension)
- `workflow/workflow.py` — Current implementation (one-turn)

---

## Next Steps

1. **Clarify Open Questions** — Decide on execution model, router prompting, history trimming
2. **Design Multi-Turn Extension** — Run `/cg-design` for detailed design
3. **Prototype LLM Router** — Implement semantic classification
4. **Test Checkpoint Resume** — Verify persistence and recovery
5. **Backward Compatibility** — Ensure one-turn workflows continue to work

---

## Summary

**Key Insight**: Multi-turn conversations require fundamentally different execution model (turn-based vs. loop-until-terminal) and router (LLM-powered vs. pure-code).

**Integration Strategy**: Extend current system with optional turn_input and semantic routing, while preserving one-turn document processing use case.

**Risk**: LLM router introduces non-determinism and latency; requires careful prompting and validation.
