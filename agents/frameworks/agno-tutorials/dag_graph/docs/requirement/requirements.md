# Requirements: State Machine Workflow System

## Functional Requirements

### Core Workflow Engine
- **Requirement**: System must support state machine workflows with defined states and transitions
- **Details**: Support arbitrary state enums, state transitions, and terminal states
- **User Flow**: Developer defines states → creates handlers → workflow processes documents → completes

### Single-Document Processing
- **Requirement**: Each workflow invocation processes a single document from INIT to terminal state
- **Details**: One-turn execution, no session interruption, runs until COMPLETE or ERROR
- **Constraint**: Loop runs continuously with max iteration safety cap (20 iterations default)

### Multi-Turn Conversation Support (Future)
- **Requirement**: System must support multi-turn conversations with user interaction
- **Details**: Accept user text input per turn; save/restore conversation checkpoints between turns
- **Turn Types**: Some steps continue immediately (no user input needed); others wait for next user turn
- **Input**: Each turn includes user input text that affects routing decisions

### Smart Multi-Turn Router (Future)
- **Requirement**: Router must classify next proposed state given turn input text and saved checkpoints
- **Details**: Router becomes LLM-powered semantic classifier, not pure-code state machine
- **Context**: Router input includes current state, turn input text, checkpoint context (conversation history)
- **Output**: Proposed next state with confidence score

### Handler Execution Model
- **Requirement**: Each state must have an associated handler function
- **Details**: Handler signature: `(state_dict) → state_dict`
- **Constraint**: Handlers must be pure functions, no side effects beyond state mutation

### Guardrail Validation
- **Requirement**: System must validate state transitions before execution
- **Details**: Guardrails can reject transitions and override routing
- **Use Case**: Prevent invalid data flow, enforce business rules, route to recovery paths

### Error Recovery Paths
- **Requirement**: System must support three automatic recovery mechanisms
  - Retry: Automatic retry on transient failures
  - Human Review: Route to manual approval when validation fails
  - Terminal Error: Graceful termination on unrecoverable errors
- **Details**: Recovery handled via guardrail fallback states

### Session Persistence
- **Requirement**: System must persist session state to enable multi-run resume
- **Details**: Full audit trail, retry counts, business data preserved across runs
- **Storage**: JsonDb-based persistence (file-based default)

### Audit Trail
- **Requirement**: Complete chronological log of all state transitions and actions
- **Details**: Captures: state names, transitions, guardrail results, audit trail entries
- **Use Case**: Compliance, debugging, replay analysis

## Non-Functional Requirements

### Architecture Quality
- **Requirement**: Infrastructure code must be reusable across multiple pipelines
- **Details**: Generic base class pattern eliminates 80%+ boilerplate per new workflow
- **Metric**: Next workflow requires ~50 lines vs. 300+ without abstraction

### Type Safety
- **Requirement**: Use strict typing (enums for states, TypedDict for state shape)
- **Details**: No magic strings, compile-time verification of transitions
- **Tooling**: Python type hints, runtime Pydantic validation

### Extensibility
- **Requirement**: Support custom guardrails, routing tables, and session initialization
- **Details**: Five hook points for subclass specialization
- **Pattern**: Template Method pattern for framework-specific behavior

### Code Quality
- **Requirement**: Zero linting issues, passes ruff checks
- **Details**: No unused imports, no unused variables, no f-string placeholders
- **Metric**: `ruff check` returns "All checks passed"

### Performance
- **Requirement**: Processing must complete in reasonable time (not explicitly bounded)
- **Details**: Single-document one-turn execution, no async/await (synchronous)
- **Assumption**: Document processing steps (fetch, validate, enrich) use LLM APIs (external)

### Observability
- **Requirement**: System must provide clear logging and audit trail visibility
- **Details**: Log state transitions, handler execution, guardrail decisions
- **Tool**: Python logging module with structured audit trail

### Developer Experience
- **Requirement**: New workflows should be simple to implement
- **Details**: Subclass base class, implement 5 hook methods, define routing table
- **Documentation**: Implementation guide with complete examples

### Conversational Latency (Future)
- **Requirement**: Multi-turn system must respond within sub-1s per turn
- **Details**: User-acceptable wait time; LLM router call dominates latency budget
- **Tool**: Prompt caching to reuse conversation context across turns

### Context Window Management (Future)
- **Requirement**: System must manage conversation history within LLM context limits
- **Details**: Implement conversation trimming; keep last N turns; summarize older turns if needed
- **Metric**: Monitor total tokens per turn; stay within 80k token context window

### Turn-Based Checkpoint Efficiency (Future)
- **Requirement**: Checkpoints must be efficient between turns
- **Details**: Store incremental changes, not full state; compress checkpoint (last N turns vs all turns)
- **Impact**: Enables quick resume after network interruption

## Notes

### Design Constraints
- **One-turn execution**: Loop runs to terminal state, no pause/resume within a single run
- **Session-based**: Across multiple `process()` calls, same session persists state
- **Synchronous**: No async/await, handlers are blocking
- **Stateless handlers**: Handlers don't maintain state; all state flows through PipelineState dict

### Known Limitations
- Max loop iterations capped at 20 (safety measure)
- No distributed execution (single-process)
- No parallel state processing

### Future Considerations
- Async/await support for I/O-bound handlers
- Distributed execution (multiple workers)
- Pause/resume within a single document processing
- Multiple simultaneous document processing
