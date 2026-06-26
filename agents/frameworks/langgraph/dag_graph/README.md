# LangGraph DAG Graph: Document Processing Pipeline

A production-ready state machine workflow implementation using LangGraph with support for single-turn and multi-turn conversations, semantic routing, and full checkpoint persistence.

## Features

- **State Machine Architecture**: Router → Guardrail → Handler pattern with terminal conditions
- **Multi-turn Conversations**: Built-in support for pause/resume workflows with conversation history
- **Semantic Routing**: LLM-powered state transitions with Claude (customizable)
- **Handler Metadata**: Declarative handler configuration via `@handler` decorator
- **Input Validation**: Prompt injection prevention with automatic escaping
- **Auto-progression**: Automatic continuation through non-blocking states
- **Checkpointing**: Full session persistence with SQLite (production-ready)
- **Comprehensive Testing**: 104+ tests covering all features

## Installation

```bash
pip install -e .
```

**Dependencies:**
- langgraph >= 1.2.6
- langchain-core >= 0.3.0
- langchain-anthropic >= 0.2.0
- pydantic >= 2.0
- python-dotenv >= 1.0

## Quick Start

### One-Turn Processing

Process a single entity through the complete workflow:

```python
from src.workflow.workflow import run_pipeline

# Process a document
result = run_pipeline(
    document_id="doc-001",
    timeout_seconds=300,
    thread_id="session-id",
    db_path="path/to/checkpoint.db"
)

print(f"Final state: {result['current_state']}")
print(f"Audit trail: {result['audit_trail']}")
```

**Execution Flow:**
```
INIT → FETCH → VALIDATE → ENRICH → STORE → COMPLETE
```

### Multi-Turn Conversation

Execute an interactive workflow with pause/resume:

```python
from src.engine.input_validation import validate_turn_input, escape_for_llm
from src.engine.handler_registry import does_state_wait_for_input

# Turn 1: Start processing
user_input = "Process this document"
validate_turn_input(user_input)  # Validates length, tokens, control chars
escaped = escape_for_llm(user_input)  # Removes injection patterns

# Turn 1 response (invoke_turn implementation in graph.py)
response1 = graph.invoke_turn("user-123", "session-456", user_input)
print(f"State: {response1['current_state']}")
print(f"Waits for input: {response1['waits_for_input']}")

# Turn 2: Continue after user feedback  
if response1["waits_for_input"]:
    response2 = graph.invoke_turn("user-123", "session-456", "Document approved")
```

**Multi-turn Features:**
- Automatic conversation history tracking
- Input validation and injection prevention
- Semantic context extraction (entities, intents, confidence)
- Auto-progression through non-blocking states
- Pause at blocking states for user feedback
- Full session resumption via checkpoints

## Architecture

### State Machine

Nine states with clear progression:

```
┌─────────────────────────────────────────────────────┐
│                  HAPPY PATH                         │
│  INIT → FETCH → VALIDATE → ENRICH → STORE → COMPLETE │
├─────────────────────────────────────────────────────┤
│  Fallbacks (controlled by guardrails):              │
│  • FETCH failure → RETRY → FETCH                    │
│  • VALIDATE failure → HUMAN_REVIEW → ENRICH        │
│  • Any unhandled error → ERROR                      │
└─────────────────────────────────────────────────────┘
```

### Core Components

**Router Node**
- Pure code routing via `HAPPY_PATH` table, or
- LLM-powered semantic routing via `DocPipelineRouter`
- Outputs: `proposed_next` state

**Guardrail Node**
- Validates proposed transition
- Applies fallback if validation fails
- Tracks decision in audit trail

**Handler Nodes**
- Execute business logic for each state
- Metadata (`@handler` decorator) controls pause/resume
- Pure functions: `(PipelineState) → PipelineState`

### Handler Metadata

The `@handler` decorator registers handler configuration:

```python
from engine.handler_registry import handler

@handler(
    state="human_review",           # Which state this handler processes
    waits_for_input=True,           # Pauses workflow for user input
    description="Wait for expert review"
)
def handle_human_review(state):
    # Business logic...
    return {**state, "current_state": "human_review"}
```

**Key Properties:**
- `state`: State enum value (one per handler)
- `waits_for_input`: If True, workflow pauses and awaits next turn
- `description`: Human-readable handler documentation

## Input Validation & Security

### Validation

```python
from engine.input_validation import validate_turn_input, escape_for_llm, InputValidationError

# Validate input (raises InputValidationError on failure)
try:
    validate_turn_input(user_input)
except InputValidationError as e:
    return {"error": str(e), "current_state": None}
```

**Checks:**
- Type: Must be string
- Length: < 10,000 characters
- Token count: < 2,000 tokens (via tiktoken)
- Control characters: None except newline/tab/return

### Injection Prevention

```python
# Escape input to prevent prompt injection
escaped = escape_for_llm(user_input)
# Removes: XML tags, role indicators, jailbreak patterns, template markers
```

## Semantic Routing

### Built-in Document Router

```python
from workflow.router import DocPipelineRouter

router = DocPipelineRouter(model="claude-haiku-4-5-20251001")

decision = router.route(
    current_state="validate",
    turn_input="Document approved, proceed",
    history=[...],
    allowed_states=["enrich", "human_review", "error"]
)
# Returns: RouterDecision with proposed_next, confidence, entities, intents
```

### Custom Routers

Create domain-specific routers by extending `DefaultSemanticRouter`:

```python
from engine.router import DefaultSemanticRouter
from pydantic import BaseModel, Field

class MyRouterOutput(BaseModel):
    proposed_next: str
    confidence: float
    semantic_entities: dict = Field(default_factory=dict)
    semantic_intents: list = Field(default_factory=list)

class MyRouter(DefaultSemanticRouter):
    output_schema = MyRouterOutput
    
    def get_instructions(self):
        return "Your domain-specific routing instructions..."

# Use in graph
graph.router = MyRouter()
```

## Checkpointing

### SQLite Checkpointer

Full session persistence out of the box:

```python
from engine.checkpointing import SqliteCheckpointer

checkpointer = SqliteCheckpointer("path/to/checkpoint.db")

# Sessions resume automatically via thread_id
response = graph.invoke_turn(
    user_id="user-123",
    session_id="session-456",
    turn_input="Continue from where we left off"
)
```

**Thread ID Format:**
- Single-turn: `process:doc-001`
- Multi-turn: `user-123:session-456`

## Project Structure

```
src/
├── engine/
│   ├── handler_registry.py   # @handler decorator + metadata registry
│   ├── router.py             # BaseSemanticRouter, DefaultSemanticRouter
│   ├── input_validation.py   # validate_turn_input, escape_for_llm
│   ├── graph.py              # StateMachineGraph base class
│   ├── checkpointing.py      # SqliteCheckpointer
│   ├── chain.py              # LCEL chain factory
│   ├── guardrail.py          # GuardrailResult, make_guardrail
│   └── session.py            # Session helpers
├── workflow/
│   ├── pipeline_state.py     # PipelineState TypedDict, new_pipeline()
│   ├── state_machine.py      # State enum, ALLOWED_TRANSITIONS
│   ├── handlers.py           # 8 handler functions with @handler decorator
│   ├── chains.py             # VALIDATE_CHAIN, ENRICH_CHAIN, REVIEW_CHAIN
│   ├── router.py             # DocPipelineRouter
│   ├── guardrails.py         # Domain-specific guardrail checks
│   ├── graph.py              # DocumentPipelineGraph
│   ├── validation.py         # Input sanitization
│   └── workflow.py           # run_pipeline() public API
└── main.py                   # Demo scenarios (one-turn + multi-turn)

tests/
├── test_handler_registry.py   # 9 tests
├── test_semantic_router.py    # 11 tests
├── test_input_validation.py   # 19 tests
├── test_doc_pipeline_router.py # 7 tests
├── test_multi_turn.py         # 10 tests
├── test_graph_methods.py      # 16 tests
├── test_handler_integration.py # 10 tests
├── test_router_integration.py  # 11 tests
├── test_integration.py        # 17 tests
└── test_main_examples.py      # 10 tests
```

## Test Coverage

**104 tests passing (Phase 1-4):**
- **Phase 1** (46 tests): Handler registry, routers, input validation
- **Phase 2** (26 tests): Multi-turn state, graph methods, auto-progression
- **Phase 3** (32 tests): Handler integration, router integration, end-to-end flows
- **Phase 4** (10 tests): Main.py examples and documentation

Run tests:
```bash
pytest tests/ -v
```

## Running Demos

Execute the demo script to see all scenarios:

```bash
python -m src.main
```

**Scenarios Demonstrated:**
1. Happy path: Successful processing through all states
2. Fetch retry: Handle transient failures with automatic retry
3. Human review: Route to human expert when validation fails
4. Checkpoint resume: Load and continue from prior session
5. Multi-turn conversation: Interactive workflow with pause/resume

## Documentation

Detailed documentation available in `docs/design/`:
- `01-feature-mapping-langgraph.md` - Complete feature design mapping from Agno
- `02-implementation-guide.md` - Step-by-step implementation guide
- `README.md` - Design overview and quick reference

## Key Design Decisions

1. **LangGraph Native**: Uses StateGraph and checkpointer natively (not custom workflow loop)
2. **Semantic Routing Optional**: Default is pure code routing; semantic router is opt-in
3. **Handler Metadata**: Declarative via `@handler` decorator for configuration
4. **Input Validation**: Multi-layer defense (length, tokens, control chars, injection patterns)
5. **Multi-turn via invoke_turn()**: Wraps compiled_graph.invoke() with turn management
6. **Auto-progression**: Continues through non-blocking states automatically
7. **Checkpointing**: SQLite for persistence across turns and sessions

## Future Enhancements

- Async/await execution for higher throughput
- PostgreSQL checkpointing for distributed deployments
- Advanced semantic routing with embedding-based similarity
- Custom agent types (e.g., specialized routers per domain)
- Metrics collection and observability dashboard
- Interactive web UI for workflow visualization

## References

- **Design Docs**: `docs/design/` folder
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Anthropic Claude API**: https://docs.anthropic.com/
