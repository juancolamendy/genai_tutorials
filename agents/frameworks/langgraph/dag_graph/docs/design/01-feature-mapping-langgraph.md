# LangGraph DAG Graph: Feature Mapping & Design Specification

**Status:** Design Phase  
**Date:** 2026-06-24  
**Based on:** Agno dag_graph architecture (last 36 commits)

---

## Executive Summary

This document maps features from the **Agno DAG Graph** implementation (36 recent commits) to the **LangGraph DAG Graph** implementation. The Agno architecture extends a basic state machine with:

1. **Multi-turn workflows** with conversation history and pause/resume
2. **Semantic routing** via LLM-powered state transitions
3. **Handler metadata** registry with `@handler` decorator
4. **Input validation** with prompt injection prevention
5. **Auto-progression** through non-blocking states
6. **Engine layer refactoring** to move multi-turn logic to base class
7. **Response building framework** for structured outputs

LangGraph's `StateGraph` and `checkpointer` provide a more native foundation than Agno's custom workflow loop. This design leverages LangGraph's strengths while adopting Agno's architectural patterns.

---

## Part 1: Feature Inventory from Last 36 Commits

### Commit History Analysis

| Category | Commits | Feature |
|----------|---------|---------|
| **Multi-turn** | 84fbd16, 8bac160, 1e463cf, 02654e4 | Conversation history, turn tracking, pause/resume |
| **Semantic routing** | d6af9ed, f4b31be, ae0601f, 575f109 | LLM-powered router, RouterDecision, allowed_states |
| **Handler metadata** | 034abcf, 28bbc0f | @handler decorator, HandlerMetadata registry |
| **Input validation** | 02654e4, fe74940 | validate_turn_input(), escape_for_llm(), injection prevention |
| **Auto-progression** | fe74940, 375f109 | _auto_progress(), waits_for_input flag |
| **Engine refactoring** | 28bbc0f, 9207f20, 19811bd | Move process(), process_turn() to base class |
| **Response building** | f045eea, 19811bd | _build_response(), _build_turn_response() |
| **Router refactoring** | d16efbd, b532b5a, 1b2caf5 | BaseSemanticRouter, DefaultSemanticRouter, router.py |

---

## Part 2: Feature-by-Feature Design

### Feature 1: Multi-turn Workflows with Conversation History

#### Agno Implementation
```python
# engine/statemachine_workflow.py
def process_turn(self, user_id: str, session_id: str, turn_input: str, timeout_sec: float = 10.0) -> dict:
    """Execute one turn of a multi-turn conversation."""
    validate_turn_input(turn_input)
    self.session_state.update({
        "turn_input": turn_input,
        "turn_number": turn_num + 1,
        "conversation_history": [...]  # Append-only list of turns
    })
    self.run(...)  # Execute state machine loop once
    self._auto_progress()
    self._trim_history()
    return self._build_turn_response()

# Per-turn structure
turn = {
    "role": "user" | "assistant",
    "content": str,
    "semantic_context": {"entities": {...}, "intents": [...]},
    "state": str,
}
```

#### LangGraph Implementation Strategy

**Design Decision:** Use LangGraph's native `StateGraph` with thread-based persistence for multi-turn resumption.

```python
# engine/graph.py: StateMachineGraph (updated)
class StateMachineGraph(BaseModel):
    """
    Extends existing graph implementation with multi-turn support.
    """
    
    def invoke_turn(self,
                   user_id: str,
                   session_id: str,
                   turn_input: str,
                   timeout_sec: float = 10.0,
                   thread_id: str = None) -> dict[str, Any]:
        """
        Execute one turn of a multi-turn conversation.
        
        Wraps invoke() with:
        1. Input validation & escaping (prompt injection prevention)
        2. Conversation history management
        3. Turn metadata preparation
        4. Auto-progression through non-blocking states
        5. History trimming
        6. Response building
        
        Uses LangGraph's thread_id for cross-turn state persistence.
        """
        # Validate & escape input
        escaped = escape_for_llm(turn_input)
        
        # Initialize state on first turn (turn_number == 0)
        if thread_id is None:
            thread_id = f"{user_id}:{session_id}"
        
        # Build config with thread_id for checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        
        # Prepare turn metadata
        state = {
            "turn_input": escaped,
            "turn_number": ...,  # Retrieved from prior state
            "conversation_history": [...],  # Retrieved from prior state
            "router_timeout_sec": timeout_sec,
            "user_id": user_id,
            "session_id": session_id,
        }
        
        # Invoke graph (single iteration)
        output_state = self.compiled_graph.invoke(state, config=config)
        
        # Auto-progress through non-blocking states
        self._auto_progress_langgraph(output_state, config)
        
        # Trim history to max_history_turns
        output_state["conversation_history"] = self._trim_history(
            output_state["conversation_history"]
        )
        
        # Build and return turn response
        return self._build_turn_response(output_state)

# workflow/pipeline_state.py: Updated PipelineState
class PipelineState(TypedDict):
    # ... existing fields ...
    # Multi-turn fields
    turn_input: Optional[str]           # Current turn's user input (escaped)
    turn_number: int                     # Turn counter (0, 1, 2, ...)
    conversation_history: list           # List of {role, content, semantic_context, state}
    max_history_turns: int              # Max turns to keep (default 10)
    router_timeout_sec: float           # Timeout for semantic router (default 10.0)
    user_id: str                         # Caller identity
    session_id: str                      # Multi-turn session ID (for audit)
    semantic_context: dict               # {entities, intents} from router
    router_confidence: float             # [0.0, 1.0]
```

**Implementation Steps:**
1. Add `turn_input`, `turn_number`, `conversation_history` to `PipelineState`
2. Implement `invoke_turn()` method in `StateMachineGraph`
3. Update handlers to append turn results to `conversation_history`
4. Implement `_auto_progress_langgraph()` for LangGraph's graph structure
5. Use `checkpointer` with `thread_id` for persistence across turns

**Key Differences from Agno:**
- Agno: Custom workflow loop with `end_condition` checking `turn_number`
- LangGraph: Native `invoke()` per turn + `checkpointer` handles persistence
- Agno: Manual history trimming via `_trim_history()`
- LangGraph: Same approach, but called from `invoke_turn()` wrapper

---

### Feature 2: Semantic Routing with LLM

#### Agno Implementation
```python
# engine/router.py
class BaseSemanticRouter(ABC):
    @abstractmethod
    def route(self, current_state: str, turn_input: str, history: list,
              allowed_states: list, timeout_sec: float = 10.0) -> RouterDecision:
        """Classify user input → determine next state."""

class DefaultSemanticRouter(BaseSemanticRouter):
    """Concrete router with common LLM logic."""
    output_schema: type  # Pydantic model (subclass override)
    
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.router_agent = make_agent(
            name=..., output_schema=..., instructions=...
        )
    
    def route(self, ...):
        # Build prompt from current_state, turn_input, history, allowed_states
        # Call router_agent.run()
        # Extract RouterDecision from response
        # Validate proposed_next in allowed_states (retry if invalid)

@dataclass
class RouterDecision:
    proposed_next: str                   # Next state
    confidence: float                    # [0.0, 1.0]
    semantic_entities: dict = field(default_factory=dict)
    semantic_intents: list = field(default_factory=list)
    reasoning: Optional[str] = None
```

#### LangGraph Implementation Strategy

**Design Decision:** Create router as a `RunnableLambda` node or separate router class called by router node.

```python
# engine/router.py: NEW FILE
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel

@dataclass
class RouterDecision:
    """LLM-powered router output."""
    proposed_next: str
    confidence: float
    semantic_entities: dict = field(default_factory=dict)
    semantic_intents: list = field(default_factory=list)
    reasoning: Optional[str] = None

class BaseSemanticRouter(ABC):
    """Abstract base for LLM-powered routers."""
    
    @abstractmethod
    def route(self,
              current_state: str,
              turn_input: str,
              history: list,
              allowed_states: list,
              timeout_sec: float = 10.0) -> RouterDecision:
        """Classify user input and determine next state."""
        raise NotImplementedError

class DefaultSemanticRouter(BaseSemanticRouter):
    """Concrete router using Claude via LangChain."""
    
    output_schema: type  # Pydantic model (subclass override)
    
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        from langchain_anthropic import ChatAnthropic
        self.llm = ChatAnthropic(model=model, temperature=0)
        
    def get_instructions(self) -> str:
        """System instructions for router. Override in subclass."""
        return """You are a state machine router for workflows.
Given the current state, user input, conversation history, and allowed next states,
determine which state the workflow should transition to next.
Always propose one of the ALLOWED NEXT STATES.
Extract relevant entities and intents from the user's input."""
    
    def build_router_prompt(self, current_state: str, turn_input: str,
                           history_text: str, allowed_states: list) -> str:
        """Build LLM prompt. Override in subclass."""
        allowed_str = ", ".join(allowed_states)
        return f"""WORKFLOW STATE MACHINE ROUTING

Current State: {current_state}
Allowed Next States: {allowed_str}

Conversation History:
{history_text}

User Input: {turn_input}

Determine the next state based on the user's intent and allowed transitions."""
    
    def route(self, current_state: str, turn_input: str, history: list,
              allowed_states: list, timeout_sec: float = 10.0) -> RouterDecision:
        """Route using LLM."""
        from langchain.schema import SystemMessage, HumanMessage
        
        # Build history text
        history_text = "\n".join([
            f"{turn['role'].title()}: {turn['content']}"
            for turn in history[-5:]  # Last 5 turns only
        ])
        
        # Build prompt
        prompt = self.build_router_prompt(current_state, turn_input,
                                         history_text, allowed_states)
        
        # Call LLM with output_schema
        with_structure = self.llm.with_structured_output(self.output_schema)
        response = with_structure.invoke([
            SystemMessage(content=self.get_instructions()),
            HumanMessage(content=prompt)
        ])
        
        # Validate proposed_next is in allowed_states (retry if not)
        if response.proposed_next not in allowed_states:
            # Retry with constraint
            response.proposed_next = self._fallback_state(allowed_states)
        
        return RouterDecision(
            proposed_next=response.proposed_next,
            confidence=response.confidence,
            semantic_entities=response.semantic_entities,
            semantic_intents=response.semantic_intents,
            reasoning=response.reasoning
        )
    
    def _fallback_state(self, allowed_states: list) -> str:
        """Fallback if LLM doesn't respect constraints."""
        return allowed_states[0] if allowed_states else "error"

# workflow/router.py: Domain-specific router
from engine.router import DefaultSemanticRouter, RouterDecision
from pydantic import BaseModel, Field

class DocRouterOutput(BaseModel):
    """Output schema for document pipeline router."""
    proposed_next: str = Field(..., description="Next state")
    confidence: float = Field(..., description="[0.0, 1.0]")
    semantic_entities: dict = Field(default_factory=dict)
    semantic_intents: list = Field(default_factory=list)
    reasoning: str = Field(default="")

class DocPipelineRouter(DefaultSemanticRouter):
    """Document pipeline semantic router."""
    output_schema = DocRouterOutput
    
    def get_instructions(self) -> str:
        return """You are a document processing workflow router.
Determine the next state based on document content and user intent.
Valid states: init, fetch, validate, enrich, store, complete, retry, error, human_review."""
    
    def build_router_prompt(self, current_state: str, turn_input: str,
                           history_text: str, allowed_states: list) -> str:
        # Domain-specific prompt for document processing
        return super().build_router_prompt(current_state, turn_input, history_text, allowed_states)
```

**Integration with Graph:**

```python
# engine/graph.py: Router node updated for multi-turn
def _semantic_router_node(self, state: PipelineState) -> dict:
    """LLM-powered router (called when turn_input is present)."""
    if not self.router:
        # Fallback to pure code router
        return self._router_node(state)
    
    current = state["current_state"]
    turn_input = state.get("turn_input", "")
    history = state.get("conversation_history", [])
    
    # Get allowed next states from routing table
    routing_table = self._build_routing_table()
    allowed_states = self._get_allowed_states(current, routing_table)
    
    # Call semantic router
    decision = self.router.route(
        current_state=current,
        turn_input=turn_input,
        history=history,
        allowed_states=allowed_states,
        timeout_sec=state.get("router_timeout_sec", 10.0)
    )
    
    # Update state
    state["proposed_next"] = decision.proposed_next
    state["semantic_context"] = {
        "entities": decision.semantic_entities,
        "intents": decision.semantic_intents,
    }
    state["router_confidence"] = decision.confidence
    
    return state
```

**Key Differences from Agno:**
- Agno: Calls `make_agent()` from agno.agent module
- LangGraph: Uses `ChatAnthropic` + `with_structured_output()` from LangChain
- Agno: Router decision stored in session_state during step execution
- LangGraph: Router decision returned as state dict update

---

### Feature 3: Handler Metadata Registry with @handler Decorator

#### Agno Implementation
```python
# engine/handler_registry.py
@dataclass
class HandlerMetadata:
    state: str
    waits_for_input: bool = False
    description: Optional[str] = None

HANDLER_MAP_METADATA: dict[str, HandlerMetadata] = {}

def handler(state: str, waits_for_input: bool = False, description: Optional[str] = None) -> Callable:
    """Decorator that registers handler metadata."""
    def decorator(func: Callable) -> Callable:
        HANDLER_MAP_METADATA[state] = HandlerMetadata(state, waits_for_input, description)
        return func
    return decorator

def does_state_wait_for_input(state: str) -> bool:
    """Check if state pauses and waits for user input."""
    meta = HANDLER_MAP_METADATA.get(state)
    return meta.waits_for_input if meta else False

# Usage in workflow
@handler(state="validate", waits_for_input=False)
def handle_validate(state: PipelineState) -> PipelineState:
    ...

@handler(state="human_review", waits_for_input=True)
def handle_human_review(state: PipelineState) -> PipelineState:
    ...
```

#### LangGraph Implementation Strategy

```python
# engine/handler_registry.py: NEW FILE (same as Agno)
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class HandlerMetadata:
    """Metadata for a state handler."""
    state: str
    waits_for_input: bool = False
    description: Optional[str] = None

# Global registry populated by @handler decorator
HANDLER_MAP_METADATA: dict[str, HandlerMetadata] = {}

def handler(state: str,
            waits_for_input: bool = False,
            description: Optional[str] = None) -> Callable:
    """
    Decorator that registers a handler function with metadata.
    
    Usage:
        @handler(state="validate", waits_for_input=False)
        def handle_validate(state: PipelineState) -> PipelineState:
            ...
    """
    def decorator(func: Callable) -> Callable:
        HANDLER_MAP_METADATA[state] = HandlerMetadata(
            state=state,
            waits_for_input=waits_for_input,
            description=description
        )
        return func
    return decorator

def get_handler_metadata(state: str) -> Optional[HandlerMetadata]:
    """Retrieve metadata for a registered state."""
    return HANDLER_MAP_METADATA.get(state)

def does_state_wait_for_input(state: str) -> bool:
    """Check if a state pauses and waits for user input."""
    meta = get_handler_metadata(state)
    return meta.waits_for_input if meta else False

# workflow/handlers.py: Updated handlers
from engine.handler_registry import handler

@handler(state="init", waits_for_input=False, description="Initialize pipeline")
def handle_init(state: PipelineState) -> PipelineState:
    state["current_state"] = "fetch"
    return state

@handler(state="fetch", waits_for_input=False, description="Fetch document")
def handle_fetch(state: PipelineState) -> PipelineState:
    ...

@handler(state="human_review", waits_for_input=True, description="Wait for human review")
def handle_human_review(state: PipelineState) -> PipelineState:
    state["current_state"] = "human_review"
    return state  # Workflow pauses here until next turn
```

**Integration with LangGraph:**

```python
# engine/graph.py
class StateMachineGraph:
    def __init__(self, ...):
        # Store handlers with metadata
        self.handlers_with_meta = {}
        for state, handler_fn in self.HANDLER_MAP.items():
            meta = get_handler_metadata(state.value if hasattr(state, 'value') else state)
            self.handlers_with_meta[state] = {
                "fn": handler_fn,
                "metadata": meta
            }
    
    def build_graph(self, ...):
        """Build LangGraph StateGraph with handler nodes."""
        # Create handler nodes using metadata
        for state, handler_info in self.handlers_with_meta.items():
            def make_handler_node(fn, state):
                def node(state_dict: PipelineState) -> PipelineState:
                    return fn(state_dict)
                return node
            
            graph_builder.add_node(
                str(state),
                make_handler_node(handler_info["fn"], state)
            )
```

**Key Differences from Agno:**
- Implementation is identical; this is a direct port
- Used by `_auto_progress()` to check if workflow should pause
- Used by guardrails to determine fallback behavior

---

### Feature 4: Input Validation & Prompt Injection Prevention

#### Agno Implementation
```python
# engine/input_validation.py
class InputValidationError(Exception):
    pass

def validate_turn_input(turn_input: str) -> None:
    """Validate user input for length, token count, etc."""
    if not turn_input or not isinstance(turn_input, str):
        raise InputValidationError("Input must be non-empty string")
    if len(turn_input) > MAX_INPUT_LENGTH:
        raise InputValidationError(f"Input too long (max {MAX_INPUT_LENGTH})")

def escape_for_llm(turn_input: str) -> str:
    """Escape input to prevent prompt injection."""
    # Strip dangerous patterns
    dangerous = ["<prompt>", "</prompt>", "System:", "Admin:"]
    escaped = turn_input
    for pattern in dangerous:
        escaped = escaped.replace(pattern, "")
    return escaped.strip()
```

#### LangGraph Implementation Strategy

```python
# engine/input_validation.py: (Same as Agno with minor tweaks)
class InputValidationError(Exception):
    """Raised when input validation fails."""
    pass

MAX_INPUT_LENGTH = 10000
MAX_TOKEN_COUNT = 2000

def validate_turn_input(turn_input: str) -> None:
    """
    Validate user input for safety and constraints.
    
    Checks:
    - Non-empty, string type
    - Length under MAX_INPUT_LENGTH
    - Token count under MAX_TOKEN_COUNT (via tiktoken estimate)
    - No control characters
    """
    if not turn_input or not isinstance(turn_input, str):
        raise InputValidationError("Input must be non-empty string")
    
    if len(turn_input) > MAX_INPUT_LENGTH:
        raise InputValidationError(
            f"Input exceeds max length ({len(turn_input)} > {MAX_INPUT_LENGTH})"
        )
    
    # Estimate token count
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = len(enc.encode(turn_input))
    if tokens > MAX_TOKEN_COUNT:
        raise InputValidationError(
            f"Input too verbose ({tokens} tokens > {MAX_TOKEN_COUNT})"
        )
    
    # Check for control characters
    if any(ord(c) < 32 and c not in '\n\t\r' for c in turn_input):
        raise InputValidationError("Input contains invalid control characters")

def escape_for_llm(turn_input: str) -> str:
    """
    Escape input to prevent prompt injection attacks.
    
    Removes:
    - XML-like tags (<prompt>, </prompt>, <system>, etc.)
    - Role indicators (System:, Admin:, User:, Assistant:)
    - Jailbreak patterns
    """
    # Dangerous patterns to remove
    patterns_to_remove = [
        r"<prompt>", r"</prompt>",
        r"<system>", r"</system>",
        r"<instruction>", r"</instruction>",
        r"System:", r"Admin:", r"User:", r"Assistant:",
        r"SYSTEM PROMPT:", r"INSTRUCTIONS:",
        r"{{", r"}}",  # Template injection
    ]
    
    escaped = turn_input
    for pattern in patterns_to_remove:
        import re
        escaped = re.sub(pattern, "", escaped, flags=re.IGNORECASE)
    
    return escaped.strip()

# Usage in process_turn()
def invoke_turn(self, user_id: str, session_id: str, turn_input: str, ...):
    try:
        validate_turn_input(turn_input)
        escaped = escape_for_llm(turn_input)
        # Continue with escaped input
    except InputValidationError as e:
        return {"error": str(e), "current_state": None, "waits_for_input": False}
```

**Integration with Graph:**

```python
# engine/graph.py
def invoke_turn(self, user_id: str, session_id: str, turn_input: str, ...):
    """Execute one turn of multi-turn conversation."""
    from engine.input_validation import validate_turn_input, escape_for_llm, InputValidationError
    
    try:
        validate_turn_input(turn_input)
        escaped = escape_for_llm(turn_input)
    except InputValidationError as e:
        return {"error": str(e), "current_state": None, "waits_for_input": False}
    
    # Continue with escaped input
    state["turn_input"] = escaped
    return self.compiled_graph.invoke(state, config=config)
```

**Key Differences from Agno:**
- Same general approach, with added tiktoken-based token counting
- Regex patterns for more robust pattern matching

---

### Feature 5: Auto-progression Through Non-blocking States

#### Agno Implementation
```python
# engine/statemachine_workflow.py
def _auto_progress(self) -> None:
    """
    Auto-progress workflow through non-blocking states.
    
    If current state has waits_for_input=False, continue running the state
    machine loop until hitting a state with waits_for_input=True or terminal.
    """
    from engine.handler_registry import does_state_wait_for_input
    
    while True:
        current = self._get_current_state(self.session_state)
        
        # Stop if terminal
        if current in self._TERMINAL_STATES:
            break
        
        # Stop if state waits for input
        if does_state_wait_for_input(current.value):
            break
        
        # Continue: run one more iteration
        self.run(input=...)  # Run state machine loop again
```

#### LangGraph Implementation Strategy

```python
# engine/graph.py
def _auto_progress_langgraph(self, state: PipelineState, config: dict) -> PipelineState:
    """
    Auto-progress through non-blocking states until hitting a pause point.
    
    For LangGraph: repeatedly call invoke() with same config until:
    - current_state is in TERMINAL_STATES
    - does_state_wait_for_input(current_state) returns True
    
    Each invoke() iteration represents one full state machine step.
    """
    from engine.handler_registry import does_state_wait_for_input
    
    while True:
        current = state.get("current_state", "init")
        
        # Stop if terminal
        if current in self._TERMINAL_STATES:
            break
        
        # Stop if state waits for input
        if does_state_wait_for_input(current):
            break
        
        # Continue: run state machine one more time
        state = self.compiled_graph.invoke(state, config=config)
    
    return state

# Usage in invoke_turn()
def invoke_turn(self, user_id: str, session_id: str, turn_input: str, ...):
    # ... validate & escape input ...
    
    # First invoke: router, guardrail, handler
    output_state = self.compiled_graph.invoke(state, config=config)
    
    # Auto-progress through non-blocking states
    output_state = self._auto_progress_langgraph(output_state, config)
    
    # Return response
    return self._build_turn_response(output_state)
```

**State Machine with Auto-progression:**

```
Example: INIT (non-blocking) → FETCH (non-blocking) → VALIDATE (blocks on failure)

Turn 1 invocation:
  1. Router: INIT → FETCH
  2. Handler: INIT → outputs current_state=FETCH
  3. Auto-progress: FETCH is non-blocking, so invoke again
  4. Router: FETCH → VALIDATE
  5. Handler: FETCH → outputs current_state=VALIDATE
  6. Auto-progress: VALIDATE waits_for_input=False (success path), continue
  7. Router: VALIDATE → ENRICH
  8. Handler: VALIDATE → outputs current_state=ENRICH
  9. Auto-progress: ENRICH waits_for_input=True, STOP
  
  Return to user: current_state=ENRICH, waits_for_input=True
```

**Key Differences from Agno:**
- Agno: Calls `self.run()` in a loop within `_auto_progress()`
- LangGraph: Calls `self.compiled_graph.invoke()` in a loop
- Both use `does_state_wait_for_input()` from handler_registry

---

### Feature 6: Engine Layer Refactoring (process() & process_turn())

#### Agno Implementation
```python
# engine/statemachine_workflow.py (base class)

def process(self, entity_id: str) -> Any:
    """Execute one complete run for an entity."""
    self._ensure_initialized()
    self.session_state.update(self._new_session_state(entity_id))
    self.run(input=entity_id)
    self._auto_progress()
    response = self._build_response(entity_id)
    self.session_state["output"].append({...})
    return response

def process_turn(self, user_id: str, session_id: str, turn_input: str, ...) -> dict:
    """Execute one turn of multi-turn conversation."""
    # Validate & escape
    # Initialize router
    # First turn: create session state
    # Prepare turn metadata
    # Run state machine
    # Auto-progress
    # Trim history
    # Return turn response
```

#### LangGraph Implementation Strategy

```python
# engine/graph.py: StateMachineGraph (base class)

class StateMachineGraph:
    """
    Base class for state machine graphs in LangGraph.
    
    Subclasses provide:
    - HAPPY_PATH routing table
    - _build_routing_table() method
    - Guardrail logic
    - Handler implementations
    
    Base class provides:
    - invoke() — wrapper around compiled_graph.invoke()
    - invoke_turn() — multi-turn support
    - process() — one-turn entity processing
    - _auto_progress_langgraph() — auto-progression
    - _build_response() & _build_turn_response() — response builders
    """
    
    def __init__(self, ...):
        self.compiled_graph = self.build_graph()
        self.checkpointer = SqliteCheckpointer(db_path)
    
    def invoke(self,
               state: PipelineState,
               config: dict = None) -> PipelineState:
        """
        Invoke compiled graph once.
        
        Args:
            state: Current pipeline state
            config: {"configurable": {"thread_id": ...}} for checkpointing
        
        Returns:
            Updated state after one full iteration (router → guardrail → handler)
        """
        if config is None:
            config = {}
        return self.compiled_graph.invoke(state, config=config)
    
    def process(self,
               entity_id: str,
               timeout_seconds: float = 300.0) -> dict[str, Any]:
        """
        Execute one complete run of workflow for an entity.
        
        Provides generic one-turn support:
        1. Create fresh state via new_pipeline()
        2. Invoke state machine loop
        3. Auto-progress through non-blocking states
        4. Return response
        
        Args:
            entity_id: Document ID, invoice ID, etc.
            timeout_seconds: Max execution time
        
        Returns:
            Response dict with final state, audit trail, etc.
        """
        # Create fresh state
        state = self.new_pipeline(entity_id, timeout_seconds)
        
        # Create thread_id for checkpointing
        thread_id = f"process:{entity_id}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run state machine
        try:
            state = self.invoke(state, config=config)
            
            # Auto-progress through non-blocking states
            state = self._auto_progress_langgraph(state, config)
            
        except Exception as e:
            state["error_message"] = str(e)
            state["current_state"] = "error"
        
        # Return response
        return self._build_response(entity_id, state)
    
    def invoke_turn(self,
                   user_id: str,
                   session_id: str,
                   turn_input: str,
                   timeout_sec: float = 10.0) -> dict[str, Any]:
        """
        Execute one turn of multi-turn conversation.
        
        Provides generic multi-turn support:
        1. Validate & escape input
        2. Initialize or resume session state
        3. Prepare turn metadata
        4. Invoke state machine
        5. Auto-progress
        6. Trim history
        7. Return turn response
        
        Args:
            user_id: Caller identity
            session_id: Multi-turn session ID
            turn_input: User's input text
            timeout_sec: Router timeout
        
        Returns:
            Turn response dict with state, entities, intents, confidence
        """
        from engine.input_validation import (
            validate_turn_input, escape_for_llm, InputValidationError
        )
        
        try:
            # Validate & escape
            validate_turn_input(turn_input)
            escaped = escape_for_llm(turn_input)
            
            # Thread ID for checkpointing
            thread_id = f"{user_id}:{session_id}"
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get or initialize state
            state = self._get_or_init_state(session_id)
            
            # Prepare turn metadata
            state["turn_input"] = escaped
            state["turn_number"] = state.get("turn_number", 0) + 1
            state["router_timeout_sec"] = timeout_sec
            state["user_id"] = user_id
            state["session_id"] = session_id
            
            # Initialize router if needed
            if not hasattr(self, 'router') or self.router is None:
                self._init_router()
            
            # Invoke state machine once
            state = self.invoke(state, config=config)
            
            # Auto-progress
            state = self._auto_progress_langgraph(state, config)
            
            # Trim history
            max_turns = state.get("max_history_turns", 10)
            history = state.get("conversation_history", [])
            if len(history) > max_turns:
                state["conversation_history"] = history[-max_turns:]
            
            # Append turn to conversation history
            state["conversation_history"].append({
                "role": "assistant",
                "content": f"Moved to {state['current_state']}",
                "semantic_context": {
                    "entities": state.get("semantic_context", {}).get("entities", {}),
                    "intents": state.get("semantic_context", {}).get("intents", [])
                },
                "state": state["current_state"]
            })
            
            # Return turn response
            return self._build_turn_response(state)
        
        except InputValidationError as e:
            return {
                "error": str(e),
                "current_state": None,
                "waits_for_input": False,
                "turn_number": 0,
                "semantic_context": {},
                "router_confidence": 0.0
            }
        except Exception as e:
            log.exception("invoke_turn failed: %s", e)
            return {
                "error": str(e),
                "current_state": "error",
                "waits_for_input": False,
                "turn_number": state.get("turn_number", 0),
                "semantic_context": {},
                "router_confidence": 0.0
            }
    
    def _get_or_init_state(self, session_id: str) -> PipelineState:
        """Get existing state or create fresh state for session."""
        # Try to load from checkpointer
        thread_id = f"invoke_turn:{session_id}"
        # ... load from checkpointer ...
        
        # If not found, create fresh
        return self.new_pipeline(session_id)
    
    def _auto_progress_langgraph(self, state: PipelineState, config: dict) -> PipelineState:
        """Auto-progress through non-blocking states."""
        from engine.handler_registry import does_state_wait_for_input
        
        while True:
            current = state.get("current_state", "init")
            if current in self.TERMINAL_STATES:
                break
            if does_state_wait_for_input(current):
                break
            state = self.invoke(state, config=config)
        
        return state
    
    def _build_response(self, entity_id: str, state: PipelineState) -> dict:
        """Build response from final state."""
        return {
            "current_state": state.get("current_state", "init"),
            "proposed_next": state.get("proposed_next"),
            "retry_count": state.get("retry_count", 0),
            "error_message": state.get("error_message"),
            "audit_trail": state.get("audit_trail", []),
            "entity_id": entity_id
        }
    
    def _build_turn_response(self, state: PipelineState) -> dict:
        """Build response from state after a turn."""
        from engine.handler_registry import does_state_wait_for_input
        
        current = state.get("current_state", "init")
        return {
            "current_state": current,
            "waits_for_input": does_state_wait_for_input(current),
            "turn_number": state.get("turn_number", 0),
            "semantic_context": state.get("semantic_context", {}),
            "router_confidence": state.get("router_confidence", 0.0),
            "error": state.get("error_message")
        }
```

**Subclass Implementation (workflow/graph.py):**

```python
# workflow/graph.py

class DocumentPipelineGraph(StateMachineGraph):
    """Document processing state machine."""
    
    HAPPY_PATH = {
        State.INIT: State.FETCH,
        State.FETCH: State.VALIDATE,
        State.VALIDATE: State.ENRICH,
        State.ENRICH: State.STORE,
        State.STORE: State.COMPLETE,
        State.RETRY: State.FETCH,
        State.HUMAN_REVIEW: State.ENRICH,
    }
    
    TERMINAL_STATES = {State.COMPLETE, State.ERROR}
    HANDLER_MAP = {...}  # Maps states to handler functions
    
    def __init__(self, db_path: str = None):
        self.router = None  # Initialize in _init_router()
        super().__init__(db_path=db_path)
    
    def _init_router(self):
        """Initialize semantic router."""
        if self.router is None:
            self.router = DocPipelineRouter()
    
    def _build_routing_table(self) -> dict:
        """Return happy-path routing table."""
        return self.HAPPY_PATH
    
    def _new_session_state(self, entity_id: str) -> dict:
        """Initialize fresh state."""
        return new_pipeline(entity_id)
    
    # Subclass overrides guardrails, handlers, etc.
```

**Key Differences from Agno:**
- Agno: `process()` and `process_turn()` methods on `StateMachineWorkflow`
- LangGraph: Same methods on `StateMachineGraph`, using `compiled_graph.invoke()`
- Agno: `self.run(input=...)` in loop
- LangGraph: `self.compiled_graph.invoke(state, config=config)` in loop
- Agno: Agno's `Workflow.run()` executes one loop iteration
- LangGraph: `StateGraph.invoke()` does the same

---

### Feature 7: Response Building Framework

#### Agno Implementation
```python
# engine/statemachine_workflow.py (base class)

def _build_response(self, entity_id: str) -> dict[str, Any]:
    """Build response dict from current session_state."""
    return {
        "current_state": self.session_state.get("current_state", "init"),
        "proposed_next": self.session_state.get("proposed_next"),
        "retry_count": self.session_state.get("retry_count", 0),
        "error_message": self.session_state.get("error_message"),
        "guardrail_ok": self.session_state.get("guardrail_ok", True),
        "audit_trail": self.session_state.get("audit_trail", []),
        "semantic_context": self.session_state.get("semantic_context", {}),
        "router_confidence": self.session_state.get("router_confidence", 0.0),
    }

def _build_turn_response(self) -> dict[str, Any]:
    """Build response dict from state after a turn."""
    from engine.handler_registry import does_state_wait_for_input
    
    current = self.session_state.get("current_state", "init")
    return {
        "current_state": current,
        "waits_for_input": does_state_wait_for_input(current),
        "turn_number": self.session_state.get("turn_number", 0),
        "semantic_context": self.session_state.get("semantic_context", {}),
        "router_confidence": self.session_state.get("router_confidence", 0.0),
        "error": self.session_state.get("error_message")
    }
```

#### LangGraph Implementation

**Already shown in Feature 6** (see `_build_response()` and `_build_turn_response()` above).

---

### Feature 8: Router Refactoring (Separate Module)

#### Agno Implementation
```python
# engine/router.py: BaseSemanticRouter, DefaultSemanticRouter
# workflow/router.py: DocPipelineRouter(DefaultSemanticRouter)
```

#### LangGraph Implementation

**Already shown in Feature 2** (see `engine/router.py` and `workflow/router.py` above).

---

## Part 3: Implementation Checklist

### Phase 1: Foundation (Weeks 1-2)

- [ ] **1.1** Extract handler metadata framework into `engine/handler_registry.py`
  - Copy `HandlerMetadata` dataclass
  - Copy `@handler` decorator
  - Copy helper functions (`get_handler_metadata`, `does_state_wait_for_input`)
  
- [ ] **1.2** Create router base classes in `engine/router.py`
  - Create `RouterDecision` dataclass
  - Create `BaseSemanticRouter` abstract class
  - Create `DefaultSemanticRouter` concrete class
  - Implement `route()` with LangChain Claude integration
  
- [ ] **1.3** Create `engine/input_validation.py`
  - Implement `validate_turn_input()` with length, token count checks
  - Implement `escape_for_llm()` for injection prevention
  - Define `InputValidationError` exception
  
- [ ] **1.4** Update `workflow/router.py`
  - Create `DocPipelineRouter` inheriting `DefaultSemanticRouter`
  - Set `output_schema` to Pydantic model
  - Override `get_instructions()` with document-specific guidance
  - Override `build_router_prompt()` with domain prompt

### Phase 2: Multi-turn Support (Weeks 3-4)

- [ ] **2.1** Extend `PipelineState` in `workflow/pipeline_state.py`
  - Add `turn_input`, `turn_number`, `conversation_history`
  - Add `max_history_turns`, `router_timeout_sec`
  - Add `user_id`, `session_id`
  - Add `semantic_context`, `router_confidence`
  
- [ ] **2.2** Implement base class methods in `engine/graph.py`
  - Add `invoke_turn()` method
  - Add `_auto_progress_langgraph()` method
  - Add `process()` one-turn method
  - Add `_build_turn_response()` response builder
  
- [ ] **2.3** Update router node to support semantic routing
  - Add `_semantic_router_node()` method (LLM-powered)
  - Keep `_router_node()` method (pure code)
  - Route based on `turn_input` presence
  
- [ ] **2.4** Update handler nodes to record turns in history
  - Each handler appends result to `conversation_history`
  - Include semantic_context and state in turn record

### Phase 3: Integration & Testing (Weeks 5-6)

- [ ] **3.1** Update existing `main.py` to use new methods
  - Keep existing `process()` calls unchanged
  - Add examples of `invoke_turn()` for multi-turn
  - Demo pause/resume via `waits_for_input`
  
- [ ] **3.2** Extend tests for multi-turn scenarios
  - Test `invoke_turn()` with conversation history
  - Test semantic router decision making
  - Test auto-progression through non-blocking states
  - Test pause on input-waiting states
  - Test history trimming
  
- [ ] **3.3** Add integration tests for end-to-end flows
  - Multi-turn conversation from INIT to COMPLETE
  - Pause/resume workflow
  - Error handling and validation
  - Semantic context extraction

### Phase 4: Documentation (Week 7)

- [ ] **4.1** Update `docs/` with new features
  - Multi-turn workflow guide
  - Semantic router customization guide
  - Handler metadata usage guide
  - Input validation and escape guide
  
- [ ] **4.2** Update `README.md` with new examples
  - One-turn example (existing)
  - Multi-turn example (new)
  - Custom router example (new)

---

## Part 4: File Structure Changes

### New Files

```
src/engine/
├── handler_registry.py     # NEW: @handler decorator, metadata registry
├── router.py               # NEW: BaseSemanticRouter, DefaultSemanticRouter
├── input_validation.py     # NEW: validate_turn_input, escape_for_llm

src/workflow/
├── router.py               # UPDATED: DocPipelineRouter(DefaultSemanticRouter)
```

### Modified Files

```
src/engine/
├── graph.py                # UPDATED: Add invoke_turn(), process(), _auto_progress_langgraph()
├── checkpointing.py        # NO CHANGE (already compatible)
├── chain.py                # NO CHANGE (already compatible)
├── guardrail.py            # NO CHANGE (already compatible)

src/workflow/
├── pipeline_state.py       # UPDATED: Add multi-turn fields
├── handlers.py             # UPDATED: Add @handler decorators
├── workflow.py             # UPDATED: New graph builder, _init_router()
├── state_machine.py        # NO CHANGE
├── guardrails.py           # NO CHANGE
├── chains.py               # NO CHANGE
```

### Test Files

```
tests/
├── test_multi_turn.py      # NEW: Multi-turn scenarios
├── test_semantic_router.py # NEW: Router decision making
├── test_handler_metadata.py# NEW: @handler decorator behavior
├── test_input_validation.py# NEW: Input escaping and validation
├── test_auto_progress.py   # NEW: Non-blocking state progression
```

---

## Part 5: Migration Path from Current Implementation

### Current State (LangGraph v1)
- Single-turn `run_pipeline()` only
- Pure code router via `HAPPY_PATH` routing table
- No multi-turn support or conversation history
- Manual handler implementation without metadata

### After Implementation (LangGraph v2)
- Both `process()` (one-turn) and `invoke_turn()` (multi-turn) supported
- Optional semantic router via `router` attribute
- Full multi-turn support with conversation history, pause/resume
- Handler metadata registry with `@handler` decorator

### Backward Compatibility
- Existing `run_pipeline()` calls continue to work via `process()` wrapper
- Existing handlers work without `@handler` decorator
- Pure code router still works for one-turn flows
- Semantic router is opt-in (only used if `self.router` is initialized)

---

## Part 6: Configuration & Dependencies

### New Dependencies
```python
# pyproject.toml
langchain-anthropic = "^0.2.0"  # Already in project
pydantic = "^2.0"               # Already in project
tiktoken = "^0.7"               # NEW: For token counting in validation
regex = "^2024.0"               # NEW: For pattern matching in escape_for_llm
```

### Environment
- Same Anthropic API key requirement
- Router uses Claude Haiku 4.5 by default (cost-efficient for routing)

---

## Part 7: Design Rationale

### Why These Features?

| Feature | Benefit | Source |
|---------|---------|--------|
| **Multi-turn** | Support long conversations with pause/resume | Agno's most-used feature |
| **Semantic routing** | LLM-guided state transitions vs hard-coded | More flexible, context-aware |
| **Handler metadata** | Declarative handler configuration | Reduces boilerplate |
| **Input validation** | Security against prompt injection attacks | Production hardening |
| **Auto-progression** | Skip through automatic states without waiting | Better UX for multi-step flows |
| **Engine refactoring** | Centralize multi-turn logic in base class | Code reuse, easier to extend |
| **Response building** | Consistent response format | Cleaner API |
| **Router refactoring** | Separate domain logic from framework | Easier subclassing |

### Why LangGraph > Agno's Custom Workflow?

| Aspect | Agno | LangGraph |
|--------|------|-----------|
| **State persistence** | Custom JsonDb | Built-in checkpointer |
| **Graph visualization** | Manual logging | Native `.get_graph().draw_*()` |
| **Node composition** | Custom Step class | LangChain RunnableSequence |
| **Streaming** | Not native | Built-in `.stream()` |
| **Production maturity** | Internal use | Open-source, widely adopted |
| **Integration** | Limited to Agno | Works with any LangChain component |

---

## Part 8: Known Limitations & Mitigations

### Limitation 1: Semantic Router Latency
**Problem:** LLM call in each multi-turn adds ~1-3 seconds per turn.  
**Mitigation:**
- Use Claude Haiku 4.5 (faster, cheaper than Opus)
- Cache router prompts/constraints in `semantic_context`
- Optional: Skip semantic router for certain states (revert to pure code)

### Limitation 2: Token Counting Accuracy
**Problem:** `tiktoken` tokenizer differs from Claude's internal tokenizer.  
**Mitigation:**
- Use conservative estimate (round up)
- Monitor actual token usage via LLM API response headers
- Adjust `MAX_TOKEN_COUNT` threshold empirically

### Limitation 3: Conversation History Size
**Problem:** Appending all turns to history → unbounded growth.  
**Mitigation:**
- Trim to `max_history_turns` (default 10) after each turn
- Optional: Summarize old turns via LLM before trim
- User can set `max_history_turns` per session

### Limitation 4: Prompt Injection Escaping
**Problem:** No 100% safe escaping; advanced attacks might slip through.  
**Mitigation:**
- Escape commonly-abused patterns (XML, role indicators, etc.)
- Validate final output: if router proposes invalid state, fallback to first allowed state
- Log suspected injection attempts for auditing

---

## Part 9: Testing Strategy

### Unit Tests

1. **handler_registry.py**
   - Test `@handler` decorator registration
   - Test `does_state_wait_for_input()` for registered/unregistered states

2. **router.py**
   - Test `BaseSemanticRouter` abstract interface
   - Test `DefaultSemanticRouter` LLM call
   - Test `RouterDecision` dataclass
   - Test constraint validation (fallback if proposed_next not in allowed_states)

3. **input_validation.py**
   - Test `validate_turn_input()` with valid/invalid inputs
   - Test `escape_for_llm()` removes dangerous patterns
   - Test `InputValidationError` raised on violation

4. **engine/graph.py**
   - Test `invoke_turn()` initializes state on first turn
   - Test `invoke_turn()` resumes state on subsequent turns
   - Test `_auto_progress_langgraph()` stops on waits_for_input or terminal
   - Test `process()` one-turn execution
   - Test `_build_turn_response()` includes all required fields

### Integration Tests

1. **Multi-turn flow**
   - Turn 1: INIT → FETCH (auto) → VALIDATE (blocks)
   - Turn 2: VALIDATE → ENRICH (auto) → STORE (blocks)
   - Turn 3: STORE → COMPLETE
   - Verify conversation_history has 3 turns

2. **Semantic router**
   - Router decides VALIDATE → HUMAN_REVIEW based on user input
   - Guardrail allows transition
   - Handler executes

3. **Input validation**
   - Turn with injection attempt → rejected
   - Turn with valid input → escaped and processed

4. **History trimming**
   - 15 turns with max_history_turns=10
   - After turn 15, history has only last 10 turns

5. **Pause/resume**
   - Handler set to waits_for_input=True
   - Auto-progress stops, returns waits_for_input=True
   - Next turn resumes from same state

### End-to-End Tests

1. Document processing with user intervention
2. Multi-turn conversation with semantic routing
3. Checkpoint and resume from middle of workflow

---

## Appendix: Code Examples

### Example 1: Using Handler Decorator

**Before (current):**
```python
def handle_validate(state: PipelineState) -> PipelineState:
    ...
    return state
```

**After:**
```python
@handler(state="validate", waits_for_input=False, description="Validate document")
def handle_validate(state: PipelineState) -> PipelineState:
    ...
    return state
```

### Example 2: Custom Semantic Router

**Before (current):** Pure code routing only

**After:**
```python
from engine.router import DefaultSemanticRouter
from pydantic import BaseModel, Field

class InvoiceRouterOutput(BaseModel):
    proposed_next: str
    confidence: float
    semantic_entities: dict = Field(default_factory=dict)
    semantic_intents: list = Field(default_factory=list)
    reasoning: str = ""

class InvoiceRouter(DefaultSemanticRouter):
    output_schema = InvoiceRouterOutput
    
    def get_instructions(self) -> str:
        return """You are an invoice processing router.
States: init, fetch, validate, enrich, store, complete.
Decide next state based on document content and user intent."""
    
    def build_router_prompt(self, current_state, turn_input, history_text, allowed_states):
        return f"""Current State: {current_state}
Allowed States: {', '.join(allowed_states)}
User Input: {turn_input}
Determine next state."""

# Usage
workflow = DocumentPipelineGraph()
workflow.router = InvoiceRouter()
response = workflow.invoke_turn("user1", "session1", "Please process invoice ABC123")
```

### Example 3: Multi-turn Conversation

**Before (current):**
```python
result = run_pipeline("doc-001", timeout_seconds=30)
# One execution, then done
```

**After:**
```python
workflow = DocumentPipelineGraph()

# Turn 1: Start processing
response1 = workflow.invoke_turn("user1", "session1", "Start processing my document")
# {"current_state": "validate", "waits_for_input": True, "turn_number": 1, ...}

# Turn 2: User provides feedback
response2 = workflow.invoke_turn("user1", "session1", "Document looks good, proceed")
# {"current_state": "store", "waits_for_input": False, "turn_number": 2, ...}

# Turn 3: Auto-progression continues until next pause point
response3 = workflow.invoke_turn("user1", "session1", "")
# {"current_state": "complete", "waits_for_input": False, "turn_number": 3, ...}
```

### Example 4: Input Validation

**Before (current):** No validation

**After:**
```python
from engine.input_validation import validate_turn_input, InputValidationError

try:
    validate_turn_input(user_input)
    response = workflow.invoke_turn("user1", "session1", user_input)
except InputValidationError as e:
    return {"error": str(e), "current_state": None}
```

---

## Next Steps

1. **Review Design:** Stakeholder approval on architecture choices
2. **Start Phase 1:** Implement handler registry, router base classes, input validation
3. **Parallel Track:** Update tests as each feature is implemented
4. **Weekly Sync:** Review progress, iterate on design as needed

---

**Document Version:** 1.0  
**Last Updated:** 2026-06-24  
**Author:** Claude Code Assistant  
**Status:** Ready for Review
