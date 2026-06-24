# Multi-Turn Conversation Implementation Plan

**Date:** 2026-06-20  
**Status:** Design Phase → Implementation Ready

---

## 1. Split Pipeline State (Control + Business)

### `src/engine/pipeline_state.py` (NEW — Control Plane)

```python
"""
Control plane state — reusable across all workflows.
Manages routing, turns, semantic context, conversation metadata.
"""

from typing import Any, Optional, TypedDict

class EngineState(TypedDict, total=False):
    # ── Multi-turn conversation control ────────────────────────────────────
    turn_input: str | None              # User input for this turn (validated, escaped)
    turn_number: int                    # 0-indexed turn count
    conversation_history: list[dict]    # [{input, output, state_from, state_to, timestamp, ...}]
    semantic_context: dict              # {entities: dict, intents: list[str]}
    conversation_id: str                # UUID for multi-turn session
    max_history_turns: int              # Configurable, default 10
    
    # ── State machine control ──────────────────────────────────────────────
    current_state: str                  # Current active state
    proposed_next: str                  # Router's candidate for next state
    retry_count: int                    # Retry counter
    error_message: Optional[str]        # Error text if in ERROR state
    guardrail_ok: bool                  # True after guardrail passed
    audit_trail: list[str]              # Append-only log

def init_engine_state() -> EngineState:
    """Return fresh engine state for multi-turn conversation."""
    return EngineState(
        turn_input=None,
        turn_number=0,
        conversation_history=[],
        semantic_context={"entities": {}, "intents": []},
        conversation_id="",
        max_history_turns=10,
        current_state="init",
        proposed_next="init",
        retry_count=0,
        error_message=None,
        guardrail_ok=True,
        audit_trail=[],
    )

def audit(state: EngineState, entry: str) -> EngineState:
    """Append entry to audit_trail (immutable-style update)."""
    return {**state, "audit_trail": state["audit_trail"] + [entry]}
```

### `src/workflow/pipeline_state.py` (MODIFY — Business Plane)

```python
"""
Business plane state — document processing payload.
Specific to the DocPipeline domain.
"""

from typing import Any, Optional, TypedDict
from engine.pipeline_state import EngineState

class WorkflowState(TypedDict, total=False):
    # ── Business payload ───────────────────────────────────────────────────
    document_id: str
    raw_data: Optional[dict[str, Any]]      # set by FETCH
    validated_data: Optional[dict[str, Any]]  # set by VALIDATE
    enriched_data: Optional[dict[str, Any]]   # set by ENRICH

# Combined state = engine control + workflow business
class PipelineState(EngineState, WorkflowState):
    """Full pipeline state = control plane + business plane."""
    pass

def new_pipeline(document_id: str, 
                 conversation_id: str = "",
                 max_history_turns: int = 10) -> PipelineState:
    """Return fresh PipelineState with engine + business defaults."""
    engine = init_engine_state()
    return PipelineState(
        # Engine fields
        **engine,
        conversation_id=conversation_id,
        max_history_turns=max_history_turns,
        current_state="init",
        
        # Business fields
        document_id=document_id,
        raw_data=None,
        validated_data=None,
        enriched_data=None,
    )
```

---

## 2. Handler Registry with @handler Decorator

### `src/engine/handler_registry.py` (NEW)

```python
"""
Handler registration and metadata.
Provides @handler decorator and registry for all workflows.
"""

from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class HandlerMetadata:
    """Metadata for a state handler."""
    state: str
    waits_for_input: bool = False
    description: str | None = None

# Global registry (populated by @handler decorator)
HANDLER_MAP_METADATA: dict[str, HandlerMetadata] = {}

def handler(state: str, 
            waits_for_input: bool = False, 
            description: str | None = None) -> Callable:
    """
    Decorator that registers handler with metadata.
    
    Args:
        state: State enum value (e.g., "validate")
        waits_for_input: If True, workflow pauses and waits for next turn
        description: Human-readable description
    
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

def get_handler_metadata(state: str) -> HandlerMetadata | None:
    """Retrieve metadata for a state."""
    return HANDLER_MAP_METADATA.get(state)

def does_state_wait_for_input(state: str) -> bool:
    """Check if state pauses for user input."""
    meta = get_handler_metadata(state)
    return meta.waits_for_input if meta else False
```

### Usage in `src/workflow/handlers.py`

```python
from engine.handler_registry import handler
from workflow.pipeline_state import PipelineState, audit

@handler(state="init", waits_for_input=False, 
         description="Initialize pipeline")
def handle_init(p: PipelineState) -> PipelineState:
    return audit({**p, "current_state": "init"}, "init OK")

@handler(state="fetch", waits_for_input=False,
         description="Fetch document")
def handle_fetch(p: PipelineState) -> PipelineState:
    try:
        # ... fetch logic
        return audit({**p, "current_state": "fetch"}, "fetch OK")
    except Exception as e:
        return audit({**p, "current_state": "error", "error_message": str(e)},
                     f"fetch EXCEPTION: {e}")

@handler(state="wait_documents_uploaded", waits_for_input=True,
         description="Wait for user to upload documents")
def handle_wait_documents(p: PipelineState) -> PipelineState:
    """Pause pipeline; wait for next turn with user providing documents."""
    log.info("[WAIT]  Pausing for document upload")
    # Don't transition state; next turn will decide based on documents
    return audit({**p, "current_state": "wait_documents_uploaded"},
                 "waiting for document upload (will resume on next turn)")
```

---

## 3. Base Semantic Router

### `src/engine/router.py` (NEW)

```python
"""
Base semantic router for LLM-powered state classification.
Subclasses implement domain-specific routing logic.
"""

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

@dataclass
class RouterDecision:
    """Output from semantic router."""
    proposed_next: str              # Next state
    confidence: float               # [0.0, 1.0]
    semantic_entities: dict         # Extracted entities
    semantic_intents: list[str]     # Extracted intents
    reasoning: str | None = None    # Optional explanation

class BaseSemanticRouter(ABC):
    """Abstract base for semantic routers."""
    
    @abstractmethod
    def route(self,
              current_state: str,
              turn_input: str,
              history: list[dict],
              allowed_states: list[str],
              timeout_sec: float = 10.0) -> RouterDecision:
        """
        Classify next state based on user input and conversation history.
        
        Args:
            current_state: Current state (e.g., "validate")
            turn_input: User's input text (validated, escaped)
            history: Last N turns from session_state["conversation_history"]
            allowed_states: Valid next states per ALLOWED_TRANSITIONS
            timeout_sec: LLM call timeout
            
        Returns:
            RouterDecision with proposed_next, confidence, entities, intents
        """
        raise NotImplementedError
```

### `src/workflow/router.py` (NEW — Domain-Specific)

```python
"""
Document pipeline semantic router.
Classifies user intents and extracts document-specific entities.
"""

from engine.router import BaseSemanticRouter, RouterDecision
from engine.agent import Agent
from agno.models.anthropic import Claude

class DocPipelineRouter(BaseSemanticRouter):
    """LLM router for document processing workflow."""
    
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """Initialize with Claude LLM."""
        self.model = Claude(id=model)
        self.agent = Agent(model=self.model)
    
    def route(self,
              current_state: str,
              turn_input: str,
              history: list[dict],
              allowed_states: list[str],
              timeout_sec: float = 10.0) -> RouterDecision:
        """
        Route using Claude LLM with domain context.
        
        Extracts:
          - Entities: amounts, items, keywords, document_ids
          - Intents: confirm, clarify, escalate, upload, cancel
        """
        # Build prompt with current state, turn input, history, allowed states
        history_text = self._format_history(history)
        allowed_text = ", ".join(allowed_states)
        
        prompt = f"""
        Current state: {current_state}
        Allowed next states: {allowed_text}
        
        Conversation history:
        {history_text}
        
        User input: {repr(turn_input)}
        
        Classify the user's intent and determine the next state:
        1. What are the user's intents? (confirm, clarify, escalate, upload, cancel, etc.)
        2. What entities did the user mention? (amounts, items, keywords, document_ids)
        3. Which allowed next state should we transition to?
        
        Respond with JSON:
        {{
          "proposed_next": "<state>",
          "confidence": 0.95,
          "semantic_intents": ["confirm", "clarify"],
          "semantic_entities": {{"amounts": ["$99.99"], "items": ["document1.pdf"]}},
          "reasoning": "User confirmed amount"
        }}
        """
        
        try:
            response = self.agent.run(prompt, timeout=timeout_sec)
            # Parse response JSON
            decision_json = self._parse_json(response.content)
            
            proposed = decision_json.get("proposed_next", allowed_states[0])
            
            # If invalid, retry with constraints
            if proposed not in allowed_states:
                decision_json["proposed_next"] = allowed_states[0]
            
            return RouterDecision(
                proposed_next=proposed,
                confidence=decision_json.get("confidence", 0.5),
                semantic_entities=decision_json.get("semantic_entities", {}),
                semantic_intents=decision_json.get("semantic_intents", []),
                reasoning=decision_json.get("reasoning")
            )
        except Exception as e:
            # On timeout/error, route to ERROR
            return RouterDecision(
                proposed_next="error",
                confidence=0.0,
                semantic_entities={},
                semantic_intents=[],
                reasoning=f"Router error: {str(e)}"
            )
    
    def _format_history(self, history: list[dict], max_turns: int = 10) -> str:
        """Format last N turns as text."""
        recent = history[-max_turns:]
        lines = []
        for turn in recent:
            lines.append(f"[{turn.get('role', '?')}]: {turn.get('content', '')}")
        return "\n".join(lines) if lines else "(no history)"
    
    def _parse_json(self, text: str) -> dict:
        """Extract JSON from LLM response."""
        import json, re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
```

---

## 4. Input Validation

### `src/engine/input_validation.py` (NEW)

```python
"""
Input validation and sanitization for multi-turn.
Prevents DoS attacks and prompt injection.
"""

class InputValidationError(Exception):
    """Raised when turn_input fails validation."""
    pass

def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token per 4 characters."""
    return len(text) // 4

def validate_turn_input(turn_input: str,
                        max_chars: int = 10_000,
                        max_tokens: int = 2_000) -> None:
    """
    Validate turn_input length and token count.
    
    Raises InputValidationError if validation fails.
    """
    if not isinstance(turn_input, str):
        raise InputValidationError("turn_input must be a string")
    
    if len(turn_input) > max_chars:
        raise InputValidationError(
            f"turn_input exceeds {max_chars} characters (got {len(turn_input)})"
        )
    
    token_count = estimate_tokens(turn_input)
    if token_count > max_tokens:
        raise InputValidationError(
            f"turn_input exceeds {max_tokens} tokens (got {token_count})"
        )

def escape_for_llm(turn_input: str) -> str:
    """
    Escape turn_input to prevent prompt injection.
    Use repr() for safe escaping.
    """
    return repr(turn_input)
```

---

## 5. Engine Workflow — Add Semantic Router Support

### `src/engine/workflow.py` (MODIFY)

```python
# Add to StateMachineWorkflow class:

def _semantic_router_step(self, step_input: StepInput) -> StepOutput:
    """
    LLM-powered router: reads current_state + turn_input → proposes next state.
    
    Used for multi-turn workflows. Single-turn workflows use _router_step.
    """
    if not hasattr(self, 'router') or self.router is None:
        raise ValueError("Semantic router not initialized")
    
    if not hasattr(self, '_STATE_KEYS'):
        raise ValueError("_STATE_KEYS not defined")
    
    current = self._get_current_state(self.session_state)
    turn_input = self.session_state.get("turn_input", "")
    history = self.session_state.get("conversation_history", [])
    
    # Get allowed transitions
    routing_table = self._build_routing_table()
    allowed_next = list(routing_table.get(current, {}).keys()) \
        if isinstance(routing_table.get(current), dict) \
        else [routing_table.get(current)]
    
    # Call semantic router
    decision = self.router.route(
        current_state=current.value if hasattr(current, 'value') else current,
        turn_input=turn_input,
        history=history,
        allowed_states=allowed_next,
        timeout_sec=self.session_state.get("router_timeout_sec", 10.0)
    )
    
    # Store decision in session
    self.session_state["proposed_next"] = decision.proposed_next
    self.session_state["semantic_context"] = {
        "entities": decision.semantic_entities,
        "intents": decision.semantic_intents
    }
    self.session_state["router_confidence"] = decision.confidence
    
    log.info("[SemanticRouter] %s → proposes %s (confidence: %.2f)",
             current, decision.proposed_next, decision.confidence)
    
    return StepOutput(content={
        "proposed_next": decision.proposed_next,
        "semantic_context": self.session_state["semantic_context"]
    })
```

**Routing Logic Decision:** Keep router in `engine/workflow.py` but make it pluggable:
- One-turn: Use `_router_step()` (pure code routing)
- Multi-turn: Use `_semantic_router_step()` (LLM routing)
- Subclass chooses via `_choose_router_step()` method

---

## 6. Workflow Creation and Initialization

### Creating a Workflow with Initial State

```python
from agno.workflow import Workflow
from agno.db.sqlite import SqliteDb
from workflow.workflow import DocPipelineWorkflow

# Option 1: One-off workflow (no persistence)
wf = DocPipelineWorkflow(
    name="DocPipeline",
    session_state={
        "document_id": "DOC-001",
        "current_state": "init",
        "conversation_history": [],
    }
)
wf.run()

# Option 2: Persistent workflow with database
db = SqliteDb(
    table_name="workflow_sessions",
    db_file="tmp/workflows.db"
)

wf = DocPipelineWorkflow(
    name="DocPipeline",
    db=db,
    # Initial state used ONLY for brand new sessions
    session_state={
        "document_id": "DOC-001",
        "current_state": "init",
        "conversation_history": [],
        "conversation_id": "",
        "max_history_turns": 10,
    }
)

# First run: Creates session, initializes state
wf.run(session_id="session_123", user_id="user_1")
print(wf.session_state)

# Second run: Loads persisted state from DB
wf.run(session_id="session_123", user_id="user_1")
print(wf.session_state)  # Will show updated state from DB
```

---

## 7. Multi-Turn Execution: process_turn()

### `src/workflow/workflow.py` (EXPAND DocPipelineWorkflow)

```python
from engine.input_validation import validate_turn_input, escape_for_llm
from engine.handler_registry import does_state_wait_for_input
from workflow.router import DocPipelineRouter

class DocPipelineWorkflow(StateMachineWorkflow):
    """Document processing with one-turn and multi-turn support."""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        # Initialize semantic router for multi-turn
        self.router = DocPipelineRouter()
    
    def process_turn(self,
                     user_id: str,
                     session_id: str,
                     turn_input: str,
                     timeout_sec: float = 10.0) -> dict:
        """
        Execute one turn of multi-turn conversation.
        
        Args:
            user_id: Caller identity
            session_id: Multi-turn session ID
            turn_input: User's input text
            timeout_sec: LLM router timeout
            
        Returns:
            {
              "current_state": str,
              "waits_for_input": bool,
              "turn_number": int,
              "semantic_context": dict,
              "error": str | None
            }
        """
        try:
            # 1. Validate input
            validate_turn_input(turn_input)
            escaped = escape_for_llm(turn_input)
            
            # 2. Initialize session if needed
            if self.session_state is None:
                self.session_state = {}
            
            # 3. Append turn input
            turn_num = self.session_state.get("turn_number", 0)
            self.session_state["turn_input"] = escaped
            self.session_state["turn_number"] = turn_num + 1
            self.session_state["router_timeout_sec"] = timeout_sec
            
            # Initialize conversation history list
            if "conversation_history" not in self.session_state:
                self.session_state["conversation_history"] = []
            
            # 4. Run workflow loop (calls _semantic_router_step, guardrail, handler)
            self.run(session_id=session_id, user_id=user_id)
            
            # 5. Trim history
            self._trim_history()
            
            # 6. Check if current state waits for input
            current = self.session_state["current_state"]
            waits = does_state_wait_for_input(current)
            
            # 7. Return response
            return {
                "current_state": current,
                "waits_for_input": waits,
                "turn_number": self.session_state["turn_number"],
                "semantic_context": self.session_state.get("semantic_context", {}),
                "router_confidence": self.session_state.get("router_confidence", 0.0),
                "error": self.session_state.get("error_message")
            }
        
        except InputValidationError as e:
            return {
                "error": str(e),
                "current_state": None,
                "waits_for_input": False,
            }
        except Exception as e:
            log.exception("process_turn failed: %s", e)
            return {
                "error": str(e),
                "current_state": "error",
                "waits_for_input": False,
            }
    
    def _trim_history(self) -> None:
        """Keep only last max_history_turns in session state."""
        max_turns = self.session_state.get("max_history_turns", 10)
        history = self.session_state.get("conversation_history", [])
        if len(history) > max_turns:
            dropped = len(history) - max_turns
            self.session_state["conversation_history"] = history[-max_turns:]
            log.info(f"Trimmed {dropped} turns; keeping last {max_turns}")
    
    def _choose_router_step(self):
        """Choose between pure-code or semantic routing."""
        # Multi-turn: use semantic router
        if self.session_state.get("turn_input"):
            return self._semantic_router_step
        # One-turn: use pure-code router
        return self._router_step
```

---

## 8. Session Management with Persistence

### `src/engine/session.py` (EXPAND)

```python
def init_control_state_defaults(session_state: dict[str, Any]) -> None:
    """
    Initialize control plane fields for multi-turn workflow.
    Called from StateMachineWorkflow._init_session_defaults().
    """
    defaults = {
        "turn_input": None,
        "turn_number": 0,
        "conversation_history": [],
        "semantic_context": {"entities": {}, "intents": []},
        "conversation_id": "",
        "max_history_turns": 10,
        "router_timeout_sec": 10.0,
        "current_state": "init",
        "proposed_next": "init",
        "retry_count": 0,
        "error_message": None,
        "guardrail_ok": True,
        "audit_trail": [],
    }
    for k, v in defaults.items():
        session_state.setdefault(k, v)

def append_turn(session_state: dict[str, Any],
                role: str,
                content: str,
                state_from: str = "",
                state_to: str = "",
                router_confidence: float = 0.0) -> None:
    """
    Append a conversation turn to session_state["conversation_history"].
    Automatically persisted by agno at end of run().
    """
    turn = {
        "role": role,
        "content": content,
        "state_from": state_from,
        "state_to": state_to,
        "router_confidence": router_confidence,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    session_state.setdefault("conversation_history", []).append(turn)
```

---

## 9. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `src/engine/pipeline_state.py` (EngineState, init_engine_state)
- [ ] Modify `src/workflow/pipeline_state.py` (split business fields)
- [ ] Create `src/engine/handler_registry.py` (@handler decorator)
- [ ] Create `src/engine/input_validation.py` (validation + escaping)
- [ ] Create `src/engine/router.py` (BaseSemanticRouter)
- [ ] Create `src/workflow/router.py` (DocPipelineRouter)

### Phase 2: Workflow Integration
- [ ] Modify `src/engine/workflow.py` (_semantic_router_step, _choose_router_step)
- [ ] Modify `src/workflow/handlers.py` (@handler decorators, exception handling)
- [ ] Modify `src/workflow/workflow.py` (process_turn, _trim_history)
- [ ] Modify `src/engine/session.py` (init_control_state_defaults, append_turn)

### Phase 3: Testing
- [ ] Add `wait_documents_uploaded` handler test
- [ ] Test one-turn backward compatibility
- [ ] Test multi-turn with persistent DB
- [ ] Test router classification
- [ ] Test input validation

---

## 10. Example Usage: Multi-Turn Conversation

```python
from agno.db.sqlite import SqliteDb
from workflow.workflow import DocPipelineWorkflow

# Setup
db = SqliteDb(table_name="sessions", db_file="tmp/sessions.db")
wf = DocPipelineWorkflow(
    name="DocPipeline",
    db=db,
    session_state={"conversation_id": "conv_123", "max_history_turns": 10}
)

# Turn 1: User uploads document
response1 = wf.process_turn(
    user_id="user_1",
    session_id="session_abc",
    turn_input="I want to process document.pdf"
)
print(f"Turn 1: {response1['current_state']}, waits={response1['waits_for_input']}")
# Output: current_state=fetch, waits=False (auto-continues)

# Turn 2: User confirms validation
response2 = wf.process_turn(
    user_id="user_1",
    session_id="session_abc",
    turn_input="Yes, this looks correct"
)
print(f"Turn 2: {response2['current_state']}, waits={response2['waits_for_input']}")
# Output: current_state=enrich, waits=False

# Turn 3: User decides to stop
response3 = wf.process_turn(
    user_id="user_1",
    session_id="session_abc",
    turn_input="Actually, cancel the process"
)
print(f"Turn 3: {response3['current_state']}, waits={response3['waits_for_input']}")
# Output: current_state=error (invalid transition) or special CANCELLED state

# Resume later: Agno loads state from DB automatically
wf2 = DocPipelineWorkflow(name="DocPipeline", db=db)
response4 = wf2.process_turn(
    user_id="user_1",
    session_id="session_abc",
    turn_input="Can we resume?"
)
# State will be whatever was persisted from Turn 3
```

---

## Summary

| Component | Location | Type | Purpose |
|-----------|----------|------|---------|
| EngineState | `engine/pipeline_state.py` | NEW | Control plane TypedDict |
| WorkflowState | `workflow/pipeline_state.py` | MODIFY | Business plane TypedDict |
| @handler decorator | `engine/handler_registry.py` | NEW | Metadata for state handlers |
| BaseSemanticRouter | `engine/router.py` | NEW | Abstract router base class |
| DocPipelineRouter | `workflow/router.py` | NEW | LLM-powered domain router |
| Input validation | `engine/input_validation.py` | NEW | Sanitization + DoS prevention |
| _semantic_router_step | `engine/workflow.py` | MODIFY | LLM routing step |
| process_turn() | `workflow/workflow.py` | NEW | Multi-turn entry point |
| Agno persistence | SqliteDb/PostgresDb | EXISTING | Auto-saves session_state |

**Key:** Agno's `session_state` is auto-persisted to DB after each `run()`. No explicit checkpoint manager needed.
