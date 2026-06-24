# New Pipeline Quick Start (5-Step Checklist)

Fast path to creating a new pipeline by reusing engine components.

---

## The 5 Things You Need to Build

### 1️⃣ State Enum
```python
# src/my_pipeline/state_machine.py
from enum import Enum

class State(str, Enum):
    INIT = "init"
    STEP_1 = "step_1"
    STEP_2 = "step_2"
    COMPLETE = "complete"
    ERROR = "error"

TERMINAL_STATES = {State.COMPLETE, State.ERROR}
```

### 2️⃣ Business State TypedDict
```python
# src/my_pipeline/pipeline_state.py
from engine.engine_state import EngineState, init_engine_state

class MyPipelineState(EngineState):
    """Add your business fields here."""
    my_field_1: str
    my_field_2: Optional[dict]

def new_pipeline_state(entity_id: str) -> MyPipelineState:
    base = init_engine_state()
    return {**base, "current_state": "init", "my_field_1": entity_id}
```

### 3️⃣ Handlers (One Per State)
```python
# src/my_pipeline/handlers.py
from engine.handler_registry import handler

@handler(state="step_1", waits_for_input=False, description="Do thing 1")
def handle_step_1(state: MyPipelineState) -> MyPipelineState:
    try:
        # Your business logic here
        return {**state, "current_state": "step_1"}
    except Exception as e:
        return {**state, "current_state": "error", "error_message": str(e)}

# Decorate all handlers, add try/catch
HANDLER_MAP = {State.STEP_1: handle_step_1, ...}
```

### 4️⃣ Domain Router
```python
# src/my_pipeline/router.py
from engine.router import BaseSemanticRouter, RouterDecision

class MyRouter(BaseSemanticRouter):
    def route(self, current_state, turn_input, history, allowed_states, timeout_sec=10.0):
        # Use LLM to classify next state
        # Return RouterDecision(proposed_next, confidence, entities, intents)
        return RouterDecision(
            proposed_next=allowed_states[0],
            confidence=0.9,
            semantic_entities={},
            semantic_intents=[],
        )
```

### 5️⃣ Workflow Class
```python
# src/my_pipeline/workflow.py
from engine.statemachine_workflow import StateMachineWorkflow
from .handlers import HANDLER_MAP
from .state_machine import State, TERMINAL_STATES
from .router import MyRouter

_ROUTING_TABLE = {
    State.INIT: State.STEP_1,
    State.STEP_1: State.STEP_2,
    State.STEP_2: State.COMPLETE,
}

class MyWorkflow(StateMachineWorkflow):
    _STATE_KEYS = ("current_state", "proposed_next", "my_field_1", "my_field_2", ...)
    _STATE_ENUM = State
    _TERMINAL_STATES = TERMINAL_STATES
    HANDLER_MAP = HANDLER_MAP

    def __post_init__(self):
        super().__post_init__()
        self.router = MyRouter()

    def _init_session_defaults(self):
        self.session_state.setdefault("current_state", "init")
        self.session_state.setdefault("my_field_1", None)

    def _build_routing_table(self) -> dict:
        return _ROUTING_TABLE

    def _get_current_state(self, session_state):
        return State(session_state.get("current_state", "init"))

    def _get_proposed_state(self, session_state):
        return State(session_state.get("proposed_next", "init"))

    def _new_session_state(self, entity_id: str) -> dict:
        return {"current_state": State.INIT.value, "entity_id": entity_id, ...}

    def _run_guardrail(self, state_dict):
        from dataclasses import dataclass
        @dataclass
        class Result:
            passed: bool = True
            reason: str = ""
        return state_dict, Result()

    # Optional: override _build_response() to return domain-specific type
    # def _build_response(self, entity_id: str) -> MyPipelineState:
    #     return MyPipelineState(...from session_state...)
    # (if not overridden, base class returns dict with standard fields)
```

---

## That's It! You Get Automatically

✅ **One-turn support** — process(entity_id) inherited from base class  
✅ **Multi-turn support** — process_turn() inherited from base class  
✅ **Semantic routing** — LLM-powered state classification  
✅ **Error handling** — Exceptions route to ERROR state  
✅ **Session persistence** — Auto-save to DB  
✅ **Input validation** — Token limits, prompt injection prevention  
✅ **Entity extraction** — Router returns semantic_context  
✅ **Conversation history** — Auto-trimmed to max_history_turns  
✅ **Pause/resume** — waits_for_input flag pauses workflow  
✅ **Response building** — _build_response() default extracts standard fields  

---

## Usage

```python
# One-turn (synchronous)
wf = MyWorkflow(name="MyPipeline")
result = wf.process(entity_id="123")

# Multi-turn (conversation)
wf = MyWorkflow(name="MyPipeline", db=SqliteDb(...))
response = wf.process_turn(
    user_id="user_1",
    session_id="session_abc",
    turn_input="User says something"
)
```

---

## File Template Checklist

- [ ] `src/my_pipeline/state_machine.py` — State enum + TERMINAL_STATES
- [ ] `src/my_pipeline/pipeline_state.py` — Business TypedDict (inherit from EngineState)
- [ ] `src/my_pipeline/handlers.py` — All handlers with @handler decorator
- [ ] `src/my_pipeline/router.py` — Domain router (inherit from BaseSemanticRouter)
- [ ] `src/my_pipeline/workflow.py` — Workflow class with 5 required hooks:
  - `_init_session_defaults()`
  - `_build_routing_table()`
  - `_get_current_state()`
  - `_get_proposed_state()`
  - `_new_session_state(entity_id)`
- [ ] `src/my_pipeline/__init__.py` — Exports (workflow factory, state types)
- [ ] `tests/my_pipeline/test_workflow_integration.py` — Integration tests

---

## Copy-Paste Reference

### Minimal Handler Template
```python
@handler(state="my_state", waits_for_input=False, description="Do something")
def handle_my_state(state: MyPipelineState) -> MyPipelineState:
    try:
        # ... your logic ...
        return audit({**state, "current_state": "my_state"}, "OK")
    except Exception as e:
        return audit({**state, "current_state": "error", "error_message": str(e)}, f"EXCEPTION: {e}")
```

### Minimal Router Template
```python
class MyRouter(BaseSemanticRouter):
    def route(self, current_state, turn_input, history, allowed_states, timeout_sec=10.0):
        # Implement your routing logic
        return RouterDecision(
            proposed_next=allowed_states[0] if allowed_states else "error",
            confidence=0.9,
            semantic_entities={},
            semantic_intents=[],
        )
```

---

## See Also

- **CREATE_NEW_PIPELINE.md** — Full detailed guide with Invoice example
- **src/workflow/** — Reference implementation (DocPipelineWorkflow)
- **CLAUDE.md** — Coding standards to follow
