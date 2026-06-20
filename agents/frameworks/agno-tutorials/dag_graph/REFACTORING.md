# Workflow Refactoring: Engine Layer Extraction

## Overview
Extracted generic, reusable state machine workflow patterns from `workflow/workflow.py` into a new `engine/workflow.py` module. This separation of concerns allows other pipelines to reuse the infrastructure without duplicating code.

## What Moved to `engine/workflow.py`

### 1. **State Marshaling Utilities**
- `serialize_session_state(session_state, keys, defaults=None)` 
  - Extracts specified keys from Agno's `session_state` dict into a typed dict
  - Supports default values for missing keys
  
- `deserialize_to_session_state(state_dict, session_state, keys)`
  - Writes typed dict values back to Agno's session_state (in-place)
  - Enables round-trip serialization without data loss

### 2. **Base Workflow Class: `StateMachineWorkflow`**
Provides the complete generic infrastructure for state machine workflows:

**Class Structure:**
```python
class StateMachineWorkflow(Workflow):
    # Required class variables (set by subclass)
    _STATE_KEYS: tuple[str, ...]           # keys to persist
    _STATE_ENUM: type                      # enum for state values
    _TERMINAL_STATES: set                  # terminal state enum values
    HANDLER_MAP: dict[Any, Callable]       # state → handler function
```

**Required Subclass Hooks:**
- `_init_session_defaults()` — Initialize session_state keys
- `_build_routing_table()` → dict[State, State] — Define state transitions
- `_get_current_state(session_state)` → State — Extract current state
- `_get_proposed_state(session_state)` → State — Extract proposed state
- `_run_guardrail(state_dict)` → (state_dict, result) — Run guardrails
- `HANDLER_MAP` class variable — Maps state → handler function

**Generic Loop Implementation:**
- Builds Agno Loop with Router, Guardrail, and DispatchHandler Steps
- Handles session state binding for all handler executors
- Terminates when reaching terminal states (closed over `self`)
- Supports max iteration safety cap

## What Stayed in `workflow/workflow.py`

### 1. **`DocPipelineWorkflow` Subclass**
Now inherits from `StateMachineWorkflow` with minimal implementation:
- 51 lines (was 296 lines before)
- Only implements required hooks with document-specific logic

### 2. **Document-Specific Components**
- `_HAPPY_PATH` — Document pipeline routing table
- `process(document_id)` — Entry point for document processing
- Guardrail integration via `run_guardrail()`
- Session initialization via `init_session_defaults(mode="pipeline")`
- Multi-run resume and audit trail persistence

### 3. **Helper Functions**
- `_ss_to_pipeline()` — Session state → PipelineState dict
- `_pipeline_to_ss()` — PipelineState dict → session state
- `build_doc_pipeline()` — Factory function

## Key Design Decisions

### 1. **State Enum Values in Session State**
Session state stores string values (e.g., `"fetch"`), but enums are used internally:
- Base class methods accept/return enum values
- Marshaling handles string ↔ enum conversion
- Handlers work with TypedDict, not enums

### 2. **Guardrail Hook Pattern**
The base class provides `_run_guardrail(state_dict)` hook:
- Returns `(updated_state_dict, result)` tuple
- Subclass can run guardrails, mutations, or validation
- Default is pass-through (all guardrails pass)

### 3. **Handler Binding**
Handlers are plain functions, not Agno Steps:
- Factory `_make_handler_executor()` wraps them in Steps
- Each executor closes over `self` to access session_state
- Enables multi-run resume and audit trail persistence

## Migration Path for Other Pipelines

To create a new state machine workflow:

```python
from engine.workflow import StateMachineWorkflow

class MyPipeline(StateMachineWorkflow):
    # 1. Define class variables
    _STATE_KEYS = ("state", "data", "retry_count", ...)
    _STATE_ENUM = MyState
    _TERMINAL_STATES = {MyState.DONE, MyState.ERROR}
    HANDLER_MAP = {MyState.STEP1: handler1, MyState.STEP2: handler2, ...}
    
    # 2. Implement required hooks
    def _init_session_defaults(self):
        self.session_state.setdefault("state", MyState.INIT.value)
        self.session_state.setdefault("data", {})
    
    def _build_routing_table(self):
        return {
            MyState.INIT: MyState.STEP1,
            MyState.STEP1: MyState.STEP2,
            MyState.STEP2: MyState.DONE,
        }
    
    def _get_current_state(self, ss):
        return MyState(ss.get("state", MyState.INIT.value))
    
    def _get_proposed_state(self, ss):
        return MyState(ss.get("proposed_next", MyState.STEP1.value))
    
    def _run_guardrail(self, state_dict):
        # Optional: run custom guardrails
        # Default behavior: all pass
        return state_dict, GuardrailResult(passed=True)
```

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `engine/workflow.py` | **NEW** | 200+ |
| `workflow/workflow.py` | Refactored | 296 → 51 |
| `workflow/pipeline_state.py` | Added `pretty_audit()` | +13 |

## Backwards Compatibility

✅ **Zero breaking changes:**
- `DocPipelineWorkflow` maintains same API
- `build_doc_pipeline()` works unchanged
- Session state schema unchanged
- Handler signatures unchanged
- Main demo script (`main.py`) requires no changes

## Testing Recommendations

1. **Smoke test:** Run existing `main.py` and verify audit trails
2. **Session resume:** Verify `demo_session_resume()` still works
3. **New pipeline:** Create a minimal state machine to verify base class API
4. **Edge cases:** 
   - Missing routing table entries
   - Missing handlers
   - Invalid state transitions
