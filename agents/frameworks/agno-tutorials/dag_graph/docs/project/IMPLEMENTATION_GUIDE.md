# Implementation Guide: Creating New State Machine Workflows

This guide shows how to create a new workflow using the `StateMachineWorkflow` base class.

## Step 1: Define Your State Machine

Create a new module (e.g., `triage_pipeline/state_machine.py`):

```python
from enum import Enum

class TriageState(str, Enum):
    INIT          = "init"
    RECEIVE       = "receive"
    CLASSIFY      = "classify"
    ROUTE         = "route"
    RESOLVE       = "resolve"
    COMPLETE      = "complete"
    ERROR         = "error"

# Allowed transitions (adjacency list)
ALLOWED_TRANSITIONS: dict[TriageState, set[TriageState]] = {
    TriageState.INIT:     {TriageState.RECEIVE},
    TriageState.RECEIVE:  {TriageState.CLASSIFY, TriageState.ERROR},
    TriageState.CLASSIFY: {TriageState.ROUTE, TriageState.ERROR},
    TriageState.ROUTE:    {TriageState.RESOLVE, TriageState.ERROR},
    TriageState.RESOLVE:  {TriageState.COMPLETE, TriageState.ERROR},
    TriageState.COMPLETE: set(),
    TriageState.ERROR:    set(),
}

TERMINAL_STATES = frozenset({TriageState.COMPLETE, TriageState.ERROR})
```

## Step 2: Define Handler Functions

Create `triage_pipeline/handlers.py`:

```python
from .state_machine import TriageState
from .pipeline_state import PipelineState

def handle_receive(state: PipelineState) -> PipelineState:
    """Receive incoming ticket/message."""
    return {
        **state,
        "current_state": TriageState.RECEIVE.value,
        "raw_message": "Incoming issue...",
        "audit_trail": state["audit_trail"] + ["received ticket"],
    }

def handle_classify(state: PipelineState) -> PipelineState:
    """Use LLM to classify the issue."""
    # In real code: call CLASSIFY_AGENT.run()
    return {
        **state,
        "current_state": TriageState.CLASSIFY.value,
        "category": "bug",
        "priority": "high",
        "audit_trail": state["audit_trail"] + ["classified as bug/high"],
    }

def handle_route(state: PipelineState) -> PipelineState:
    """Route to appropriate team."""
    team = "backend" if state["category"] == "bug" else "product"
    return {
        **state,
        "current_state": TriageState.ROUTE.value,
        "assigned_team": team,
        "audit_trail": state["audit_trail"] + [f"routed to {team}"],
    }

def handle_resolve(state: PipelineState) -> PipelineState:
    """Resolve the issue."""
    return {
        **state,
        "current_state": TriageState.RESOLVE.value,
        "resolution": "assigned to team",
        "audit_trail": state["audit_trail"] + ["assigned"],
    }

def handle_complete(state: PipelineState) -> PipelineState:
    """Mark as complete."""
    return {
        **state,
        "current_state": TriageState.COMPLETE.value,
        "audit_trail": state["audit_trail"] + ["complete"],
    }

def handle_error(state: PipelineState) -> PipelineState:
    """Terminal error state."""
    return {
        **state,
        "current_state": TriageState.ERROR.value,
        "audit_trail": state["audit_trail"] + ["error"],
    }

# Export the handler map
HANDLER_MAP = {
    TriageState.RECEIVE:  handle_receive,
    TriageState.CLASSIFY: handle_classify,
    TriageState.ROUTE:    handle_route,
    TriageState.RESOLVE:  handle_resolve,
    TriageState.COMPLETE: handle_complete,
    TriageState.ERROR:    handle_error,
}
```

## Step 3: Define State Shape

Create `triage_pipeline/pipeline_state.py`:

```python
from typing import TypedDict, Optional

class PipelineState(TypedDict):
    # Control plane
    current_state: str
    proposed_next: str
    error_message: Optional[str]
    audit_trail: list[str]
    
    # Business plane
    ticket_id: str
    raw_message: str
    category: Optional[str]
    priority: Optional[str]
    assigned_team: Optional[str]
    resolution: Optional[str]

def new_pipeline(ticket_id: str) -> PipelineState:
    """Create a fresh pipeline state."""
    return PipelineState(
        current_state="init",
        proposed_next="receive",
        error_message=None,
        audit_trail=[f"init ticket_id={ticket_id}"],
        ticket_id=ticket_id,
        raw_message="",
        category=None,
        priority=None,
        assigned_team=None,
        resolution=None,
    )
```

## Step 4: Define Guardrails (Optional)

Create `triage_pipeline/guardrails.py`:

```python
from engine.guardrail import GuardrailResult, GuardrailFn, GUARDRAIL_PASS
from .pipeline_state import PipelineState
from .state_machine import TriageState, ALLOWED_TRANSITIONS

def check_transition_allowed(state: PipelineState) -> GuardrailResult:
    """Ensure proposed transition is legal."""
    current = TriageState(state["current_state"])
    proposed = TriageState(state["proposed_next"])
    
    if proposed in ALLOWED_TRANSITIONS.get(current, set()):
        return GUARDRAIL_PASS
    
    return GuardrailResult(
        passed=False,
        reason=f"Illegal transition {current.value} → {proposed.value}",
        fallback=TriageState.ERROR,
    )

def check_has_category(state: PipelineState) -> GuardrailResult:
    """Require category before routing."""
    if state["proposed_next"] == "route" and not state.get("category"):
        return GuardrailResult(
            passed=False,
            reason="Missing category; must classify first",
            fallback=TriageState.CLASSIFY,
        )
    return GUARDRAIL_PASS

GUARDRAILS = {
    TriageState.ROUTE: lambda s: check_transition_allowed(s),
    # ... other guardrails
}

def run_guardrail(state: PipelineState) -> tuple[PipelineState, GuardrailResult]:
    """Run guardrail for proposed_next."""
    proposed = TriageState(state["proposed_next"])
    guard = GUARDRAILS.get(proposed, lambda _: GUARDRAIL_PASS)
    result = guard(state)
    
    if result.passed:
        return state, result
    
    fallback_state = (result.fallback or TriageState.ERROR).value
    return {
        **state,
        "proposed_next": fallback_state,
        "error_message": result.reason,
    }, result
```

## Step 5: Create Your Workflow Class

Create `triage_pipeline/workflow.py`:

```python
from agno.db.json.json_db import JsonDb
from engine.workflow import StateMachineWorkflow

from .handlers import HANDLER_MAP
from .guardrails import run_guardrail
from .pipeline_state import PipelineState, new_pipeline
from .session import init_session_defaults
from .state_machine import TriageState, TERMINAL_STATES

# Define state keys (what to persist)
_STATE_KEYS = (
    "current_state", "proposed_next", "error_message", "audit_trail",
    "ticket_id", "raw_message", "category", "priority", 
    "assigned_team", "resolution",
)

# Routing table (happy path)
_ROUTING_TABLE = {
    TriageState.INIT:     TriageState.RECEIVE,
    TriageState.RECEIVE:  TriageState.CLASSIFY,
    TriageState.CLASSIFY: TriageState.ROUTE,
    TriageState.ROUTE:    TriageState.RESOLVE,
    TriageState.RESOLVE:  TriageState.COMPLETE,
}

class TriageWorkflow(StateMachineWorkflow):
    """Triage workflow for support tickets."""
    
    # Bind class variables
    _STATE_KEYS = _STATE_KEYS
    _STATE_ENUM = TriageState
    _TERMINAL_STATES = TERMINAL_STATES
    HANDLER_MAP = HANDLER_MAP
    
    # Implement hooks
    def _init_session_defaults(self) -> None:
        """Initialize session with triage defaults."""
        init_session_defaults(self.session_state, mode="triage")
    
    def _build_routing_table(self) -> dict[TriageState, TriageState]:
        """Return the happy-path routing table."""
        return _ROUTING_TABLE
    
    def _get_current_state(self, session_state):
        """Extract current state."""
        return TriageState(session_state.get("current_state", "init"))
    
    def _get_proposed_state(self, session_state):
        """Extract proposed state."""
        return TriageState(session_state.get("proposed_next", "receive"))
    
    def _run_guardrail(self, state_dict):
        """Run guardrails."""
        return run_guardrail(state_dict)
    
    # Business logic
    def process(self, ticket_id: str) -> PipelineState:
        """Process a support ticket."""
        fresh = new_pipeline(ticket_id)
        _pipeline_to_ss(fresh, self.session_state)
        self.run(input=ticket_id)
        final = _ss_to_pipeline(self.session_state)
        print(f"Ticket {ticket_id}: {final['current_state'].upper()}")
        return final


# Helper functions
def _ss_to_pipeline(ss):
    return PipelineState(
        current_state=ss.get("current_state", "init"),
        proposed_next=ss.get("proposed_next", "receive"),
        error_message=ss.get("error_message"),
        audit_trail=ss.get("audit_trail", []),
        ticket_id=ss.get("ticket_id", ""),
        raw_message=ss.get("raw_message", ""),
        category=ss.get("category"),
        priority=ss.get("priority"),
        assigned_team=ss.get("assigned_team"),
        resolution=ss.get("resolution"),
    )

def _pipeline_to_ss(p, ss):
    for k in _STATE_KEYS:
        if k in p:
            ss[k] = p[k]

def build_triage_workflow(session_id: str) -> TriageWorkflow:
    """Create or resume a triage workflow."""
    return TriageWorkflow(
        name="TriageWorkflow",
        session_id=session_id,
        db=JsonDb(db_path=".triage_sessions"),
    )
```

## Step 6: Use Your Workflow

```python
from triage_pipeline.workflow import build_triage_workflow

# Create a new workflow
workflow = build_triage_workflow("session-001")

# Process a ticket
result = workflow.process("TICKET-123")

print(f"Final state: {result['current_state']}")
print(f"Audit trail: {result['audit_trail']}")
```

## Key Patterns

### 1. **Immutable-Style Updates**
```python
def handle_something(state: PipelineState) -> PipelineState:
    return {
        **state,  # Copy everything
        "current_state": NewState.value,  # Override fields
        "field": new_value,
    }
```

### 2. **Audit Trail**
```python
def handle_step(state: PipelineState) -> PipelineState:
    return {
        **state,
        "current_state": State.NEXT.value,
        "audit_trail": state["audit_trail"] + [f"step: did something"],
    }
```

### 3. **Error Handling**
```python
def handle_risky_step(state: PipelineState) -> PipelineState:
    try:
        result = risky_operation()
        return {**state, "current_state": State.NEXT.value, "result": result}
    except Exception as e:
        return {
            **state,
            "current_state": State.ERROR.value,
            "error_message": str(e),
        }
```

### 4. **LLM Integration**
```python
def handle_classify(state: PipelineState) -> PipelineState:
    agent = get_agent("classifier")
    result = agent.run(state["raw_message"]).content
    return {
        **state,
        "current_state": State.NEXT.value,
        "category": result.category,
        "confidence": result.confidence,
    }
```

## Testing

```python
def test_triage_workflow():
    wf = build_triage_workflow("test-session")
    
    # Process a ticket
    result = wf.process("TEST-001")
    
    # Verify final state
    assert result["current_state"] == "complete"
    assert len(result["audit_trail"]) > 0
    assert result["category"] is not None
    
    # Verify persistence (resume same session)
    resumed = build_triage_workflow("test-session")
    assert len(resumed.session_state.get("pipeline_runs", [])) > 0
```

## Checklist

- [ ] Define State enum with all states
- [ ] Define ALLOWED_TRANSITIONS (adjacency list)
- [ ] Define TERMINAL_STATES
- [ ] Create handler function for each state
- [ ] Build HANDLER_MAP dict
- [ ] Define PipelineState TypedDict
- [ ] Create new_pipeline() factory
- [ ] Define _STATE_KEYS tuple
- [ ] Create routing table dict
- [ ] Subclass StateMachineWorkflow
- [ ] Implement 5 hook methods
- [ ] Define helper functions (_ss_to_pipeline, _pipeline_to_ss)
- [ ] Create workflow factory function
- [ ] Write tests
- [ ] Document state transitions

## Common Mistakes

❌ **Don't:**
- Modify state in-place (use `{**state, ...}`)
- Forget to set `current_state` in handlers
- Forget to append audit trail entries
- Have handlers that can fail silently
- Use hardcoded state strings instead of enums

✅ **Do:**
- Use immutable-style updates
- Always set `current_state` to current state's value
- Log all transitions in audit_trail
- Handle exceptions and transition to ERROR
- Use State enums throughout
