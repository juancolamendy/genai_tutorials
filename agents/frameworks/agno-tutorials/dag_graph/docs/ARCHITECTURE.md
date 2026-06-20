# Workflow Architecture

## Layered Design

```
┌────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                          │
│  main.py — Demo application using DocPipelineWorkflow          │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│              WORKFLOW / BUSINESS LOGIC LAYER                   │
│  workflow/                                                      │
│  ├── workflow.py          DocPipelineWorkflow (51 lines)       │
│  │   ├── _HAPPY_PATH                                           │
│  │   ├── process()        — Entry point                        │
│  │   └── hooks             — Business logic implementations    │
│  ├── handlers.py          Handler functions (HANDLER_MAP)      │
│  ├── guardrails.py        Guardrail rules & runner             │
│  ├── state_machine.py     State enum & transitions             │
│  ├── pipeline_state.py    PipelineState TypedDict             │
│  └── session.py           Session initialization               │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│              ENGINE / INFRASTRUCTURE LAYER                     │
│  engine/                                                        │
│  ├── workflow.py          StateMachineWorkflow base class      │
│  │   ├── serialize_session_state()                             │
│  │   ├── deserialize_to_session_state()                        │
│  │   └── Loop + Router + Guardrail + Dispatch pattern         │
│  ├── agent.py             Agent factory & LLM step builder     │
│  ├── session.py           Conversation history helpers         │
│  └── guardrail.py         GuardrailResult & composition        │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│            FRAMEWORK LAYER (Agno + External)                  │
│  - Agno Workflow, Step, Router, Loop                          │
│  - Agno Agents (Claude)                                        │
│  - Agno JsonDb for session persistence                         │
│  - Python standard library                                     │
└────────────────────────────────────────────────────────────────┘
```

## Component Interaction

### Workflow Execution Loop
```
StateMachineWorkflow.run()
    ↓
Loop (max_iterations=20, end_condition=is_terminal)
    ↓
    Step(Router)          — Reads session_state["current_state"]
        ↓                   Calls _router_step() → proposes next
    Step(Guardrail)       — Validates proposed_next
        ↓                   Calls _guardrail_step() → may override
    Router(DispatchHandler)
        ↓                 — Selects handler Step via _dispatch()
    Step(Handler)         — Executes handler (fetch, validate, enrich, etc.)
        ↓                   Calls handler from HANDLER_MAP
    Updates session_state & persists to JsonDb
        ↓
[Loop continues or terminates if _is_terminal() = True]
```

### State Flow: Document Pipeline Example
```
INIT (fresh session)
  ↓ [Router: INIT → FETCH]
FETCH (simulates document fetch)
  ↓ [Router: FETCH → VALIDATE]
VALIDATE (LLM validates raw_data)
  ↓
  ├→ [If valid] VALIDATE → ENRICH
  └→ [If invalid] Guardrail overrides → HUMAN_REVIEW
      ↓
      HUMAN_REVIEW (LLM simulates human review)
        ↓
        ├→ [If approved] HUMAN_REVIEW → ENRICH
        └→ [If rejected] ERROR
ENRICH (LLM enriches validated_data)
  ↓ [Router: ENRICH → STORE]
STORE (persists enriched_data)
  ↓ [Router: STORE → COMPLETE]
COMPLETE (terminal state)
```

## Extensibility Points

### For New Pipelines
1. Subclass `StateMachineWorkflow`
2. Define:
   - State enum & transitions
   - Handler map
   - Routing table
   - Session initialization
   - Optional: custom guardrails

### For New Handlers
1. Define handler function: `(state_dict) → state_dict`
2. Add to `HANDLER_MAP`
3. Can call LLM agents, external APIs, etc.

### For New Guardrails
1. Compose checks with `make_guardrail(check1, check2, ...)`
2. Register in `GUARDRAILS` dict
3. Runs before state transition; can override proposed_next

## Data Flow: Session State

```
session_state (Agno persisted dict)
    ├── Control plane:
    │   ├── current_state    : str      (where we are)
    │   ├── proposed_next    : str      (where router wants to go)
    │   ├── retry_count      : int      (failed attempts)
    │   ├── error_message    : str|None (last error)
    │   ├── guardrail_ok     : bool     (last guardrail result)
    │   └── audit_trail      : list[str] (chronological log)
    │
    └── Business plane:
        ├── document_id      : str      (which doc we're processing)
        ├── raw_data         : dict|None (after FETCH)
        ├── validated_data   : dict|None (after VALIDATE)
        └── enriched_data    : dict|None (after ENRICH)

+ pipeline_runs         : list[dict]  (history of all completed runs)
```

## Type Safety

### PipelineState TypedDict
```python
class PipelineState(TypedDict):
    current_state   : str
    proposed_next   : str
    retry_count     : int
    error_message   : str | None
    guardrail_ok    : bool
    audit_trail     : list[str]
    document_id     : str
    raw_data        : dict | None
    validated_data  : dict | None
    enriched_data   : dict | None
```

### State Enum
```python
class State(str, Enum):
    INIT, FETCH, VALIDATE, ENRICH, STORE, COMPLETE, RETRY, ERROR, HUMAN_REVIEW
```

### Handler Signature
```python
Handler: (state_dict: PipelineState) -> PipelineState
```

## Session Persistence

```
build_doc_pipeline(session_id)
    ↓
JsonDb(db_path=".doc_sessions")
    ↓
session_state → JSON file (.doc_sessions/{session_id}.json)
    ↓
[Resume later with same session_id → loads from disk]
```

This enables:
- Multi-run session resume
- Full audit trail recovery
- Disaster recovery
- Testing with known state snapshots
