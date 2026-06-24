# Creating a New Pipeline: Complete Guide

This guide shows how to create a new multi-turn workflow by reusing the engine layer components. We'll use **Invoice Processing** as an example.

---

## Architecture Overview

```
Reusable Engine Layer (one-time setup)
├── StateMachineWorkflow      (base class for all workflows)
├── EngineState               (control plane fields)
├── BaseSemanticRouter        (LLM router interface)
├── @handler decorator        (metadata for handlers)
├── Input validation          (validate_turn_input, escape_for_llm)
└── Multi-turn helpers        (_prepare_turn_metadata, _trim_history, _build_turn_response)

Your Pipeline Layer (create these)
├── InvoiceState              (inherit from EngineState, add business fields)
├── State enum                (FETCH, VALIDATE, ENRICH, etc.)
├── Handlers                  (handle_fetch, handle_validate, etc.)
├── InvoiceRouter             (inherit from BaseSemanticRouter)
└── InvoiceWorkflow           (inherit from StateMachineWorkflow)
```

---

## Step-by-Step: Create Invoice Processing Pipeline

### Step 1: Define Business State

Create `src/invoice_pipeline/pipeline_state.py`:

```python
"""
invoice_pipeline/pipeline_state.py
Business state for invoice processing workflow.
"""

from typing import Any, Optional, TypedDict

from engine.engine_state import EngineState, init_engine_state


class InvoiceState(EngineState):
    """
    Full invoice processing state = control plane (EngineState) + business payload.
    
    Adds invoice-specific fields.
    """

    # Business payload
    invoice_id: str
    vendor_name: Optional[str]
    invoice_amount: Optional[float]
    invoice_date: Optional[str]
    raw_invoice: Optional[dict[str, Any]]       # set by FETCH
    validated_invoice: Optional[dict[str, Any]]  # set by VALIDATE
    approved_invoice: Optional[dict[str, Any]]   # set by APPROVE


def new_invoice_state(invoice_id: str) -> InvoiceState:
    """Return a fresh InvoiceState ready to start at INIT."""
    base = init_engine_state()
    return {
        **base,
        "current_state": "init",
        "proposed_next": "fetch",
        "audit_trail": [f"init  invoice_id={invoice_id}"],
        "invoice_id": invoice_id,
        "vendor_name": None,
        "invoice_amount": None,
        "invoice_date": None,
        "raw_invoice": None,
        "validated_invoice": None,
        "approved_invoice": None,
    }


def audit(state: InvoiceState, entry: str) -> InvoiceState:
    """Return state with `entry` appended to audit_trail."""
    return {**state, "audit_trail": state["audit_trail"] + [entry]}
```

---

### Step 2: Define State Machine

Create `src/invoice_pipeline/state_machine.py`:

```python
"""
invoice_pipeline/state_machine.py
State machine definition for invoice processing.
"""

from enum import Enum


class State(str, Enum):
    """Invoice processing states."""
    
    INIT = "init"
    FETCH = "fetch"          # Retrieve invoice from source
    VALIDATE = "validate"    # Validate schema, amounts, dates
    EXTRACT = "extract"      # Extract key entities (vendor, amount, date)
    ENRICH = "enrich"        # Add business context (vendor history, tax data)
    APPROVE = "approve"      # Get approval (human or automated)
    STORE = "store"          # Persist to database
    COMPLETE = "complete"    # Terminal: success
    RETRY = "retry"          # Transient failure, retry
    REJECT = "reject"        # Terminal: rejected by business rules
    ERROR = "error"           # Terminal: unhandled error


TERMINAL_STATES = {
    State.COMPLETE,
    State.REJECT,
    State.ERROR,
}
```

---

### Step 3: Define Handlers

Create `src/invoice_pipeline/handlers.py`:

```python
"""
invoice_pipeline/handlers.py
Handler functions for each invoice processing state.
"""

import logging
from typing import Any

from engine.handler_registry import handler
from .pipeline_state import InvoiceState, audit
from .state_machine import State
from .agents import VALIDATE_AGENT, ENRICH_AGENT, APPROVE_AGENT

log = logging.getLogger(__name__)


@handler(state="fetch", waits_for_input=False, description="Fetch invoice from source system")
def handle_fetch(state: InvoiceState) -> InvoiceState:
    """Retrieve invoice from external source (API, email, file)."""
    invoice_id = state["invoice_id"]
    log.info("[FETCH] invoice_id=%s", invoice_id)

    try:
        # Simulate fetching from invoice system
        raw_invoice = {
            "id": invoice_id,
            "vendor": "ACME Corp",
            "amount": 1500.00,
            "date": "2026-06-20",
            "items": [
                {"description": "Software license", "qty": 1, "unit_price": 1500.00}
            ]
        }
        
        return audit(
            {**state, "current_state": State.FETCH.value, "raw_invoice": raw_invoice},
            f"fetch OK  vendor={raw_invoice['vendor']} amount=${raw_invoice['amount']}"
        )

    except Exception as exc:
        log.error("[FETCH] exception: %s", exc)
        return audit(
            {**state, "current_state": State.ERROR.value, "error_message": str(exc)},
            f"fetch EXCEPTION: {exc}"
        )


@handler(state="validate", waits_for_input=False, description="Validate invoice schema and amounts")
def handle_validate(state: InvoiceState) -> InvoiceState:
    """LLM validates invoice format, required fields, amount consistency."""
    log.info("[VALIDATE] invoice_id=%s", state["invoice_id"])

    try:
        raw = state["raw_invoice"] or {}
        result = VALIDATE_AGENT.run(f"<invoice>{raw}</invoice>").content

        if result.is_valid:
            validated = {**result.cleaned_data, "_validated": True}
            return audit(
                {**state, "current_state": State.VALIDATE.value, "validated_invoice": validated},
                f"validate OK  amount=${result.amount}"
            )

        log.warning("[VALIDATE] FAILED issues=%s", result.issues)
        return audit(
            {**state, "current_state": State.VALIDATE.value, "validated_invoice": None},
            f"validate FAILED  issues={result.issues}"
        )

    except Exception as exc:
        log.error("[VALIDATE] exception: %s", exc)
        return audit(
            {**state, "current_state": State.ERROR.value, "error_message": str(exc)},
            f"validate EXCEPTION: {exc}"
        )


@handler(state="extract", waits_for_input=False, description="Extract key entities from invoice")
def handle_extract(state: InvoiceState) -> InvoiceState:
    """Extract vendor name, amount, date, line items."""
    log.info("[EXTRACT] invoice_id=%s", state["invoice_id"])

    try:
        validated = state.get("validated_invoice") or state.get("raw_invoice") or {}
        
        return audit(
            {
                **state,
                "current_state": State.EXTRACT.value,
                "vendor_name": validated.get("vendor"),
                "invoice_amount": validated.get("amount"),
                "invoice_date": validated.get("date")
            },
            f"extract OK  vendor={validated.get('vendor')}"
        )

    except Exception as exc:
        log.error("[EXTRACT] exception: %s", exc)
        return audit(
            {**state, "current_state": State.ERROR.value, "error_message": str(exc)},
            f"extract EXCEPTION: {exc}"
        )


@handler(state="enrich", waits_for_input=False, description="Enrich with vendor history and tax context")
def handle_enrich(state: InvoiceState) -> InvoiceState:
    """Add business context: vendor history, tax treatment, cost center."""
    log.info("[ENRICH] invoice_id=%s", state["invoice_id"])

    try:
        validated = state.get("validated_invoice") or {}
        
        # Simulate LLM enrichment
        result = ENRICH_AGENT.run(f"<invoice>{validated}</invoice>").content

        enriched = {
            **validated,
            "vendor_history": result.vendor_history,
            "tax_treatment": result.tax_treatment,
            "cost_center": result.cost_center,
            "_enriched": True
        }

        return audit(
            {**state, "current_state": State.ENRICH.value, "approved_invoice": enriched},
            f"enrich OK  cost_center={result.cost_center}"
        )

    except Exception as exc:
        log.error("[ENRICH] exception: %s", exc)
        return audit(
            {**state, "current_state": State.ERROR.value, "error_message": str(exc)},
            f"enrich EXCEPTION: {exc}"
        )


@handler(state="approve", waits_for_input=True, description="Route to approval (pauses workflow)")
def handle_approve(state: InvoiceState) -> InvoiceState:
    """
    Route to approval queue (human or automated).
    Pauses workflow; next turn will resume based on approval result.
    """
    log.warning("[APPROVE] 🔍 invoice_id=%s  amount=${:.2f}  routing to approval",
                state["invoice_id"], state.get("invoice_amount", 0))

    return audit(
        {**state, "current_state": State.APPROVE.value},
        "approval pending (waiting for decision)"
    )


@handler(state="store", waits_for_input=False, description="Persist approved invoice to database")
def handle_store(state: InvoiceState) -> InvoiceState:
    """Store approved invoice in the invoice management system."""
    log.info("[STORE] invoice_id=%s", state["invoice_id"])

    try:
        # Simulate storing to database
        record_id = f"INV-{state['invoice_id']}-STORED"
        
        return audit(
            {**state, "current_state": State.STORE.value},
            f"store OK  record_id={record_id}"
        )

    except Exception as exc:
        log.error("[STORE] exception: %s", exc)
        return audit(
            {**state, "current_state": State.ERROR.value, "error_message": str(exc)},
            f"store EXCEPTION: {exc}"
        )


@handler(state="complete", waits_for_input=False, description="Terminal success")
def handle_complete(state: InvoiceState) -> InvoiceState:
    """Terminal success: invoice processed and stored."""
    log.info("[COMPLETE] ✅ invoice_id=%s", state["invoice_id"])
    return audit({**state, "current_state": State.COMPLETE.value}, "COMPLETE ✅")


@handler(state="retry", waits_for_input=False, description="Retry failed operation")
def handle_retry(state: InvoiceState) -> InvoiceState:
    """Increment retry counter and go back to fetch."""
    new_count = state["retry_count"] + 1
    log.info("[RETRY] attempt #%d", new_count)
    return audit(
        {**state, "current_state": State.RETRY.value, "retry_count": new_count, "raw_invoice": None},
        f"retry #{new_count}"
    )


@handler(state="reject", waits_for_input=False, description="Terminal rejection")
def handle_reject(state: InvoiceState) -> InvoiceState:
    """Terminal rejection: business rules or approval failed."""
    reason = state.get("error_message", "approval rejected")
    log.error("[REJECT] 🔴 invoice_id=%s  reason=%s", state["invoice_id"], reason)
    return audit({**state, "current_state": State.REJECT.value}, f"REJECT 🔴  reason={reason}")


@handler(state="error", waits_for_input=False, description="Terminal error")
def handle_error(state: InvoiceState) -> InvoiceState:
    """Terminal error state."""
    reason = state.get("error_message", "unknown error")
    log.error("[ERROR] 🔴 invoice_id=%s  reason=%s", state["invoice_id"], reason)
    return audit({**state, "current_state": State.ERROR.value}, f"ERROR 🔴  reason={reason}")


# HANDLER_MAP: Maps state → handler function
HANDLER_MAP = {
    State.FETCH: handle_fetch,
    State.VALIDATE: handle_validate,
    State.EXTRACT: handle_extract,
    State.ENRICH: handle_enrich,
    State.APPROVE: handle_approve,
    State.STORE: handle_store,
    State.COMPLETE: handle_complete,
    State.RETRY: handle_retry,
    State.REJECT: handle_reject,
    State.ERROR: handle_error,
}
```

---

### Step 4: Define Domain Router

Create `src/invoice_pipeline/router.py`:

```python
"""
invoice_pipeline/router.py
Domain-specific semantic router for invoice processing.
"""

import json
import logging
from typing import Optional

from engine.router import BaseSemanticRouter, RouterDecision

log = logging.getLogger(__name__)


class InvoiceRouter(BaseSemanticRouter):
    """
    LLM router for invoice processing.
    
    Classifies approval decisions, detects fraud/anomalies,
    recommends next state based on invoice context.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model

    def route(self,
              current_state: str,
              turn_input: str,
              history: list,
              allowed_states: list,
              timeout_sec: float = 10.0) -> RouterDecision:
        """
        Route based on:
        - Current state (fetch, validate, enrich, approve)
        - User input (approval decision, questions, flags)
        - Invoice context (amount, vendor, history)
        - Allowed transitions
        """
        try:
            # Build prompt
            history_text = "\n".join([f"[{t.get('role')}]: {t.get('content')}" for t in history[-5:]])
            allowed_str = ", ".join(allowed_states)

            prompt = f"""
You are an invoice processing router. Decide the next state.

CURRENT STATE: {current_state}
ALLOWED NEXT STATES: {allowed_str}
USER INPUT: {repr(turn_input)}
HISTORY:
{history_text or "(no history)"}

Classify the user's input:
1. Is this an approval/rejection decision?
2. Are there concerns (amount too high, vendor unknown, date issue)?
3. What's the next state?

Respond with JSON:
{{
  "proposed_next": "<state>",
  "confidence": 0.95,
  "semantic_intents": ["approve"],
  "semantic_entities": {{"decision": "approved"}},
  "reasoning": "User approved invoice for $1500"
}}
"""

            # Mock LLM call (replace with actual Claude call)
            response_json = {
                "proposed_next": allowed_states[0],
                "confidence": 0.85,
                "semantic_intents": ["approve"],
                "semantic_entities": {},
                "reasoning": "Processing invoice"
            }

            proposed = response_json.get("proposed_next", allowed_states[0])
            if proposed not in allowed_states:
                proposed = allowed_states[0]

            return RouterDecision(
                proposed_next=proposed,
                confidence=float(response_json.get("confidence", 0.5)),
                semantic_entities=response_json.get("semantic_entities", {}),
                semantic_intents=response_json.get("semantic_intents", []),
                reasoning=response_json.get("reasoning")
            )

        except Exception as e:
            log.exception(f"Router error: {e}")
            return RouterDecision(
                proposed_next="error",
                confidence=0.0,
                semantic_entities={},
                semantic_intents=[],
                reasoning=f"Router error: {e}"
            )
```

---

### Step 5: Define Workflow Class

Create `src/invoice_pipeline/workflow.py`:

```python
"""
invoice_pipeline/workflow.py
InvoiceWorkflow — invoice processing state machine.
"""

from typing import Any

from engine.statemachine_workflow import StateMachineWorkflow
from .handlers import HANDLER_MAP
from .pipeline_state import InvoiceState, new_invoice_state
from .router import InvoiceRouter
from .state_machine import State, TERMINAL_STATES


# Routing table: default transitions
_INVOICE_PATH = {
    State.INIT:     State.FETCH,
    State.FETCH:    State.VALIDATE,
    State.VALIDATE: State.EXTRACT,
    State.EXTRACT:  State.ENRICH,
    State.ENRICH:   State.APPROVE,
    State.APPROVE:  State.STORE,
    State.STORE:    State.COMPLETE,
    State.RETRY:    State.FETCH,
}

# Fields to persist to DB
_INVOICE_KEYS = (
    "current_state", "proposed_next", "guardrail_ok",
    "retry_count", "error_message", "audit_trail",
    "invoice_id", "vendor_name", "invoice_amount", "invoice_date",
    "raw_invoice", "validated_invoice", "approved_invoice",
)


class InvoiceWorkflow(StateMachineWorkflow):
    """Invoice processing workflow with semantic routing."""

    _STATE_KEYS = _INVOICE_KEYS
    _STATE_ENUM = State
    _TERMINAL_STATES = TERMINAL_STATES
    HANDLER_MAP = HANDLER_MAP

    def __post_init__(self) -> None:
        """Initialize base class and semantic router."""
        super().__post_init__()
        self.router = InvoiceRouter()

    def _init_session_defaults(self) -> None:
        """Initialize session with defaults."""
        if self.session_state is None:
            self.session_state = {}
        self.session_state.setdefault("current_state", State.INIT.value)
        self.session_state.setdefault("proposed_next", State.FETCH.value)
        self.session_state.setdefault("turn_number", 0)
        self.session_state.setdefault("conversation_history", [])
        self.session_state.setdefault("retry_count", 0)
        self.session_state.setdefault("error_message", None)
        self.session_state.setdefault("audit_trail", [])

    def _build_routing_table(self) -> dict[State, State]:
        """Return the routing table."""
        return _INVOICE_PATH

    def _get_current_state(self, session_state: dict[str, Any]) -> State:
        """Extract current state."""
        return State(session_state.get("current_state", State.INIT.value))

    def _get_proposed_state(self, session_state: dict[str, Any]) -> State:
        """Extract proposed next state."""
        return State(session_state.get("proposed_next", State.FETCH.value))

    def _run_guardrail(self, state_dict: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        """Run guardrails (optional: implement business rule validation)."""
        from dataclasses import dataclass

        @dataclass
        class Result:
            passed: bool = True
            reason: str = ""

        # Example: Block high-value invoices from auto-approval
        if state_dict.get("invoice_amount", 0) > 5000:
            return state_dict, Result(passed=False, reason="High-value invoice requires manual review")

        return state_dict, Result()

    def _new_session_state(self, entity_id: str) -> dict[str, Any]:
        """Initialize fresh session state for a new invoice."""
        return new_invoice_state(entity_id)

    def _build_response(self, entity_id: str) -> InvoiceState:
        """Build InvoiceState from current session_state."""
        final = InvoiceState(
            current_state=self.session_state["current_state"],
            proposed_next=self.session_state["proposed_next"],
            retry_count=self.session_state["retry_count"],
            error_message=self.session_state.get("error_message"),
            guardrail_ok=self.session_state.get("guardrail_ok", True),
            audit_trail=self.session_state["audit_trail"],
            invoice_id=self.session_state["invoice_id"],
            vendor_name=self.session_state.get("vendor_name"),
            invoice_amount=self.session_state.get("invoice_amount"),
            invoice_date=self.session_state.get("invoice_date"),
            raw_invoice=self.session_state.get("raw_invoice"),
            validated_invoice=self.session_state.get("validated_invoice"),
            approved_invoice=self.session_state.get("approved_invoice"),
        )
        print(f"\n✅ Invoice {entity_id} processed to state: {final['current_state']}")
        return final
```

---

### Step 6: (Optional) Agents

Create `src/invoice_pipeline/agents.py` with LLM agents:

```python
"""
invoice_pipeline/agents.py
LLM agents for invoice processing tasks.
"""

from agno.agent import Agent
from agno.models.anthropic import Claude


# Agent 1: Validates invoice format and amounts
VALIDATE_AGENT = Agent(
    model=Claude(id="claude-haiku-4-5-20251001"),
    instructions="""
    Validate the invoice:
    - Check required fields: vendor, amount, date
    - Verify amount is a positive number
    - Check date format
    - Return: {is_valid, cleaned_data, amount, issues}
    """
)

# Agent 2: Enriches invoice with business context
ENRICH_AGENT = Agent(
    model=Claude(id="claude-haiku-4-5-20251001"),
    instructions="""
    Enrich the invoice:
    - Determine tax treatment (taxable, tax-exempt, etc.)
    - Look up vendor history (first time? repeat vendor?)
    - Suggest cost center
    - Return: {vendor_history, tax_treatment, cost_center}
    """
)

# Agent 3: Makes approval decisions
APPROVE_AGENT = Agent(
    model=Claude(id="claude-haiku-4-5-20251001"),
    instructions="""
    Review and approve the invoice:
    - Check for red flags (unusual vendor, amount spike, etc.)
    - Make approval recommendation
    - Return: {approved, confidence, recommendation}
    """
)
```

---

## Step 7: Create Handler Stubs (Optional)

For quick prototyping, create minimal handlers without LLM calls. The handlers above show the pattern.

---

## Usage Examples

### One-Turn Processing (Synchronous)

```python
from src.invoice_pipeline.workflow import InvoiceWorkflow

wf = InvoiceWorkflow(name="InvoiceProcessor")

# Process a single invoice end-to-end
result = wf.process(invoice_id="INV-2026-001")

print(f"Status: {result['current_state']}")
print(f"Amount: ${result['invoice_amount']}")
print(f"Vendor: {result['vendor_name']}")
```

### Multi-Turn Processing (Conversation)

```python
from agno.db.sqlite import SqliteDb
from src.invoice_pipeline.workflow import InvoiceWorkflow

db = SqliteDb(table_name="invoices", db_file="tmp/invoices.db")
wf = InvoiceWorkflow(name="InvoiceProcessor", db=db)

# Turn 1: Start processing
response1 = wf.process_turn(
    user_id="user_1",
    session_id="invoice_session_001",
    turn_input="Process invoice INV-2026-001"
)
print(f"Turn 1: {response1['current_state']}")
print(f"Entities: {response1['semantic_context']['entities']}")

# Turn 2: User provides approval (workflow paused at APPROVE)
response2 = wf.process_turn(
    user_id="user_1",
    session_id="invoice_session_001",
    turn_input="Approved. Vendor is legitimate, amount is reasonable."
)
print(f"Turn 2: {response2['current_state']}")

# Turn 3: Continue processing
response3 = wf.process_turn(
    user_id="user_1",
    session_id="invoice_session_001",
    turn_input="Store the invoice"
)
print(f"Turn 3: {response3['current_state']}")  # COMPLETE
```

---

## Key Patterns to Follow

| Pattern | Implementation |
|---------|-----------------|
| **State Definition** | Create State enum with all states |
| **Business State** | Create TypedDict that inherits from EngineState |
| **Handlers** | Decorate with @handler, implement try/catch → ERROR |
| **Router** | Inherit from BaseSemanticRouter, implement route() |
| **Workflow** | Inherit from StateMachineWorkflow, implement 4 hooks |
| **Routing Table** | Map State → State transitions |
| **Session Keys** | Define _STATE_KEYS tuple for DB persistence |

---

## Reusable Components Checklist

✅ **From Engine Layer (use as-is):**
- StateMachineWorkflow (base class)
- EngineState (control plane TypedDict)
- BaseSemanticRouter (router interface)
- @handler decorator
- Input validation (validate_turn_input, escape_for_llm)
- Multi-turn helpers (_prepare_turn_metadata, _trim_history, _build_turn_response)

✅ **Create for Your Pipeline:**
- State enum (domain-specific states)
- Business TypedDict (inherit from EngineState)
- Handlers (one per state, decorated with @handler)
- Domain Router (inherit from BaseSemanticRouter)
- Workflow class (inherit from StateMachineWorkflow)

---

## File Structure for New Pipeline

```
src/invoice_pipeline/
├── __init__.py
├── pipeline_state.py      (InvoiceState, new_invoice_state)
├── state_machine.py       (State enum, TERMINAL_STATES)
├── handlers.py            (@handler decorated handlers)
├── router.py              (InvoiceRouter)
├── workflow.py            (InvoiceWorkflow)
├── agents.py              (LLM agents for validation, enrichment)
└── guardrails.py          (optional: business rule validation)

tests/invoice_pipeline/
├── __init__.py
├── test_invoice_state.py
├── test_handlers.py
├── test_router.py
└── test_workflow_integration.py
```

---

## Summary

To create a new pipeline, you only need to define **5 things**:

1. **State enum** — What states exist?
2. **Business state TypedDict** — What business data do you track?
3. **Handlers** — What happens in each state? (reuses error handling pattern)
4. **Router** — How do you decide next state? (reuses semantic routing interface)
5. **Workflow class** — How do you tie it together? (reuses StateMachineWorkflow)

Everything else (routing logic, multi-turn support, error handling, input validation) is already built and reused! 🎯
