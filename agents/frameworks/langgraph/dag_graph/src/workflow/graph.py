"""LangGraph StateGraph assembly and graph execution engine."""

from langgraph.graph import StateGraph, END
from src.workflow.state_machine import State, PipelineState
from src.engine.router import router
from src.workflow.guardrails import GUARDRAILS
from src.workflow.handlers import (
    handle_fetch_template,
    handle_validate_template,
    handle_enrich_template,
    handle_store_template,
    handle_retry_template,
    handle_human_review_template,
    handle_complete_template,
    handle_error_template,
)

HANDLER_MAP = {
    State.FETCH: handle_fetch_template,
    State.VALIDATE: handle_validate_template,
    State.ENRICH: handle_enrich_template,
    State.STORE: handle_store_template,
    State.RETRY: handle_retry_template,
    State.HUMAN_REVIEW: handle_human_review_template,
    State.COMPLETE: handle_complete_template,
    State.ERROR: handle_error_template,
}

TERMINAL_STATES = {State.COMPLETE, State.ERROR}


def guardrail_node(state: PipelineState) -> PipelineState:
    """Guardrail node: validate transition, apply fallback if needed.

    Args:
        state: PipelineState with proposed_next set by router

    Returns:
        Updated state with guardrail result applied
    """
    current = State(state["current_state"])

    # Skip guardrail for INIT state (always proceeds)
    if current == State.INIT:
        state["audit_trail"].append(f"guardrail SKIP (init) → {state['proposed_next']}")
        state["fallback_depth"] = 0
        return state

    # Run guardrail checks
    guardrail_fn = GUARDRAILS.get(current)
    if not guardrail_fn:
        return state

    result = guardrail_fn(state)

    if not result.passed:
        # Guardrail failed: apply fallback
        fallback_state = result.fallback.value if result.fallback else "error"
        state = {
            **state,
            "proposed_next": fallback_state,
            "error_message": result.reason,
            "audit_trail": state["audit_trail"]
            + [f"guardrail FAIL → {fallback_state} ({result.reason})"],
            "fallback_depth": state.get("fallback_depth", 0) + 1,
        }
    else:
        # Guardrail passed: reset fallback_depth
        state = {
            **state,
            "audit_trail": state["audit_trail"]
            + [f"guardrail PASS → {state['proposed_next']}"],
            "fallback_depth": 0,
        }

    # Cap audit trail at 1000 entries
    if len(state["audit_trail"]) > 1000:
        state["audit_trail"] = state["audit_trail"][-1000:]

    return state


def guardrail_router(state: PipelineState) -> str:
    """Conditional edge router: select handler based on proposed_next.

    Args:
        state: PipelineState with proposed_next set

    Returns:
        Handler node name (state value)
    """
    return state["proposed_next"]


def build_graph() -> StateGraph:
    """Build the LangGraph StateGraph.

    Graph structure:
    - Entry: router node
    - router → guardrail → (conditional) handler nodes
    - Non-terminal handlers → router (loop back)
    - Terminal handlers → END

    Returns:
        Compiled StateGraph ready for invocation
    """
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("router", router)
    workflow.add_node("guardrail", guardrail_node)

    for state_enum, handler in HANDLER_MAP.items():
        workflow.add_node(state_enum.value, handler)

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    workflow.add_edge("router", "guardrail")

    # Conditional routing from guardrail to handlers
    workflow.add_conditional_edges(
        "guardrail",
        guardrail_router,
        {state.value: state.value for state in HANDLER_MAP.keys()},
    )

    # Loop-back edges: non-terminal handlers → router
    for state in [
        State.FETCH,
        State.VALIDATE,
        State.ENRICH,
        State.STORE,
        State.RETRY,
        State.HUMAN_REVIEW,
    ]:
        workflow.add_edge(state.value, "router")

    # Terminal edges
    workflow.add_edge(State.COMPLETE.value, END)
    workflow.add_edge(State.ERROR.value, END)

    return workflow.compile()
