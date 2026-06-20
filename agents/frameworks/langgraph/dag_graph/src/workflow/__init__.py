"""LangGraph workflow package: framework-specific integration and business logic.

The workflow layer combines:
  • Generic engine patterns (src.engine) for reusable state machine logic
  • LangGraph StateGraph for execution
  • Domain-specific handlers, guardrails, and routing for document processing
"""

# Graph building and nodes (LangGraph-specific)
from src.workflow.graph import (
    DocumentPipelineGraph,
    build_graph,
    guardrail_node,
    guardrail_router,
    router_node,
    HAPPY_PATH,
    HANDLER_MAP,
    TERMINAL_STATES,
)

# Workflow entrypoint
from src.workflow.workflow import (
    run_pipeline,
    run_pipeline_with_checkpoint,
)

# State machine (domain-specific core logic)
from src.workflow.state_machine import (
    State,
    PipelineState,
    GuardrailResult,
    ALLOWED_TRANSITIONS,
    is_transition_allowed,
)

# Handlers (business logic for each state)
from src.workflow.handlers import (
    handle_fetch,
    handle_validate,
    handle_enrich,
    handle_store,
    handle_retry,
    handle_human_review,
    handle_complete,
    handle_error,
)

# Guardrails (validation checks)
from src.workflow.guardrails import (
    GUARDRAILS,
    check_transition_allowed,
    check_retry_budget,
    check_raw_data_present,
    check_validated_data_present,
    check_enriched_data_present,
)

__all__ = [
    # Graph building
    "DocumentPipelineGraph",
    "build_graph",
    "router_node",
    "guardrail_node",
    "guardrail_router",
    "HAPPY_PATH",
    "HANDLER_MAP",
    "TERMINAL_STATES",
    # Workflow entrypoint
    "run_pipeline",
    "run_pipeline_with_checkpoint",
    # State machine
    "State",
    "PipelineState",
    "GuardrailResult",
    "ALLOWED_TRANSITIONS",
    "is_transition_allowed",
    # Handlers
    "handle_fetch",
    "handle_validate",
    "handle_enrich",
    "handle_store",
    "handle_retry",
    "handle_human_review",
    "handle_complete",
    "handle_error",
    # Guardrails
    "GUARDRAILS",
    "check_transition_allowed",
    "check_retry_budget",
    "check_raw_data_present",
    "check_validated_data_present",
    "check_enriched_data_present",
]
