"""LangGraph workflow package: framework-specific integration and business logic.

The workflow layer combines:
  • Generic engine patterns (src.engine) for reusable state machine logic
  • LangGraph StateGraph for execution
  • Domain-specific handlers, guardrails, and routing for document processing
"""

# Graph building and nodes (LangGraph-specific)
from src.engine.guardrail import GuardrailResult

# LLM chains (domain-specific)
from src.workflow.chains import (
    ENRICH_CHAIN,
    REVIEW_CHAIN,
    VALIDATE_CHAIN,
    EnrichmentResult,
    ReviewDecision,
    ValidationResult,
)
from src.workflow.graph import (
    HANDLER_MAP,
    HAPPY_PATH,
    TERMINAL_STATES,
    DocumentPipelineGraph,
    build_graph,
)

# Guardrails (validation checks)
from src.workflow.guardrails import (
    GUARDRAILS,
    check_enriched_data_present,
    check_raw_data_present,
    check_retry_budget,
    check_transition_allowed,
    check_validated_data_present,
)

# Handlers (business logic for each state)
from src.workflow.handlers import (
    handle_complete,
    handle_enrich,
    handle_error,
    handle_fetch,
    handle_human_review,
    handle_retry,
    handle_store,
    handle_validate,
)

# Pipeline state and guardrail types
from src.workflow.pipeline_state import PipelineState

# State machine (domain-specific core logic)
from src.workflow.state_machine import (
    ALLOWED_TRANSITIONS,
    State,
    is_transition_allowed,
)

# Workflow entrypoint
from src.workflow.workflow import run_pipeline

__all__ = [
    # Graph building
    "DocumentPipelineGraph",
    "build_graph",
    "HAPPY_PATH",
    "HANDLER_MAP",
    "TERMINAL_STATES",
    # Workflow entrypoint
    "run_pipeline",
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
    # LLM chains
    "VALIDATE_CHAIN",
    "ENRICH_CHAIN",
    "REVIEW_CHAIN",
    "ValidationResult",
    "EnrichmentResult",
    "ReviewDecision",
]
