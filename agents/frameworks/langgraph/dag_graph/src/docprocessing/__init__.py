"""Document processing package: framework-specific integration and business logic.

The docprocessing layer combines:
  • Generic engine patterns (src.engine) for reusable state machine logic
  • LangGraph StateGraph for execution
  • Domain-specific handlers, guardrails, and routing for document processing
"""

# Graph building and nodes (LangGraph-specific)
# LLM chains (domain-specific)
from src.docprocessing.chains import (
    EnrichmentResult,
    ReviewDecision,
    ValidationResult,
    enrich_chain,
    review_chain,
    validate_chain,
)
from src.docprocessing.graph import (
    Graph,
    build_graph,
)

# Guardrails (validation checks)
from src.docprocessing.guardrails import (
    check_enriched_data_present,
    check_raw_data_present,
    check_retry_budget,
    check_transition_allowed,
    check_validated_data_present,
    guardrails,
)

# Handlers (business logic for each state)
from src.docprocessing.handlers import (
    handle_complete,
    handle_enrich,
    handle_error,
    handle_fetch,
    handle_human_review,
    handle_retry,
    handle_store,
    handle_upload_documents,
    handle_validate,
    handler_map,
)

# Pipeline state and guardrail types
from src.docprocessing.session_state import SessionState, new_session_state

# State machine (domain-specific core logic)
from src.docprocessing.state_transitions import (
    State,
    allowed_transitions,
    happy_path,
    is_transition_allowed,
    terminal_states,
)
from src.engine.guardrail import GuardrailResult

__all__ = [
    # Graph building
    "Graph",
    "build_graph",
    "happy_path",
    "terminal_states",
    # State machine
    "State",
    "SessionState",
    "new_session_state",
    "GuardrailResult",
    "allowed_transitions",
    "is_transition_allowed",
    # Handlers
    "handle_fetch",
    "handle_upload_documents",
    "handle_validate",
    "handle_enrich",
    "handle_store",
    "handle_retry",
    "handle_human_review",
    "handle_complete",
    "handle_error",
    "handler_map",
    # Guardrails
    "guardrails",
    "check_transition_allowed",
    "check_retry_budget",
    "check_raw_data_present",
    "check_validated_data_present",
    "check_enriched_data_present",
    # LLM chains
    "validate_chain",
    "enrich_chain",
    "review_chain",
    "ValidationResult",
    "EnrichmentResult",
    "ReviewDecision",
]
