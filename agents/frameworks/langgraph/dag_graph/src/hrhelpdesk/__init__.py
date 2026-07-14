"""HR helpdesk hub + sticky-topic chatbot package."""

from src.hrhelpdesk.chains import (
    EscapeOutput,
    RouterOutput,
    RouterTopic,
    booking_agent,
    escape_chain,
    escalate_agent,
    faq_agent,
    run_escape,
    run_router,
    route_chain,
)
from src.hrhelpdesk.graph import Graph, build_graph
from src.hrhelpdesk.guardrails import guardrails
from src.hrhelpdesk.services import (
    check_desk_availability,
    confirm_booking,
    create_ticket,
    reset_providers,
    retrieve_policy,
)
from src.hrhelpdesk.session_state import HelpdeskState, new_helpdesk_session_state
from src.hrhelpdesk.state_transitions import (
    State,
    allowed_transitions,
    happy_path,
    is_transition_allowed,
    terminal_states,
)

__all__ = [
    "State",
    "HelpdeskState",
    "new_helpdesk_session_state",
    "happy_path",
    "terminal_states",
    "allowed_transitions",
    "is_transition_allowed",
    "guardrails",
    "Graph",
    "build_graph",
    "RouterTopic",
    "RouterOutput",
    "EscapeOutput",
    "route_chain",
    "escape_chain",
    "run_router",
    "run_escape",
    "faq_agent",
    "escalate_agent",
    "booking_agent",
    "retrieve_policy",
    "create_ticket",
    "check_desk_availability",
    "confirm_booking",
    "reset_providers",
]
