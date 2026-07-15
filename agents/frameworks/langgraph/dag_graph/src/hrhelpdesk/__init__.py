"""HR helpdesk hub + sticky-topic chatbot package."""

from src.engine.escape_checker import (
    DefaultEscapeChecker,
    EscapeDecision,
    EscapeOutput,
)
from src.hrhelpdesk.chains import (
    BookingDecision,
    EscalateDecision,
    FaqAnswer,
    RouterOutput,
    RouterTopic,
    booking_chain,
    escalate_chain,
    faq_chain,
)
from src.hrhelpdesk.graph import Graph, build_graph
from src.hrhelpdesk.guardrails import guardrails
from src.hrhelpdesk.handlers import run_escape
from src.hrhelpdesk.router import HelpdeskSemanticRouter
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
    "FaqAnswer",
    "EscalateDecision",
    "BookingDecision",
    "EscapeDecision",
    "EscapeOutput",
    "DefaultEscapeChecker",
    "run_escape",
    "HelpdeskSemanticRouter",
    "faq_chain",
    "escalate_chain",
    "booking_chain",
    "retrieve_policy",
    "create_ticket",
    "check_desk_availability",
    "confirm_booking",
    "reset_providers",
]
