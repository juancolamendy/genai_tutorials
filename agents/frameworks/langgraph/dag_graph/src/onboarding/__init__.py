"""New-hire onboarding package: mirrors docprocessing's structure, built to
exercise the engine-event-extension design (aemit_event, arun_to_completion,
wait_kind/expected_events).

Handlers and graph wiring land in a later phase; this package currently
exposes the state machine skeleton (state_transitions, session_state,
guardrails, chains).
"""

from src.onboarding.chains import (
    UsernameSelection,
    collect_agent,
    submit_new_hire,
    username_chain,
)
from src.onboarding.guardrails import (
    check_not_timeout_escalation,
    check_transition_allowed,
    guardrails,
)
from src.onboarding.session_state import OnboardingState, new_onboarding_session_state
from src.onboarding.state_transitions import (
    State,
    allowed_transitions,
    happy_path,
    is_transition_allowed,
    terminal_states,
)

__all__ = [
    # State machine
    "State",
    "OnboardingState",
    "new_onboarding_session_state",
    "happy_path",
    "terminal_states",
    "allowed_transitions",
    "is_transition_allowed",
    # Guardrails
    "guardrails",
    "check_transition_allowed",
    "check_not_timeout_escalation",
    # LLM chains
    "UsernameSelection",
    "submit_new_hire",
    "collect_agent",
    "username_chain",
]
