from typing import Any, Dict, Optional

from src.engine.engine_session_state import EngineSessionState, new_engine_session_state

from .state_transitions import State

# ─────────────────────────────────────────────────────────────────────────────
# ONBOARDING STATE
# ─────────────────────────────────────────────────────────────────────────────

# structures
class OnboardingState(EngineSessionState):
    """New-hire onboarding pipeline state.

    Inherits common control plane and multi-turn fields from EngineSessionState.
    Adds business-specific payload for onboarding (design spec §4).

    Business Payload Fields:
      • new_hire_details: Collected new-hire information (name, role, start
        date, etc.), gathered by the COLLECT state's tool-calling agent
      • welcome_sent / it_provisioned / schedule_sent / hr_notified: guard
        flags — each side-effect handler sets its own flag, independent of
        expected_events legality, so a resume never double-applies a
        side effect
      • username_prefix: Selected during IT_PROVISIONED
      • hardware_tracking_id: Set once hardware delivery is scheduled
    """

    # ─ Business Payload (onboarding-specific) ─────────────────────────────
    """Collected new-hire details. Set by the COLLECT state's tool-calling agent."""
    new_hire_details: Optional[Dict[str, Any]]

    """True once the welcome message has actually been sent. Guard flag —
    independent of expected_events legality, so a resumed/retried dispatch
    never re-sends it."""
    welcome_sent: bool

    """True once IT provisioning (username selection) has completed."""
    it_provisioned: bool

    """True once the hardware delivery schedule has been sent."""
    schedule_sent: bool

    """True once HR has been notified of onboarding completion."""
    hr_notified: bool

    """Username prefix selected during IT_PROVISIONED."""
    username_prefix: Optional[str]

    """Tracking ID for the hardware delivery, set once scheduled."""
    hardware_tracking_id: Optional[str]


# functions
# ── Constructor ───────────────────────────────────────────────────────────────
def new_onboarding_session_state() -> OnboardingState:
    """Return a fresh OnboardingState ready to start at INIT.

    Returns:
        Fresh OnboardingState with all fields initialized
    """
    engine_session_state = new_engine_session_state()
    onboarding_state: OnboardingState = {
        **engine_session_state,
        "current_state": State.INIT.value,
        "proposed_next": State.COLLECT.value,
        "new_hire_details": None,
        "welcome_sent": False,
        "it_provisioned": False,
        "schedule_sent": False,
        "hr_notified": False,
        "username_prefix": None,
        "hardware_tracking_id": None,
    }
    return onboarding_state
