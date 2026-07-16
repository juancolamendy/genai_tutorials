"""Tests for the onboarding domain's state machine skeleton (design spec
§3, §4, §8) — state enum/transitions, session state, and guardrails.
Mirrors docprocessing's own test coverage pattern: chains.py (LLM
integration) has no dedicated unit tests in this codebase, matched here.
"""

from src.onboarding.guardrails import check_not_timeout_escalation, guardrails
from src.onboarding.session_state import OnboardingState, new_onboarding_session_state
from src.onboarding.state_transitions import (
    State,
    allowed_transitions,
    happy_path,
    is_transition_allowed,
    terminal_states,
)


def test_state_enum_has_all_states_from_chart():
    expected = {
        "init",
        "collect",
        "welcome_sent",
        "await_documents_signed",
        "it_provisioned",
        "await_hardware_delivered",
        "schedule_sent",
        "complete",
        "escalated",
        "error",
    }
    actual = {s.value for s in State}
    assert actual == expected


def test_terminal_states_are_complete_escalated_and_error():
    assert terminal_states == {State.COMPLETE, State.ESCALATED, State.ERROR}


def test_happy_path_matches_state_chart():
    assert happy_path[State.INIT] == State.COLLECT
    assert happy_path[State.COLLECT] == State.WELCOME_SENT
    assert happy_path[State.WELCOME_SENT] == State.AWAIT_DOCUMENTS_SIGNED
    assert happy_path[State.AWAIT_DOCUMENTS_SIGNED] == State.IT_PROVISIONED
    assert happy_path[State.IT_PROVISIONED] == State.AWAIT_HARDWARE_DELIVERED
    assert happy_path[State.AWAIT_HARDWARE_DELIVERED] == State.SCHEDULE_SENT
    assert happy_path[State.SCHEDULE_SENT] == State.COMPLETE
    # Terminal/park-only states propose nothing forward
    assert State.COMPLETE not in happy_path
    assert State.ESCALATED not in happy_path


def test_collect_self_loop_is_allowed():
    """collect --> collect: guardrail fallback when details are incomplete
    (§3) — a real transition, not just an aemit_event-level "ignored"."""
    assert is_transition_allowed(State.COLLECT, State.COLLECT)


def test_await_states_can_transition_to_escalated():
    assert is_transition_allowed(State.AWAIT_DOCUMENTS_SIGNED, State.ESCALATED)
    assert is_transition_allowed(State.AWAIT_HARDWARE_DELIVERED, State.ESCALATED)


def test_terminal_states_have_no_outgoing_transitions():
    for terminal in terminal_states:
        assert allowed_transitions.get(terminal, set()) == set()


def test_collect_guardrail_excludes_fallback_depth():
    """Round-1 finding (design spec §8): check_fallback_depth (max_depth=2)
    would kill a normal three-clarifying-turn COLLECT conversation to
    ERROR even though nothing is actually looping — must not be present."""
    from src.engine.guardrail import make_fallback_depth_guardrail

    check_fallback_depth = make_fallback_depth_guardrail(max_depth=2, error_state=State.ERROR)

    state = {
        "current_state": State.COLLECT.value,
        "proposed_next": State.COLLECT.value,
        "fallback_depth": 5,  # would trip check_fallback_depth if it were present
    }
    result = guardrails[State.COLLECT](state)
    assert result.passed is True

    # Sanity: prove check_fallback_depth really would fail at this depth,
    # so the above assertion is meaningful and not vacuously true.
    assert check_fallback_depth(state).passed is False


def test_it_provisioned_guardrail_diverts_on_timeout_escalation():
    state = {
        "current_state": State.AWAIT_DOCUMENTS_SIGNED.value,
        "proposed_next": State.IT_PROVISIONED.value,
        "event_type": "timeout_escalation",
        "fallback_depth": 0,
    }
    result = guardrails[State.IT_PROVISIONED](state)
    assert result.passed is False
    assert result.fallback == State.ESCALATED


def test_schedule_sent_guardrail_diverts_on_timeout_escalation():
    """The second AWAIT state's happy-path target needs the identical
    timeout-diversion guardrail — easy to forget since only IT_PROVISIONED
    is shown in the design's worked example (§8)."""
    state = {
        "current_state": State.AWAIT_HARDWARE_DELIVERED.value,
        "proposed_next": State.SCHEDULE_SENT.value,
        "event_type": "timeout_escalation",
        "fallback_depth": 0,
    }
    result = guardrails[State.SCHEDULE_SENT](state)
    assert result.passed is False
    assert result.fallback == State.ESCALATED


def test_it_provisioned_guardrail_passes_on_legal_event():
    state = {
        "current_state": State.AWAIT_DOCUMENTS_SIGNED.value,
        "proposed_next": State.IT_PROVISIONED.value,
        "event_type": "document_signed",
        "fallback_depth": 0,
    }
    result = guardrails[State.IT_PROVISIONED](state)
    assert result.passed is True


def test_check_not_timeout_escalation_standalone():
    passing = check_not_timeout_escalation({"event_type": "document_signed"})
    assert passing.passed is True

    failing = check_not_timeout_escalation({"event_type": "timeout_escalation"})
    assert failing.passed is False
    assert failing.fallback == State.ESCALATED


def test_new_onboarding_session_state_has_business_fields():
    state = new_onboarding_session_state()
    assert state["current_state"] == State.INIT.value
    assert state["new_hire_details"] is None
    assert state["welcome_sent"] is False
    assert state["it_provisioned"] is False
    assert state["schedule_sent"] is False
    assert state["hr_notified"] is False
    assert state["username_prefix"] is None
    assert state["hardware_tracking_id"] is None


def test_onboarding_state_inherits_engine_session_state_fields():
    state = new_onboarding_session_state()
    assert state["session_status"] == "ok"
    assert state["audit_trail"] == ["init session state"]
    assert state["output_messages"] == []


def test_onboarding_state_type_is_importable():
    assert OnboardingState is not None
