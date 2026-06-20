"""LangGraph state machine engine (reusable, domain-agnostic)."""

from .state_machine import (
    State,
    PipelineState,
    GuardrailResult,
    ALLOWED_TRANSITIONS,
    is_transition_allowed,
)

__all__ = [
    "State",
    "PipelineState",
    "GuardrailResult",
    "ALLOWED_TRANSITIONS",
    "is_transition_allowed",
]
