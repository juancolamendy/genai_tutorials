"""
engine/workflow.py
────────────────────────────────────────────────────────────────────────────
Reusable state machine workflow base class and utilities.

Provides:
  • serialize_session_state()   — extract state dict keys into typed dict
  • deserialize_to_session_state() — write typed dict back into session_state
  • StateMachineWorkflow        — base class for state machine workflows

The base class handles generic patterns:
  • Loop setup with guardrails and routing
  • Handler registration and binding
  • Session state lifecycle
  • Terminal condition checking
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

from agno.workflow import Loop, Router, Step, Workflow
from agno.workflow.types import StepInput, StepOutput

log = logging.getLogger(__name__)

_MAX_LOOP_ITERS = 20


StateT = TypeVar("StateT", bound=dict[str, Any])
StateEnum = TypeVar("StateEnum")


def serialize_session_state(
    session_state: dict[str, Any],
    keys: tuple[str, ...],
    defaults: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Extract specified keys from session_state into a new dict.

    Args:
        session_state: The Agno session_state dict
        keys: Tuple of keys to extract (order matters for positional access)
        defaults: Optional dict of {key: default_value} for missing keys

    Returns:
        A new dict containing only the specified keys with their values
        from session_state (or defaults if missing).
    """
    defaults = defaults or {}
    return {
        k: session_state.get(k, defaults.get(k))
        for k in keys
    }


def deserialize_to_session_state(
    state_dict: dict[str, Any],
    session_state: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    """
    Write state_dict values back into session_state for specified keys.

    Args:
        state_dict: The state dict to read from
        session_state: The Agno session_state dict to write to (in-place)
        keys: Tuple of keys to sync
    """
    for k in keys:
        if k in state_dict:
            session_state[k] = state_dict[k]


class StateMachineWorkflow(Workflow):
    """
    Generic state machine workflow base class.

    Subclasses must:
      • Define _STATE_KEYS tuple (keys to persist from state dict)
      • Define _STATE_ENUM (enum type for state values)
      • Define _TERMINAL_STATES (set of terminal state enum values)
      • Define HANDLER_MAP (dict[StateEnum, Callable])
      • Implement _init_session_defaults() — initialize session_state
      • Implement _build_routing_table() — return dict[StateEnum, StateEnum]
      • Implement _get_current_state(session_state) → StateEnum
      • Implement _get_proposed_state(session_state) → StateEnum
    """

    # Subclasses must override these
    _STATE_KEYS: tuple[str, ...] = ()
    _STATE_ENUM: type = None
    _TERMINAL_STATES: set = set()
    HANDLER_MAP: dict[Any, Callable] = {}

    def __post_init__(self) -> None:
        if self.session_state is None:
            self.session_state = {}
        self._init_session_defaults()
        self._init_steps()

    def _ensure_initialized(self) -> None:
        """Ensure session_state and steps are initialized (idempotent)."""
        if self.session_state is None:
            self.session_state = {}
        if self.steps is None or len(self.steps) == 0:
            self._init_steps()

    def _init_steps(self) -> None:
        """Initialize the workflow steps (called from __post_init__ and when needed)."""
        # Build handler Steps bound to `self` so they can access session_state.
        handler_steps: dict[Any, Step] = {
            state: Step(
                name=f"{state.value.title().replace('_', '')}Handler",
                executor=self._make_handler_executor(state),
            )
            for state in self.HANDLER_MAP
        }

        # end_condition closes over self — no need for CEL expressions.
        def _is_terminal(outputs: list) -> bool:
            current = self._get_current_state(self.session_state)
            return current in self._TERMINAL_STATES

        self.steps = [
            Loop(
                name="StateMachineLoop",
                max_iterations=_MAX_LOOP_ITERS,
                end_condition=_is_terminal,
                steps=[
                    Step(name="Router", executor=self._router_step),
                    Step(name="Guardrail", executor=self._guardrail_step),
                    Router(
                        name="DispatchHandler",
                        selector=self._dispatch,
                        choices=list(handler_steps.values()),
                    ),
                ],
            )
        ]

        self._handler_steps = handler_steps

    # ── Subclass hooks (must be overridden) ────────────────────────────────────

    def _init_session_defaults(self) -> None:
        """Initialize session_state with required keys. Override in subclass."""
        pass

    def _build_routing_table(self) -> dict[Any, Any]:
        """Return {current_state: next_state} routing table. Override in subclass."""
        return {}

    def _get_current_state(self, session_state: dict[str, Any]) -> Any:
        """Extract current state from session_state. Override in subclass."""
        raise NotImplementedError

    def _get_proposed_state(self, session_state: dict[str, Any]) -> Any:
        """Extract proposed next state from session_state. Override in subclass."""
        raise NotImplementedError

    def _run_guardrail(self, state_dict: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        """
        Run guardrails on proposed transition.

        Returns:
            (updated_state_dict, result) where result has .passed and .reason.
        Override in subclass to add guardrail logic; default is pass-through.
        """
        from dataclasses import dataclass

        @dataclass
        class Result:
            passed: bool = True
            reason: str = ""

        return state_dict, Result()

    # ── Step: Router ──────────────────────────────────────────────────────────

    def _router_step(self, step_input: StepInput) -> StepOutput:
        """Pure code router: reads current_state → proposes next state."""
        routing_table = self._build_routing_table()
        current = self._get_current_state(self.session_state)
        if current not in routing_table:
            raise ValueError(f"Current state {current} not in routing table")
        proposal = routing_table[current]

        proposal_val = proposal.value if hasattr(proposal, 'value') else proposal
        self.session_state["proposed_next"] = proposal_val

        log.info("[Router]   %s → proposes %s", current, proposal)
        return StepOutput(content={"proposed_next": proposal})

    # ── Step: Guardrail ───────────────────────────────────────────────────────

    def _guardrail_step(self, step_input: StepInput) -> StepOutput:
        """Validates the proposed transition."""
        state_dict = serialize_session_state(self.session_state, self._STATE_KEYS)
        new_state_dict, result = self._run_guardrail(state_dict)
        deserialize_to_session_state(new_state_dict, self.session_state, self._STATE_KEYS)

        proposed = self._get_proposed_state(self.session_state)
        if result.passed:
            log.info("[Guardrail] ✅  %s", proposed)
        else:
            log.warning("[Guardrail] ❌  fallback → %s  (%s)", proposed, result.reason)

        return StepOutput(content={"guardrail_ok": result.passed, "proposed_next": proposed})

    # ── Router: DispatchHandler ───────────────────────────────────────────────

    def _dispatch(self, step_input: StepInput) -> list[Step]:
        """Routes to handler Step based on proposed_next."""
        proposed = self._get_proposed_state(self.session_state)
        if proposed not in self._handler_steps:
            raise ValueError(f"No handler for state: {proposed}")
        chosen = self._handler_steps[proposed]
        log.info("[Dispatch] → %s", chosen.name)
        return [chosen]

    # ── Handler executor factory ──────────────────────────────────────────────

    def _make_handler_executor(self, state: Any):
        """Return a Step executor that calls HANDLER_MAP[state]."""
        handler_fn = self.HANDLER_MAP[state]

        def _executor(step_input: StepInput) -> StepOutput:
            state_dict = serialize_session_state(self.session_state, self._STATE_KEYS)
            new_state = handler_fn(state_dict)
            deserialize_to_session_state(new_state, self.session_state, self._STATE_KEYS)
            current = self._get_current_state(self.session_state)
            return StepOutput(content={"current_state": current})

        return _executor

    # ── Multi-turn helpers ──────────────────────────────────────────────────────

    def _prepare_turn_metadata(self, turn_input: str, timeout_sec: float) -> None:
        """Prepare session_state for a new turn."""
        turn_num = self.session_state.get("turn_number", 0)
        self.session_state.update({
            "turn_input": turn_input,
            "turn_number": turn_num + 1,
            "router_timeout_sec": timeout_sec,
        })

    def _trim_history(self) -> None:
        """Keep only last max_history_turns in session_state."""
        max_turns = self.session_state.get("max_history_turns", 10)
        history = self.session_state.get("conversation_history", [])
        if len(history) > max_turns:
            dropped = len(history) - max_turns
            self.session_state["conversation_history"] = history[-max_turns:]
            log.info(f"Trimmed {dropped} turns; keeping last {max_turns}")

    def _build_turn_response(self) -> dict[str, Any]:
        """Build response dict from current session_state."""
        from engine.handler_registry import does_state_wait_for_input

        current = self.session_state.get("current_state", "init")
        return {
            "current_state": current,
            "waits_for_input": does_state_wait_for_input(current),
            "turn_number": self.session_state.get("turn_number", 0),
            "semantic_context": self.session_state.get("semantic_context", {}),
            "router_confidence": self.session_state.get("router_confidence", 0.0),
            "error": self.session_state.get("error_message")
        }
