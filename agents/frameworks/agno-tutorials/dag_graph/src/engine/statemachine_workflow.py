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
        from engine.handler_registry import does_state_wait_for_input

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
            # Stop if terminal state
            if current in self._TERMINAL_STATES:
                return True
            # Stop if state waits for user input (multi-turn pause point)
            current_val = current.value if hasattr(current, 'value') else current
            if does_state_wait_for_input(current_val):
                return True
            return False

        self.steps = [
            Loop(
                name="StateMachineLoop",
                max_iterations=_MAX_LOOP_ITERS,
                end_condition=_is_terminal,
                steps=[
                    Step(name="Router", executor=self._dispatch_router_step),
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

    def _new_session_state(self, entity_id: str) -> dict[str, Any]:
        """
        Create a fresh session state for processing an entity.

        Subclasses must override to initialize their domain-specific state.

        Args:
            entity_id: The entity being processed (document_id, invoice_id, etc.)

        Returns:
            A new session state dict ready for processing
        """
        raise NotImplementedError("Subclass must implement _new_session_state()")

    def _build_response(self, entity_id: str) -> dict[str, Any]:
        """
        Build response dict from current session_state.

        Extracts standard fields (current_state, retry_count, error_message, etc.)
        into a response dict. Subclasses can override to return a custom response
        type (e.g., PipelineState, InvoiceState) if needed.

        Args:
            entity_id: The entity being processed (document_id, invoice_id, etc.)

        Returns:
            Response dict with current_state, audit_trail, errors, etc.
        """
        return {
            "current_state": self.session_state.get("current_state", "init"),
            "proposed_next": self.session_state.get("proposed_next"),
            "retry_count": self.session_state.get("retry_count", 0),
            "error_message": self.session_state.get("error_message"),
            "guardrail_ok": self.session_state.get("guardrail_ok", True),
            "audit_trail": self.session_state.get("audit_trail", []),
            "semantic_context": self.session_state.get("semantic_context", {}),
            "router_confidence": self.session_state.get("router_confidence", 0.0),
        }

    # ── Step: Router (Pure Code or Semantic) ──────────────────────────────────

    def _dispatch_router_step(self, step_input: StepInput) -> StepOutput:
        """
        Dispatch to appropriate router based on mode.
        - Multi-turn (has turn_input): Use semantic router with LLM
        - One-turn (no turn_input): Use pure code router
        """
        is_multiturn = self.session_state.get("turn_input") is not None
        if is_multiturn:
            return self._semantic_router_step(step_input)
        else:
            return self._router_step(step_input)

    def _semantic_router_step(self, step_input: StepInput) -> StepOutput:
        """
        LLM-powered router: reads current_state + turn_input → proposes next state.

        Used for multi-turn workflows. Requires self.router to be initialized.
        Falls back to _router_step if router not available.
        """
        if not hasattr(self, 'router') or self.router is None:
            log.warning("Semantic router not initialized; using pure code routing")
            return self._router_step(step_input)

        current = self._get_current_state(self.session_state)
        turn_input = self.session_state.get("turn_input", "")
        history = self.session_state.get("conversation_history", [])

        # Get allowed next states from routing table
        routing_table = self._build_routing_table()
        routing_entry = routing_table.get(current)
        if routing_entry is None:
            raise ValueError(f"Current state {current} not in routing table")

        # Extract allowed states (handle both single state and dict of paths)
        if isinstance(routing_entry, dict):
            allowed = list(routing_entry.keys())
        else:
            allowed = [routing_entry]

        # Convert to string values
        allowed_str = [s.value if hasattr(s, 'value') else str(s) for s in allowed]

        # Call semantic router
        timeout_sec = self.session_state.get("router_timeout_sec", 10.0)
        decision = self.router.route(
            current_state=current.value if hasattr(current, 'value') else str(current),
            turn_input=turn_input,
            history=history,
            allowed_states=allowed_str,
            timeout_sec=timeout_sec
        )

        # Store decision in session_state
        self.session_state["proposed_next"] = decision.proposed_next
        self.session_state["semantic_context"] = {
            "entities": decision.semantic_entities,
            "intents": decision.semantic_intents,
        }
        self.session_state["router_confidence"] = decision.confidence

        log.info(
            "[SemanticRouter] %s + '%s...' → %s (confidence: %.2f)",
            current, turn_input[:30] if turn_input else "(empty)",
            decision.proposed_next, decision.confidence
        )

        return StepOutput(content={"proposed_next": decision.proposed_next})

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

    def _auto_progress(self) -> None:
        """
        Auto-progress workflow through non-blocking states.

        If current state has waits_for_input=False, continue running the state
        machine loop until we hit a state with waits_for_input=True or a
        terminal state. This allows workflows to skip through automatic
        processing steps without waiting for user input.
        """
        from engine.handler_registry import does_state_wait_for_input

        while True:
            current = self._get_current_state(self.session_state)

            # Stop if we hit a terminal state
            if current in self._TERMINAL_STATES:
                break

            # Stop if we hit a state that waits for input
            if does_state_wait_for_input(current.value if hasattr(current, 'value') else current):
                break

            # Continue: state is non-blocking, so run one more iteration
            log.debug(f"[AutoProgress] {current} has waits_for_input=False, continuing...")
            self.run(input=self.session_state.get("turn_input", ""))

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

    # ── One-turn entry point ────────────────────────────────────────────────

    def process(self, entity_id: str) -> Any:
        """
        Execute one complete run of the workflow for an entity.

        Provides generic one-turn support for any workflow:
          1. Initialize fresh session state via _new_session_state()
          2. Execute state machine loop (routing, handlers, guardrails)
          3. Auto-progress through non-blocking states (waits_for_input=False)
          4. Build response object via _build_response()
          5. Record run in output history
          6. Return response

        Args:
            entity_id: The entity identifier (document_id, invoice_id, etc.)

        Returns:
            Response object of the workflow's domain type
        """
        self._ensure_initialized()
        self.session_state.update(self._new_session_state(entity_id))
        self.run(input=entity_id)
        self._auto_progress()
        response = self._build_response(entity_id)
        if self.session_state.get("output") is None:
            self.session_state["output"] = []
        self.session_state["output"].append({
            "entity_id": entity_id,
            "final_state": self.session_state.get("current_state"),
        })
        return response

    # ── Multi-turn entry point ──────────────────────────────────────────────

    def process_turn(self,
                     user_id: str,
                     session_id: str,
                     turn_input: str,
                     timeout_sec: float = 10.0) -> dict[str, Any]:
        """
        Execute one turn of a multi-turn conversation.

        Provides generic multi-turn support for any workflow:
          1. Validate input (token count, length, injection prevention)
          2. Escape for LLM (prompt injection safety)
          3. Prepare turn metadata (turn_input, turn_number, router_timeout)
          4. Execute state machine loop (routing, handlers, guardrails)
          5. Auto-progress through non-blocking states (waits_for_input=False)
          6. Trim conversation history to max_history_turns
          7. Return response with state, entities, intents, confidence

        Args:
            user_id: Caller identity (for audit trail)
            session_id: Multi-turn session ID (for persistence)
            turn_input: User's input text (will be validated & escaped)
            timeout_sec: LLM router timeout in seconds

        Returns:
            {
              "current_state": str,           # Current state after execution
              "waits_for_input": bool,        # True if workflow paused (waits_for_input=True)
              "turn_number": int,             # Turn counter (incremented each call)
              "semantic_context": dict,       # {entities, intents} from router
              "router_confidence": float,     # LLM router confidence [0.0, 1.0]
              "error": str | None             # Error message if validation failed
            }
        """
        from engine.input_validation import validate_turn_input, escape_for_llm, InputValidationError

        try:
            validate_turn_input(turn_input)
            escaped = escape_for_llm(turn_input)
            self._ensure_initialized()

            # Initialize router if subclass provides one (for semantic routing in multi-turn)
            if hasattr(self, '_init_router'):
                self._init_router()

            # Initialize session state on first turn (turn_number == 0)
            if self.session_state.get("turn_number", 0) == 0:
                # First turn: initialize fresh session state
                entity_id = session_id  # Use session_id as entity identifier for multi-turn
                self.session_state.update(self._new_session_state(entity_id))

            self._prepare_turn_metadata(escaped, timeout_sec)
            self.run(session_id=session_id, user_id=user_id)
            self._auto_progress()
            self._trim_history()

            return self._build_turn_response()

        except InputValidationError as e:
            return {"error": str(e), "current_state": None, "waits_for_input": False}
        except Exception as e:
            log.exception("process_turn failed: %s", e)
            return {"error": str(e), "current_state": "error", "waits_for_input": False}
