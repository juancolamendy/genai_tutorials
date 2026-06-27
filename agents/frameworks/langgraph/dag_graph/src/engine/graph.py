"""Generic state machine graph base class for LangGraph.

Provides:
  • StateMachineGraph — base class for state machine workflows
  • State serialization/deserialization helpers
  • Router, guardrail, and handler dispatch patterns
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=dict[str, Any])
StateEnum = TypeVar("StateEnum")


# ─────────────────────────────────────────────────────────────────────────
# ERROR HANDLING & SAFETY
# ─────────────────────────────────────────────────────────────────────────

def safe_node(func: Callable) -> Callable:
    """Wrap node function with error handling.

    Catches exceptions and returns error state without propagating.
    """
    @wraps(func)
    def wrapper(state: dict[str, Any]) -> dict[str, Any]:
        try:
            return func(state)
        except Exception as e:
            log.error(f"Node {func.__name__} failed: {e}", exc_info=True)
            return {
                **state,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "proposed_next": "error",  # Route to error state
            }
    return wrapper


def serialize_session_state(
    session_state: dict[str, Any],
    keys: tuple[str, ...],
    defaults: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Extract specified keys from session_state into a new dict.

    Args:
        session_state: The LangGraph state dict
        keys: Tuple of keys to extract (order matters for positional access)
        defaults: Optional dict of {key: default_value} for missing keys

    Returns:
        A new dict containing only the specified keys with their values
        from session_state (or defaults if missing).
    """
    defaults = defaults or {}
    return {k: session_state.get(k, defaults.get(k)) for k in keys}


def deserialize_to_session_state(
    state_dict: dict[str, Any],
    session_state: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    """Write state_dict values back into session_state for specified keys.

    Args:
        state_dict: The state dict to read from
        session_state: The LangGraph state dict to write to (in-place)
        keys: Tuple of keys to sync
    """
    for k in keys:
        if k in state_dict:
            session_state[k] = state_dict[k]


class StateMachineGraph:
    """Generic state machine graph base class for LangGraph.

    Implements the production pattern:
      Router → Guardrail → Handler → (loop or end)

    Subclasses must override:
      • _STATE_KEYS tuple (keys to persist from state dict)
      • _STATE_ENUM (enum type for state values)
      • _TERMINAL_STATES (set of terminal state enum values)
      • HANDLER_MAP (dict[StateEnum, Callable])
      • _build_routing_table() → dict[StateEnum, StateEnum]
      • _get_current_state(state) → StateEnum
      • _get_proposed_state(state) → StateEnum
      • _get_guardrails() → dict[StateEnum, GuardrailFn]

    Optional:
      • semantic_router — LLM-powered router (if None, uses routing table)
    """

    # Subclasses must override these
    _STATE_KEYS: tuple[str, ...] = ()
    _STATE_ENUM: type = None
    _TERMINAL_STATES: set = set()
    HANDLER_MAP: dict[Any, Callable] = {}
    semantic_router: Optional[Any] = None

    def _build_routing_table(self) -> dict[Any, Any]:
        """Return {current_state: next_state} routing table. Override in subclass."""
        raise NotImplementedError

    def _get_current_state(self, state: dict[str, Any]) -> Any:
        """Extract current state from state dict. Override in subclass."""
        raise NotImplementedError

    def _get_proposed_state(self, state: dict[str, Any]) -> Any:
        """Extract proposed next state from state dict. Override in subclass."""
        raise NotImplementedError

    def _get_guardrails(self) -> dict[Any, Callable]:
        """Return guardrail registry. Override in subclass."""
        return {}

    # ─────────────────────────────────────────────────────────────────────────
    # GENERIC NODES (used by all subclasses)
    # ─────────────────────────────────────────────────────────────────────────

    def _router_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Route: current_state → proposed_next via semantic or code router.

        If semantic_router is available, uses LLM routing. Otherwise falls back
        to pure code routing via routing table.

        Args:
            state: State dict with current_state set

        Returns:
            Updated state with proposed_next set
        """
        current = self._get_current_state(state)

        # Try semantic router first (if available)
        if self.semantic_router is not None:
            try:
                router_decision = self.semantic_router.route(state)
                proposal = router_decision.proposed_next
                proposal_val = proposal.value if hasattr(proposal, "value") else proposal

                log.info(
                    "[ROUTER] semantic: %s → %s (confidence=%.2f)",
                    current,
                    proposal,
                    router_decision.confidence,
                )

                # Store semantic context in state
                semantic_state = {
                    **state,
                    "proposed_next": proposal_val,
                    "semantic_context": {
                        "entities": router_decision.semantic_entities,
                        "intents": router_decision.semantic_intents,
                    },
                    "router_confidence": router_decision.confidence,
                    "audit_trail": state.get("audit_trail", [])
                    + [
                        f"router: semantic {current} → {proposal} "
                        f"(conf={router_decision.confidence:.2f})"
                    ],
                }

                # Add reasoning if available
                if router_decision.reasoning:
                    semantic_state["router_reasoning"] = router_decision.reasoning

                return semantic_state

            except Exception as e:
                log.warning("[ROUTER] semantic routing failed (%s); falling back to code router", e)

        # Fallback: code-based routing via routing table
        routing_table = self._build_routing_table()
        if current not in routing_table:
            raise ValueError(f"Current state {current} not in routing table")

        proposal = routing_table[current]
        proposal_val = proposal.value if hasattr(proposal, "value") else proposal

        log.info("[ROUTER] code: %s → proposes %s", current, proposal)

        return {
            **state,
            "proposed_next": proposal_val,
            "audit_trail": state.get("audit_trail", []) + [f"router: code {current} → {proposal}"],
        }

    def _guardrail_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Validate proposed_next; apply fallback if needed.

        Args:
            state: State dict with proposed_next set

        Returns:
            Updated state with guardrail result applied
        """
        proposed = self._get_proposed_state(state)
        guardrails = self._get_guardrails()
        guard = guardrails.get(proposed, lambda _: type("Result", (), {"passed": True})())
        result = guard(state)

        if result.passed:
            log.info("[GUARDRAIL] ✅  %s passed", proposed)
            return {
                **state,
                "audit_trail": state.get("audit_trail", []) + [f"guardrail PASS → {proposed}"],
            }

        fallback_val = (result.fallback or self._STATE_ENUM.ERROR).value
        log.warning(
            "[GUARDRAIL] ❌  %s failed (%s) → fallback: %s",
            proposed,
            result.reason,
            fallback_val,
        )
        return {
            **state,
            "proposed_next": fallback_val,
            "error_message": result.reason,
            "audit_trail": state.get("audit_trail", [])
            + [f"guardrail FAIL → {proposed} ({result.reason}) → fallback {fallback_val}"],
        }

    def _guardrail_router(self, state: dict[str, Any]) -> str:
        """Route to handler based on proposed_next.

        Args:
            state: State dict with proposed_next set

        Returns:
            State name (string) to route to
        """
        return state["proposed_next"]

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH BUILDING (generic, reusable for all subclasses)
    # ─────────────────────────────────────────────────────────────────────────

    def build_graph(self, state_schema: Any, checkpointer: Optional[Any] = None) -> Any:
        """Build and compile the LangGraph state machine.

        Pattern: Router → Guardrail → Handler → (loop or end)

        Args:
            state_schema: TypedDict or dict defining the state structure
            checkpointer: Optional BaseCheckpointSaver for checkpointing

        Returns:
            Compiled StateGraph ready for invocation
        """
        g = StateGraph(state_schema)

        # Add nodes with error handling
        g.add_node("router", safe_node(self._router_node))
        g.add_node("guardrail", safe_node(self._guardrail_node))

        # Add handler nodes with error handling
        for state_enum, handler_fn in self.HANDLER_MAP.items():
            g.add_node(state_enum.value, safe_node(handler_fn))

        # Entry point
        g.set_entry_point("router")

        # Edges: router → guardrail (always)
        g.add_edge("router", "guardrail")

        # Edges: guardrail → handler (conditional: based on proposed_next)
        g.add_conditional_edges(
            "guardrail",
            self._guardrail_router,
            {state.value: state.value for state in self.HANDLER_MAP.keys()},
        )

        # Edges: handlers → router (loop for non-terminal) or END (for terminal or blocking)
        def _should_continue(state: dict[str, Any]) -> str:
            """Route handler output: stop if blocking, loop if non-blocking."""
            from engine.handler_registry import does_state_wait_for_input

            current = state.get("current_state", "init")

            # Terminal states always end
            if current in [s.value for s in self._TERMINAL_STATES]:
                return END

            # Blocking states end (waits for input)
            if does_state_wait_for_input(current):
                return END

            # Non-blocking states continue to router
            return "router"

        for state_enum in self.HANDLER_MAP.keys():
            g.add_conditional_edges(
                state_enum.value,
                _should_continue,
                {END: END, "router": "router"},
            )

        # Compile with optional checkpointer
        return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-TURN SUPPORT METHODS (new in Phase 2)
    # ─────────────────────────────────────────────────────────────────────────

    def invoke(
        self,
        state: dict[str, Any],
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Invoke compiled graph once.

        Args:
            state: Current pipeline state
            config: {"configurable": {"thread_id": ...}} for checkpointing

        Returns:
            Updated state after one full iteration (router → guardrail → handler)
        """
        if config is None:
            config = {}
        return self.compiled_graph.invoke(state, config=config)

    def _auto_progress_langgraph(
        self,
        state: dict[str, Any],
        config: dict[str, Any],
        max_auto_iters: int = 10,
    ) -> dict[str, Any]:
        """Auto-progress through non-blocking states.

        If current state has waits_for_input=False, continue running state machine
        until hitting a state with waits_for_input=True or a terminal state.

        Args:
            state: Current pipeline state
            config: Graph invocation config (with thread_id for checkpointing)

        Returns:
            Updated state after auto-progression
        """
        from engine.handler_registry import does_state_wait_for_input

        iters = 0

        while iters < max_auto_iters:
            current = state.get("current_state", "init")

            # Stop if terminal state
            if current in self._TERMINAL_STATES:
                log.debug(f"[auto_progress] Stopped at terminal state {current}")
                break

            # Stop if state waits for input
            if does_state_wait_for_input(current):
                log.debug(f"[auto_progress] Stopped at input-waiting state {current}")
                break

            # Continue: run state machine one more time
            log.debug(f"[auto_progress] {current} is non-blocking; continuing...")
            state = self.compiled_graph.invoke(state, config=config)
            iters += 1

        if iters >= max_auto_iters:
            log.warning("[auto_progress] Reached max iterations (%d); stopping", max_auto_iters)

        return state

    def process(
        self,
        entity_id: str,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        """Execute one complete workflow run for an entity.

        Workflow:
        1. Create fresh state
        2. Run state machine loop
        3. Auto-progress through non-blocking states
        4. Return response

        Args:
            entity_id: Document ID, invoice ID, etc.
            timeout_seconds: Max execution time

        Returns:
            Response dict with current_state, audit_trail, errors, etc.
        """
        try:
            # Create fresh state
            state = self.new_pipeline(entity_id, timeout_seconds)

            # Thread ID for checkpointing
            thread_id = f"process:{entity_id}"
            config = {"configurable": {"thread_id": thread_id}}

            # Run state machine
            state = self.compiled_graph.invoke(state, config=config)

            # Auto-progress
            state = self._auto_progress_langgraph(state, config)

        except Exception as e:
            log.exception("[process] Error: %s", e)
            state = self.new_pipeline(entity_id, timeout_seconds)
            state["current_state"] = "error"
            state["error_message"] = str(e)

        return self._build_response(entity_id, state)

    def invoke_turn(
        self,
        user_id: str,
        session_id: str,
        turn_input: str,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        """Execute one turn of multi-turn conversation.

        Workflow:
        1. Validate and escape user input
        2. Get or initialize state for session
        3. Prepare turn metadata
        4. Run state machine once
        5. Auto-progress through non-blocking states
        6. Trim conversation history
        7. Append turn to history
        8. Return turn response

        Args:
            user_id: Caller identity (for audit)
            session_id: Multi-turn session ID
            turn_input: User's input text
            timeout_sec: LLM router timeout

        Returns:
            {
                "current_state": str,
                "waits_for_input": bool,
                "turn_number": int,
                "semantic_context": dict,
                "router_confidence": float,
                "error": str or None,
            }
        """
        from engine.input_validation import (
            InputValidationError,
            escape_for_llm,
            validate_turn_input,
        )

        try:
            # Validate and escape input
            validate_turn_input(turn_input)
            escaped = escape_for_llm(turn_input)

            # Thread ID for checkpointing across turns
            thread_id = f"{user_id}:{session_id}"
            config = {"configurable": {"thread_id": thread_id}}

            # Get or initialize state
            state = self._get_or_init_state(session_id)

            # Prepare turn metadata
            state["turn_input"] = escaped
            state["turn_number"] = state.get("turn_number", 0) + 1
            state["router_timeout_sec"] = timeout_sec
            state["user_id"] = user_id
            state["session_id"] = session_id

            # Initialize router if available and needed
            if hasattr(self, "_init_router"):
                self._init_router()

            # First invoke: router → guardrail → handler
            state = self.compiled_graph.invoke(state, config=config)

            # Auto-progress through non-blocking states
            state = self._auto_progress_langgraph(state, config)

            # Trim history
            max_turns = state.get("max_history_turns", 10)
            history = state.get("conversation_history", [])
            if len(history) > max_turns:
                dropped = len(history) - max_turns
                state["conversation_history"] = history[-max_turns:]
                log.info(f"[invoke_turn] Trimmed {dropped} old turns; keeping {max_turns}")

            # Append turn result to history
            state["conversation_history"].append({
                "role": "assistant",
                "content": f"Transitioned to {state['current_state']}",
                "semantic_context": {
                    "entities": state.get("semantic_context", {}).get("entities", {}),
                    "intents": state.get("semantic_context", {}).get("intents", []),
                },
                "state": state["current_state"],
                "turn_number": state["turn_number"],
            })

            return self._build_turn_response(state)

        except InputValidationError as e:
            return {
                "error": str(e),
                "current_state": None,
                "waits_for_input": False,
                "turn_number": 0,
                "semantic_context": {},
                "router_confidence": 0.0,
            }
        except Exception as e:
            log.exception("[invoke_turn] Error: %s", e)
            return {
                "error": str(e),
                "current_state": "error",
                "waits_for_input": False,
                "turn_number": 0,
                "semantic_context": {},
                "router_confidence": 0.0,
            }

    def _get_or_init_state(self, session_id: str) -> dict[str, Any]:
        """Get existing state or create fresh state for session.

        Tries to load from checkpointer first; if not found, creates fresh state.

        Args:
            session_id: Session identifier

        Returns:
            PipelineState dict
        """
        thread_id = f"invoke_turn:{session_id}"

        # Try to load from checkpointer (if available)
        try:
            if hasattr(self, "checkpointer") and self.checkpointer:
                checkpoint = self.checkpointer.get_tuple(thread_id)
                if checkpoint and hasattr(checkpoint, "values"):
                    log.info(f"[invoke_turn] Loaded state from checkpoint {thread_id}")
                    return checkpoint.values
        except Exception as e:
            log.debug(f"[invoke_turn] Checkpoint load failed ({e}); creating fresh state")

        # Create fresh state
        return self.new_pipeline(session_id)

    def new_pipeline(self, entity_id: str, timeout_seconds: float = 300.0) -> dict[str, Any]:
        """Create fresh pipeline state. Override in subclass if needed."""
        raise NotImplementedError

    def _build_response(
        self,
        entity_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build response dict from final state after process().

        Args:
            entity_id: Entity being processed
            state: Final state

        Returns:
            Response dict
        """
        return {
            "current_state": state.get("current_state", "init"),
            "proposed_next": state.get("proposed_next"),
            "retry_count": state.get("retry_count", 0),
            "error_message": state.get("error_message"),
            "audit_trail": state.get("audit_trail", []),
            "entity_id": entity_id,
        }

    def _build_turn_response(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build response dict from state after invoke_turn().

        Args:
            state: PipelineState after turn execution

        Returns:
            Turn response dict
        """
        from engine.handler_registry import does_state_wait_for_input

        current = state.get("current_state", "init")
        return {
            "current_state": current,
            "waits_for_input": does_state_wait_for_input(current),
            "turn_number": state.get("turn_number", 0),
            "semantic_context": state.get("semantic_context", {}),
            "router_confidence": state.get("router_confidence", 0.0),
            "error": state.get("error_message"),
        }
