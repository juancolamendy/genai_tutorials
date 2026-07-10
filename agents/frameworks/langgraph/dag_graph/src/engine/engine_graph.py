"""Generic state machine graph base class for LangGraph.

Provides:
  • EngineGraph — base class for state machine workflows
  • Router, guardrail, and handler dispatch patterns
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Optional

from langgraph.graph import END, StateGraph

log = logging.getLogger(__name__)


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


class EngineGraph:
    """Generic state machine graph base class for LangGraph.

    Implements the production pattern:
      Router → Guardrail → Handler → (loop or end)

    Subclasses must override:
      • state_enum (enum type for state values)
      • terminal_states (set of terminal state enum values)
      • handler_map (dict[StateEnum, Callable])
      • _build_routing_table() → dict[StateEnum, StateEnum]
      • _get_current_state(state) → StateEnum
      • _get_proposed_state(state) → StateEnum
      • _get_guardrails() → dict[StateEnum, GuardrailFn]

    Optional:
      • semantic_router — LLM-powered router (if None, uses routing table)
    """

    # Subclasses must override these
    state_enum: type = None
    terminal_states: set = set()
    handler_map: dict[Any, Callable] = {}
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

    def _get_allowed_states(self, current_state: Any) -> list[str]:
        """Get allowed next states for current state.

        Override in subclass to provide state machine's allowed transitions.
        Default returns all possible states (permissive).

        Args:
            current_state: Current state enum or string

        Returns:
            List of allowed state strings
        """
        # Default: permissive - allow any state. Subclass should override.
        return []

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
                # Extract arguments for semantic router
                current_state = state.get("current_state", "init")
                turn_input = state.get("turn_input", "")
                history = state.get("conversation_history", [])
                timeout_sec = state.get("router_timeout_sec", 10.0)

                # Get allowed next states from state machine
                allowed_states = self._get_allowed_states(current)

                router_decision = self.semantic_router.route(
                    current_state=current_state,
                    turn_input=turn_input,
                    history=history,
                    allowed_states=allowed_states,
                    timeout_sec=timeout_sec,
                )
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

        If the (possibly fallback-adjusted) target state waits for input, park
        current_state there without running its handler — the handler only
        runs once, on the turn that actually supplies fresh input for it (see
        the resume-at-blocking-state step in invoke()).

        Args:
            state: State dict with proposed_next set

        Returns:
            Updated state with guardrail result applied
        """
        from src.engine.handler_registry import does_state_wait_for_input

        proposed = self._get_proposed_state(state)
        guardrails = self._get_guardrails()
        guard = guardrails.get(proposed, lambda _: type("Result", (), {"passed": True})())
        result = guard(state)

        if result.passed:
            log.info("[GUARDRAIL] ✅  %s passed", proposed)
            new_state = {
                **state,
                "fallback_depth": 0,
                "audit_trail": state.get("audit_trail", []) + [f"guardrail PASS → {proposed}"],
            }
        else:
            fallback_val = (result.fallback or self.state_enum.ERROR).value
            fallback_depth = state.get("fallback_depth", 0) + 1
            log.warning(
                "[GUARDRAIL] ❌  %s failed (%s) → fallback: %s (depth=%d)",
                proposed,
                result.reason,
                fallback_val,
                fallback_depth,
            )
            new_state = {
                **state,
                "proposed_next": fallback_val,
                "error_message": result.reason,
                "fallback_depth": fallback_depth,
                "audit_trail": state.get("audit_trail", [])
                + [f"guardrail FAIL → {proposed} ({result.reason}) → fallback {fallback_val}"],
            }

        target = new_state["proposed_next"]
        if does_state_wait_for_input(target):
            new_state["current_state"] = target

        return new_state

    def _guardrail_router(self, state: dict[str, Any]) -> str:
        """Route to handler based on proposed_next, or stop if it waits for input.

        Args:
            state: State dict with proposed_next set

        Returns:
            State name (string) to route to, or END
        """
        from src.engine.handler_registry import does_state_wait_for_input

        proposed = state["proposed_next"]
        if does_state_wait_for_input(proposed):
            return END
        return proposed

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
        for state_val, handler_fn in self.handler_map.items():
            g.add_node(state_val.value, safe_node(handler_fn))

        # Entry point
        g.set_entry_point("router")

        # Edges: router → guardrail (always)
        g.add_edge("router", "guardrail")

        # Edges: guardrail → handler (conditional: based on proposed_next), or
        # END if proposed_next waits for input — its handler never dispatches
        # here; see _guardrail_router/_guardrail_node.
        g.add_conditional_edges(
            "guardrail",
            self._guardrail_router,
            {END: END, **{state.value: state.value for state in self.handler_map.keys()}},
        )

        # Edges: handlers → router (loop) or END (terminal). A handler is
        # never dispatched to for a waits_for_input state (guardrail stops
        # before that), so only the terminal check applies here.
        def _should_continue(state: dict[str, Any]) -> str:
            """Route handler output: stop if terminal, loop otherwise."""
            current = state.get("current_state", "init")

            if current in [s.value for s in self.terminal_states]:
                return END

            return "router"

        for state_val in self.handler_map.keys():
            g.add_conditional_edges(
                state_val.value,
                _should_continue,
                {END: END, "router": "router"},
            )

        # Compile with optional checkpointer
        return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-TURN SUPPORT METHODS (new in Phase 2)
    # ─────────────────────────────────────────────────────────────────────────

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
        from src.engine.handler_registry import does_state_wait_for_input

        iters = 0

        while iters < max_auto_iters:
            current = state.get("current_state", "init")

            # Stop if terminal state
            if current in self.terminal_states:
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

    def invoke(
        self,
        user_id: str,
        session_id: str,
        turn_input: str,
        state_delta: Optional[dict[str, Any]] = None,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        """Execute one turn of multi-turn conversation.

        Workflow:
        1. Validate and escape user input
        2. Get or initialize state for session
        3. Prepare turn metadata (and merge state_delta, if given)
        4. Run state machine once
        5. Auto-progress through non-blocking states
        6. Trim conversation history
        7. Append turn to history
        8. Return turn response

        Args:
            user_id: Caller identity (for audit)
            session_id: Multi-turn session ID
            turn_input: User's input text
            state_delta: Optional extra fields to merge into state alongside
                turn_input (e.g. business payload supplied this turn), before
                the graph is invoked
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
        from src.engine.input_validation import (
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

            # Get or initialize state (use thread_id format for consistency)
            state = self._get_or_init_state(session_id, user_id=user_id)

            # Prepare turn metadata
            state["turn_input"] = escaped
            current_turn_number = state.get("turn_number", 0) + 1
            state["turn_number"] = current_turn_number
            state["router_timeout_sec"] = timeout_sec
            state["user_id"] = user_id
            state["session_id"] = session_id

            # Merge caller-supplied state updates before the graph runs
            if state_delta is not None:
                state.update(state_delta)

            # Resuming at a blocking state: run its own handler with the fresh
            # turn_input before the graph executes. The router only proposes
            # transitions FORWARD from current_state via the routing table, so
            # without this it would skip past the blocking state on resume and
            # silently discard the new turn's input (e.g. an upload) instead of
            # letting the handler process it.
            from src.engine.handler_registry import does_state_wait_for_input

            current_state_str = state.get("current_state", "init")
            if does_state_wait_for_input(current_state_str):
                handler_fn = self.handler_map.get(self.state_enum(current_state_str))
                if handler_fn is not None:
                    state = handler_fn(state)

            # Initialize router if available and needed
            if hasattr(self, "_init_router"):
                self._init_router()

            # First invoke: router → guardrail → handler
            state = self.compiled_graph.invoke(state, config=config)

            # CRITICAL: Restore turn_number after invoke (LangGraph may reset it from checkpoint)
            state["turn_number"] = current_turn_number

            # Auto-progress through non-blocking states
            state = self._auto_progress_langgraph(state, config)

            # CRITICAL: Restore turn_number again after auto-progress
            # (additional invoke calls may reset it)
            state["turn_number"] = current_turn_number

            # Trim history
            max_turns = state.get("max_history_turns", 10)
            history = state.get("conversation_history", [])
            if len(history) > max_turns:
                dropped = len(history) - max_turns
                state["conversation_history"] = history[-max_turns:]
                log.info(f"[invoke] Trimmed {dropped} old turns; keeping {max_turns}")

            # Append user input to history
            state["conversation_history"].append({
                "role": "user",
                "content": escaped,
                "turn_number": state["turn_number"],
            })

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

            # AUTO-SAVE PAUSE POINT (hidden from user)
            from src.engine.handler_registry import does_state_wait_for_input

            if does_state_wait_for_input(state.get("current_state")):
                # Mark this checkpoint as a pause point for automatic resumption
                checkpointer = getattr(self.compiled_graph, "checkpointer", None)
                if checkpointer:
                    try:
                        # Get the latest checkpoint ID from the session file
                        session_data = checkpointer.export_session(thread_id)
                        if session_data:
                            checkpoint_id = session_data.get("latest_checkpoint_id")
                            if checkpoint_id:
                                checkpointer.save_pause_point(config, checkpoint_id)
                                log.info(
                                    f"[invoke] Auto-saved pause point "
                                    f"{checkpoint_id} at {state['current_state']}"
                                )
                    except Exception as e:
                        log.debug(f"[invoke] Could not auto-save pause point: {e}")

            return state

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
            log.exception("[invoke] Error: %s", e)
            return {
                "error": str(e),
                "current_state": "error",
                "waits_for_input": False,
                "turn_number": 0,
                "semantic_context": {},
                "router_confidence": 0.0,
            }

    def _get_or_init_state(self, session_id: str, user_id: str = "") -> dict[str, Any]:
        """Get existing state or create fresh state for session.

        Auto-loads from pause point if available, otherwise tries latest checkpoint.

        Args:
            session_id: Session identifier
            user_id: User identifier (used for thread_id in checkpointing)

        Returns:
            SessionState dict
        """
        thread_id = f"{user_id}:{session_id}" if user_id else session_id

        # Try to load from pause point first (hidden auto-resumption)
        try:
            if hasattr(self.compiled_graph, "checkpointer") and self.compiled_graph.checkpointer:
                config = {"configurable": {"thread_id": thread_id}}

                # Check for pause point first
                pause_checkpoint_id = self.compiled_graph.checkpointer.get_pause_point(config)
                if pause_checkpoint_id:
                    log.info(
                        f"[invoke] Found pause point {pause_checkpoint_id}; "
                        "resuming from there"
                    )
                    config["configurable"]["checkpoint_id"] = pause_checkpoint_id
                    checkpoint_tuple = self.compiled_graph.checkpointer.get_tuple(config)
                    if checkpoint_tuple:
                        pause_state = self._extract_state_from_checkpoint(checkpoint_tuple)
                        if pause_state is not None:
                            return pause_state

                # Fallback to latest checkpoint if no pause point
                checkpoint_tuple = self.compiled_graph.checkpointer.get_tuple(config)
                if checkpoint_tuple:
                    state = self._extract_state_from_checkpoint(checkpoint_tuple)
                    if state is not None:
                        log.info(
                            "[invoke] Loaded state from latest checkpoint "
                            f"with turn_number={state.get('turn_number', 0)}"
                        )
                        return state
        except Exception as e:
            log.debug(f"[invoke] Checkpoint load failed ({e}); creating fresh state")

        # Create fresh state if no checkpoint found
        log.info(f"[invoke] Creating fresh state for session_id={session_id}")
        return self._new_session_state()

    def _extract_state_from_checkpoint(self, checkpoint_tuple: Any) -> Optional[dict[str, Any]]:
        """Extract state dict from a CheckpointTuple.

        Args:
            checkpoint_tuple: CheckpointTuple from checkpointer.get_tuple()

        Returns:
            State dict, or None if extraction fails
        """
        try:
            checkpoint_data = checkpoint_tuple.checkpoint
            if isinstance(checkpoint_data, dict) and "channel_values" in checkpoint_data:
                # LangGraph's full checkpoint format has channel_values
                return checkpoint_data.get("channel_values", {})
            elif isinstance(checkpoint_data, dict) and "values" in checkpoint_data:
                # Fallback for alternate checkpoint format
                return checkpoint_data["values"]
        except Exception as e:
            log.debug(f"[invoke] State extraction failed: {e}")

        return None

    def _new_session_state(self) -> dict[str, Any]:
        """Create fresh session state. Override in subclass if needed."""
        raise NotImplementedError
