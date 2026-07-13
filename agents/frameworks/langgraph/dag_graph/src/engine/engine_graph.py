"""Generic state machine graph base class for LangGraph.

Provides:
  • EngineGraph — base class for state machine workflows
  • Router, guardrail, and handler dispatch patterns
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# ERROR HANDLING & SAFETY
# ─────────────────────────────────────────────────────────────────────────

def safe_node(func: Callable) -> Callable:
    """Wrap node function with error handling.

    Catches exceptions and returns error state without propagating. If func
    declares a second parameter, the LangGraph invocation config is forwarded
    to it — used by nodes that need a per-invocation runtime setting (e.g.
    router timeout) without persisting it in the checkpointed session state.
    """
    accepts_config = len(inspect.signature(func).parameters) > 1

    @wraps(func)
    def wrapper(state: dict[str, Any], config: Optional[RunnableConfig] = None) -> dict[str, Any]:
        try:
            return func(state, config) if accepts_config else func(state)
        except Exception as e:
            log.error(f"Node {func.__name__} failed: {e}", exc_info=True)
            return {
                **state,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "status": "error",
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
      • max_history_turns — message retention policy (default: 10). Lives
        here, not in session state, because it's the same for every session
        of this graph rather than data that changes as a session progresses.
    """

    # Subclasses must override these
    state_enum: type = None
    terminal_states: set = set()
    handler_map: dict[Any, Callable] = {}
    semantic_router: Optional[Any] = None
    max_history_turns: int = 10

    # Set externally by the domain's build_graph() factory function after
    # construction (e.g. graph.compiled_graph = graph.build_graph(...)) —
    # declared as Any (not Optional[Any]) so mypy knows the attribute
    # exists without treating every .compiled_graph.foo() call as a
    # possible None-access; every real call site only ever runs after
    # build_graph() has set it.
    compiled_graph: Any = None

    def __init__(self) -> None:
        # Per-thread lock, keyed by thread_id — serializes concurrent
        # aemit_event calls for the SAME thread within this process, so a
        # duplicate/racing delivery can't interleave with an in-flight one.
        # Correct only within a single long-lived process (the same
        # tradeoff already accepted for SqliteCheckpointer's async
        # serialization) — a real multi-worker deployment would need
        # cross-process locking instead.
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # Lazily constructed on first use by aemit_event (mirrors
        # DefaultSemanticRouter._get_llm()'s lazy-load pattern) — a domain's
        # build_graph() may assign its own EventLedger(ledger_dir=...)
        # before that, matching how compiled_graph/semantic_router are set
        # externally after construction.
        self._ledger: Optional[Any] = None

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

    def _router_node(
        self, state: dict[str, Any], config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Route: current_state → proposed_next via semantic or code router.

        If semantic_router is available, uses LLM routing. Otherwise falls back
        to pure code routing via routing table.

        Args:
            state: State dict with current_state set
            config: LangGraph invocation config; configurable.router_timeout_sec
                overrides the semantic router's default LLM call timeout

        Returns:
            Updated state with proposed_next set
        """
        current = self._get_current_state(state)

        # Try semantic router first (if available) — but never for a
        # system-sourced turn. An LLM can never decide a system-event
        # transition: this is structural, not a prompt-level instruction,
        # so it holds regardless of whether a semantic_router is attached.
        if self.semantic_router is not None and state.get("current_event_source") != "system":
            try:
                # Extract arguments for semantic router
                current_state = state.get("current_state", "init")
                input_message = state.get("input_message", "")
                history = state.get("messages", [])
                timeout_sec = (config or {}).get("configurable", {}).get("router_timeout_sec", 10.0)

                # Get allowed next states from state machine
                allowed_states = self._get_allowed_states(current)

                router_decision = self.semantic_router.route(
                    current_state=current_state,
                    input_message=input_message,
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
                    "proposed_next": proposal_val,
                    "semantic_context": {
                        "entities": router_decision.semantic_entities,
                        "intents": router_decision.semantic_intents,
                    },
                    "router_confidence": router_decision.confidence,
                    "audit_trail": [
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
            "proposed_next": proposal_val,
            "audit_trail": [f"router: code {current} → {proposal}"],
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
            # proposed_next is unchanged from what the router set — no need to re-set it.
            target = proposed.value if hasattr(proposed, "value") else proposed
            new_state = {
                "fallback_depth": 0,
                "audit_trail": [f"guardrail PASS → {proposed}"],
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
            target = fallback_val
            new_state = {
                "proposed_next": fallback_val,
                "error_message": result.reason,
                "fallback_depth": fallback_depth,
                "audit_trail": [
                    f"guardrail FAIL → {proposed} ({result.reason}) → fallback {fallback_val}"
                ],
            }

        if does_state_wait_for_input(target):
            # Parking at a blocking state is a normal pause, not a failure —
            # and it's never the ERROR state (ERROR isn't waits_for_input).
            new_state["current_state"] = target
            new_state["status"] = "ok"

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
    # HANDLER DISPATCH (centralizes what every handler call needs, so
    # handlers never set current_state/status themselves)
    # ─────────────────────────────────────────────────────────────────────────

    def _dispatch_handler(self, state_val: Any, state: dict[str, Any]) -> dict[str, Any]:
        """Run the handler registered for state_val and stamp its result.

        A handler is only ever run because state_val's node was dispatched to
        (whether by the graph or, on multi-turn resume, directly by invoke()),
        so current_state is fully determined by state_val — no handler needs
        to (or should) set it. status flips to "error" only when state_val is
        the domain's designated ERROR state; every other dispatch resets it
        to "ok", so a later successful state self-heals a prior error.

        Returns ONLY the handler's delta plus the stamped fields — never a
        full merged state. This matters for reducer-backed fields (messages,
        audit_trail): a handler dispatch never touches them, so they must be
        absent from this delta, not resubmitted via a full state spread —
        resubmitting an unchanged list through a reducer is not a no-op (e.g.
        add_messages would treat it as new messages to merge/append).

        Args:
            state_val: The state enum whose handler should run
            state: Current state dict, passed through to the handler to read

        Returns:
            The handler's delta, plus current_state/status stamped on top
        """
        handler_fn = self.handler_map[state_val]
        result = handler_fn(state)
        return {
            **result,
            "current_state": state_val.value,
            "status": "error" if state_val == self.state_enum.ERROR else "ok",
        }

    def _make_handler_node(self, state_val: Any) -> Callable:
        """Build a 1-arg node function bound to state_val, for add_node()."""

        def node(state: dict[str, Any]) -> dict[str, Any]:
            return self._dispatch_handler(state_val, state)

        return node

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
        for state_val in self.handler_map:
            g.add_node(state_val.value, safe_node(self._make_handler_node(state_val)))

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

        The state passed into each compiled_graph.invoke() call here has
        audit_trail/messages reset to their identity ([]) first — the state
        we're holding is always the (already checkpointed) result of a
        prior invoke() call on this same thread, so LangGraph already has
        the true accumulated value; feeding it back in full would
        double-count it against what's already persisted. See
        _get_or_init_state's docstring for the same reasoning applied there.

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
            call_state = {**state, "audit_trail": [], "messages": []}
            state = self.compiled_graph.invoke(call_state, config=config)
            iters += 1

        if iters >= max_auto_iters:
            log.warning("[auto_progress] Reached max iterations (%d); stopping", max_auto_iters)

        return state

    async def _auto_progress_langgraph_async(
        self,
        state: dict[str, Any],
        config: dict[str, Any],
        max_auto_iters: int = 10,
    ) -> dict[str, Any]:
        """Async twin of _auto_progress_langgraph — identical logic, using
        await compiled_graph.ainvoke(...) so ainvoke() never falls back to
        a blocking sync graph call partway through a turn."""
        from src.engine.handler_registry import does_state_wait_for_input

        iters = 0

        while iters < max_auto_iters:
            current = state.get("current_state", "init")

            if current in self.terminal_states:
                log.debug(f"[auto_progress] Stopped at terminal state {current}")
                break

            if does_state_wait_for_input(current):
                log.debug(f"[auto_progress] Stopped at input-waiting state {current}")
                break

            log.debug(f"[auto_progress] {current} is non-blocking; continuing...")
            call_state = {**state, "audit_trail": [], "messages": []}
            state = await self.compiled_graph.ainvoke(call_state, config=config)
            iters += 1

        if iters >= max_auto_iters:
            log.warning("[auto_progress] Reached max iterations (%d); stopping", max_auto_iters)

        return state

    def invoke(
        self,
        user_id: str,
        session_id: str,
        input_message: str,
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
        6. Trim message history
        7. Append turn to message history
        8. Return turn response

        Args:
            user_id: Caller identity (for audit)
            session_id: Multi-turn session ID
            input_message: User's input text
            state_delta: Optional extra fields to merge into state alongside
                input_message (e.g. business payload supplied this turn),
                before the graph is invoked
            timeout_sec: LLM router timeout

        Returns:
            Final SessionState after this turn's execution.
        """
        from src.engine.input_validation import InputValidationError

        try:
            escaped = self._prepare_input(input_message)

            # Thread ID for checkpointing across turns
            thread_id = self._thread_id(user_id, session_id)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "router_timeout_sec": timeout_sec,
                }
            }

            # Get or initialize state (use thread_id format for consistency)
            state = self._get_or_init_state(session_id, user_id=user_id)

            # Prepare turn metadata
            state["input_message"] = escaped
            current_turn_number = state.get("turn_number", 0) + 1
            state["turn_number"] = current_turn_number
            state["user_id"] = user_id
            state["session_id"] = session_id

            # Merge caller-supplied state updates before the graph runs
            if state_delta is not None:
                state.update(state_delta)

            # Resuming at a blocking state: run its own handler with the fresh
            # input_message before the graph executes. The router only proposes
            # transitions FORWARD from current_state via the routing table, so
            # without this it would skip past the blocking state on resume and
            # silently discard the new turn's input (e.g. an upload) instead of
            # letting the handler process it.
            from src.engine.handler_registry import does_state_wait_for_input

            current_state_str = state.get("current_state", "init")
            if does_state_wait_for_input(current_state_str):
                delta = self._dispatch_handler(self.state_enum(current_state_str), state)
                # delta["audit_trail"] is just the new entry (not the full
                # accumulated list) — plain-merging it here is intentional:
                # it correctly overwrites state's audit_trail (which
                # _get_or_init_state already reset to [] for a
                # loaded-from-checkpoint state) with just this new entry,
                # ready for the compiled_graph.invoke() call below. See
                # _get_or_init_state's docstring for why.
                state = {**state, **delta}

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

            # Trim history (this runs outside the graph's node/reducer cycle,
            # directly on the accumulated list, so plain list ops are correct)
            messages = state.get("messages", [])
            if len(messages) > self.max_history_turns:
                dropped = len(messages) - self.max_history_turns
                messages = messages[-self.max_history_turns:]
                state["messages"] = messages
                log.info(
                    f"[invoke] Trimmed {dropped} old messages; keeping {self.max_history_turns}"
                )

            # New messages for this turn, per the output_messages convention
            # (§11) — see _build_new_messages(). A system-sourced turn with
            # nothing to say produces new_messages == [], leaving messages
            # completely untouched below.
            new_messages = self._build_new_messages(escaped, state)
            if new_messages:
                messages.extend(new_messages)
                state["messages"] = messages

            # Persist the new messages now. Nothing does this automatically:
            # unlike audit_trail (written by graph nodes, so every node's
            # checkpoint already includes it), messages are appended here in
            # plain Python, after the graph has already finished running and
            # checkpointing for this turn. Without this, the next turn loads
            # a checkpoint from before these messages existed, and message
            # history silently resets every turn. update_state() applies the
            # update through the "messages" channel's add_messages reducer
            # and persists it via the checkpointer, exactly like a node would
            # — so only the new messages are passed, not the full list.
            # This must run BEFORE the pause-point save below, so the pause
            # point marks the checkpoint that actually includes them.
            checkpointer = getattr(self.compiled_graph, "checkpointer", None)
            if checkpointer and new_messages:
                try:
                    self.compiled_graph.update_state(config, {"messages": new_messages})
                except Exception as e:
                    log.debug(f"[invoke] Could not persist new messages: {e}")

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
                "error_message": str(e),
                "status": "error",
                "current_state": None,
                "turn_number": 0,
                "semantic_context": {},
                "router_confidence": 0.0,
            }
        except Exception as e:
            log.exception("[invoke] Error: %s", e)
            return {
                "error_message": str(e),
                "status": "error",
                "current_state": "error",
                "turn_number": 0,
                "semantic_context": {},
                "router_confidence": 0.0,
            }

    async def ainvoke(
        self,
        user_id: str,
        session_id: str,
        input_message: str,
        state_delta: Optional[dict[str, Any]] = None,
        timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        """Async twin of invoke() — same algorithm, using
        await compiled_graph.ainvoke(...) throughout (including
        auto-progression), sharing _thread_id()/_prepare_input() with
        invoke() so both agree on thread identity and input handling.

        Called by aemit_event() and arun_to_completion() (later phases),
        always with user_id="" — see invoke()'s docstring for the shared
        argument semantics.
        """
        from src.engine.input_validation import InputValidationError

        try:
            escaped = self._prepare_input(input_message)

            thread_id = self._thread_id(user_id, session_id)
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "router_timeout_sec": timeout_sec,
                }
            }

            state = self._get_or_init_state(session_id, user_id=user_id)

            state["input_message"] = escaped
            current_turn_number = state.get("turn_number", 0) + 1
            state["turn_number"] = current_turn_number
            state["user_id"] = user_id
            state["session_id"] = session_id

            if state_delta is not None:
                state.update(state_delta)

            from src.engine.handler_registry import does_state_wait_for_input

            current_state_str = state.get("current_state", "init")
            if does_state_wait_for_input(current_state_str):
                delta = self._dispatch_handler(self.state_enum(current_state_str), state)
                state = {**state, **delta}

            if hasattr(self, "_init_router"):
                self._init_router()

            state = await self.compiled_graph.ainvoke(state, config=config)

            state["turn_number"] = current_turn_number

            state = await self._auto_progress_langgraph_async(state, config)

            state["turn_number"] = current_turn_number

            messages = state.get("messages", [])
            if len(messages) > self.max_history_turns:
                dropped = len(messages) - self.max_history_turns
                messages = messages[-self.max_history_turns:]
                state["messages"] = messages
                log.info(
                    f"[ainvoke] Trimmed {dropped} old messages; keeping {self.max_history_turns}"
                )

            # New messages for this turn, per the output_messages convention
            # (§11) — see _build_new_messages(). A system-sourced turn with
            # nothing to say produces new_messages == [], leaving messages
            # completely untouched below.
            new_messages = self._build_new_messages(escaped, state)
            if new_messages:
                messages.extend(new_messages)
                state["messages"] = messages

            # Message/pause-point persistence stays on the sync checkpointer
            # API here (update_state/export_session/save_pause_point are
            # JsonCheckpointer-specific bookkeeping helpers, not part of
            # LangGraph's core async execution path fixed in an earlier
            # phase) — mirrors invoke()'s equivalent block exactly.
            checkpointer = getattr(self.compiled_graph, "checkpointer", None)
            if checkpointer and new_messages:
                try:
                    self.compiled_graph.update_state(config, {"messages": new_messages})
                except Exception as e:
                    log.debug(f"[ainvoke] Could not persist new messages: {e}")

            from src.engine.handler_registry import does_state_wait_for_input

            if does_state_wait_for_input(state.get("current_state")):
                checkpointer = getattr(self.compiled_graph, "checkpointer", None)
                if checkpointer:
                    try:
                        session_data = checkpointer.export_session(thread_id)
                        if session_data:
                            checkpoint_id = session_data.get("latest_checkpoint_id")
                            if checkpoint_id:
                                checkpointer.save_pause_point(config, checkpoint_id)
                                log.info(
                                    f"[ainvoke] Auto-saved pause point "
                                    f"{checkpoint_id} at {state['current_state']}"
                                )
                    except Exception as e:
                        log.debug(f"[ainvoke] Could not auto-save pause point: {e}")

            return state

        except InputValidationError as e:
            return {
                "error_message": str(e),
                "status": "error",
                "current_state": None,
                "turn_number": 0,
                "semantic_context": {},
                "router_confidence": 0.0,
            }
        except Exception as e:
            log.exception("[ainvoke] Error: %s", e)
            return {
                "error_message": str(e),
                "status": "error",
                "current_state": "error",
                "turn_number": 0,
                "semantic_context": {},
                "router_confidence": 0.0,
            }

    async def aemit_event(
        self,
        thread_id: str,
        source: str,
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
        event_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """The unified event gate — both a human chat message and a
        system/webhook/timeout event flow through here.

        thread_id IS identity (never source): ainvoke() is always called
        with user_id="" from here, so a human turn and a system turn for
        the same thread_id land in the same checkpointed session, never
        two.

        Returns a dict with a "status" key distinguishable by callers:
        "ok", "duplicate", "ignored", "not_waiting", "already_terminal",
        or "waiting" (from _describe_wait — a human message against a
        wait_kind="system_event" state, read-only, state untouched).
        """
        if source == "system" and not event_id:
            raise ValueError("system-sourced events must supply event_id")

        if self._ledger is None:
            from src.engine.event_ledger import EventLedger

            self._ledger = EventLedger()

        async with self._locks[thread_id]:
            already_seen = False
            if source == "system" and event_id is not None:
                already_seen = await self._ledger.is_processed(event_id)
            if already_seen:
                return {"status": "duplicate", "event_id": event_id}

            from src.engine.handler_registry import does_state_wait_for_input, get_handler_metadata

            state = self._get_or_init_state(session_id=thread_id, user_id="")
            current = state.get("current_state", "init")
            meta = get_handler_metadata(current)

            if current in self.terminal_states:
                if event_id:
                    await self._ledger.mark_processed(event_id)
                return {"status": "already_terminal", "current_state": current}
            if not does_state_wait_for_input(current):
                return {"status": "not_waiting", "current_state": current}

            if source == "human":
                if meta and meta.wait_kind == "system_event":
                    return self._describe_wait(state, meta)
                result = await self.ainvoke(
                    user_id="",
                    session_id=thread_id,
                    input_message=(payload or {}).get("text", ""),
                    state_delta={"current_event_source": "human", "current_event_type": "message"},
                )
                return {"status": "ok", **result}

            # source == "system"
            expected = meta.expected_events or [] if meta else []
            legal = meta and meta.wait_kind != "human" and event_type in expected
            if not legal:
                # Consume the delivery so a legitimate retry of a
                # since-moved-past event gets reclassified as "ignored"
                # rather than retried forever.
                if event_id:
                    await self._ledger.mark_processed(event_id)
                return {"status": "ignored", "current_state": current, "event_type": event_type}

            result = await self.ainvoke(
                user_id="",
                session_id=thread_id,
                input_message="",
                state_delta={
                    **(payload or {}),
                    "current_event_source": "system",
                    "current_event_type": event_type,
                },
            )
            # Marked AFTER success, not before: a crash between mark and
            # invoke would otherwise silently swallow a provider's retry
            # with no recovery path. A retry landing in that gap here just
            # gets correctly reclassified as "ignored" next time (state
            # already moved past that event), not reprocessed.
            if event_id:
                await self._ledger.mark_processed(event_id)
            return {"status": "ok", **result}

    def _describe_wait(self, state: dict[str, Any], meta: Any) -> dict[str, Any]:
        """Read-only reply for a human message against a wait_kind="system_event"
        state — state stays untouched, but the caller gets more than silence."""
        return {
            "status": "waiting",
            "current_state": state.get("current_state"),
            "wait_kind": meta.wait_kind,
            "expected_events": meta.expected_events,
        }

    def _build_new_messages(self, escaped: str, state: dict[str, Any]) -> list[Any]:
        """Per-turn message convention (§11) — replaces the previous
        unconditional [HumanMessage(...), AIMessage(f"Transitioned to
        {state}")] append on every turn, shared by invoke() and ainvoke().

        A handler's output_messages (if any) become AI messages; otherwise
        a human-sourced turn falls back to the generic status message it
        always got (backward compatible: no existing handler sets
        output_messages, so every docprocessing turn still hits this
        branch, producing the same message it does today). A
        system-sourced turn with nothing to say produces no messages at
        all — no meaningless empty-turn noise in the checkpoint.
        """
        turn = state["turn_number"]
        current_state = state["current_state"]

        new_messages: list[Any] = []
        if escaped:
            new_messages.append(
                HumanMessage(content=escaped, additional_kwargs={"turn_number": turn})
            )

        outs = state.get("output_messages") or []
        if outs:
            kwargs = {"turn_number": turn, "state": current_state}
            new_messages.extend(AIMessage(content=m, additional_kwargs=kwargs) for m in outs)
        elif state.get("current_event_source", "human") == "human":
            new_messages.append(
                AIMessage(
                    content=f"Transitioned to {current_state}",
                    additional_kwargs={
                        "turn_number": turn,
                        "state": current_state,
                        "semantic_context": {
                            "entities": state.get("semantic_context", {}).get("entities", {}),
                            "intents": state.get("semantic_context", {}).get("intents", []),
                        },
                    },
                )
            )
        # else: system-sourced turn with nothing to say — messages untouched.

        return new_messages

    def _thread_id(self, user_id: str, session_id: str) -> str:
        """Compute the checkpointer thread_id for (user_id, session_id).

        The single source of truth for this formula — invoke() and
        _get_or_init_state() previously computed it independently and
        disagreed whenever user_id was falsy (invoke() always produced
        f":{session_id}", _get_or_init_state() produced session_id), so a
        turn's state would load from one thread_id and checkpoint under a
        different one, silently starting a fresh session every call. Any
        future caller (e.g. ainvoke()) must go through this method rather
        than re-deriving the formula.
        """
        return f"{user_id}:{session_id}" if user_id else session_id

    def _prepare_input(self, input_message: str) -> str:
        """Validate+escape non-empty input; skip validation entirely for
        empty input.

        System-sourced and bg-run turns have nothing a human said this
        turn, so they call invoke()/ainvoke() with input_message="" by
        design (see aemit_event/arun_to_completion in later phases).
        validate_turn_input() unconditionally rejects an empty string as
        invalid input, which would reject every one of those calls before
        the graph ever runs — that's a validation-scope bug, not a
        legitimate rejection, since there is no human-authored text to
        validate in the first place.
        """
        if not input_message:
            return ""
        from src.engine.input_validation import escape_for_llm, validate_turn_input

        validate_turn_input(input_message)
        return escape_for_llm(input_message)

    def _get_or_init_state(self, session_id: str, user_id: str = "") -> dict[str, Any]:
        """Get existing state or create fresh state for session.

        Auto-loads from pause point if available, otherwise tries latest checkpoint.

        A state loaded from a checkpoint has its reducer-backed fields
        (audit_trail, messages) reset to their identity (empty list) before
        being returned. This matters because invoke() feeds the result of
        this method into compiled_graph.invoke() shortly after: LangGraph
        merges whatever we pass for a reducer-backed channel against what
        it has ALREADY persisted for this thread — it does not treat our
        input as a replacement. Everything in a just-loaded checkpoint's
        audit_trail/messages is, by definition, already persisted, so
        feeding it back in full would double-count it. Any handler that
        runs directly on this state before the graph does (see invoke()'s
        resume-at-blocking-state step) must therefore also only ever set
        these fields to new entries, never a full accumulated copy.

        Args:
            session_id: Session identifier
            user_id: User identifier (used for thread_id in checkpointing)

        Returns:
            SessionState dict
        """
        thread_id = self._thread_id(user_id, session_id)

        # Try to load from pause point first (hidden auto-resumption)
        try:
            if hasattr(self.compiled_graph, "checkpointer") and self.compiled_graph.checkpointer:
                config = {"configurable": {"thread_id": thread_id}}

                # Check for pause point first — but only trust it if it's
                # still the latest checkpoint. save_pause_point() is only
                # called when a turn ends AT a wait state (see invoke()'s
                # "AUTO-SAVE PAUSE POINT" block); a turn that resumes past
                # it and auto-progresses all the way to a non-wait terminal
                # state (e.g. onboarding's AWAIT_* -> a terminal COMPLETE
                # that never itself pauses) never records a new pause
                # point, leaving the OLD one in place — pointing at a state
                # the thread has since moved past. latest_checkpoint_id is
                # unconditionally updated on every put(), so comparing
                # against it is how a stale pause point gets detected
                # (found while making aemit_event's already_terminal check
                # actually observable — this bug predates this design and
                # already latently affects any invoke()-only thread that
                # reaches completion via auto-progress from a pause point).
                pause_checkpoint_id = self.compiled_graph.checkpointer.get_pause_point(config)
                if pause_checkpoint_id:
                    checkpointer = self.compiled_graph.checkpointer
                    export_session = getattr(checkpointer, "export_session", None)
                    session_data = export_session(thread_id) if export_session else None
                    latest_checkpoint_id = (session_data or {}).get("latest_checkpoint_id")
                    pause_point_is_current = (
                        latest_checkpoint_id is None or latest_checkpoint_id == pause_checkpoint_id
                    )
                    if pause_point_is_current:
                        log.info(
                            f"[invoke] Found pause point {pause_checkpoint_id}; "
                            "resuming from there"
                        )
                        config["configurable"]["checkpoint_id"] = pause_checkpoint_id
                        checkpoint_tuple = self.compiled_graph.checkpointer.get_tuple(config)
                        if checkpoint_tuple:
                            pause_state = self._extract_state_from_checkpoint(checkpoint_tuple)
                            if pause_state is not None:
                                return {**pause_state, "audit_trail": [], "messages": []}
                    else:
                        log.info(
                            f"[invoke] Pause point {pause_checkpoint_id} is stale "
                            f"(latest is {latest_checkpoint_id}); using latest instead"
                        )

                # Fallback to latest checkpoint if no pause point
                checkpoint_tuple = self.compiled_graph.checkpointer.get_tuple(config)
                if checkpoint_tuple:
                    state = self._extract_state_from_checkpoint(checkpoint_tuple)
                    if state is not None:
                        log.info(
                            "[invoke] Loaded state from latest checkpoint "
                            f"with turn_number={state.get('turn_number', 0)}"
                        )
                        return {**state, "audit_trail": [], "messages": []}
        except Exception as e:
            log.debug(f"[invoke] Checkpoint load failed ({e}); creating fresh state")

        # Create fresh state if no checkpoint found — nothing is persisted
        # yet for this thread, so its initial audit_trail is genuinely new.
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

    def get_active_sessions(self) -> list[dict[str, Any]]:
        """Enumerate sessions as [{"thread_id": ..., "state": {...}}].

        checkpointer.get_sessions() (JsonCheckpointer-specific) returns raw
        session-file dicts — {"thread_id", "checkpoints": {checkpoint_id:
        {"checkpoint", "metadata", ...}}, "latest_checkpoint_id",
        "pause_checkpoint_id"?, ...} — with no top-level "state" key.
        Callers like a timeout sweep need the actual state dict, not that
        internal shape, so this reuses the same checkpoint-unwrapping logic
        _get_or_init_state() already relies on rather than making every
        caller reach into the checkpointer's file format directly.
        """
        from langgraph.checkpoint.base import CheckpointTuple

        checkpointer = getattr(self.compiled_graph, "checkpointer", None)
        get_sessions = getattr(checkpointer, "get_sessions", None)
        if get_sessions is None:
            return []

        results = []
        for session in get_sessions():
            checkpoints = session.get("checkpoints") or {}
            checkpoint_id = session.get("pause_checkpoint_id") or session.get(
                "latest_checkpoint_id"
            )
            cp_entry = checkpoints.get(checkpoint_id) if checkpoint_id else None
            if not cp_entry:
                continue
            checkpoint_tuple = CheckpointTuple(
                config={},
                checkpoint=cp_entry.get("checkpoint", {}),
                metadata=cp_entry.get("metadata", {}),
            )
            state = self._extract_state_from_checkpoint(checkpoint_tuple)
            if state is not None:
                results.append({"thread_id": session["thread_id"], "state": state})
        return results

    def _new_session_state(self) -> dict[str, Any]:
        """Create fresh session state. Override in subclass if needed."""
        raise NotImplementedError
