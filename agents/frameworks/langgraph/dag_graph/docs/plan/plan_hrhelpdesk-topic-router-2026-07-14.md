# Wire hrhelpdesk Topic Classification into ChatEngineGraph's TopicRouter Seam — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish wiring `ChatEngineGraph`'s already-declared-but-unused `TopicRouter`/`classify_utterance` seam by adding a reusable `DefaultTopicRouter` engine base class and a domain `HelpdeskSemanticRouter`, replacing the ad hoc `run_router()` calls currently inline in `src/hrhelpdesk/handlers.py`.

**Architecture:** `src/engine/router.py` gains `DefaultTopicRouter` (topic+confidence classification via `engine/chains.py::make_chain`, mirroring the existing `BaseSemanticRouter`/`DefaultSemanticRouter` pair but for topics instead of states). `src/hrhelpdesk/router.py` (new) adds `HelpdeskSemanticRouter(DefaultTopicRouter)` supplying the domain's schema/prompt. `hrhelpdesk/graph.py` and `hrhelpdesk/handlers.py` wire that router into `Graph.topic_router` / the handlers' `_topic_fanout` instance, and `_route_human_message`/`handle_topic_booking` call `classify_utterance(...)` instead of the free `run_router()` function, which is deleted along with `route_chain`.

**Tech Stack:** Python, LangChain/LangGraph, Pydantic, pytest (`pytest-asyncio`), `uv`, `ruff`.

**Spec:** `docs/design/design_hrhelpdesk-topic-router-2026-07-14.md`

**Run tests as:** `PYTHONPATH=. uv run pytest <path>` — a bare `pytest` fails every module with `ModuleNotFoundError: No module named 'src'` (see project CLAUDE.md).

---

### Task 1: `DefaultTopicRouter` engine base class

**Files:**
- Modify: `src/engine/router.py:1-17` (imports/module docstring), append new class at end of file (currently ends at line 287)
- Test: `tests/test_topic_router.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_topic_router.py`:

```python
"""Tests for DefaultTopicRouter (engine-level topic+confidence classifier base)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from src.engine.chat_engine_graph import TopicDecision
from src.engine.router import DefaultTopicRouter


class _MockTopicOutput(BaseModel):
    topic: str
    confidence: float


class _MockTopicRouter(DefaultTopicRouter):
    output_schema = _MockTopicOutput
    system_prompt = "Classify the topic."
    chain_name = "MockTopicRouterChain"


def _with_fake_chain(router: DefaultTopicRouter, fake_chain: MagicMock) -> DefaultTopicRouter:
    router._chain = fake_chain
    return router


def test_requires_output_schema():
    class _NoSchemaRouter(DefaultTopicRouter):
        pass

    with pytest.raises(NotImplementedError):
        _NoSchemaRouter()


def test_defaults():
    router = _MockTopicRouter()
    assert router.model_role == "router"
    assert router.unclear_topic == "unclear"


def test_classify_returns_topic_decision():
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = _MockTopicOutput(topic="faq", confidence=0.92)
    router = _with_fake_chain(_MockTopicRouter(), fake_chain)

    decision = router.classify("How much PTO do I get?")

    assert decision == TopicDecision(topic="faq", confidence=0.92)
    fake_chain.invoke.assert_called_once_with({"input": "How much PTO do I get?"})


def test_classify_normalizes_dict_response():
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {"topic": "booking", "confidence": 0.8}
    router = _with_fake_chain(_MockTopicRouter(), fake_chain)

    decision = router.classify("book a desk")

    assert decision.topic == "booking"
    assert decision.confidence == 0.8


def test_classify_clamps_confidence():
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = _MockTopicOutput(topic="faq", confidence=1.5)
    router = _with_fake_chain(_MockTopicRouter(), fake_chain)

    decision = router.classify("anything")

    assert decision.confidence == 1.0


def test_classify_falls_back_to_unclear_on_exception():
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("boom")
    router = _with_fake_chain(_MockTopicRouter(), fake_chain)

    decision = router.classify("anything")

    assert decision == TopicDecision(topic="unclear", confidence=0.0)


def test_get_chain_builds_via_make_chain(monkeypatch):
    captured = {}

    def fake_make_chain(*, name, system_prompt, output_schema, model_id):
        captured.update(
            name=name,
            system_prompt=system_prompt,
            output_schema=output_schema,
            model_id=model_id,
        )
        return MagicMock()

    monkeypatch.setattr("src.engine.router.make_chain", fake_make_chain)
    monkeypatch.setattr("src.engine.router.get_model", lambda role: f"model-for-{role}")

    router = _MockTopicRouter()
    router._get_chain()

    assert captured["name"] == "MockTopicRouterChain"
    assert captured["system_prompt"] == "Classify the topic."
    assert captured["output_schema"] is _MockTopicOutput
    assert captured["model_id"] == "model-for-router"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_topic_router.py -v`
Expected: collection error — `ImportError: cannot import name 'DefaultTopicRouter' from 'src.engine.router'`

- [ ] **Step 3: Implement `DefaultTopicRouter`**

Modify `src/engine/router.py` lines 1-17 (module docstring + imports) — current:

```python
"""
Semantic router base classes for LLM-powered state transitions.

Provides:
- RouterDecision: Output structure with decision metadata
- BaseSemanticRouter: Abstract interface
- DefaultSemanticRouter: Concrete implementation with common LLM logic
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)
```

Replace with:

```python
"""
Semantic router base classes for LLM-powered state transitions.

Provides:
- RouterDecision: Output structure with decision metadata
- BaseSemanticRouter: Abstract interface
- DefaultSemanticRouter: Concrete implementation with common LLM logic
- DefaultTopicRouter: Concrete implementation for topic+confidence
  classification (chatbot hub routing, as opposed to state routing)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.engine.chains import get_model, make_chain
from src.engine.chat_engine_graph import TopicDecision

log = logging.getLogger(__name__)
```

Append this class at the end of `src/engine/router.py` (after `DefaultSemanticRouter.route()`, which currently ends the file at line 287):

```python


class DefaultTopicRouter:
    """Concrete ``TopicRouter`` (structural) for LLM topic+confidence classification.

    Unlike ``DefaultSemanticRouter`` (which proposes a graph *state* and rolls
    its own LLM call), this classifies an utterance into a *topic* and reuses
    ``engine.chains.make_chain`` — the same helper every domain's
    ``chains.py`` already uses — instead of a second hand-rolled
    LLM-invocation path.

    Subclasses set:
        output_schema: Pydantic model with ``topic`` (str|Enum) and
            ``confidence`` (float) fields. Required.
        system_prompt / get_instructions(): classification system prompt.
        model_role: passed to ``get_model(role)`` for model selection
            (default ``"router"``).
        unclear_topic: topic value used when classification fails or the
            model itself returns it (default ``"unclear"``).
        chain_name: name passed to ``make_chain`` for its process-level
            cache (default ``"TopicRouterChain"`` — override per subclass
            so multiple topic routers don't collide in the shared cache).

    Example:
        class MyTopicRouter(DefaultTopicRouter):
            output_schema = MyRouterOutput
            chain_name = "MyTopicRouterChain"

            def get_instructions(self):
                return "Classify the message into: a | b | unclear..."
    """

    output_schema: type = None  # Subclasses MUST set this to a Pydantic model
    system_prompt: str = ""  # or override get_instructions()
    model_role: str = "router"
    unclear_topic: str = "unclear"
    chain_name: str = "TopicRouterChain"

    def __init__(self) -> None:
        if self.output_schema is None:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} must set output_schema to a Pydantic model"
            )
        self._chain = None  # Lazy-built in _get_chain()

    def get_instructions(self) -> str:
        """Return the LLM system prompt. Subclasses override or set ``system_prompt``."""
        return self.system_prompt

    def _get_chain(self):
        """Lazily build (and cache) the classification chain via ``make_chain``."""
        if self._chain is None:
            self._chain = make_chain(
                name=self.chain_name,
                system_prompt=self.get_instructions(),
                output_schema=self.output_schema,
                model_id=get_model(self.model_role),
            )
        return self._chain

    def classify(
        self,
        input_message: str,
        history: Optional[list] = None,
    ) -> TopicDecision:
        """Classify ``input_message`` into a ``TopicDecision``.

        ``history`` is accepted for ``TopicRouter`` protocol conformance but
        unused by this default implementation. On any failure (LLM error,
        malformed response), falls back to ``TopicDecision(unclear_topic,
        0.0)`` rather than raising — callers already treat unclear/low
        confidence uniformly via ``ChatEngineGraph.should_clarify``, so a
        swallowed error degrades to "ask the user to clarify" instead of
        surfacing as a handler error.
        """
        try:
            raw = self._get_chain().invoke({"input": input_message})
            response = raw if isinstance(raw, self.output_schema) else self.output_schema(**raw)
            topic = response.topic
            topic_val = topic.value if hasattr(topic, "value") else str(topic)
            confidence = max(0.0, min(1.0, float(response.confidence)))
            return TopicDecision(topic=topic_val, confidence=confidence)
        except Exception as exc:
            log.exception("[DefaultTopicRouter] classify() failed: %s", exc)
            return TopicDecision(topic=self.unclear_topic, confidence=0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_topic_router.py -v`
Expected: 7 passed

- [ ] **Step 5: Ruff check and commit**

Run: `ruff check src/engine/router.py tests/test_topic_router.py --fix`

```bash
git add src/engine/router.py tests/test_topic_router.py
git commit -m "feat(engine): add DefaultTopicRouter for chatbot topic classification"
```

---

### Task 2: `HelpdeskSemanticRouter` domain router

**Files:**
- Create: `src/hrhelpdesk/router.py`
- Test: `tests/test_hrhelpdesk_router.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_hrhelpdesk_router.py`:

```python
"""Tests for HelpdeskSemanticRouter (hrhelpdesk's TopicRouter wiring)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.hrhelpdesk.chains import RouterOutput, RouterTopic
from src.hrhelpdesk.router import HelpdeskSemanticRouter


def test_helpdesk_router_config():
    router = HelpdeskSemanticRouter()
    assert router.output_schema is RouterOutput
    assert router.unclear_topic == RouterTopic.UNCLEAR.value
    assert "faq" in router.get_instructions()
    assert "booking" in router.get_instructions()


def test_helpdesk_router_classify_faq():
    router = HelpdeskSemanticRouter()
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = RouterOutput(topic=RouterTopic.FAQ, confidence=0.88)
    router._chain = fake_chain

    decision = router.classify("How much PTO do I get?")

    assert decision.topic == "faq"
    assert decision.confidence == 0.88


def test_helpdesk_router_classify_falls_back_to_unclear_on_error():
    router = HelpdeskSemanticRouter()
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("boom")
    router._chain = fake_chain

    decision = router.classify("anything")

    assert decision.topic == "unclear"
    assert decision.confidence == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. uv run pytest tests/test_hrhelpdesk_router.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.hrhelpdesk.router'`

- [ ] **Step 3: Implement `HelpdeskSemanticRouter`**

Create `src/hrhelpdesk/router.py`:

```python
"""Topic-classification router for the HR helpdesk chatbot."""

from __future__ import annotations

from src.engine.router import DefaultTopicRouter

from .chains import RouterOutput, RouterTopic


class HelpdeskSemanticRouter(DefaultTopicRouter):
    """Classifies a helpdesk utterance into faq | escalate | booking | unclear."""

    output_schema = RouterOutput
    unclear_topic = RouterTopic.UNCLEAR.value
    chain_name = "HelpdeskRouteChain"

    def get_instructions(self) -> str:
        return """You are a semantic router for an HR helpdesk assistant.

Classify the user's message into exactly one topic:
- faq: policy / benefits / PTO questions answerable from a knowledge base
- escalate: complaints, HR issues, or requests needing a human ticket
- booking: desk / office seat reservations
- unclear: ambiguous or off-topic

Respond ONLY with valid JSON:
{{
  "topic": "faq" | "escalate" | "booking" | "unclear",
  "confidence": <float 0.0-1.0>
}}"""


__all__ = ["HelpdeskSemanticRouter"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. uv run pytest tests/test_hrhelpdesk_router.py -v`
Expected: 3 passed

- [ ] **Step 5: Ruff check and commit**

Run: `ruff check src/hrhelpdesk/router.py tests/test_hrhelpdesk_router.py --fix`

```bash
git add src/hrhelpdesk/router.py tests/test_hrhelpdesk_router.py
git commit -m "feat(hrhelpdesk): add HelpdeskSemanticRouter on DefaultTopicRouter"
```

---

### Task 3: Wire `HelpdeskSemanticRouter` into `Graph`/`handlers`, delete `run_router`, update tests

These files must change together — `handlers.py` stops importing `run_router` in the
same commit that `chains.py` deletes it, otherwise the module fails to import. Run the
full hrhelpdesk test suite only at the end of this task.

**Files:**
- Modify: `src/hrhelpdesk/chains.py:39-80` (delete `route_chain`, `run_router`), `:148-162` (`__all__`)
- Modify: `src/hrhelpdesk/__init__.py` (imports + `__all__`)
- Modify: `src/hrhelpdesk/graph.py:17-33` (add `topic_router` class attribute)
- Modify: `src/hrhelpdesk/handlers.py` (imports, `_topic_fanout` setup, `_route_human_message`, `handle_topic_booking`)
- Modify: `tests/test_hrhelpdesk_flow.py` (imports, `_router` → `_decision` helper, all `run_router` patch sites)

- [ ] **Step 1: Delete `route_chain`/`run_router` from `src/hrhelpdesk/chains.py`**

Delete lines 39-57 (the `route_chain = make_chain(...)` block and the blank line after it) — current:

```python
route_chain = make_chain(
    name="HelpdeskRouteChain",
    system_prompt="""You are a semantic router for an HR helpdesk assistant.

Classify the user's message into exactly one topic:
- faq: policy / benefits / PTO questions answerable from a knowledge base
- escalate: complaints, HR issues, or requests needing a human ticket
- booking: desk / office seat reservations
- unclear: ambiguous or off-topic

Respond ONLY with valid JSON:
{{
  "topic": "faq" | "escalate" | "booking" | "unclear",
  "confidence": <float 0.0-1.0>
}}""",
    output_schema=RouterOutput,
    model_id=get_model("router"),
)

escape_chain = make_chain(
```

Becomes (this text now leads directly into `escape_chain`):

```python
escape_chain = make_chain(
```

Delete the `run_router` function (currently lines 74-81, between `escape_chain`'s closing `)` and `def run_escape`) — current:

```python
def run_router(text: str) -> RouterOutput:
    """Invoke the router chain (patchable in tests)."""
    result = route_chain.invoke({"input": text})
    if isinstance(result, RouterOutput):
        return result
    return RouterOutput(**result)


def run_escape(text: str) -> EscapeOutput:
```

Becomes:

```python
def run_escape(text: str) -> EscapeOutput:
```

Update `__all__` (currently lines 148-162):

```python
__all__ = [
    "RouterTopic",
    "RouterOutput",
    "EscapeOutput",
    "route_chain",
    "escape_chain",
    "run_router",
    "run_escape",
    "create_ticket_tool",
    "check_desk_availability",
    "confirm_booking",
    "faq_agent",
    "escalate_agent",
    "booking_agent",
]
```

Remove `"route_chain"` and `"run_router"`:

```python
__all__ = [
    "RouterTopic",
    "RouterOutput",
    "EscapeOutput",
    "escape_chain",
    "run_escape",
    "create_ticket_tool",
    "check_desk_availability",
    "confirm_booking",
    "faq_agent",
    "escalate_agent",
    "booking_agent",
]
```

- [ ] **Step 2: Update `src/hrhelpdesk/__init__.py`**

Current:

```python
from src.hrhelpdesk.chains import (
    EscapeOutput,
    RouterOutput,
    RouterTopic,
    booking_agent,
    escape_chain,
    escalate_agent,
    faq_agent,
    run_escape,
    run_router,
    route_chain,
)
from src.hrhelpdesk.graph import Graph, build_graph
```

Replace with:

```python
from src.hrhelpdesk.chains import (
    EscapeOutput,
    RouterOutput,
    RouterTopic,
    booking_agent,
    escape_chain,
    escalate_agent,
    faq_agent,
    run_escape,
)
from src.hrhelpdesk.graph import Graph, build_graph
from src.hrhelpdesk.router import HelpdeskSemanticRouter
```

In `__all__`, current:

```python
    "RouterTopic",
    "RouterOutput",
    "EscapeOutput",
    "route_chain",
    "escape_chain",
    "run_router",
    "run_escape",
```

Replace with:

```python
    "RouterTopic",
    "RouterOutput",
    "EscapeOutput",
    "escape_chain",
    "run_escape",
    "HelpdeskSemanticRouter",
```

- [ ] **Step 3: Wire `topic_router` into `src/hrhelpdesk/graph.py`**

Current (lines 1-33):

```python
"""Domain-specific LangGraph configuration for the HR helpdesk chatbot."""

from __future__ import annotations

from typing import Any, Callable

from src.engine.chat_engine_graph import ChatEngineGraph
from src.engine.handler_registry import get_handler_metadata
from src.engine.json_checkpointer import JsonCheckpointer

from .guardrails import guardrails
from .handlers import handler_map, set_ledger
from .session_state import HelpdeskState, new_helpdesk_session_state
from .state_transitions import State, happy_path, terminal_states


class Graph(ChatEngineGraph):
    """HR helpdesk hub + sticky-topic workflow on ChatEngineGraph."""

    state_enum = State
    terminal_states = terminal_states
    handler_map = handler_map

    idle_state = State.IDLE
    clarify_state = State.HUB_CLARIFY
    notify_state = State.NOTIFY_USER
    confidence_threshold = 0.7
    unclear_topic = "unclear"
    topic_to_state = {
        "faq": State.TOPIC_FAQ,
        "escalate": State.TOPIC_ESCALATE,
        "booking": State.TOPIC_BOOKING,
    }
```

Replace with:

```python
"""Domain-specific LangGraph configuration for the HR helpdesk chatbot."""

from __future__ import annotations

from typing import Any, Callable

from src.engine.chat_engine_graph import ChatEngineGraph
from src.engine.handler_registry import get_handler_metadata
from src.engine.json_checkpointer import JsonCheckpointer

from .guardrails import guardrails
from .handlers import handler_map, set_ledger
from .router import HelpdeskSemanticRouter
from .session_state import HelpdeskState, new_helpdesk_session_state
from .state_transitions import State, happy_path, terminal_states


class Graph(ChatEngineGraph):
    """HR helpdesk hub + sticky-topic workflow on ChatEngineGraph."""

    state_enum = State
    terminal_states = terminal_states
    handler_map = handler_map

    idle_state = State.IDLE
    clarify_state = State.HUB_CLARIFY
    notify_state = State.NOTIFY_USER
    confidence_threshold = 0.7
    unclear_topic = "unclear"
    topic_router = HelpdeskSemanticRouter()
    topic_to_state = {
        "faq": State.TOPIC_FAQ,
        "escalate": State.TOPIC_ESCALATE,
        "booking": State.TOPIC_BOOKING,
    }
```

(Only the two added lines — `from .router import HelpdeskSemanticRouter` and
`topic_router = HelpdeskSemanticRouter()` — change; the rest is shown for exact
placement.)

- [ ] **Step 4: Update imports in `src/hrhelpdesk/handlers.py`**

Current (lines 1-45):

```python
"""Handler functions for the HR helpdesk chatbot."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage

from src.engine.chat_engine_graph import ChatEngineGraph, TopicDecision
from src.engine.event_ledger import effect_key
from src.engine.handler_registry import handler
from src.engine.utils import (
    close_topic_delta,
    content_hash,
    find_tool_call,
    last_ai_content,
    ledger_is_processed_sync,
    ledger_mark_processed_sync,
    log_handler_enter,
    log_handler_exit,
)

from .chains import (
    booking_agent,
    escalate_agent,
    faq_agent,
    run_escape,
    run_router,
)
from .services import confirm_booking, create_ticket, retrieve_policy
from .session_state import HelpdeskState
from .state_transitions import State

log = logging.getLogger(__name__)

ROUTER_CONFIDENCE_THRESHOLD = 0.7

# Typed-field fan-out helper (same threshold as Graph); avoids importing Graph
# from handlers (circular: graph → handlers → graph).
_topic_fanout = ChatEngineGraph()
_topic_fanout.confidence_threshold = ROUTER_CONFIDENCE_THRESHOLD

_ledger: Any = None
```

Replace with:

```python
"""Handler functions for the HR helpdesk chatbot."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage

from src.engine.chat_engine_graph import ChatEngineGraph
from src.engine.event_ledger import effect_key
from src.engine.handler_registry import handler
from src.engine.utils import (
    close_topic_delta,
    content_hash,
    find_tool_call,
    last_ai_content,
    ledger_is_processed_sync,
    ledger_mark_processed_sync,
    log_handler_enter,
    log_handler_exit,
)

from .chains import (
    booking_agent,
    escalate_agent,
    faq_agent,
    run_escape,
)
from .router import HelpdeskSemanticRouter
from .services import confirm_booking, create_ticket, retrieve_policy
from .session_state import HelpdeskState
from .state_transitions import State

log = logging.getLogger(__name__)

ROUTER_CONFIDENCE_THRESHOLD = 0.7

# Typed-field fan-out + classification helper (same config as Graph); avoids
# importing Graph from handlers (circular: graph → handlers → graph).
_topic_fanout = ChatEngineGraph()
_topic_fanout.confidence_threshold = ROUTER_CONFIDENCE_THRESHOLD
_topic_fanout.topic_router = HelpdeskSemanticRouter()

_ledger: Any = None
```

(Removed the now-unused `TopicDecision` import and `run_router` import; added the
`HelpdeskSemanticRouter` import and the `_topic_fanout.topic_router = ...` line.)

- [ ] **Step 5: Rewrite `_route_human_message`**

Current (lines 141-176):

```python
def _route_human_message(
    state: HelpdeskState,
    input_message: str,
    *,
    source: str,
) -> dict[str, Any]:
    """Semantic-route a human utterance into a lane (or stay in clarify).

    Shared by IDLE (first classification) and HUB_CLARIFY (user's answer
    after a disambiguation ask). Never guesses: unclear / low confidence
    keeps ``pending_clarify`` and re-asks.
    """
    try:
        router_out = run_router(input_message)
    except Exception as exc:
        log.error("[HANDLER] router failed (%s): %s", source, exc)
        return {
            "handler_status": "error",
            "error_message": str(exc),
            "audit_trail": [f"{source}: router failed: {exc}"],
        }

    decision = TopicDecision(
        topic=router_out.topic.value,
        confidence=router_out.confidence,
    )
    base = _topic_fanout.topic_decision_to_delta(decision, source=source)
    if base.get("pending_clarify"):
        base["output_messages"] = [CLARIFY_PROMPT]
        return base

    if decision.topic == "booking":
        booking_delta = _process_booking_turn({**state, **base}, input_message)
        base.update(booking_delta)
    return base
```

Replace with:

```python
def _route_human_message(
    state: HelpdeskState,
    input_message: str,
    *,
    source: str,
) -> dict[str, Any]:
    """Semantic-route a human utterance into a lane (or stay in clarify).

    Shared by IDLE (first classification) and HUB_CLARIFY (user's answer
    after a disambiguation ask). Never guesses: unclear / low confidence
    keeps ``pending_clarify`` and re-asks. A router failure degrades to
    ``unclear`` (handled inside ``HelpdeskSemanticRouter.classify``) rather
    than surfacing as ``handler_status="error"``.
    """
    decision = _topic_fanout.classify_utterance(input_message, state.get("messages"))
    base = _topic_fanout.topic_decision_to_delta(decision, source=source)
    if base.get("pending_clarify"):
        base["output_messages"] = [CLARIFY_PROMPT]
        return base

    if decision.topic == "booking":
        booking_delta = _process_booking_turn({**state, **base}, input_message)
        base.update(booking_delta)
    return base
```

- [ ] **Step 6: Rewrite the escape-reroute branch in `handle_topic_booking`**

Current (lines 346-382):

```python
@handler(
    state=State.TOPIC_BOOKING.value,
    waits_for_input=True,
    wait_kind="human",
    description="Sticky desk booking specialist",
)
def handle_topic_booking(state: HelpdeskState) -> HelpdeskState:
    log_handler_enter("topic_booking", state)
    input_message = state.get("input_message") or ""
    messages = state.get("messages") or []

    try:
        if run_escape(input_message).escape:
            cleared = close_topic_delta(messages, "User changed topic.")
            cleared["pending_clarify"] = False
            try:
                router_out = run_router(input_message)
            except Exception as exc:
                return log_handler_exit(
                    "topic_booking",
                    {
                        "handler_status": "error",
                        "error_message": str(exc),
                        **cleared,
                    },
                )
            decision = TopicDecision(
                topic=router_out.topic.value,
                confidence=router_out.confidence,
            )
            routed = _topic_fanout.topic_decision_to_delta(
                decision, source="topic_booking"
            )
            cleared.update(routed)
            if cleared.get("pending_clarify"):
                cleared["output_messages"] = [CLARIFY_PROMPT]
            cleared["audit_trail"] = ["topic_booking: escape — re-routed"]
            return log_handler_exit("topic_booking", cleared)
    except Exception as exc:
        log.error("[HANDLER] escape check failed: %s", exc)

    delta = _process_booking_turn(state, input_message)
    return log_handler_exit("topic_booking", delta)
```

Replace with:

```python
@handler(
    state=State.TOPIC_BOOKING.value,
    waits_for_input=True,
    wait_kind="human",
    description="Sticky desk booking specialist",
)
def handle_topic_booking(state: HelpdeskState) -> HelpdeskState:
    log_handler_enter("topic_booking", state)
    input_message = state.get("input_message") or ""
    messages = state.get("messages") or []

    try:
        if run_escape(input_message).escape:
            cleared = close_topic_delta(messages, "User changed topic.")
            cleared["pending_clarify"] = False
            decision = _topic_fanout.classify_utterance(input_message, messages)
            routed = _topic_fanout.topic_decision_to_delta(
                decision, source="topic_booking"
            )
            cleared.update(routed)
            if cleared.get("pending_clarify"):
                cleared["output_messages"] = [CLARIFY_PROMPT]
            cleared["audit_trail"] = ["topic_booking: escape — re-routed"]
            return log_handler_exit("topic_booking", cleared)
    except Exception as exc:
        log.error("[HANDLER] escape check failed: %s", exc)

    delta = _process_booking_turn(state, input_message)
    return log_handler_exit("topic_booking", delta)
```

(`run_escape` is unaffected — escape-detection stays out of scope, per the design
doc. Only the inner router-failure branch collapses, since `classify_utterance`
no longer raises.)

- [ ] **Step 7: Update `tests/test_hrhelpdesk_flow.py` imports and helper**

Current (lines 1-46):

```python
"""End-to-end HR helpdesk workflow tests with mocked LLM chains/agents."""

from __future__ import annotations

import importlib
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.engine.handler_registry import get_handler_metadata
from src.hrhelpdesk.chains import EscapeOutput, RouterOutput, RouterTopic
from src.hrhelpdesk.graph import build_graph as _build_helpdesk_graph
from src.hrhelpdesk.services import _booking_store, reset_providers
from src.hrhelpdesk.state_transitions import State


@pytest.fixture(autouse=True)
def _ensure_handlers_registered():
    if get_handler_metadata(State.IDLE.value) is None:
        import src.hrhelpdesk.handlers as hd_handlers

        importlib.reload(hd_handlers)


@pytest.fixture(autouse=True)
def _reset_provider_stores():
    reset_providers()
    yield
    reset_providers()


def build_graph():
    sessions_dir = tempfile.mkdtemp(prefix=f"hd_test_{uuid4()}_")
    return _build_helpdesk_graph(sessions_dir=sessions_dir)


def _router(topic: str, confidence: float = 0.9):
    return RouterOutput(topic=RouterTopic(topic), confidence=confidence)


def _escape(escape: bool = False):
    return EscapeOutput(escape=escape)
```

Replace with:

```python
"""End-to-end HR helpdesk workflow tests with mocked LLM chains/agents."""

from __future__ import annotations

import importlib
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.engine.chat_engine_graph import TopicDecision
from src.engine.handler_registry import get_handler_metadata
from src.hrhelpdesk.chains import EscapeOutput
from src.hrhelpdesk.graph import build_graph as _build_helpdesk_graph
from src.hrhelpdesk.services import _booking_store, reset_providers
from src.hrhelpdesk.state_transitions import State


@pytest.fixture(autouse=True)
def _ensure_handlers_registered():
    if get_handler_metadata(State.IDLE.value) is None:
        import src.hrhelpdesk.handlers as hd_handlers

        importlib.reload(hd_handlers)


@pytest.fixture(autouse=True)
def _reset_provider_stores():
    reset_providers()
    yield
    reset_providers()


def build_graph():
    sessions_dir = tempfile.mkdtemp(prefix=f"hd_test_{uuid4()}_")
    return _build_helpdesk_graph(sessions_dir=sessions_dir)


def _decision(topic: str, confidence: float = 0.9) -> TopicDecision:
    return TopicDecision(topic=topic, confidence=confidence)


def _escape(escape: bool = False):
    return EscapeOutput(escape=escape)
```

(`RouterOutput`/`RouterTopic` are no longer imported here — they were only used by
the now-removed `_router()` helper. `TopicDecision` and `_decision()` replace it.)

- [ ] **Step 8: Replace every `run_router` patch site with `_topic_fanout.classify_utterance`**

There are 7 occurrences of `patch("src.hrhelpdesk.handlers.run_router", ...)` /
`"src.hrhelpdesk.handlers.run_router"` across the file. Replace each target string
and each `_router(...)` call with `_decision(...)`, one test at a time:

In `test_faq_happy_path_clears_active_topic`:

```python
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("faq")),
```
→
```python
        patch("src.hrhelpdesk.handlers._topic_fanout.classify_utterance", return_value=_decision("faq")),
```

In `test_hub_clarify_reroutes_user_reply_to_faq`:

```python
        patch(
            "src.hrhelpdesk.handlers.run_router",
            side_effect=[_router("unclear", 0.4), _router("faq")],
        ),
```
→
```python
        patch(
            "src.hrhelpdesk.handlers._topic_fanout.classify_utterance",
            side_effect=[_decision("unclear", 0.4), _decision("faq")],
        ),
```

In `test_escalate_creates_one_ticket_ledger_dedupes`:

```python
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("escalate")),
```
→
```python
        patch("src.hrhelpdesk.handlers._topic_fanout.classify_utterance", return_value=_decision("escalate")),
```

In `test_booking_sticky_two_turn_confirm`:

```python
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("booking")),
```
→
```python
        patch("src.hrhelpdesk.handlers._topic_fanout.classify_utterance", return_value=_decision("booking")),
```

In `test_escape_mid_booking_reroutes` (two occurrences — the initial booking turn,
then the escape-to-faq turn):

```python
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("booking")),
```
→
```python
        patch("src.hrhelpdesk.handlers._topic_fanout.classify_utterance", return_value=_decision("booking")),
```

```python
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("faq")),
```
→
```python
        patch("src.hrhelpdesk.handlers._topic_fanout.classify_utterance", return_value=_decision("faq")),
```

In `test_faq_path_never_calls_booking_tools`:

```python
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("faq")),
```
→
```python
        patch("src.hrhelpdesk.handlers._topic_fanout.classify_utterance", return_value=_decision("faq")),
```

All other lines in the file (`run_escape` patches, `test_illegal_system_event_ignored`,
`test_topic_timeout_clears_booking`) are unchanged — neither test touches topic
classification.

- [ ] **Step 9: Run the full hrhelpdesk + engine test suite**

Run: `PYTHONPATH=. uv run pytest tests/test_hrhelpdesk_flow.py tests/test_hrhelpdesk_router.py tests/test_topic_router.py tests/test_chat_engine_graph.py tests/test_engine_astream.py -v`
Expected: all pass (8 in `test_hrhelpdesk_flow.py`, 3 in `test_hrhelpdesk_router.py`, 7 in `test_topic_router.py`, plus existing counts for the other two files — no failures, no new errors)

- [ ] **Step 10: Run the entire project test suite**

Run: `PYTHONPATH=. uv run pytest`
Expected: all pass (no regressions in `docprocessing`/`onboarding`/`triageprocessing`, which are untouched)

- [ ] **Step 11: Ruff check and commit**

Run: `ruff check src/hrhelpdesk/ tests/test_hrhelpdesk_flow.py --fix`

```bash
git add src/hrhelpdesk/chains.py src/hrhelpdesk/__init__.py src/hrhelpdesk/graph.py \
        src/hrhelpdesk/handlers.py tests/test_hrhelpdesk_flow.py
git commit -m "feat(hrhelpdesk): wire HelpdeskSemanticRouter into Graph/handlers, drop run_router"
```

---

## Post-implementation checklist

- [ ] All three commits landed (Task 1, Task 2, Task 3)
- [ ] `PYTHONPATH=. uv run pytest` is fully green
- [ ] `ruff check .` clean (or no new findings beyond pre-existing debt)
- [ ] No remaining references to `run_router` / `route_chain` anywhere: `grep -rn "run_router\|route_chain" src/ tests/` returns nothing
