# Design: Wire hrhelpdesk Topic Classification into `ChatEngineGraph`'s `TopicRouter` Seam

_Date: 2026-07-14 · Status: Approved for implementation planning_

## Context

`ChatEngineGraph` (`src/engine/chat_engine_graph.py`) already declares a `TopicRouter`
Protocol and a `topic_router` / `classify_utterance()` seam for pluggable topic
classification. `src/hrhelpdesk/` never wires anything into it: `handlers.py` calls a
free function `run_router()` (wrapping `route_chain` from `hrhelpdesk/chains.py`)
directly at three call sites (`_route_human_message`, and the escape-reroute branch
inside `handle_topic_booking`). The seam is dead code; classification logic is
duplicated ad hoc in handlers instead.

This design finishes wiring that seam, and — mirroring how `docprocessing/router.py`'s
`DocPipelineRouter` already inherits reusable LLM-routing boilerplate from
`src/engine/router.py`'s `DefaultSemanticRouter` — adds the equivalent reusable base
for **topic** classification (as opposed to **state** classification) so future chat
domains get it for free.

## Decisions locked in brainstorming

| Decision | Choice |
|---|---|
| Boilerplate location | Engine-level `DefaultTopicRouter` in `src/engine/router.py`, next to the existing `BaseSemanticRouter`/`DefaultSemanticRouter` pair. |
| Scope | Topic classification only. Escape-detection (`run_escape`/`escape_chain`) stays as-is — different shape of problem (binary intent-to-leave, not topic+confidence), out of scope for this pass. |
| LLM call plumbing | `DefaultTopicRouter.classify()` builds its chain via `engine/chains.py::make_chain` (the helper every domain's `chains.py` already uses, including hrhelpdesk's own `route_chain`) — **not** a hand-rolled `init_chat_model`/`with_structured_output` call like `DefaultSemanticRouter` has. Avoids a third parallel LLM-invocation pattern. |
| Protocol conformance | `DefaultTopicRouter` satisfies the existing `TopicRouter` Protocol structurally (matches its `classify(input_message, history=None) -> TopicDecision` signature) — no new ABC hierarchy. |
| Router failure fallback | `classify()` swallows exceptions internally and returns `TopicDecision(unclear_topic, 0.0)`, matching `DefaultSemanticRouter`'s existing fallback-on-error convention. **Behavior change** (see below). |

## 1. Architecture

```
src/engine/router.py
├── BaseSemanticRouter / DefaultSemanticRouter   (existing — state routing, own LLM call)
└── DefaultTopicRouter                            (new — topic routing, via make_chain)

src/hrhelpdesk/router.py                          (new — mirrors docprocessing/router.py)
└── HelpdeskSemanticRouter(DefaultTopicRouter)
```

`ChatEngineGraph._resolve_proposed_next` is unaffected — it still only reads typed
fields (`active_topic`, `pending_clarify`) and never touches message text. This change
only replaces *how* a `TopicDecision` gets produced inside a handler, before that
delta is written via `topic_decision_to_delta`.

## 2. Components

### `src/engine/router.py` — `DefaultTopicRouter` (new)

```python
class DefaultTopicRouter:
    """Concrete TopicRouter (structural) for LLM-powered topic+confidence classification.

    Subclasses set:
      output_schema — Pydantic model with `topic` (str|Enum) and `confidence` (float) fields
      get_instructions() / system_prompt — classification system prompt
      model_role — passed to get_model(role) for model selection (default "router")
      unclear_topic — topic value used for the clarify branch (default "unclear")
      chain_name — name passed to make_chain (default "TopicRouterChain")
    """

    output_schema: type          # must be set by subclass
    system_prompt: str = ""      # or override get_instructions()
    model_role: str = "router"
    unclear_topic: str = "unclear"
    chain_name: str = "TopicRouterChain"

    def __init__(self) -> None:
        self._chain = None  # lazy, built on first classify()

    def get_instructions(self) -> str:
        return self.system_prompt

    def _get_chain(self):
        if self._chain is None:
            self._chain = make_chain(
                name=self.chain_name,
                system_prompt=self.get_instructions(),
                output_schema=self.output_schema,
                model_id=get_model(self.model_role),
            )
        return self._chain

    def classify(self, input_message: str, history=None) -> TopicDecision:
        try:
            response = self._get_chain().invoke({"input": input_message})
            topic = response.topic
            topic_val = topic.value if hasattr(topic, "value") else str(topic)
            confidence = max(0.0, min(1.0, float(response.confidence)))
            return TopicDecision(topic=topic_val, confidence=confidence)
        except Exception:
            log.exception("[DefaultTopicRouter] classify() failed")
            return TopicDecision(topic=self.unclear_topic, confidence=0.0)
```

`TopicDecision` is imported from `src.engine.chat_engine_graph` (no circular import:
`chat_engine_graph.py` does not import `router.py`). `history` is accepted for Protocol
conformance and future use but unused by the default implementation — hrhelpdesk's
current classification is stateless per-utterance, matching today's behavior exactly.

### `src/hrhelpdesk/router.py` (new)

```python
from src.engine.router import DefaultTopicRouter
from .chains import RouterOutput, RouterTopic

class HelpdeskSemanticRouter(DefaultTopicRouter):
    output_schema = RouterOutput
    unclear_topic = RouterTopic.UNCLEAR.value
    chain_name = "HelpdeskRouteChain"

    def get_instructions(self) -> str:
        return <verbatim system_prompt currently inline in chains.py::route_chain>
```

### `src/hrhelpdesk/chains.py`

Delete `route_chain` and `run_router()` (superseded). Keep `RouterOutput`/`RouterTopic`
(imported by the new `router.py`) and leave `escape_chain`/`run_escape` untouched.

### `src/hrhelpdesk/graph.py`

Add `topic_router = HelpdeskSemanticRouter()` as a `Graph` class attribute, alongside
the existing `topic_to_state`/`confidence_threshold` class attributes.

### `src/hrhelpdesk/handlers.py`

The existing `_topic_fanout = ChatEngineGraph()` module-level singleton (already
present solely to reuse `topic_decision_to_delta()` without importing `Graph` and
creating a `graph → handlers → graph` circular import) additionally gets
`_topic_fanout.topic_router = HelpdeskSemanticRouter()` set next to its existing
`.confidence_threshold = ROUTER_CONFIDENCE_THRESHOLD` line.

`_route_human_message` replaces its direct `run_router(input_message)` call with
`_topic_fanout.classify_utterance(input_message, state.get("messages"))`. The
escape-reroute branch inside `handle_topic_booking` gets the same substitution.

## 3. Data flow

Unchanged at the graph/router-node level. Classification still happens inside a
handler (`handle_idle`, `handle_hub_clarify`, or the escape branch of
`handle_topic_booking`), which converts the `TopicDecision` to typed fields via
`topic_decision_to_delta` before returning. The `ChatEngineGraph` router node
(`_resolve_proposed_next`) only ever reads those typed fields on the next pass — the
"router never re-reads message text" invariant documented in `chat_engine_graph.py`'s
module docstring is preserved.

## 4. Error handling — behavior change

**Today:** if `run_router()` raises, `_route_human_message` returns
`handler_status="error"`, which the guardrail's `check_handler_status` diverts to the
terminal `ERROR` state — one flaky LLM call ends the session.

**After this change:** `DefaultTopicRouter.classify()` swallows the exception itself
and returns `TopicDecision(unclear_topic, 0.0)`, which `topic_decision_to_delta`
treats identically to a genuinely low-confidence/ambiguous classification — the user
sees the ordinary `CLARIFY_PROMPT` ("I'm not sure I understood...") instead of the
session terminating. This was explicitly reviewed and accepted: a transient router
failure degrading to "please clarify" is better UX than killing the session, and it
matches `DefaultSemanticRouter`'s existing error-fallback convention.

## 5. Testing

- New unit tests for `DefaultTopicRouter.classify()` (valid response → `TopicDecision`;
  malformed/exception → unclear fallback; confidence clamping; enum-vs-str `topic`
  coercion) — mirrors the style of `tests/test_chat_engine_graph.py`.
- Update `tests/test_hrhelpdesk_flow.py`: existing mocks that patch `run_router`/
  `route_chain` directly must be updated to patch `HelpdeskSemanticRouter.classify` (or
  the chain it builds) instead. Audit all patch sites before implementation.
- No change expected to `tests/test_chat_engine_graph.py` or `tests/test_engine_astream.py`
  (neither touches topic classification).

## Out of scope (explicitly deferred)

- Escape-detection (`run_escape`/`escape_chain`) — stays ad hoc in handlers.
- Any change to `ChatEngineGraph._resolve_proposed_next` or the typed-field fan-out
  contract.
- Retrofitting `docprocessing`'s `DefaultSemanticRouter` to use `make_chain` (existing
  duplication between it and `chains.py::make_chain` is left as-is; not this design's
  concern).
