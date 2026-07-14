# Design: HR Helpdesk Assistant (Hub + Semantic Router + Topic States)

_Date: 2026-07-14 · Status: Approved for implementation planning_

## Goal

Extend the engine with **generic conversational (chatbot) primitives**, then implement `src/hrhelpdesk/` as the reference hub + sticky-topic workflow: FAQ (RAG), Escalate (ticket), Book-desk (multi-turn slots).

## Decisions locked in brainstorming

| Decision | Choice |
|---|---|
| Fidelity | **Option A** — JsonCheckpointer, file keyed ledger, CLI, fake providers, mocked models in tests. Conversational chatbot (not a linear pipeline with chat bolted on). |
| Engine vs domain | Thin **chat engine layer** + domain package; no helpdesk vocabulary on base `EngineSessionState`. |
| Ledger | **Generalize** existing ledger into one keyed store; namespace keys (`event:…`, `effect:…`); keep `EventLedger` alias. |
| Message reset | Engine helper for **any** chatbot workflow (trim + segment reset with summary). |
| Topology | Approach 1 — hub/topic as `EngineGraph` states; specialists via `make_llm_agent`. |
| Topic state fields | `ChatEngineSessionState` (`chat_engine_session_state.py`); graph fan-out via `ChatEngineGraph`. |
| Router audit field | **No `router_last`** — use `router_confidence` / `semantic_context` / `audit_trail`. |
| HITL | No `interrupt()` in v1; end-and-reenter parks. |
| Async streaming | **In scope** — engine streaming entry points parallel to `ainvoke` / `aemit_event` (graph `.astream()`). Helpdesk CLI exercises them by printing LLM generations as they stream (no FastAPI). |
| Out of scope v1 | FastAPI (do not add), Postgres/AsyncPostgresSaver, real RAG/APIs, LangSmith Platform, routing-eval CI gate. |

---

## 1. Architecture & boundaries

### Stack (Option A)

- Persistence: `JsonCheckpointer` (sessions dir `.hrhelpdesk_sessions`)
- Dedup / effects: generalized keyed ledger (sibling ledger dir)
- Entry: CLI `chat` / `event` / `sweep` / `status` / `serve` (onboarding shape)
- Providers: in-process fakes for policy KB, tickets, desk booking
- Tests: mocked LLM chains/agents; real graph + checkpointer

### Engine (generic)

| Addition | Purpose |
|---|---|
| `ChatEngineSessionState` + `new_chat_session_state()` | Shared sticky-lane fields for chatbot workflows |
| Message helpers (`utils.py`) | Trim prompt window; segment reset + one-line summary |
| Generalized keyed ledger | Delivery IDs and write-effect idempotency keys |
| `get_model(role)` | Role-based model selection for `make_chain` / `make_llm_agent` |
| `ChatEngineGraph._resolve_proposed_next` | Hub fan-out from typed fields (topic / clarify / system) |
| `_is_system_event_legal` (overridable) | Predicate-based system-event legality (parallel open items) |
| `astream` / `aemit_event_stream` (names TBD) | Async streaming entry points parallel to `ainvoke` / `aemit_event` |

Linear workflows (onboarding, triage, docprocessing) keep using `EngineSessionState` unchanged.

### Domain (`src/hrhelpdesk/`)

Hub routing policy, escape-check, topic specialists, tools, legality predicates, sweep thresholds, CLI, fake providers.

### Outer vs specialist topology

```
User/system → aemit_event → EngineGraph (router → guardrail → handler)
                              │
                              ├─ route / clarify / notify: deterministic or make_chain
                              └─ topic handlers invoke make_llm_agent specialists
                                    (compiled create_agent subgraphs, scoped tools)
```

Supervisor = outer `EngineGraph`. Workers = per-topic `make_llm_agent` graphs. No unbounded outer ReAct loop; no cross-topic tool binding.

### Production-report alignment (adopt / defer)

**Adopt:** lean typed state; checkpoint by thread; specialized workers + controllable routing; loop/call caps; IDs not document blobs in state; async token streaming via graph `.astream()` as a first-class engine entry point.  
**Defer:** Postgres/Redis, LangSmith Platform, `interrupt()` HITL.  
**Reject for v1:** monolithic chatbot↔ToolNode ReAct with all tools on every turn.

### Async streaming (engine + CLI)

Add streaming counterparts to the existing turn APIs so callers can yield LLM output text as it is produced:

| Blocking / result API | Streaming API (new) |
|---|---|
| `ainvoke(...)` | e.g. `astream(...)` — same session/input contract; yields chunks from compiled graph `.astream()` |
| `aemit_event(...)` | e.g. `aemit_event_stream(...)` — same funnel (lock, dedupe, legality); streams user-visible tokens/updates for that turn |

Requirements:

- Same durability, ledger, and legality semantics as the non-streaming path (streaming is a delivery mode, not a second state machine).
- Prefer `stream_mode` that can surface LLM tokens (e.g. `messages`) and/or node updates; filter to user-facing nodes so router/internal steps are not dumped as chat text.
- Final checkpointed state must match what `ainvoke` / `aemit_event` would have produced for the same input.
- **Helpdesk CLI is the v1 consumer:** `chat` (and interactive `serve` turns) use the streaming path and print generations to stdout as chunks arrive. No HTTP/FastAPI layer.
- Tests cover streaming on the engine and CLI-level generation printing (mocked model chunks acceptable).

---

## 2. State machine & durable state

### Graph states

| State | Role | waits_for_input |
|---|---|---|
| `IDLE` | Hub park + human-turn routing (escape / semantic route) | yes (human / either) |
| `HUB_CLARIFY` | Disambiguate when unclear / low confidence | yes (human) |
| `TOPIC_FAQ` | Policy Q&A specialist | no (one-shot then close → IDLE) |
| `TOPIC_ESCALATE` | Ticket specialist | no (one-shot then close → IDLE) |
| `TOPIC_BOOKING` | Sticky desk booking | yes (human); self-loop until done/escape |
| `NOTIFY_USER` | Render legal system events into chat | no |
| `ERROR` | Terminal failure | — |

> Note: a separate `ROUTE` node was dropped — hub routing lives on `IDLE` + `Graph._resolve_proposed_next`.

### Per-turn routing (domain policy)

```
if incoming == system_event:  # funnel already validated via legality hook
    → NOTIFY_USER → IDLE
elif active_topic:
    if escape_check(msg).escape:
        clear topic (message segment reset) → semantic_router(msg)
    else:
        → TOPIC_NODES[active_topic]
else:
    r = router(msg)  # structured, validated
    if r.topic == unclear or r.confidence < 0.7:
        → HUB_CLARIFY → IDLE
    else:
        set active_topic (booking sticky; faq/escalate one-shot)
        → topic node → close topic if one-shot → IDLE
```

### Durable state

```python
class ChatEngineSessionState(EngineSessionState):
    """Base for hub + sticky-topic conversational workflows."""
    schema_version: int
    active_topic: Optional[str]       # sticky lane id; None at hub
    topic_started_at: Optional[str]   # sweep / TTL
    topic_data: dict                  # lean lane scratch

class HelpdeskState(ChatEngineSessionState):
    open_tickets: list[str]           # IDs only
    bookings: list[dict]              # confirmed lean records
    last_event: Optional[str]
    last_event_at: Optional[str]
```

**Invariants**

- Routing and business transitions read typed fields only — never `messages`.
- Prompts see a trimmed window (engine helper); topic close uses segment reset + one-line summary.
- No raw KB chunks or full LLM payloads in checkpoints — source IDs / slot fields only.
- Thread id: `hd-{tenant}-{employee_id}` (CLI may use `hd-{employee_id}`).

---

## 3. Events, legality, effects & resume

### Entry

Reuse `aemit_event` for human `message` and system events. No `interrupt()` in v1.

### System-event legality map (exhaustive)

| Event | Legal when | Effect |
|---|---|---|
| `ticket_resolved` | `ticket_id ∈ open_tickets` | remove from open_tickets; `NOTIFY_USER` |
| `booking_cancelled_by_system` | `booking_id ∈ bookings` (active) | mark cancelled; `NOTIFY_USER` |
| `topic_timeout` | `active_topic == "booking"` and stale | clear topic/slots; `NOTIFY_USER` |
| other | — | ignore + anomaly log; consume delivery id |

Predicates live in domain; engine exposes `_is_system_event_legal(state, event_type, payload)` so legality is not limited to a single park state's `expected_events` list (IDLE can coexist with open tickets/bookings).

### Human path

End-and-reenter at `IDLE` or sticky `TOPIC_BOOKING`. Booking confirmation is a **code-validated tool execution**, not an interrupt gate.

### Keyed ledger

| Kind | Key pattern |
|---|---|
| Delivery dedupe | `event:{event_id}` |
| Create ticket | `effect:{thread}:ticket:{content_hash}` |
| Confirm booking | `effect:{thread}:booking:{date}:{location}` |

Same file-marker store; namespaces prevent collisions. `EventLedger` remains a compatible alias/API for existing callers.

### Sweep

Domain `sweep.py` via `get_active_sessions()`:

| Condition | Threshold | Action |
|---|---|---|
| Sticky booking inactive | 48 h | emit `topic_timeout` |
| Open tickets unresolved | 5 business days | reminder stub (log / fake provider) |

### Conflict rule

Retain check for interrupts before state merges (v1 always end-and-reenter; prepares for future approver topics).

---

## 4. Model & tool layer

### Roles (`get_model(role)`)

| Role | Use | Notes |
|---|---|---|
| `router` | Semantic route + escape-check | Structured output only; small/fast model |
| `topic` | FAQ / escalate / booking specialists | Mid-tier; tutorial may share DEFAULT_MODEL |

Hard cap: **≤3 model calls per user turn**. No unbounded outer agent loops.

### Factories & tool scoping

| Component | Factory | Tools |
|---|---|---|
| Semantic router | `make_chain` → `{topic, confidence}` | none |
| Escape check | `make_chain` → `{escape: bool}` | none |
| Hub clarify | handler / tiny chain | none |
| FAQ specialist | `make_llm_agent(..., tools=[])` | none; handler runs fake retriever |
| Escalate specialist | `make_llm_agent(..., tools=[create_ticket])` | write, ledgered in handler |
| Booking specialist | `make_llm_agent(..., tools=[check_desk_availability, confirm_booking])` | read + write; write ledgered |
| Notify | deterministic handler | none |

Handlers intercept tool calls/results (onboarding `submit_new_hire` pattern) and apply validated state deltas + ledger marks. Model prose never transitions business state.

### Validation

- Router parse failure / `unclear` / `confidence < 0.7` → `HUB_CLARIFY` (never guess).
- Booking: slots validated in code; `confirm_booking` only when complete + user confirmed.
- FAQ: KB content in delimited untrusted blocks; source citations; no tools on FAQ agent.

---

## 5. Package layout, CLI, tests & NFRs

### Engine touch points

- `chat_engine_session_state.py`: `ChatEngineSessionState`, constructors
- `chat_engine_graph.py`: `ChatEngineGraph` (topic + confidence → typed fields → `proposed_next`)
- `utils.py`: message trim / segment-reset helpers
- Ledger generalization (`event_ledger.py` or rename with alias)
- `chains.py`: `get_model(role)` integration
- `engine_graph.py`: overridable next-state + system-event legality hooks; `astream` / `aemit_event_stream` entry points

### Domain package `src/hrhelpdesk/`

```
__init__.py
graph.py
session_state.py
state_transitions.py
handlers.py
guardrails.py
chains.py          # router/escape chains + make_llm_agent specialists
providers.py       # fake RAG, ticket, desk APIs
sweep.py
cli.py
README.md
```

### CLI

Subcommands: `chat`, `event`, `sweep`, `status`, `serve`.  
Sessions: `.hrhelpdesk_sessions` (+ ledger sibling).  
`chat` / human turns in `serve` call `aemit_event_stream` (or equivalent) and stream assistant generations to the terminal; `event` / `sweep` may stay non-streaming (system notifies are short, deterministic).

### Golden tests (workflow correctness)

- Happy paths: FAQ answer; escalate → one ticket; booking confirm → one booking
- Adversarial: in FAQ, “book me a desk” → zero booking tool calls
- Sticky resume: 2/3 slots survive a later turn
- Escape mid-booking → hub → re-route same message
- Duplicate ticket/booking keys → exactly once
- Illegal/unknown events → ignored
- `topic_timeout` clears slots + notifies once
- Engine unit tests: chat state helpers, ledger namespaces, message segment reset, streaming entry point yields chunks and final state matches non-streaming path

### NFRs (tutorial-scaled)

- ≤3 model calls / turn; no outer unbounded loops
- Effect-bearing paths use ledger + existing sync durability patterns
- 48 h booking sweep; lean checkpoints

### Follow-ups (not v1)

AsyncPostgresSaver; real providers; LangSmith; routing-eval dataset (≥150 utterances) as deploy gate; `interrupt()` approver topics. **Do not add FastAPI** in this tutorial track — CLI + engine stream APIs are sufficient.

---

## 6. Always / Ask / Never

### Always

- Route from typed state; messages never drive routing or transitions
- Validate model outputs (router, escape, slots) before they touch state
- Ledger every write effect; scoped tools per specialist agent
- Give sticky topics and open items a sweep policy

### Ask first

- Adding a topic (eval-set extension when evals exist)
- Binding any new write tool to any node
- Changing router/escape prompts or confidence threshold
- Introducing `interrupt()` gates
- Promoting more fields onto `ChatEngineSessionState`

### Never

- Let LLM prose transition business state
- Bind write tools on FAQ / hub / router
- Guess on `unclear`
- Let one topic agent see another topic's tools
- Exceed 3 model calls per turn / add unbounded outer agent loops

---

## Open questions (deferred)

1. Channel identity/auth specifics (CLI thread id sufficient for v1).
2. Whether `ticket_resolved` should reopen a conversational turn or only notify (v1: notify → IDLE).
3. Real desk-booking provider idempotency (v1: fake + ledger keys).
4. Whether “payroll question” is FAQ or its own lane (v1: FAQ).
