# HR Helpdesk Assistant

Hub + semantic-router chatbot built on `ChatEngineGraph` with sticky topic lanes
(FAQ, escalate, book-desk).

## Layout

| Module | Role |
|---|---|
| `graph.py` | `ChatEngineGraph` subclass — hub roles, topic map, system-event legality |
| `handlers.py` | State handlers (hub park, topic specialists, notify) |
| `chains.py` | Topic agents + tools (escape lives in `engine.escape_checker`) |
| `services.py` | In-memory fake RAG, tickets, desk booking |
| `cli.py` | `chat`, `event`, `sweep`, `status`, `serve` |
| `sweep.py` | 48h booking timeout sweep |

## CLI

```bash
export SESSIONS=.hrhelpdesk_sessions
export THREAD_ID=hd-emp-1

# FAQ (one-shot → idle)
uv run python -m src.hrhelpdesk.cli --sessions-dir "$SESSIONS" \
  chat "$THREAD_ID" "How much PTO do I get?"

# Escalate (creates a ticket), then resolve it
uv run python -m src.hrhelpdesk.cli --sessions-dir "$SESSIONS" \
  chat "$THREAD_ID" "My paycheck is wrong"
uv run python -m src.hrhelpdesk.cli --sessions-dir "$SESSIONS" status "$THREAD_ID"
# use open_tickets id from logs/state, e.g. TICKET-1
uv run python -m src.hrhelpdesk.cli --sessions-dir "$SESSIONS" \
  event "$THREAD_ID" ticket_resolved --event-id evt-1 --payload ticket_id=TICKET-1

# Booking sticky + optional sweep (only useful after a parked booking goes stale)
uv run python -m src.hrhelpdesk.cli --sessions-dir "$SESSIONS" \
  chat "$THREAD_ID" "Book me a desk downtown next Tuesday"
uv run python -m src.hrhelpdesk.cli --sessions-dir "$SESSIONS" sweep

uv run python -m src.hrhelpdesk.cli --sessions-dir "$SESSIONS" serve
```

Human turns (`chat`, `serve`) stream assistant tokens via `aemit_event_stream`.

Sessions persist under `.hrhelpdesk_sessions` with a sibling `.hrhelpdesk_sessions_ledger`.
Use the same `--sessions-dir` for every command (default matches `$SESSIONS` above).

## Sticky topics

A **sticky topic** is a conversation lane that stays active across multiple turns
until it finishes, the user escapes, or a timeout clears it. Without stickiness,
every follow-up (e.g. “window”) would re-enter IDLE’s semantic router and might
be misclassified.

| Topic | Sticky? | Behavior |
|---|---|---|
| `TOPIC_BOOKING` | Yes | `waits_for_input=True`; parks with `active_topic="booking"` until done / escape / timeout |
| `TOPIC_FAQ` | No | One-shot → `close_topic_delta` → IDLE |
| `TOPIC_ESCALATE` | No | One-shot → close → IDLE |

Example (sticky booking):

```
You: Book a desk downtown Tuesday
Bot: Window or aisle?          ← still in TOPIC_BOOKING

You: window                    ← not re-routed as a new hub turn
Bot: Desk booked…
                               ← close_topic_delta → IDLE
```

Engine signals: `active_topic` / `topic_data` / `topic_started_at` on
`ChatEngineSessionState`, fan-out via `topic_to_state`, escape checker, and
sweep → `topic_timeout` while `active_topic == "booking"`.

### Leaving a sticky topic (booking)

1. **User escape** — Before the booking agent runs, `run_escape(text)`
   (`engine.escape_checker`) detects leave-intent (“actually, what’s PTO?”,
   “cancel”, “never mind”). Then:
   - `close_topic_delta` clears sticky fields
   - In `handle_topic_booking`: re-classify that same message → FAQ / escalate /
     clarify / new booking
   - In `handle_idle` (if `active_topic` still set): clear topic, then normal hub routing

2. **Natural completion** — All slots confirmed → `_process_booking_turn` calls
   `close_topic_delta` → fan-out to IDLE (not escape, but ends the lane).

3. **Timeout** — `sweep` emits `topic_timeout` for a stale booking → handler
   short-circuits (no booking LLM) → `NOTIFY_USER` clears the topic and tells
   the user the session timed out.

## Mental model: reasoning about `ChatEngineGraph`


Use this when reading helpdesk or designing a similar hub + sticky-topic chatbot.

### Two layers, two clocks

One checkpoint, two responsibilities:

| Layer | Job | Reads | Writes |
|---|---|---|---|
| **Handlers** | Understand the turn (LLM, tools, side effects) | `input_message`, history, topic data | Typed fields + `output_messages` |
| **Router fan-out** (`_resolve_proposed_next`) | Decide where to park next | Typed fields only | `proposed_next` |

**Invariant:** transitions never read message text. Handlers may call LLMs; the graph router must not. If you want “message contains X → go to Y,” put that in a handler that sets a typed field.

### Thought process for a new chatbot domain

1. **Name the parks** — where can the session wait between turns?
   - Hub park → `idle_state`
   - “I didn’t understand” → `clarify_state`
   - Async push notifications → `notify_state`
   - Multi-turn slots → sticky `TOPIC_*` with `waits_for_input=True`
   - One-shot specialists → `waits_for_input=False` (run, then return to idle)

2. **Separate topic id from graph state**
   - **Topic** = semantic lane (`faq`, `booking`) — classifier output
   - **State** = graph node (`topic_faq`, `topic_booking`) — machine location  
   Flow: utterance → `{topic, confidence}` → typed fields (`active_topic`, `pending_clarify`) → fan-out → state. Do not let the LLM name a graph state directly (that’s linear `EngineGraph` territory).

3. **Classify each turn type before coding handlers**

   | Kind | Example | Handler must… |
   |---|---|---|
   | Fresh human @ IDLE | “How much PTO?” | Classify → set typed fields (maybe booking bootstrap) |
   | Sticky continue | “window seat” | Skip re-route; run specialist |
   | Escape | “actually, PTO?” | Clear topic → re-classify |
   | Clarify reply | “policy question” | Re-classify (don’t ignore input) |
   | System | `ticket_resolved` | Short-circuit LLMs; stamp event → NOTIFY |

   If a park can receive system events, short-circuit on `current_event_source == "system"` before any model call.

4. **Fan-out priority** (check in this order when debugging)

   ```
   system? → notify (then idle)
   pending_clarify? → clarify
   active_topic mapped? → topic_*
   else → idle
   ```

5. **Side effects** — tools/ledger in handlers; legality on the graph (`_is_system_event_legal`); sweeps emit system events (don’t invent a second machine).

6. **User-visible text** — handler sets `output_messages` (this turn); engine builds `AIMessage`s into `messages` (history). Streaming may print live LLM tokens or fall back to `output_messages` (e.g. the hardcoded clarify prompt).

### Debugging checklist

1. Where are we parked? `current_state` + `active_topic` + `pending_clarify`
2. What event is this? `current_event_source` / `current_event_type`
3. Did the handler short-circuit? (system before LLM)
4. What typed fields did the handler write?
5. What did fan-out propose? (audit trail `router: …`)
6. Did guardrails allow it? (`allowed_transitions` — missing edge often → ERROR)
7. Did we stop because `waits_for_input`? (expected park vs unexpected END)

### When to use this vs linear `EngineGraph`

| Use `ChatEngineGraph` when… | Use linear `EngineGraph` when… |
|---|---|
| Ongoing conversation, hub home | Case progresses INIT → … → COMPLETE |
| Multiple lanes + sticky multi-turn | Mostly auto-advance + a few waits |
| Topic classifier + escape | Code happy-path (or state-proposing router) |
| System events interrupt chat | System events unblock pipeline parks |

Helpdesk = chatbot. Onboarding/triage = pipelines. Same engine core; different session shape and fan-out.

### Design sketch template

1. Parks: IDLE / CLARIFY / NOTIFY / sticky topics?
2. Topics + `topic_to_state`
3. Per topic: one-shot vs sticky; tools; close via `close_topic_delta`
4. Escape checker (default or override prompt)
5. Legal system events + notify copy
6. Sweep thresholds if sticky can go stale
7. CLI: stream human turns; one-shot events for system

Reason **flags → park → handler → flags → park**, not “what did the user say → next node.”

## Design

See `docs/design/design_hrhelpdesk-2026-07-14.md`.
