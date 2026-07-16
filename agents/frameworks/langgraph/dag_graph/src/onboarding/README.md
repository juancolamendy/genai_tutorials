# Onboarding pipeline

Linear new-hire onboarding built on `EngineGraph` (not `ChatEngineGraph`).
Code-routed happy path with human and system park states — no semantic router
for transitions (design: “No LLM output transitions state”).

## Layout

| Module | Role |
|---|---|
| `graph.py` | `EngineGraph` subclass — happy path, guardrails, handlers |
| `handlers.py` | Per-state business logic (collect, welcome, awaits, provision, …) |
| `state_transitions.py` | `State`, `happy_path`, `allowed_transitions`, terminals |
| `guardrails.py` | Transition + completeness checks (e.g. collect self-loop) |
| `chains.py` | Collect agent + username chain |
| `cli.py` | `chat`, `event`, `sweep`, `status`, `serve` |
| `sweep.py` | Timeout escalation for document / hardware park states |

## How to run

```bash
export SESSIONS=.onboarding_sessions
export THREAD_ID=thread-10

uv run python -c "
import asyncio
from src.onboarding.graph import build_graph
g = build_graph(sessions_dir='$SESSIONS')
r = asyncio.run(g.ainvoke(user_id='', session_id='$THREAD_ID', input_message='start'))
print(r['current_state'])
"

uv run python -m src.onboarding.cli --sessions-dir "$SESSIONS" status $THREAD_ID
# expect: current_state=collect

uv run python -m src.onboarding.cli --sessions-dir "$SESSIONS" chat $THREAD_ID "Jane Doe, Engineer, 2026-08-01"

uv run python -m src.onboarding.cli --sessions-dir "$SESSIONS" event $THREAD_ID document_signed --event-id "$THREAD_ID-evt-1"

uv run python -m src.onboarding.cli --sessions-dir "$SESSIONS" event $THREAD_ID hardware_delivered --event-id "$THREAD_ID-evt-2"

uv run python -m src.onboarding.cli --sessions-dir "$SESSIONS" status $THREAD_ID

uv run python -m src.onboarding.cli --sessions-dir "$SESSIONS" sweep
```

Use the same `--sessions-dir` for bootstrap and CLI (default:
`.onboarding_sessions`). Mismatched dirs look like “stuck at init.”

## Happy path (onboarding)

```
INIT → COLLECT → WELCOME_SENT → AWAIT_DOCUMENTS_SIGNED
     → IT_PROVISIONED → AWAIT_HARDWARE_DELIVERED → SCHEDULE_SENT → COMPLETE
```

| State kind | Examples | `waits_for_input` | What unblocks |
|---|---|---|---|
| Auto side-effect | `WELCOME_SENT`, `IT_PROVISIONED`, `SCHEDULE_SENT` | No | Runs and advances |
| Human park | `COLLECT` | Yes (`human`) | `chat` / human message |
| System park | `AWAIT_DOCUMENTS_SIGNED`, `AWAIT_HARDWARE_DELIVERED` | Yes (`system_event`) | `event` / sweep timeout |
| Terminal | `COMPLETE`, `ESCALATED`, `ERROR` | — | Session done |

## Mental model: reasoning about linear `EngineGraph`

Use this when reading onboarding (or triage/docprocessing) or designing a
similar case pipeline.

### One case, one line

Unlike a chatbot hub, a linear graph is a **case that progresses**:

```
start → … → park → … → park → … → COMPLETE | ESCALATED | ERROR
```

There is no IDLE home to return to. After `COMPLETE` / `ESCALATED` / `ERROR`,
further turns hit terminal behavior (`already_terminal`). Use a **new
`thread_id`** for a new hire / case.

### Two layers (same engine, different contract)

| Layer | Job | Typical source of `proposed_next` |
|---|---|---|
| **Router** | Propose next state | `happy_path[current]` (code) — or optional semantic router that proposes a *state* |
| **Guardrails** | Accept, divert, or reject | Completeness checks, `allowed_transitions`, timeout → escalate |
| **Handlers** | Do the work for *this* state | Partial deltas only; never stamp `current_state` themselves |

**Invariant for onboarding:** the LLM may collect data or pick a username, but
it does **not** choose the next graph state. The router proposes; the guardrail
may keep you in `COLLECT` (incomplete details) or divert to `ESCALATED` /
`ERROR`.

### Thought process for a new linear domain

1. **Draw the happy path** — ordered states from INIT to COMPLETE.
   Put that in `happy_path`. Everything else is a branch (`ESCALATED`,
   `ERROR`, self-loops).

2. **Label each state** before writing handlers:

   | Kind | Meaning | Handler focus |
   |---|---|---|
   | Auto | Side effect, then continue | Idempotent guard flags (`welcome_sent`, …) |
   | Human park | Need user text | `waits_for_input=True`, `wait_kind="human"` |
   | System park | Need external event | `wait_kind="system_event"`, `expected_events=[…]` |
   | Either park | Human *or* system (see triage approval) | `wait_kind="either"` |
   | Terminal | Stop | Empty `allowed_transitions` |

3. **Widen with `allowed_transitions`** — not just the happy edge. Include
   self-loops (`COLLECT → COLLECT`), escalate, and error for every state that
   can legally go there. Missing edges fail as guardrail fallback to `ERROR`,
   not as an authoring-time error.

4. **Handlers return deltas** — update business fields + `output_messages` /
   `audit_trail`. Engine stamps `current_state` / `status` on dispatch.
   Use guard flags so retries never double-send welcome / provision / schedule.

5. **System events** — emit via `aemit_event(event_source="system", event_type=…,
   event_id=…)`. Ledger dedupes by `event_id`. Payload merges into state
   before the parked handler runs. Sweep emits `timeout_escalation` for
   stale parks (onboarding: docs 7d, hardware 3d).

6. **Auto-progress** — after a turn, the engine keeps invoking while
   `waits_for_input=False` until it parks or hits a terminal. That’s why
   one `chat` after collect can run welcome → await documents in one go.

### Turn types (onboarding examples)

| Turn | Command | Effect |
|---|---|---|
| Bootstrap | `ainvoke(…, 'start')` | INIT → park at `collect` |
| Human | `cli chat … "Jane Doe, …"` | Collect agent → if complete, auto through welcome → `await_documents_signed` |
| System | `cli event … document_signed` | Unblock docs → IT provision → park at `await_hardware_delivered` |
| System | `cli event … hardware_delivered` | Unblock hardware → schedule → `complete` |
| Sweep | `cli sweep` | Stale park → `timeout_escalation` → escalate path |

### Debugging checklist

1. Same `sessions_dir` for every process? (CLI default vs bootstrap)
2. `current_state` — parked where you expect, or terminal?
3. Was this human or system? (`event_source` / `event_type`)
4. Did the handler set the business fields / guard flags?
5. What did the router propose? (`proposed_next` / happy path)
6. Did a guardrail divert or self-loop? (e.g. incomplete collect)
7. Is `event_id` unique? (duplicate system events are ledger-deduped)
8. Auto-progress: did non-waiting states run after the park handler?

### When to use linear `EngineGraph` vs `ChatEngineGraph`

| Use linear `EngineGraph` when… | Use `ChatEngineGraph` when… |
|---|---|
| Case progresses INIT → … → COMPLETE | Ongoing conversation, hub home (IDLE) |
| Mostly auto-advance + a few waits | Multiple lanes + sticky multi-turn |
| Code happy-path (or state-proposing router) | Topic classifier + escape |
| System events unblock pipeline parks | System events interrupt chat (`NOTIFY_USER`) |

Onboarding / triage / docprocessing = pipelines. Helpdesk = chatbot.
Same engine core; different session shape and routing contract.

### Design sketch template

1. States + `happy_path` + `terminal_states`
2. `allowed_transitions` (happy + loops + escalate + error)
3. Per state: auto / human park / system park; `expected_events`
4. Handlers + idempotent side-effect flags
5. Guardrails for completeness and diversions
6. Sweep thresholds for system parks
7. CLI: `chat` / `event` / `status` / `sweep` (share `--sessions-dir`)

Reason **current → propose → guard → handle → auto-progress or park**,
not “what did the user say → next node.”
