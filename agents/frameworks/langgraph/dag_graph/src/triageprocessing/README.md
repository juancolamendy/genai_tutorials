# triageprocessing — Sentry triage-and-fix harness

Uses the same EngineGraph patterns as `onboarding/`. The review gate is
`AWAIT_APPROVAL` (`wait_kind=either`) — **not** LangGraph `interrupt()`.

## Identity: `thread_id`

One LangGraph / EngineGraph **thread** = one task run.

| Name in docs / shell | Engine field | Meaning |
|----------------------|--------------|---------|
| `THREAD_ID` | `session_id` / `thread_id` | Checkpoint key for this run |
| (state) `run_id` | same string | Business label stored in TriageState |

`run` writes the active thread to `$SESSIONS/.current_thread_id`. Follow-up
commands default to that thread — same idea as onboarding’s `$THREAD_ID`,
without re-typing it each time.

## How to run

```bash
export SESSIONS=.triage_sessions
export ARTS=.triage_artifacts
export THREAD_ID=thread-1

# Start (or resume a known thread): parks at await_approval
uv run python -m src.triageprocessing.cli --sessions-dir "$SESSIONS" --artifacts-dir "$ARTS" \
  run --repo /tmp/does-not-need-to-exist --issue PROJ-142 --thread-id "$THREAD_ID"

# Defaults to the current thread saved by `run` (here: thread-1)
uv run python -m src.triageprocessing.cli --sessions-dir "$SESSIONS" pending "$THREAD_ID"
uv run python -m src.triageprocessing.cli --sessions-dir "$SESSIONS" approve "$THREAD_ID" --by alice
# or: reject --by alice --note "needs tests"
# or human path: chat "approve"

uv run python -m src.triageprocessing.cli --sessions-dir "$SESSIONS" report
```

If you omit `--thread-id` on `run`, a new id like `run-<hex>` is generated and
saved as the current thread.
