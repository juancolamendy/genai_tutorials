# Design: triageprocessing (Sentry triage-and-fix harness)

**Approved gate mode:** Option C — `AWAIT_APPROVAL` uses `wait_kind="either"` with `expected_events=["approve","reject"]`.

## Goal

Port the build-your-own-harness LangGraph illustration onto this repo’s `EngineGraph` domain pattern (`onboarding/`), **without** LangGraph `interrupt()`.

## Gate

- Pre-gate work (intake → gather → patch → diff) auto-progresses and checkpoints.
- Park at `AWAIT_APPROVAL`.
- Resume via human (`chat` / text approve|reject) **or** system (`event_type` approve|reject + `event_id`).
- Publish / reject / report happen only after the gate; `published` is idempotent.

## States

`INIT → INTAKE → GATHER_CONTEXT → EXECUTE_PATCH → COLLECT_DIFF → AWAIT_APPROVAL`
→ approve path: `PUBLISH → WRITE_REPORT → COMPLETE`
→ reject path: guardrail divert to `REJECT → WRITE_REPORT → REJECTED`
→ `ERROR` terminal

## Layout

`state_transitions`, `session_state`, `artifacts`, `clients` (not adapters), `worker`, `worktree`, `handlers`, `guardrails`, `graph`, `cli`, tests.
