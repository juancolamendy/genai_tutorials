# Plan: Rename status fields

## Goal
- Session health: `status` → `session_status` (engine-owned on state)
- Handler outcome: keep `handler_status`
- Emit/API gate result: `status` → `emit_status` on `aemit_event` / stream chunks / `arun_to_completion`

## Do NOT rename
- Booking record `status` (`confirmed` / `cancelled`)
- Stream chunk `type: "status"` (chunk kind)
- CLI subcommand `status`

## Tasks
1. `EngineSessionState` + `new_engine_session_state`: `session_status`
2. `engine_graph.py`: safe_node, dispatch, guardrail park, invoke errors, aemit/astream/arun keys
3. Domains: docprocessing handlers that set `"status": "error"` → `handler_status`; CLIs/READMEs
4. Tests: assert `emit_status` on emit results; `session_status` on state
5. Run related pytest

## Verify
`PYTHONPATH=. uv run pytest tests/test_aemit_event.py tests/test_hrhelpdesk_flow.py tests/test_engine_session_state.py tests/test_onboarding_flow.py tests/test_triageprocessing.py -q --no-cov`
