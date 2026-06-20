# Project Standards — DAG Graph

## Python & Code Style

- **Python Version:** 3.9+
- **Type Hints:** Required for all functions (PEP 484)
- **Imports:** Use `from __future__ import annotations` for forward references
- **Formatting:** Follow PEP 8; max line length 100 characters
- **Docstrings:** Google-style docstrings for modules, classes, and public functions

## Architecture

- **Engine Layer** (`src/engine/`): Reusable, framework-agnostic infrastructure (state management, routing, handlers, validation)
- **Workflow Layer** (`src/workflow/`): Domain-specific logic for document processing pipeline

## Testing

- **Framework:** pytest
- **Structure:** Arrange → Act → Assert in every test
- **Location:** `tests/` directory with structure mirroring `src/`
- **Coverage:** All public functions tested; happy path + error cases
- **Naming:** `test_<function_name>_<scenario>.py` or `test_<module>.py`

## File Naming

- **Modules:** `snake_case` (e.g., `pipeline_state.py`)
- **Classes:** `PascalCase` (e.g., `EngineState`, `SemanticRouter`)
- **Functions:** `snake_case` (e.g., `init_engine_state()`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `HANDLER_MAP_METADATA`)

## Documentation

- **Module docstring:** First line explains purpose; see examples in `src/engine/workflow.py`
- **Type hints:** Always include return types and argument types
- **Complex logic:** Brief inline comment explaining "why", not "what" (code is self-documenting)

## Git & Commits

- **Branch naming:** `phase/<description>-<number>` (e.g., `phase/multi-turn-conversation-1`)
- **Commit type:** `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`
- **Message:** Imperative mood; reference design spec sections when applicable (e.g., "spec §4.1")
- **Atomicity:** One logical change per commit; phases commit together

## Backward Compatibility

- **One-turn workflows** must continue to work unchanged
- **New multi-turn fields** are optional in TypedDict (use `total=False`)
- **process() method** must remain stable; `process_turn()` is new
- **Tests must verify:** Both one-turn and multi-turn paths work

## Dependencies

- **Agno:** State machine workflow framework (see `src/engine/workflow.py`)
- **Claude API:** Via `agno.models.anthropic.Claude` for LLM calls
- **Type checking:** `typing`, `typing_extensions` for TypedDict, Protocol, etc.
- **Testing:** pytest with standard fixtures

## Error Handling

- **Validation errors:** Raise specific exceptions (e.g., `InputValidationError`)
- **Handler exceptions:** Caught and routed to ERROR state (not re-raised)
- **Logging:** Use `logging` module; log at INFO (normal flow), WARNING (guardrail fallback), ERROR (exceptions)

## Code Limits

- **Functions:** Maximum 70 lines of code per function (excluding docstrings)
- **Classes:** Keep single responsibility; avoid God objects
- **Complexity:** Prefer simple over clever; three similar lines > one premature abstraction
