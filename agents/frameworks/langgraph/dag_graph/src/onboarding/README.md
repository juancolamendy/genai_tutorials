# README

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
