# HR Helpdesk Assistant

Hub + semantic-router chatbot built on `EngineGraph` with sticky topic lanes
(FAQ, escalate, book-desk).

## Layout

| Module | Role |
|---|---|
| `graph.py` | `EngineGraph` subclass — routing override, system-event legality |
| `handlers.py` | State handlers (hub park, topic specialists, notify) |
| `chains.py` | Router/escape chains + `make_llm_agent` specialists |
| `providers.py` | In-memory fake RAG, tickets, desk booking |
| `cli.py` | `chat`, `event`, `sweep`, `status`, `serve` |
| `sweep.py` | 48h booking timeout sweep |

## CLI

```bash
python -m src.hrhelpdesk.cli chat hd-emp-1 "How much PTO do I get?"
python -m src.hrhelpdesk.cli event hd-emp-1 ticket_resolved --event-id evt-1 --payload ticket_id=TICKET-1
python -m src.hrhelpdesk.cli sweep
python -m src.hrhelpdesk.cli status hd-emp-1
python -m src.hrhelpdesk.cli serve
```

Human turns (`chat`, `serve`) stream assistant tokens via `aemit_event_stream`.

Sessions persist under `.hrhelpdesk_sessions` with a sibling `.hrhelpdesk_sessions_ledger`.

## Design

See `docs/design/design_hrhelpdesk-2026-07-14.md`.
