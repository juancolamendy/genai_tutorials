# Design: HR helpdesk escalate via structured output

## Problem
`create_ticket_tool` and `handle_topic_escalate` both called `services.create_ticket`, producing two ticket ids per turn. Tool agents also cost an extra LLM round-trip for a one-shot lane.

## Decision
Replace escalate tool-agent with `make_chain` + `EscalateDecision` structured output. Handler alone creates the ticket and updates `open_tickets`. Booking stays agent+tools.

## Schema
- `should_create: bool`
- `subject: str`, `body: str` (when creating)
- `reply: str` (user-facing; handler appends real ticket id)

## Handler
1. `escalate_chain.invoke({"input": message})`
2. If `should_create`: ledger → one `create_ticket` → `open_tickets`
3. `close_topic_delta` + `output_messages`

## Out of scope
Booking double-`confirm_booking` harvest; FAQ.
