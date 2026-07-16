# Async Handlers (Hybrid) Design

**Date:** 2026-07-16  
**Status:** Implemented (option C)

## Decision

Handlers may be sync **or** async. Convert I/O-heavy handlers to `async def` + `await chain.ainvoke(...)`. Trivial handlers stay sync.

## Engine

- `safe_node` wraps async funcs with an async error wrapper
- `_dispatch_handler` (sync) rejects coroutine handlers with `TypeError`
- `_adispatch_handler` awaits coroutine handlers; runs sync handlers inline
- `_make_handler_node` registers async nodes for async handlers (ainvoke-only)
- `ainvoke` / `astream` resume-at-park use `_adispatch_handler`

## Domains converted

| Domain | Async handlers (LLM) |
|---|---|
| hrhelpdesk | idle, clarify, topic_faq, topic_escalate, topic_booking (+ booking/route helpers); classify/escape via `aclassify` / `acheck` |
| onboarding | collect, it_provisioned |
| docprocessing | validate, enrich, human_review |

Left sync (no LLM): notify/error/complete/store/fetch/upload/welcome/await_*, in-memory fake services (`create_ticket`, `confirm_booking`, …).

Router/escape engine APIs: sync `classify`/`check` kept for unit tests; handlers use async twins.

## Caller rule

If any handler on the turn path is async, use `ainvoke` / `aemit_event` / `astream` — not sync `invoke`.
