# Design: Remove make_llm_agent; structured chains everywhere in hrhelpdesk

## Decision
- Convert FAQ + booking to `make_chain` + Pydantic decisions (same as escalate).
- Handlers alone call `services.*` (availability, confirm_booking, create_ticket).
- Delete engine `make_llm_agent` / `invoke_agent` / `ainvoke_agent` / `astream_agent` / `get_agent` and `find_tool_call` / `find_tool_message`.

## Out of scope
- Onboarding `collect_agent` still uses LangChain `create_agent` directly (not `make_llm_agent`).
