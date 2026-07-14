"""Agent factory / registry module.

This is the code you shared, plus the minimal header it needs to run
(imports, DEFAULT_MODEL, _agent_registry, log). If you already have this
in your project, point the tests at your module instead.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Type

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

log = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

_agent_registry: dict[str, Any] = {}


# ── Agent factory ─────────────────────────────────────────────────────────────
def make_llm_agent(
    name: str,
    system_prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
    model_id: str = DEFAULT_MODEL,
    tools: Optional[Sequence[Any]] = None,
) -> Any:
    """Return a cached tool-calling agent built with ``create_agent``."""
    if name in _agent_registry:
        return _agent_registry[name]

    agent = create_agent(
        model=model_id,
        tools=list(tools) if tools else None,
        system_prompt=system_prompt,
        response_format=output_schema,
        name=name,
    )
    _agent_registry[name] = agent
    log.debug("[engine] registered agent '%s'", name)
    return agent


def get_agent(name: str) -> Any:
    """Retrieve a registered agent by name; raises KeyError if not found."""
    if name not in _agent_registry:
        raise KeyError(f"Agent '{name}' has not been registered.")
    return _agent_registry[name]


async def ainvoke_agent(
    name: str,
    user_prompt: str,
    *,
    messages: Optional[list[Any]] = None,
) -> Any:
    """Async-invoke a registered agent with a user prompt (or message list)."""
    agent = get_agent(name)
    if messages is None:
        messages = [HumanMessage(content=user_prompt)]
    return await agent.ainvoke({"messages": messages})

