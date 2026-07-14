"""Generic chain/agent factory and LLM step builder using LangChain LCEL.

Provides:
  • make_chain() — cached LCEL chain factory
  • make_llm_chain() — wrap an LLM + prompt-builder into a chain
  • make_llm_agent() — cached tool-calling agent factory (create_agent)
  • ainvoke_agent() — async invoke a registered agent
  • render_as_xml() — generic list-of-dicts → XML block renderer
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional, Sequence, Type

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

log = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

# Role → model id. Env overrides: ENGINE_MODEL_ROUTER, ENGINE_MODEL_TOPIC.
_MODEL_BY_ROLE: dict[str, str] = {
    "router": os.getenv("ENGINE_MODEL_ROUTER", DEFAULT_MODEL),
    "topic": os.getenv("ENGINE_MODEL_TOPIC", DEFAULT_MODEL),
}


def get_model(role: str) -> str:
    """Return the model id for a logical role (``router``, ``topic``, …).

    Unknown roles fall back to ``DEFAULT_MODEL``. Domains and factories
    should prefer roles over hard-coded model strings so router vs topic
    models can swap independently.
    """
    return _MODEL_BY_ROLE.get(role, DEFAULT_MODEL)

# Process-level chain cache — each chain is created once.
_chain_registry: dict[str, Any] = {}

# Process-level agent cache — each agent is created once.
_agent_registry: dict[str, Any] = {}

# Global LLM instance
_llm: Optional[BaseChatModel] = None

# Types
# Callable alias so type annotations are concise everywhere.
PromptBuilder = Callable[[dict[str, Any]], str]


def _get_llm(model_id: str = DEFAULT_MODEL) -> BaseChatModel:
    """Get or create a cached chat model instance (provider-agnostic).

    model_id follows init_chat_model's "provider:model" format
    (e.g. "anthropic:claude-haiku-4-5-20251001", "openai:gpt-4o-mini").
    The provider's API key is picked up from environment automatically.
    """
    global _llm
    if _llm is None:
        _llm = init_chat_model(model_id)
    return _llm


# ── Chain factory ─────────────────────────────────────────────────────────────
def make_chain(
    name: str,
    system_prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
    model_id: str = DEFAULT_MODEL,
    user_prompt: str = "{input}",
) -> Any:
    """Return a cached LCEL chain using the configured chat model.

    Calling this twice with the same `name` returns the same instance so no
    duplicate LLM clients are created across the codebase.

    Args:
        name: Chain identifier for caching
        system_prompt: System message for the model
        output_schema: Optional Pydantic model for structured output
        model_id: Chat model ID in "provider:model" format
            (e.g. "anthropic:claude-haiku-4-5-20251001")
        user_prompt: User message template (default "{input}")

    Returns:
        LCEL chain (prompt | llm | parser)
    """
    if name in _chain_registry:
        return _chain_registry[name]

    llm = _get_llm(model_id)

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])

    # Choose parser based on output schema
    if output_schema is not None:
        parser = JsonOutputParser(pydantic_object=output_schema)
    else:
        parser = StrOutputParser()

    # Build LCEL chain using pipe operator
    chain = prompt | llm | parser
    _chain_registry[name] = chain
    log.debug("[engine] registered chain '%s'", name)
    return chain


def get_chain(name: str) -> Any:
    """Retrieve a registered chain by name; raises KeyError if not found."""
    if name not in _chain_registry:
        raise KeyError(f"Chain '{name}' has not been registered.")
    return _chain_registry[name]


def chain_field(result: Any, name: str, default: Any = None) -> Any:
    """Read a field off a chain result that may be a dict or a Pydantic object.

    JsonOutputParser can hand back either shape depending on parser/model
    version, so callers would otherwise repeat this isinstance check per field.
    """
    return result.get(name, default) if isinstance(result, dict) else getattr(result, name, default)


# ── Step builder ──────────────────────────────────────────────────────────────
def make_llm_chain(
    name: str,
    build_prompt: PromptBuilder,
    output_schema: Optional[Type[BaseModel]] = None,
    model_id: str = DEFAULT_MODEL,
) -> Callable[[dict[str, Any]], Any]:
    """Wrap an LLM + prompt into a chain function.

    The `build_prompt` callable assembles the full prompt string from the
    state dict; the LLM response is parsed and returned.

    Args:
        name: Chain identifier
        build_prompt: Callable that builds prompt from state dict
        output_schema: Optional Pydantic model for structured output
        model_id: Chat model ID in "provider:model" format
            (e.g. "anthropic:claude-haiku-4-5-20251001")

    Returns:
        Chain executor function
    """

    def _executor(state: dict[str, Any]) -> Any:
        prompt_text = build_prompt(state)
        llm = _get_llm(model_id)

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("user", prompt_text),
        ])

        # Choose parser
        if output_schema is not None:
            parser = JsonOutputParser(pydantic_object=output_schema)
        else:
            parser = StrOutputParser()

        # Build and execute chain
        chain = prompt | llm | parser
        return chain.invoke({})

    return _executor


# ── Agent factory ─────────────────────────────────────────────────────────────
def make_llm_agent(
    name: str,
    system_prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
    model_id: str = DEFAULT_MODEL,
    tools: Optional[Sequence[Any]] = None,
) -> Any:
    """Return a cached tool-calling agent built with ``create_agent``.

    Calling this twice with the same ``name`` returns the same instance so no
    duplicate agent graphs are created across the codebase.

    In current LangChain, ``create_agent`` returns a ``CompiledStateGraph``
    (not an ``AgentExecutor``). Use ``ainvoke_agent`` / ``agent.ainvoke`` to run
    it; when ``output_schema`` is set, the result includes
    ``structured_response``.

    Args:
        name: Agent identifier for caching
        system_prompt: Top-level instructions, constraints, and personality
        output_schema: Optional Pydantic model bound as ``response_format``
        model_id: Chat model ID in "provider:model" format
            (e.g. "anthropic:claude-haiku-4-5-20251001")
        tools: Optional sequence of tools the agent may call

    Returns:
        Compiled agent graph (``create_agent`` result)
    """
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
    """Async-invoke a registered agent with a user prompt (or message list).

    Equivalent to the older ``AgentExecutor.ainvoke`` pattern: looks up the
    agent from ``_agent_registry`` and calls ``ainvoke`` on the compiled graph.

    Args:
        name: Registered agent name (from ``make_llm_agent``)
        user_prompt: User message content (used when ``messages`` is omitted)
        messages: Optional full messages list. Each item may be a LangChain
            message, a ``(role, content)`` tuple, or a
            ``{"role": ..., "content": ...}`` dict. When omitted, a single
            human message is built from ``user_prompt``.

    Returns:
        Agent result dict. With ``output_schema``, includes
        ``structured_response``.
    """
    agent = get_agent(name)
    if messages is None:
        messages = [HumanMessage(content=user_prompt)]
    return await agent.ainvoke({"messages": messages})


# ── Prompt helpers ────────────────────────────────────────────────────────────
def render_as_xml(
    tag: str,
    items: list[dict],
    max_items: int = 10,
    *,
    role_key: str = "role",
    content_key: str = "content",
    attrs: tuple[str, ...] = (),
) -> str:
    """Generic list-of-dicts → XML block renderer.

    Args:
        tag:         Outer XML tag (e.g. "history", "documents").
        items:       List of dicts to render.
        max_items:   Truncate to the last N items.
        role_key:    Dict key for the element's `role` attribute.
        content_key: Dict key for the element's text content.
        attrs:       Extra dict keys to include as XML attributes.

    Example (tag="history", role_key="role", content_key="content"):
        <history>
          <turn role="user">I have a question</turn>
          <turn role="assistant">I can help with that…</turn>
        </history>
    """
    recent = items[-max_items:]
    if not recent:
        return ""

    inner_tag = "item" if tag in {"documents", "entries"} else "turn"
    lines: list[str] = []
    for item in recent:
        role_val = item.get(role_key, "")
        content_val = item.get(content_key, "")
        attr_str = f' {role_key}="{role_val}"' if role_val else ""
        for a in attrs:
            if item.get(a):
                attr_str += f' {a}="{item[a]}"'
        lines.append(f"  <{inner_tag}{attr_str}>{content_val}</{inner_tag}>")

    return f"<{tag}>\n" + "\n".join(lines) + f"\n</{tag}>"
