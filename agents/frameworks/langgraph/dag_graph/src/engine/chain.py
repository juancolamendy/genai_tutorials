"""Generic chain factory and LLM step builder using LangChain LCEL.

Provides:
  • make_chain() — cached LCEL chain factory
  • make_llm_chain() — wrap an LLM + prompt-builder into a chain
  • render_as_xml() — generic list-of-dicts → XML block renderer
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Type

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Process-level chain cache — each chain is created once.
_CHAIN_REGISTRY: dict[str, Any] = {}

# Global LLM instance
_llm: Optional[ChatAnthropic] = None

# Types
# Callable alias so type annotations are concise everywhere.
PromptBuilder = Callable[[dict[str, Any]], str]


def _get_llm(model_id: str = DEFAULT_MODEL) -> ChatAnthropic:
    """Get or create a cached ChatAnthropic instance."""
    import os

    global _llm
    if _llm is None:
        # ChatAnthropic will automatically use ANTHROPIC_API_KEY from environment
        _llm = ChatAnthropic(model=model_id, api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _llm


# ── Chain factory ─────────────────────────────────────────────────────────────
def make_chain(
    name: str,
    description: str,
    system_prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
    model_id: str = DEFAULT_MODEL,
) -> Any:
    """Return a cached LCEL chain using Claude.

    Calling this twice with the same `name` returns the same instance so no
    duplicate LLM clients are created across the codebase.

    Args:
        name: Chain identifier for caching
        description: Chain purpose (for logging)
        system_prompt: System message for the model
        output_schema: Optional Pydantic model for structured output
        model_id: Claude model ID to use

    Returns:
        LCEL chain (prompt | llm | parser)
    """
    if name in _CHAIN_REGISTRY:
        return _CHAIN_REGISTRY[name]

    llm = _get_llm(model_id)

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
    ])

    # Choose parser based on output schema
    if output_schema is not None:
        parser = JsonOutputParser(pydantic_object=output_schema)
    else:
        parser = StrOutputParser()

    # Build LCEL chain using pipe operator
    chain = prompt | llm | parser
    _CHAIN_REGISTRY[name] = chain
    log.debug("[engine] registered chain '%s'", name)
    return chain


def get_chain(name: str) -> Any:
    """Retrieve a registered chain by name; raises KeyError if not found."""
    if name not in _CHAIN_REGISTRY:
        raise KeyError(f"Chain '{name}' has not been registered.")
    return _CHAIN_REGISTRY[name]


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
        model_id: Claude model ID to use

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
