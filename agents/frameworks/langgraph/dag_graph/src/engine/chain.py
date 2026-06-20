"""Generic chain factory and LLM step builder for LangChain.

Provides:
  • make_chain() — cached chain factory
  • make_llm_chain() — wrap an LLM + prompt-builder into a chain
  • render_as_xml() — generic list-of-dicts → XML block renderer
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Process-level chain cache — each chain is created once.
_CHAIN_REGISTRY: dict[str, Any] = {}

# Types
# Callable alias so type annotations are concise everywhere.
PromptBuilder = Callable[[dict[str, Any]], str]


# ── Chain factory ─────────────────────────────────────────────────────────────
def make_chain(
    name: str,
    description: str,
    llm: Any,
    output_schema: Optional[Type[BaseModel]] = None,
) -> Any:
    """Return a cached LLM chain.

    Calling this twice with the same `name` returns the same instance so no
    duplicate LLM clients are created across the codebase.
    """
    if name in _CHAIN_REGISTRY:
        return _CHAIN_REGISTRY[name]

    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    # Create a basic prompt template (can be customized per use case)
    prompt = PromptTemplate(
        input_variables=["input"],
        template="{input}",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
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
    llm: Any,
    build_prompt: PromptBuilder,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap an LLM in a named chain function.

    The `build_prompt` callable assembles the full prompt string from the
    state dict; the LLM response is stored in the state.
    """

    def _executor(state: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(state)
        response = llm.invoke(prompt)
        return {"content": response.content if hasattr(response, "content") else str(response)}

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