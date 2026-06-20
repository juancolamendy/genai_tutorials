"""
agents.py
────────────────────────────────────────────────────────────────────────────
Reusable Agno agent factory and step builder.

Migrated from triage/lib.py so all Agno-based pipelines can share:
  • make_agent()       — cached Agent factory
  • make_llm_step()    — wrap an agent + prompt-builder into an Agno Step
  • render_as_xml()    — generic list-of-dicts → XML block renderer
  • PromptBuilder      — callable alias for prompt builder functions
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.workflow import Step
from agno.workflow.types import StepInput, StepOutput

# variables
log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Process-level agent cache — each agent is created once.
_AGENT_REGISTRY: dict[str, Agent] = {}

# types
# Callable alias so type annotations are concise everywhere.
PromptBuilder = Callable[[StepInput], str]

# functions
# ── Agent factory ─────────────────────────────────────────────────────────────
def make_agent(
    name:          str,
    description:   str,
    instructions:  list[str],
    output_schema: Optional[Type[BaseModel]] = None,
    model_id:      str = DEFAULT_MODEL,
) -> Agent:
    """
    Return a cached Claude Agent.

    Calling this twice with the same `name` returns the same instance so no
    duplicate model clients are created across the codebase.
    """
    if name in _AGENT_REGISTRY:
        return _AGENT_REGISTRY[name]

    kwargs: dict[str, Any] = dict(
        name         = name,
        model        = Claude(id=model_id),
        description  = description,
        instructions = instructions,
    )
    if output_schema is not None:
        kwargs["output_schema"] = output_schema

    agent = Agent(**kwargs)
    _AGENT_REGISTRY[name] = agent
    log.debug("[lib] registered agent '%s'", name)
    return agent


def get_agent(name: str) -> Agent:
    """Retrieve a registered agent by name; raises KeyError if not found."""
    if name not in _AGENT_REGISTRY:
        raise KeyError(f"Agent '{name}' has not been registered.")
    return _AGENT_REGISTRY[name]


# ── Step builder ──────────────────────────────────────────────────────────────
def make_llm_step(
    name:         str,
    agent:        Agent,
    build_prompt: PromptBuilder,
) -> Step:
    """
    Wrap an Agent in a named Agno Step.

    The `build_prompt` callable assembles the full prompt string from the
    StepInput; the agent's `.content` (typed Pydantic object or string) is
    stored in StepOutput.content.
    """
    def _executor(step_input: StepInput) -> StepOutput:
        prompt = build_prompt(step_input)
        result = agent.run(prompt)
        return StepOutput(content=result.content)

    return Step(name=name, executor=_executor)


# ── Prompt helpers ────────────────────────────────────────────────────────────
def render_as_xml(
    tag:      str,
    items:    list[dict],
    max_items: int = 10,
    *,
    role_key:    str = "role",
    content_key: str = "content",
    attrs:       tuple[str, ...] = (),
) -> str:
    """
    Generic list-of-dicts → XML block renderer.

    Args:
        tag:         Outer XML tag (e.g. "history", "documents").
        items:       List of dicts to render.
        max_items:   Truncate to the last N items.
        role_key:    Dict key for the element's `role` attribute.
        content_key: Dict key for the element's text content.
        attrs:       Extra dict keys to include as XML attributes.

    Example (tag="history", role_key="role", content_key="content"):
        <history>
          <turn role="user" intent="billing">I was charged twice</turn>
          <turn role="assistant">I can help with that…</turn>
        </history>
    """
    recent = items[-max_items:]
    if not recent:
        return ""

    inner_tag = "item" if tag in {"documents", "entries"} else "turn"
    lines: list[str] = []
    for item in recent:
        role_val    = item.get(role_key, "")
        content_val = item.get(content_key, "")
        attr_str    = f' {role_key}="{role_val}"' if role_val else ""
        for a in attrs:
            if item.get(a):
                attr_str += f' {a}="{item[a]}"'
        lines.append(f"  <{inner_tag}{attr_str}>{content_val}</{inner_tag}>")

    return f"<{tag}>\n" + "\n".join(lines) + f"\n</{tag}>"

