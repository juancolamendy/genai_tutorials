"""LLM model registry — engine infrastructure.

Provides the types and registry mechanism. Model registrations (which key
maps to which provider and model name) live in the application layer (main.py)
so the engine stays provider-agnostic.

Usage in main.py::

    from engine.llm_config import ModelProvider, ModelSpec, register_model, load_model

    CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
    register_model(CLAUDE_HAIKU, ModelSpec(ModelProvider.ANTHROPIC, "claude-haiku-4-5-20251001"))

Usage in agent markdown frontmatter::

    model: claude-haiku-4-5-20251001
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"


@dataclass(frozen=True)
class ModelSpec:
    provider: ModelProvider
    model_name: str
    thinking: bool = False


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(key: str, spec: ModelSpec) -> None:
    """Add or overwrite a model entry in the registry.

    Args:
        key: The model-id string used in code and agent markdown frontmatter.
        spec: Provider and model name for this key.
    """
    _REGISTRY[key] = spec


def load_model(key: str, **kwargs: Any) -> Any:
    """Return an Agno model instance for the given registry key.

    Args:
        key: A key previously registered via register_model().
        **kwargs: Extra arguments forwarded to the model constructor
            (e.g. cache_system_prompt=True).

    Raises:
        KeyError: If key has not been registered.
    """
    spec = _REGISTRY[key]
    if spec.provider == ModelProvider.ANTHROPIC:
        from agno.models.anthropic import Claude
        return Claude(id=spec.model_name, thinking=spec.thinking, **kwargs)
    if spec.provider == ModelProvider.GOOGLE:
        from agno.models.google.gemini import Gemini
        return Gemini(id=spec.model_name, **kwargs)
    if spec.provider == ModelProvider.GROQ:
        from agno.models.groq import Groq
        return Groq(id=spec.model_name, **kwargs)
    raise ValueError(f"Unhandled provider: {spec.provider}")
