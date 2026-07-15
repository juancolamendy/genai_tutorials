"""Sticky-topic escape detection for chatbot workflows.

Binary leave-intent check (``escape: bool``), complementary to
``DefaultTopicRouter`` (topic + confidence). Domains override
``get_instructions()`` / ``system_prompt`` when needed; the default prompt
matches the original helpdesk escape chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from src.engine.chains import get_model, make_chain

log = logging.getLogger(__name__)

DEFAULT_ESCAPE_INSTRUCTIONS = """You detect whether the user wants to leave the current topic lane.

Return escape=true if they clearly want to switch topics, cancel, or start over.
Return escape=false if they are continuing the current conversation lane.

Respond ONLY with valid JSON:
{{
  "escape": <bool>
}}"""


class EscapeOutput(BaseModel):
    """Structured LLM output for escape detection."""

    escape: bool


@dataclass(frozen=True)
class EscapeDecision:
    """Result of sticky-topic escape detection."""

    escape: bool


class EscapeChecker(Protocol):
    """Minimal protocol for leave-intent classifiers."""

    def check(self, text: str) -> EscapeDecision: ...


class DefaultEscapeChecker:
    """LLM escape checker reusable across chatbot domains.

    Subclasses override ``get_instructions()`` / ``system_prompt`` (or set
    ``chain_name`` / ``model_role``) when domain wording differs. On any
    failure, returns ``EscapeDecision(escape=False)`` so a flaky model keeps
    the user in the sticky lane rather than falsely clearing it.
    """

    output_schema: type = EscapeOutput
    system_prompt: str = DEFAULT_ESCAPE_INSTRUCTIONS
    model_role: str = "router"
    chain_name: str = "EscapeCheckerChain"

    def __init__(self) -> None:
        self._chain = None

    def get_instructions(self) -> str:
        return self.system_prompt

    def _get_chain(self):
        if self._chain is None:
            self._chain = make_chain(
                name=self.chain_name,
                system_prompt=self.get_instructions(),
                output_schema=self.output_schema,
                model_id=get_model(self.model_role),
            )
        return self._chain

    def check(self, text: str) -> EscapeDecision:
        """Classify whether ``text`` is leaving the current sticky topic."""
        try:
            raw = self._get_chain().invoke({"input": text})
            response: Any = (
                raw if isinstance(raw, self.output_schema) else self.output_schema(**raw)
            )
            return EscapeDecision(escape=bool(response.escape))
        except Exception as exc:
            log.exception("[DefaultEscapeChecker] check() failed: %s", exc)
            return EscapeDecision(escape=False)


__all__ = [
    "DEFAULT_ESCAPE_INSTRUCTIONS",
    "EscapeOutput",
    "EscapeDecision",
    "EscapeChecker",
    "DefaultEscapeChecker",
]
