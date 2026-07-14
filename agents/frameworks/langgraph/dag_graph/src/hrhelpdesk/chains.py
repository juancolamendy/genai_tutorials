"""LLM chains and tool-calling agents for the HR helpdesk chatbot."""

from __future__ import annotations

from enum import Enum

from langchain.tools import tool
from pydantic import BaseModel, Field

from src.engine.chains import get_model, make_chain, make_llm_agent

from . import services

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────


class RouterTopic(str, Enum):
    FAQ = "faq"
    ESCALATE = "escalate"
    BOOKING = "booking"
    UNCLEAR = "unclear"


class RouterOutput(BaseModel):
    topic: RouterTopic
    confidence: float = Field(ge=0.0, le=1.0)


class EscapeOutput(BaseModel):
    escape: bool


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER / ESCAPE CHAINS
# ─────────────────────────────────────────────────────────────────────────────

escape_chain = make_chain(
    name="HelpdeskEscapeChain",
    system_prompt="""You detect whether the user wants to leave the current topic lane.

Return escape=true if they clearly want to switch topics, cancel, or start over.
Return escape=false if they are continuing the current conversation lane.

Respond ONLY with valid JSON:
{{
  "escape": <bool>
}}""",
    output_schema=EscapeOutput,
    model_id=get_model("router"),
)


def run_escape(text: str) -> EscapeOutput:
    """Invoke the escape chain (patchable in tests)."""
    result = escape_chain.invoke({"input": text})
    if isinstance(result, EscapeOutput):
        return result
    return EscapeOutput(**result)


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────


@tool
def create_ticket_tool(subject: str, body: str) -> str:
    """Create an HR support ticket. Call once the issue is understood."""
    return services.create_ticket(subject, body)


@tool
def check_desk_availability(date: str, location: str) -> bool:
    """Check whether a desk is available on the given date and location."""
    return services.check_desk_availability(date, location)


@tool
def confirm_booking(date: str, location: str, seat_pref: str) -> str:
    """Confirm a desk booking once date, location, and seat preference are known."""
    return services.confirm_booking(date, location, seat_pref)


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIST AGENTS
# ─────────────────────────────────────────────────────────────────────────────

faq_agent = make_llm_agent(
    name="HelpdeskFaqAgent",
    system_prompt="""You answer HR policy questions using only the provided knowledge snippets.

Cite source ids in your answer. Do not invent policy. If snippets are insufficient,
say what is missing. Never offer to book desks or create tickets.""",
    tools=[],
    model_id=get_model("topic"),
)

escalate_agent = make_llm_agent(
    name="HelpdeskEscalateAgent",
    system_prompt="""You help employees escalate HR issues.

Gather a concise subject and body, then call create_ticket_tool once.
Do not call the tool until both subject and body are clear.""",
    tools=[create_ticket_tool],
    model_id=get_model("topic"),
)

booking_agent = make_llm_agent(
    name="HelpdeskBookingAgent",
    system_prompt="""You help employees book a desk.

Collect date, location, and seat preference across turns. Use check_desk_availability
before confirming. Call confirm_booking only when all three slots are confirmed by the user.""",
    tools=[check_desk_availability, confirm_booking],
    model_id=get_model("topic"),
)


__all__ = [
    "RouterTopic",
    "RouterOutput",
    "EscapeOutput",
    "escape_chain",
    "run_escape",
    "create_ticket_tool",
    "check_desk_availability",
    "confirm_booking",
    "faq_agent",
    "escalate_agent",
    "booking_agent",
]
