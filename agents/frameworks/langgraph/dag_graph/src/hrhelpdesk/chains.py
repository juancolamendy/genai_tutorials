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


class EscalateDecision(BaseModel):
    """Structured escalate decision — handler alone applies side effects."""

    should_create: bool = Field(
        description="True when the user describes a concrete issue or asks to open a ticket"
    )
    subject: str = Field(default="", description="Ticket subject when should_create")
    body: str = Field(default="", description="Ticket body when should_create")
    reply: str = Field(
        description="User-facing confirmation or refusal; do not invent a ticket id"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS (booking lane only — escalate uses structured output)
# ─────────────────────────────────────────────────────────────────────────────


@tool
def check_desk_availability(date: str, location: str) -> bool:
    """Check whether a desk is available on the given date and location."""
    return services.check_desk_availability(date, location)


@tool
def confirm_booking(date: str, location: str, seat_pref: str) -> str:
    """Confirm a desk booking once date, location, and seat preference are known."""
    return services.confirm_booking(date, location, seat_pref)


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIST CHAINS / AGENTS
# ─────────────────────────────────────────────────────────────────────────────

faq_agent = make_llm_agent(
    name="HelpdeskFaqAgent",
    system_prompt="""You answer HR policy questions using only the provided knowledge snippets.

Cite source ids in your answer. Do not invent policy. If snippets are insufficient,
say what is missing. Never offer to book desks or create tickets.""",
    tools=[],
    model_id=get_model("topic"),
)

escalate_chain = make_chain(
    name="HelpdeskEscalateChain",
    system_prompt="""You help employees escalate HR issues by deciding whether to create
a support ticket. This lane is one-shot: one user message, then the topic closes.

Set should_create=true whenever the user describes a concrete problem OR explicitly
asks to open/create a ticket. Build subject and body from what they already said
(infer reasonable defaults; do not ask clarifying questions). Example:
subject "Payroll shortfall", body summarizing date/amount/issue from their message.

Set should_create=false only when there is nothing actionable (empty / unrelated).

reply: brief user-facing text. If creating, confirm you are filing a ticket but
do NOT invent a ticket id — the system assigns one. If not creating, explain briefly.

Respond ONLY with valid JSON matching this structure:
{{
  "should_create": <bool>,
  "subject": <str>,
  "body": <str>,
  "reply": <str>
}}""",
    output_schema=EscalateDecision,
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
    "EscalateDecision",
    "check_desk_availability",
    "confirm_booking",
    "faq_agent",
    "escalate_chain",
    "booking_agent",
]
