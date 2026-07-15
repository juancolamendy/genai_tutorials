"""LLM chains for the HR helpdesk chatbot (structured output; no tool agents)."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.engine.chains import get_model, make_chain

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


class FaqAnswer(BaseModel):
    """FAQ specialist reply grounded in provided snippets."""

    answer: str = Field(description="Policy answer citing source ids from the snippets")


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


class BookingDecision(BaseModel):
    """Booking slot fill / confirm decision — handler alone calls booking services."""

    date: Optional[str] = Field(default=None, description="ISO date if known this turn")
    location: Optional[str] = Field(default=None, description="Office location if known")
    seat_pref: Optional[str] = Field(default=None, description="Seat preference if known")
    confirm: bool = Field(
        default=False,
        description="True only when date, location, and seat_pref are all confirmed",
    )
    reply: str = Field(description="User-facing progress or confirmation text")


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIST CHAINS
# ─────────────────────────────────────────────────────────────────────────────

faq_chain = make_chain(
    name="HelpdeskFaqChain",
    system_prompt="""You answer HR policy questions using only the provided knowledge snippets.

Cite source ids in your answer. Do not invent policy. If snippets are insufficient,
say what is missing. Never offer to book desks or create tickets.

Respond ONLY with valid JSON matching this structure:
{{
  "answer": <str>
}}""",
    output_schema=FaqAnswer,
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

booking_chain = make_chain(
    name="HelpdeskBookingChain",
    system_prompt="""You help employees book a desk across multiple turns.

The user message includes currently known slots (date, location, seat_pref) and
their latest utterance. Extract any new slot values. Set confirm=true ONLY when
all three of date, location, and seat_pref are known and the user has confirmed
them (do not invent missing slots).

reply: ask for the next missing slot, or confirm you will book when confirm=true.
Do not invent booking ids.

Respond ONLY with valid JSON matching this structure:
{{
  "date": <str or null>,
  "location": <str or null>,
  "seat_pref": <str or null>,
  "confirm": <bool>,
  "reply": <str>
}}""",
    output_schema=BookingDecision,
    model_id=get_model("topic"),
)


__all__ = [
    "RouterTopic",
    "RouterOutput",
    "FaqAnswer",
    "EscalateDecision",
    "BookingDecision",
    "faq_chain",
    "escalate_chain",
    "booking_chain",
]
