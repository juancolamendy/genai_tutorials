"""
Agno — DAG Pattern: Ticket Triage
─────────────────────────────────────────────────────────────────────────────
Shape:  Classify → Route → Handle → Validate → Respond
        Linear, fixed, no back-transitions.
        Agno Workflow with Step + Router + code-owned transitions.

Structured output:
        Each LLM step declares an `output_schema` (a Pydantic model), so Agno
        constrains the model to that shape and hands back a validated object —
        no manual JSON parsing, no fence-stripping, no fallbacks-on-garbage.
        The typed object flows between steps as `previous_step_content`.

Install:
    uv add agno python-dotenv

Usage:
    uv run agno_dag_router.py
"""

from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.workflow import Router, Step, Workflow
from agno.workflow.types import StepInput, StepOutput

load_dotenv()

MODEL_ID = "claude-sonnet-4-6"

# ── Models (structured output schemas) ────────────────────────────────────────

class Category(str, Enum):
    BILLING   = "billing"
    TECHNICAL = "technical"
    GENERAL   = "general"


class Classification(BaseModel):
    category:   Category
    confidence: float


class Resolution(BaseModel):
    response: str
    escalate: bool


# ── LLM Agents (each is one focused task) ─────────────────────────────────────
# Setting `output_schema` makes Agno return a validated Pydantic instance as the
# run's `.content` — instructions no longer need to beg for "valid JSON only".

classifier_agent = Agent(
    name="ClassifierAgent",
    model=Claude(id=MODEL_ID),
    description="Classifies support tickets.",
    output_schema=Classification,
    instructions=[
        "Classify the ticket into billing, technical, or general.",
        "Set confidence to your certainty in the classification (0.0–1.0).",
    ],
)

billing_agent = Agent(
    name="BillingAgent",
    model=Claude(id=MODEL_ID),
    description="Resolves billing issues.",
    output_schema=Resolution,
    instructions=[
        "You are a billing support specialist.",
        "Generate a resolution for the billing ticket.",
        "Set escalate=true only if a refund over $100 is needed.",
    ],
)

technical_agent = Agent(
    name="TechnicalAgent",
    model=Claude(id=MODEL_ID),
    description="Resolves technical issues.",
    output_schema=Resolution,
    instructions=[
        "You are a technical support specialist.",
        "Generate a resolution for the technical ticket.",
        "Set escalate=true if engineering intervention is required.",
    ],
)

general_agent = Agent(
    name="GeneralAgent",
    model=Claude(id=MODEL_ID),
    description="Handles general inquiries.",
    output_schema=Resolution,
    instructions=[
        "You handle general support inquiries. Be helpful and concise.",
        "Set escalate=false unless the request clearly needs a human.",
    ],
)

responder_agent = Agent(
    name="ResponderAgent",
    model=Claude(id=MODEL_ID),
    description="Formats the final customer reply.",
    instructions=[
        "You are a customer communications writer.",
        "Turn the resolution into a polished customer-facing reply.",
        "If escalate is true: tell the customer their ticket has been escalated.",
        "Output only the final message — no JSON, no labels.",
    ],
)


# ── Handler steps (global routing table) ──────────────────────────────────────
# The Router selects one of these by category; the workflow engine then executes
# the returned step (that is where the chosen handler agent is actually called).
# Each handler emits a `Resolution`, consumed directly by FormatResponse.

HANDLER_STEPS: dict[Category, Step] = {
    Category.BILLING:   Step(name="BillingHandler",   agent=billing_agent),
    Category.TECHNICAL: Step(name="TechnicalHandler", agent=technical_agent),
    Category.GENERAL:   Step(name="GeneralHandler",   agent=general_agent),
}


# ── Step 1: Classify (LLM call) ───────────────────────────────────────────────

def classify_ticket(step_input: StepInput) -> StepOutput:
    """
    LLM call: extract category and confidence from the raw ticket.
    The Classification object is stored in StepOutput.content for the Router.
    """
    ticket = step_input.input or step_input.previous_step_content or ""
    result = classifier_agent.run(ticket)
    classification: Classification = result.content

    print(f"\n[CLASSIFY] → {classification.category.value} "
          f"({classification.confidence:.0%})")

    return StepOutput(content=classification)


# ── Step 2: Route (pure code) ─────────────────────────────────────────────────
# Router reads the classification object and returns the matching handler step.
# No LLM involved — this is a dict lookup. The workflow engine runs the result.

def route_by_category(step_input: StepInput) -> list[Step]:
    """
    Code-owned routing: read the typed classification → return the handler step.
    This is the core of the DAG pattern — code decides which branch executes.
    """
    classification = step_input.previous_step_content
    category = (
        classification.category
        if isinstance(classification, Classification)
        else Category.GENERAL
    )

    selected = HANDLER_STEPS.get(category, HANDLER_STEPS[Category.GENERAL])
    print(f"[ROUTE] → {selected.name}")
    return [selected]


# ── Step 3: Format response (LLM call) ────────────────────────────────────────

def format_response(step_input: StepInput) -> StepOutput:
    """
    Turn the structured Resolution into a customer-facing reply.
    The escalation check is pure code reading a typed boolean.
    """
    resolution = step_input.previous_step_content

    # Defensive: a handler should always emit a Resolution, but never crash if not.
    if not isinstance(resolution, Resolution):
        return StepOutput(content=str(resolution))

    # Code makes the escalation decision — not the LLM.
    if resolution.escalate:
        print("[VALIDATE] → Escalating to human agent")
        msg = (
            f"Your ticket requires specialist attention. "
            f"We have escalated it and you will hear back within 24 hours.\n\n"
            f"Reference: {resolution.response}"
        )
        return StepOutput(content=msg)

    result = responder_agent.run(resolution.model_dump_json())
    return StepOutput(content=result.content)


# ── DAG Workflow ───────────────────────────────────────────────────────────────
# Steps execute in order. Code owns every transition.
# LLM is called only in classify, the selected handler, and format_response.

dag = Workflow(
    name="TicketTriageDAG",
    steps=[
        Step(
            name="Classify",
            executor=classify_ticket,
        ),
        Router(
            name="RouteByCategory",
            selector=route_by_category,
            choices=list(HANDLER_STEPS.values()),
        ),
        Step(
            name="FormatResponse",
            executor=format_response,
        ),
    ],
)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tickets = [
        "I was charged twice for my subscription last month. "
        "Order #12345. The duplicate charge was $49.99.",

        "The app crashes every time I try to export a PDF. "
        "I'm on macOS 14.3, version 3.2.1.",

        "What are your business hours? Do you offer phone support?",
    ]

    for ticket in tickets:
        print(f"\n{'═'*60}")
        print(f"TICKET: {ticket}")
        print(f"{'═'*60}")
        dag.print_response(ticket)
