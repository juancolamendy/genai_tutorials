"""
ADK — DAG Pattern: Ticket Triage (single-handler routing via CustomAgent)
─────────────────────────────────────────────────────────────────────────────
Shape:  Classify → Route → Handle → Respond
        Code controls every transition; LLM only extracts/generates.

Routing logic (Python-controlled, not the LLM):
    1. Call the classifier            → writes state["classification"]
    2. classification.category == key in the handlers dict
    3. Get the one valid handler for that category
    4. Call ONLY that handler         → writes state["resolution"]
    5. Call the responder_agent        → writes state["final_response"]

Why a CustomAgent instead of SequentialAgent:
    SequentialAgent runs ALL sub_agents in order, so every handler fires on
    every ticket (the old "smart handler" no-op hack). A CustomAgent lets us
    select exactly one handler in code — true single-handler routing.

Install:
    uv add google-adk python-dotenv

Usage:
    uv run adk_dag_router.py
"""

import asyncio
import json
from typing import AsyncGenerator

from dotenv import load_dotenv
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types
from typing_extensions import override

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash"
APP   = "ticket-triage-dag"


# ── Step 1: Classify ─────────────────────────────────────────────────────────
# Narrow LLM task: read ticket → write category + confidence to shared state.

classifier_agent = LlmAgent(
    name="ClassifierAgent",
    model=MODEL,
    description="Classifies a support ticket into billing, technical, or general.",
    instruction="""
You are a ticket classifier. Read the ticket and output ONLY this JSON — no prose, no markdown:

{
  "category": "<billing|technical|general>",
  "confidence": <0.0-1.0>
}
""",
    output_key="classification",   # written to session.state["classification"]
)


# ── Step 2: Billing handler ───────────────────────────────────────────────────

billing_agent = LlmAgent(
    name="BillingAgent",
    model=MODEL,
    description="Handles billing tickets.",
    instruction="""
You are a billing support specialist.

Ticket classification: {classification}

Read the original ticket and produce a resolution. Output ONLY this JSON:

{
  "response": "<your resolution text>",
  "escalate": <true if refund > $100 is needed, else false>
}
""",
    output_key="resolution",
)


# ── Step 2: Technical handler ─────────────────────────────────────────────────

technical_agent = LlmAgent(
    name="TechnicalAgent",
    model=MODEL,
    description="Handles technical tickets.",
    instruction="""
You are a technical support specialist.

Ticket classification: {classification}

Read the original ticket and produce a resolution. Output ONLY this JSON:

{
  "response": "<your resolution text>",
  "escalate": <true if engineering intervention is needed, else false>
}
""",
    output_key="resolution",
)


# ── Step 2: General handler ───────────────────────────────────────────────────

general_agent = LlmAgent(
    name="GeneralAgent",
    model=MODEL,
    description="Handles general tickets.",
    instruction="""
You are a general support specialist. Be helpful and concise.

Ticket classification: {classification}

Read the original ticket and produce a resolution. Output ONLY this JSON:

{
  "response": "<your resolution text>",
  "escalate": false
}
""",
    output_key="resolution",
)


# ── Step 3: Response formatter ────────────────────────────────────────────────
# Final LLM call: turn the structured resolution into a polished customer reply.

responder_agent = LlmAgent(
    name="ResponderAgent",
    model=MODEL,
    description="Formats the final customer-facing response.",
    instruction="""
You are a customer communications writer.

Resolution JSON: {resolution}

If escalate is true: write a short, empathetic message telling the customer
their ticket has been escalated to a specialist and they will hear back within
24 hours.

If escalate is false: write a friendly, professional reply using the
resolution text.

Output ONLY the final customer-facing message. No JSON, no labels.
""",
    output_key="final_response",
)


# ── DAG Orchestrator: CustomAgent with single-handler routing ─────────────────
# A CustomAgent is a BaseAgent subclass whose _run_async_impl drives the flow in
# plain Python. Sub-agents are run with `sub_agent.run_async(ctx)` and share the
# same session.state, so output_key values flow between steps exactly as before.
#
# Unlike SequentialAgent, here WE decide which agents run — so exactly one
# handler is selected per ticket based on the classifier's category.

def _parse_category(raw: str) -> str:
    """Extract category from the classifier's JSON output, tolerating fences."""
    text = (raw or "").strip()
    # Strip ```json ... ``` or ``` ... ``` fences if the model added them.
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
        text = text.lstrip("json").strip()
    try:
        return str(json.loads(text).get("category", "")).lower().strip()
    except (json.JSONDecodeError, AttributeError):
        return ""


class TicketTriageAgent(BaseAgent):
    """Classify → route to ONE handler → respond."""

    classifier: LlmAgent
    handlers: dict[str, LlmAgent]
    responder: LlmAgent

    def __init__(
        self,
        name: str,
        classifier: LlmAgent,
        handlers: dict[str, LlmAgent],
        responder: LlmAgent,
    ):
        super().__init__(
            name=name,
            description="Routes a ticket to a single handler by category.",
            classifier=classifier,
            handlers=handlers,
            responder=responder,
            # Register every sub-agent with the framework, even though we only
            # run a subset on any given invocation.
            sub_agents=[classifier, *handlers.values(), responder],
        )

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # 1. Call the classifier → writes state["classification"].
        async for event in self.classifier.run_async(ctx):
            yield event

        # 2. classification.category == key in the handlers dict.
        category = _parse_category(ctx.session.state.get("classification", ""))

        # 3. Get the one valid handler (fall back to general if unknown).
        handler = self.handlers.get(category) or self.handlers["general"]
        print(f"ROUTED: category={category or '?'} → {handler.name}")

        # 4. Call ONLY that handler → writes state["resolution"].
        async for event in handler.run_async(ctx):
            yield event

        # 5. Call the responder → writes state["final_response"].
        async for event in self.responder.run_async(ctx):
            yield event


dag_pipeline = TicketTriageAgent(
    name="TicketTriageDAG",
    classifier=classifier_agent,
    handlers={
        "billing": billing_agent,
        "technical": technical_agent,
        "general": general_agent,
    },
    responder=responder_agent,
)

root_agent = dag_pipeline


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_ticket(ticket: str) -> None:
    runner = InMemoryRunner(agent=dag_pipeline, app_name=APP)

    # Create the session on the runner's own service (InMemoryRunner owns one).
    await runner.session_service.create_session(
        app_name=APP, user_id="user-1", session_id="session-1"
    )

    print(f"\n{'─'*60}")
    print(f"TICKET: {ticket}")
    print(f"{'─'*60}")

    async for event in runner.run_async(
        user_id="user-1",
        session_id="session-1",
        new_message=types.Content(role="user", parts=[types.Part(text=ticket)]),
    ):
        # Surface the final formatted response only
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    # Only print the responder's output (last agent)
                    if event.author == "ResponderAgent":
                        print(f"\nFINAL RESPONSE:\n{part.text}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tickets = [
        # Billing ticket
        "I was charged twice for my subscription last month. "
        "Order #12345. The duplicate charge was $49.99.",

        # Technical ticket
        "The app crashes every time I try to export a PDF. "
        "I'm on macOS 14.3, version 3.2.1. This started after the last update.",

        # General ticket
        "What are your business hours and do you offer phone support?",
    ]

    for ticket in tickets:
        asyncio.run(run_ticket(ticket))
