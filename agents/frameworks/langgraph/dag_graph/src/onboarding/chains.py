"""Domain-specific LLM chains for the onboarding pipeline.

Two states require LLM assistance (design spec §10):
  COLLECT       — a tool-calling agent gathers new-hire details; tool
                  execution (not model prose) performs the transition, the
                  handler node intercepts the tool call and applies the
                  guarded, idempotent state delta itself
  IT_PROVISIONED — a plain, deterministic, single-shot LCEL chain selects
                  a username prefix

All chains are created via engine.chains.make_chain (LCEL) or
langchain.agents.create_agent (tool-calling), reusing the same cached
Claude client pattern already established in docprocessing/chains.py.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel

from src.engine.chains import DEFAULT_MODEL, make_chain

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class UsernameSelection(BaseModel):
    """Result of IT_PROVISIONED's username prefix selection."""

    username_prefix: str
    reasoning: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT — tool-calling agent
# ─────────────────────────────────────────────────────────────────────────────

@tool
def submit_new_hire(full_name: str, role: str, start_date: str) -> dict:
    """Submit the new hire's collected details once name, role, and start
    date are all known. Do not call this until all three are confirmed.

    Args:
        full_name: The new hire's full name
        role: The new hire's job title / role
        start_date: The new hire's start date (any human-readable format)

    Returns:
        The submitted details, echoed back so the handler node can read
        them straight off the tool call rather than re-parsing model prose.
    """
    return {"full_name": full_name, "role": role, "start_date": start_date}


collect_agent = create_agent(
    model=DEFAULT_MODEL,
    tools=[submit_new_hire],
    system_prompt="""You are an onboarding assistant collecting details about a new hire.

Your job is to gather exactly three pieces of information through natural
conversation:
1. Full name
2. Role / job title
3. Start date

Ask for whatever is still missing, one turn at a time. Once all three are
confirmed, call submit_new_hire with them — do not call it before all
three are known, and do not invent values the human hasn't provided.""",
)


# ─────────────────────────────────────────────────────────────────────────────
# IT_PROVISIONED — deterministic LCEL chain
# ─────────────────────────────────────────────────────────────────────────────

username_chain = make_chain(
    name="UsernameChain",
    system_prompt="""You select a username prefix for a newly provisioned IT account.

Your job is to:
1. Derive a short, lowercase username prefix from the new hire's full name
   (e.g. "Jane Doe" -> "jdoe")
2. Briefly explain your reasoning

Respond ONLY with valid JSON matching this structure:
{{
  "username_prefix": <str>,
  "reasoning": <str>
}}""",
    output_schema=UsernameSelection,
)


__all__ = [
    "UsernameSelection",
    "submit_new_hire",
    "collect_agent",
    "username_chain",
]
