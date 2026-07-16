"""Domain-specific LLM chains for the onboarding pipeline.

Two states require LLM assistance:
  COLLECT        — structured extraction of new-hire details (no tool agent)
  IT_PROVISIONED — username prefix selection

Both use engine.chains.make_chain (LCEL + structured output), matching
hrhelpdesk/docprocessing.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.engine.chains import make_chain

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────


class NewHireDetails(BaseModel):
    """Structured collect decision — handler alone applies state deltas."""

    full_name: Optional[str] = Field(default=None, description="Full name if known")
    role: Optional[str] = Field(default=None, description="Job title / role if known")
    start_date: Optional[str] = Field(
        default=None, description="Start date if known (any clear format)"
    )
    complete: bool = Field(
        default=False,
        description="True only when full_name, role, and start_date are all confirmed",
    )
    reply: str = Field(
        description="User-facing ask for missing fields, or short confirmation when complete"
    )


class UsernameSelection(BaseModel):
    """Result of IT_PROVISIONED's username prefix selection."""

    username_prefix: str
    reasoning: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT — structured extraction
# ─────────────────────────────────────────────────────────────────────────────

collect_chain = make_chain(
    name="CollectNewHireChain",
    system_prompt="""You are an onboarding assistant collecting details about a new hire.

Gather exactly three pieces of information:
1. full_name
2. role / job title
3. start_date

Rules:
- Extract only values the human has clearly provided; do not invent fields.
- Set complete=true only when all three are known and confirmed.
- When anything is missing, set complete=false and put a short, natural
  question in reply asking only for what is still missing.
- When complete, reply may briefly confirm the three fields.

Respond ONLY with valid JSON matching this structure:
{{
  "full_name": <str or null>,
  "role": <str or null>,
  "start_date": <str or null>,
  "complete": <bool>,
  "reply": <str>
}}""",
    output_schema=NewHireDetails,
)


# ─────────────────────────────────────────────────────────────────────────────
# IT_PROVISIONED — username prefix
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
    "NewHireDetails",
    "UsernameSelection",
    "collect_chain",
    "username_chain",
]
