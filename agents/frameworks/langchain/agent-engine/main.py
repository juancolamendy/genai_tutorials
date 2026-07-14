# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain>=1.0",
#     "langchain-anthropic>=1.0",
#     "pydantic>=2",
# ]
# ///
"""Live end-to-end demo of engine.py against the real Anthropic API.

Exercises all three pieces:
  1. make_llm_agent  — with a tool AND a structured output schema
  3. ainvoke_agent   — both the single-prompt and full-messages forms

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run demo_live.py
"""

import asyncio
import os
import sys

from pydantic import BaseModel, Field

import engine


# ── A tool the agent can call ────────────────────────────────────────────────
def get_word_count(text: str) -> int:
    """Count the number of words in the given text."""
    return len(text.split())


# ── Structured output schema ─────────────────────────────────────────────────
class Summary(BaseModel):
    """Structured summary of the user's text."""

    title: str = Field(description="A short title for the text")
    word_count: int = Field(description="Word count, from the get_word_count tool")
    key_points: list[str] = Field(description="2-3 key points")


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first: export ANTHROPIC_API_KEY=sk-ant-...")

    # 1) Build (and cache) the agent
    engine.make_llm_agent(
        name="summarizer",
        system_prompt=(
            "You summarize text. Always call get_word_count on the input "
            "before answering, and use its result for word_count."
        ),
        output_schema=Summary,
        tools=[get_word_count],
    )

    # 2) Simple invocation — a HumanMessage is built for you
    text = (
        "The James Webb Space Telescope has detected carbon dioxide in the "
        "atmosphere of an exoplanet for the first time, a milestone for the "
        "search for habitable worlds beyond our solar system."
    )
    result = await engine.ainvoke_agent("summarizer", f"Summarize this:\n{text}")
    print("── result ─────────────────────────")
    print(result)

    structured: Summary = result["structured_response"]
    print("── structured_response ─────────────────────────")
    print(f"title:      {structured.title}")
    print(f"word_count: {structured.word_count} (actual: {get_word_count(text)})")
    for p in structured.key_points:
        print(f"  • {p}")

    ## Show the tool call actually happened
    #tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    #print(f"\ntool calls made: {len(tool_msgs)}")

    #followup = await engine.ainvoke_agent(
    #    "summarizer",
    #    "",  # ignored when messages= is given
    #    messages=[
    #        ("user", f"{history_block}\n\nSummarize what I said I'm researching."),
    #    ],
    #)
    #print("\n── follow-up structured_response ───────────────")
    #print(followup["structured_response"].model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
