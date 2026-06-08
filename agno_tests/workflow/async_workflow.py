from typing import AsyncIterator
import asyncio

from agno.agent import Agent
from agno.tools.hackernews import HackerNewsTools
from agno.workflow import Condition, Step, Workflow, StepInput
from agno.run.workflow import WorkflowRunOutput, WorkflowRunOutputEvent, WorkflowRunEvent
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response

# === BASIC AGENTS ===
researcher = Agent(
    name="Researcher",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="Research the given topic and provide detailed findings.",
    tools=[HackerNewsTools()],
    debug_mode = True,
)

summarizer = Agent(
    name="Summarizer",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="Create a clear summary of the research findings.",
    debug_mode = True,
)

fact_checker = Agent(
    name="Fact Checker",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="Verify facts and check for accuracy in the research.",
    tools=[HackerNewsTools()],
    debug_mode = True,
)

writer = Agent(
    name="Writer",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="Write a comprehensive article based on all available research and verification.",
    debug_mode = True,
)

# === CONDITION EVALUATOR ===
def needs_fact_checking(step_input: StepInput) -> bool:
    """Determine if the research contains claims that need fact-checking"""
    summary = step_input.previous_step_content or ""

    # Look for keywords that suggest factual claims
    fact_indicators = [
        "study shows",
        "breakthroughs",
        "research indicates",
        "according to",
        "statistics",
        "data shows",
        "survey",
        "report",
        "million",
        "billion",
        "percent",
        "%",
        "increase",
        "decrease",
    ]

    return any(indicator in summary.lower() for indicator in fact_indicators)


# === BASIC LINEAR WORKFLOW ===
async_workflow = Workflow(
    name="Basic Linear Workflow",
    description="Research -> Summarize -> Condition(Fact Check) -> Write Article",
    steps=[
        researcher,
        summarizer,
        Condition(
            name="fact_check_condition",
            description="Check if fact-checking is needed",
            evaluator=needs_fact_checking,
            steps=[fact_checker],
        ),
        writer,
    ],
    debug_mode = True,
)

async def main():
    try:
        response: WorkflowRunOutput = await async_workflow.arun(
            input="Recent breakthroughs in AI",
        )
        pprint_run_response(response, markdown=True)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
