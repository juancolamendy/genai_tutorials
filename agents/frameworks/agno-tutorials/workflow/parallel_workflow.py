from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.hackernews import HackerNewsTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.workflow import Parallel, Step, Workflow
from agno.run.workflow import WorkflowRunOutput
from agno.utils.pprint import pprint_run_response

# Define the HackerNews researcher agent
hn_researcher = Agent(
    name="HackerNews Researcher",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HackerNewsTools()],
    role="Search HackerNews for relevant tech discussions and news",
    instructions=[
        "Search for the latest discussions and stories on the given topic",
        "Extract key points, trends, and community insights",
        "Focus on technical discussions and developer perspectives"
    ],
    debug_mode=True,
)

# Define the Web researcher agent
web_researcher = Agent(
    name="Web Researcher",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    role="Search the web for comprehensive information and latest news",
    instructions=[
        "Search the web for the latest news and information on the given topic",
        "Find authoritative sources and recent developments",
        "Focus on factual information and industry trends"
    ],
    debug_mode=True,
)

# Define the synthesizer agent to combine results
synthesizer = Agent(
    name="Research Synthesizer",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Synthesize research findings into a comprehensive report",
    instructions=[
        "Combine insights from all research sources",
        "Create a well-structured, comprehensive report",
        "Highlight key findings, trends, and actionable insights",
        "Cross-reference information from different sources",
        "Present the information in a clear, organized manner"
    ],
    debug_mode=True,
)

# Create the workflow with parallel execution
workflow = Workflow(
    name="Parallel Research Pipeline",
    description="Demonstrates parallel execution: HackerNews and Web research run simultaneously, then results are synthesized",
    steps=[
        Parallel(
            Step(name="HackerNews Research", agent=hn_researcher),
            Step(name="Web Research", agent=web_researcher),
            name="Research Step"
        ),
        Step(name="Synthesis", agent=synthesizer),  # Combines the results and produces a report
    ],
    debug_mode=True,
)

# Run the workflow
if __name__ == "__main__":
    print("=" * 80)
    print("PARALLEL WORKFLOW DEMONSTRATION")
    print("=" * 80)
    print("\nThis workflow demonstrates parallel execution:")
    print("1. HackerNews Research and Web Research run SIMULTANEOUSLY")
    print("2. Once both complete, the Synthesizer combines their findings\n")
    print("=" * 80)

    response: WorkflowRunOutput = workflow.run(
        input="Write about the latest AI developments and breakthroughs in 2026",
        markdown=True
    )

    pprint_run_response(response, markdown=True)
