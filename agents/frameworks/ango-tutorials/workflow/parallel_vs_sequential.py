"""
Parallel vs Sequential Execution Comparison
Demonstrates the performance benefits of parallel execution
"""
import time
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.hackernews import HackerNewsTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.workflow import Parallel, Step, Workflow
from agno.run.workflow import WorkflowRunOutput
from agno.utils.pprint import pprint_run_response

# === DEFINE AGENTS ===
agent1 = Agent(
    name="HackerNews Researcher",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HackerNewsTools()],
    role="Search HackerNews for tech discussions",
    instructions=["Search and summarize the top discussions on the topic"],
    debug_mode=False,  # Disable debug for cleaner output
)

agent2 = Agent(
    name="Web Researcher",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    role="Search the web for information",
    instructions=["Search the web and provide relevant findings"],
    debug_mode=False,
)

agent3 = Agent(
    name="Trend Analyzer",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Analyze trends from research",
    instructions=["Analyze and identify key trends from the research"],
    debug_mode=False,
)

synthesizer = Agent(
    name="Synthesizer",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Synthesize all findings into a report",
    instructions=["Create a comprehensive report from all findings"],
    debug_mode=False,
)

# === SEQUENTIAL WORKFLOW ===
sequential_workflow = Workflow(
    name="Sequential Workflow",
    description="Tasks run one after another (slower)",
    steps=[
        Step(name="HN Research", agent=agent1),
        Step(name="Web Research", agent=agent2),
        Step(name="Trend Analysis", agent=agent3),
        Step(name="Synthesis", agent=synthesizer),
    ],
    debug_mode=False,
)

# === PARALLEL WORKFLOW ===
parallel_workflow = Workflow(
    name="Parallel Workflow",
    description="Research tasks run simultaneously (faster)",
    steps=[
        Parallel(
            Step(name="HN Research", agent=agent1),
            Step(name="Web Research", agent=agent2),
            Step(name="Trend Analysis", agent=agent3),
            name="Parallel Research"
        ),
        Step(name="Synthesis", agent=synthesizer),
    ],
    debug_mode=False,
)

def run_workflow_timed(workflow: Workflow, input_text: str, workflow_type: str):
    """Run a workflow and measure execution time"""
    print(f"\n{'=' * 80}")
    print(f"Running {workflow_type} Workflow")
    print(f"{'=' * 80}")

    start_time = time.time()
    response: WorkflowRunOutput = workflow.run(input=input_text, markdown=True)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"\n⏱️  Execution Time: {elapsed_time:.2f} seconds")
    print(f"{'=' * 80}\n")

    return response, elapsed_time

# === MAIN EXECUTION ===
if __name__ == "__main__":
    topic = "Python async programming trends in 2026"

    print("\n" + "=" * 80)
    print("PARALLEL vs SEQUENTIAL EXECUTION COMPARISON")
    print("=" * 80)
    print(f"\nTopic: {topic}\n")
    print("This demo shows how parallel execution speeds up workflows by running")
    print("independent tasks simultaneously instead of one after another.\n")

    # Run Sequential Workflow
    print("\n🐢 SEQUENTIAL EXECUTION (Tasks run one at a time)")
    print("   Flow: HN Research → Web Research → Trend Analysis → Synthesis")
    seq_response, seq_time = run_workflow_timed(
        sequential_workflow,
        topic,
        "SEQUENTIAL"
    )

    # Run Parallel Workflow
    print("\n🚀 PARALLEL EXECUTION (Research tasks run simultaneously)")
    print("   Flow: [HN Research ∥ Web Research ∥ Trend Analysis] → Synthesis")
    par_response, par_time = run_workflow_timed(
        parallel_workflow,
        topic,
        "PARALLEL"
    )

    # Compare Results
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"\n{'Sequential Time:':<20} {seq_time:.2f} seconds")
    print(f"{'Parallel Time:':<20} {par_time:.2f} seconds")

    if seq_time > par_time:
        speedup = seq_time / par_time
        time_saved = seq_time - par_time
        improvement = ((seq_time - par_time) / seq_time) * 100

        print(f"\n✅ Parallel workflow was {speedup:.2f}x faster!")
        print(f"✅ Time saved: {time_saved:.2f} seconds ({improvement:.1f}% improvement)")
    else:
        print(f"\n⚠️  Results may vary due to network latency and API response times")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAY")
    print("=" * 80)
    print("""
When you have independent tasks (tasks that don't depend on each other's output),
use Parallel() to run them simultaneously. This significantly reduces total
execution time, especially for I/O-bound operations like API calls, web searches,
and database queries.

Use Sequential (default) when:
  - Tasks depend on previous results
  - Tasks must execute in a specific order

Use Parallel when:
  - Tasks are independent
  - Tasks can run simultaneously
  - You want faster execution
    """)

    print("=" * 80 + "\n")
