"""
Advanced Parallel Workflow Example
Demonstrates multiple parallel execution patterns and complex orchestration
"""
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.hackernews import HackerNewsTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow import Parallel, Step, Workflow
from agno.run.workflow import WorkflowRunOutput
from agno.utils.pprint import pprint_run_response

# === RESEARCH AGENTS (Run in Parallel) ===
tech_news_researcher = Agent(
    name="Tech News Researcher",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HackerNewsTools()],
    role="Research technology news and developer discussions",
    instructions=[
        "Search for the latest tech news on the given topic",
        "Focus on technical implementations and developer perspectives",
        "Extract trending discussions and community sentiment"
    ],
    debug_mode=True,
)

market_researcher = Agent(
    name="Market Researcher",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    role="Research market trends and industry analysis",
    instructions=[
        "Search for market analysis and industry trends",
        "Find information about market size, growth, and predictions",
        "Look for expert opinions and analyst reports"
    ],
    debug_mode=True,
)

financial_analyst = Agent(
    name="Financial Analyst",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[YFinanceTools()],
    role="Analyze financial data and stock performance of related companies",
    instructions=[
        "Look up stock prices and financial metrics for relevant companies",
        "Analyze trends in stock performance",
        "Provide financial context to the research topic"
    ],
    debug_mode=True,
)

# === ANALYSIS AGENTS (Run in Parallel after research) ===
technical_analyst = Agent(
    name="Technical Analyst",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Analyze technical aspects and implications",
    instructions=[
        "Review all research findings",
        "Analyze technical implementations and architecture",
        "Identify technical trends and patterns",
        "Assess feasibility and technical challenges"
    ],
    debug_mode=True,
)

business_analyst = Agent(
    name="Business Analyst",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Analyze business implications and opportunities",
    instructions=[
        "Review all research findings",
        "Analyze business models and market opportunities",
        "Identify competitive advantages and risks",
        "Assess ROI and business viability"
    ],
    debug_mode=True,
)

# === FINAL SYNTHESIS ===
executive_synthesizer = Agent(
    name="Executive Synthesizer",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Create executive summary with actionable insights",
    instructions=[
        "Synthesize all research and analysis",
        "Create a comprehensive executive summary",
        "Provide clear, actionable recommendations",
        "Structure information for decision-makers",
        "Include both technical and business perspectives"
    ],
    debug_mode=True,
)

# === ADVANCED PARALLEL WORKFLOW ===
advanced_workflow = Workflow(
    name="Advanced Parallel Research & Analysis Pipeline",
    description="""
    Demonstrates multi-stage parallel execution:
    Stage 1: Three researchers work in parallel (Tech, Market, Financial)
    Stage 2: Two analysts work in parallel (Technical, Business)
    Stage 3: Executive synthesizer creates final report
    """,
    steps=[
        # Stage 1: Parallel Research
        Parallel(
            Step(name="Tech News Research", agent=tech_news_researcher),
            Step(name="Market Research", agent=market_researcher),
            Step(name="Financial Analysis", agent=financial_analyst),
            name="Parallel Research Phase"
        ),
        # Stage 2: Parallel Analysis
        Parallel(
            Step(name="Technical Analysis", agent=technical_analyst),
            Step(name="Business Analysis", agent=business_analyst),
            name="Parallel Analysis Phase"
        ),
        # Stage 3: Final Synthesis
        Step(name="Executive Synthesis", agent=executive_synthesizer),
    ],
    debug_mode=True,
)

# === RUN THE WORKFLOW ===
if __name__ == "__main__":
    print("=" * 100)
    print("ADVANCED PARALLEL WORKFLOW DEMONSTRATION")
    print("=" * 100)
    print("\nThis workflow demonstrates multi-stage parallel execution:\n")
    print("STAGE 1 - Parallel Research (3 agents run simultaneously):")
    print("  ├─ Tech News Researcher    → Searches HackerNews for tech discussions")
    print("  ├─ Market Researcher       → Searches web for market trends")
    print("  └─ Financial Analyst       → Analyzes stock data and financials")
    print("\n↓ (Wait for all to complete)")
    print("\nSTAGE 2 - Parallel Analysis (2 agents run simultaneously):")
    print("  ├─ Technical Analyst       → Analyzes technical implications")
    print("  └─ Business Analyst        → Analyzes business opportunities")
    print("\n↓ (Wait for all to complete)")
    print("\nSTAGE 3 - Final Synthesis (1 agent):")
    print("  └─ Executive Synthesizer   → Creates comprehensive report")
    print("\n" + "=" * 100 + "\n")

    # Run the workflow
    response: WorkflowRunOutput = advanced_workflow.run(
        input="Analyze the impact and opportunities of Generative AI in enterprise software for 2026",
        markdown=True
    )

    pprint_run_response(response, markdown=True)

    print("\n" + "=" * 100)
    print("WORKFLOW COMPLETE")
    print("=" * 100)
