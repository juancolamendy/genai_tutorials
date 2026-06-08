from agno.agent import Agent
from agno.models.openai import OpenAIChat
#from agno.db.sqlite import SqliteDb
from agno.team import Team
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.workflow import Step, Workflow
from agno.run.workflow import WorkflowRunOutput
from agno.utils.pprint import pprint_run_response

# Define agents
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
    debug_mode = True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[YFinanceTools()],
    role="Get stock prices and financial data",
    debug_mode = True,
)

# Define research team for complex analysis
research_team = Team(
    name="Research Team",
    members=[hackernews_agent, finance_agent],
    instructions="Research tech topics and related stocks",
    debug_mode = True,
)

content_planner = Agent(
    name="Content Planner",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=[
        "Plan a content schedule over 4 weeks for the provided topic and research content",
        "Ensure that I have posts for 3 posts per week",
    ],
    debug_mode = True,
)

content_workflow = Workflow(
    name="Content Creation Workflow",
    description="Automated content creation from blog posts to social media",
    #db=SqliteDb(db_file="sessions/workflow.db"),
    steps=[research_team, content_planner],
    debug_mode = True,
)

# Create and use workflow
if __name__ == "__main__":
    response: WorkflowRunOutput = content_workflow.run(
        input="AI trends in 2026",
        markdown=True,
    )

    pprint_run_response(response, markdown=True)
