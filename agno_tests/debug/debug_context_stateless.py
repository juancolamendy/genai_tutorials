from typing import Dict, Any, Optional

from rich.pretty import pprint

from agno.agent import Agent
from agno.models.google import Gemini

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
# Functions
def create_agent():
    agent = Agent(
        model=Gemini(id="gemini-3-flash-preview"),
        # system prompt
        # role / instructions / output
        description="You are a helpful AI assistant",
        instructions=["Use the information provided above to answer questions."],
        # output
        # system prompt extra
        add_datetime_to_context=True,

        # debug
        debug_mode=True,
    )
    return agent


def execute_turn(agent, turn: str, user_id: Optional[str]=None, session_id: Optional[str]=None):
    response = agent.run(turn, user_id=user_id, session_id=session_id)
    pprint(f"Run ID: {response.run_id}\nAgent ID: {response.agent_id}\nSession ID: {response.session_id}\nContent: {response.content}")
    return response

agent = create_agent()

user_id = 'user_01'
session_id = 'session_01'

while True:
    user_input = input("\nYou: ")
    if user_input == "/exit":
        break
    execute_turn(agent, user_input, user_id, session_id)
