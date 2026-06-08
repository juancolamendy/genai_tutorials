from typing import Dict, Any, Optional

from rich.pretty import pprint

from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
DB_FILE = "simple_context_data.db"

# Functions
def add(num1: float, num2: float) -> float:
    """Add two numbers together.

    Args:
        num1: The first number
        num2: The second number

    Returns:
        The sum of num1 and num2
    """
    return num1 + num2


def multiply(num1: float, num2: float) -> float:
    """Multiply two numbers together.

    Args:
        num1: The first number
        num2: The second number

    Returns:
        The product of num1 and num2
    """
    return num1 * num2


def create_agent():
    agent = Agent(
        model=Gemini(id="gemini-3-flash-preview"),
        # system prompt
        # role / instructions / output
        description="You are a helpful AI assistant",
        instructions=["Use the information provided above to answer questions.",
                      "If you need to add two numbers, use the function add",
                      "If you need to multiply two numbers, use the function multiply"],
        # output
        # system prompt extra
        add_datetime_to_context=True,

        # session storage
        db=SqliteDb(db_file=DB_FILE),
        # session history
        add_history_to_context=True,
        num_history_runs=2,

        # tools
        tools=[add, multiply],

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
