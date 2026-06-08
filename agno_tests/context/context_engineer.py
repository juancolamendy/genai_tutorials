from rich.pretty import pprint

from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
DB_FILE = "structured_output_data.db"

# Data structures
class MovieRecommendation(BaseModel):
    title: str = Field(description="Movie title")
    genre: str = Field(description="Movie genre")
    reason: str = Field(description="Why this movie is recommended")

# Functions
def create_agent():
    agent = Agent(
        model=Gemini(id="gemini-3-flash-preview"),
        description="You are a helpful movie recommendation assistant",
        instructions=["Always be enthusiastic about movies", "Consider user preferences"],
        output_schema=MovieRecommendation,
        db=SqliteDb(db_file=DB_FILE),
        add_history_to_context=True,
        num_history_runs=2,
        debug_mode=True,
    )
    return agent


def execute_turn(agent, turn: str):
    response = agent.run(turn)

    # Print complete message structure
    print("\n=== Complete Message Structure ===")
    for msg in response.messages:
        pprint(msg.model_dump())

agent = create_agent()

execute_turn(agent, "What's a good sci-fi movie?")
execute_turn(agent, "How about something with time travel?")
execute_turn(agent, "Give me one more recommendation")

