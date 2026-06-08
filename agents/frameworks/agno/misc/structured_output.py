from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.google import Gemini

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        #add_history_to_context=True,
        #num_history_runs=1,
        debug_mode=True,
    )
    return agent


def execute_turn(agent, turn: str):
    response = agent.run(turn)


agent = create_agent()

execute_turn(agent, "My name is Juan. I'd like to know about movies")
execute_turn(agent, "How about something with time travel?")
execute_turn(agent, "Give me one more recommendation. Answer using my name")
