from typing import Dict, Any, Optional

from rich.pretty import pprint

from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb
from agno.tools import tool
from agno.compression.manager import CompressionManager
from agno.session import SessionSummaryManager

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
DB_FILE = "context_compression_data.db"

# Functions
@tool
def add(x: float, y: float) -> str:
    """
    Add two numbers together and return detailed result information.
    This tool intentionally returns verbose output to demonstrate compression.
    
    Args:
        x: First number to add
        y: Second number to add
        
    Returns:
        Detailed calculation result with metadata
    """
    result = x + y
    
    # Return verbose output to simulate real-world tool results
    # that benefit from compression
    return f"""
    ════════════════════════════════════════════════════
    CALCULATION REPORT - Addition Operation
    ════════════════════════════════════════════════════
    
    Operation Type: Addition
    Timestamp: 2025-01-24 10:30:00 UTC
    
    Input Parameters:
    ├─ Operand 1 (x): {x}
    ├─ Operand 2 (y): {y}
    └─ Operation: x + y
    
    Calculation Details:
    ├─ Method: Standard arithmetic addition
    ├─ Precision: Double precision floating point
    └─ Algorithm: Direct summation
    
    Result:
    ├─ Sum: {result}
    ├─ Result Type: {'Integer' if result == int(result) else 'Float'}
    └─ Significant Digits: {len(str(result).replace('.', '').replace('-', ''))}
    
    Validation:
    ├─ Overflow Check: PASSED
    ├─ NaN Check: PASSED
    └─ Infinity Check: PASSED
    
    Performance Metrics:
    ├─ Execution Time: 0.001ms
    ├─ Memory Used: 64 bytes
    └─ CPU Cycles: ~100
    
    ════════════════════════════════════════════════════
    END OF REPORT
    ════════════════════════════════════════════════════
    """

def create_agent():
    compression_manager = CompressionManager(
        model=Gemini(id="gemini-3-flash-preview"),
        # config tool compression
        compress_tool_results=True,
        compress_tool_results_limit=3,  # Compress after 3 tool calls
        compress_tool_call_instructions="""
        Ultra-concise summary format:
        "x+y={result}"
        Nothing else.        
        """
    )
    
    # Use a cheaper model for summarization
    summary_manager = SessionSummaryManager(
        model=Gemini(id="gemini-3-flash-preview"),

        # Custom summarization prompt
        session_summary_prompt="""
        You are a conversation summarizer.

        Create a concise summary that captures:
        1. User's key personal information
        2. Main topics discussed
        3. Important decisions or preferences
        4. Action items or follow-ups

        Keep the summary under 200 words.
        """,
    )

    agent = Agent(
        model=Gemini(id="gemini-3-flash-preview"),
        # system prompt
        # role / instructions / output
        description="You are a helpful AI assistant",
        instructions=["Use the information in your context to answer questions",
                      "Use add function to add numbers if the request is to add numbers",
                      "Use general knowledge to answer general questions not involving numbers"],
        # system prompt extra
        add_datetime_to_context=True,

        # session storage
        db=SqliteDb(db_file=DB_FILE),
        # session history
        add_history_to_context=True,
        # Keep last 2 conversation turns
        num_history_runs=2,
        # session summary
        enable_session_summaries=True,
        add_session_summary_to_context=True,
        session_summary_manager=summary_manager,  # Custom manager

        # tools
        tools=[add],
        # tool compression manager
        compression_manager=compression_manager,

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
