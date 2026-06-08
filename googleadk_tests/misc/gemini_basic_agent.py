# Agent using Google Gemini API

"""
Install required libraries:
pip install -U google-genai python-dotenv
"""

from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CALCULATOR TOOL
# ============================================================================

class CalculatorTool():
    """A tool for performing mathematical calculations"""

    def get_schema(self):
        """Return Gemini function declaration format"""
        return types.FunctionDeclaration(
            name="calculator",
            description="Performs basic mathematical calculations, use also for simple additions",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "expression": {
                        "type": "STRING",
                        "description": "Mathematical expression to evaluate (e.g., '2+2', '10*5')"
                    }
                },
                "required": ["expression"]
            }
        )

    def execute(self, expression):
        """
        Evaluate mathematical expressions.
        WARNING: This tutorial uses eval() for simplicity but it is not recommended for production use.

        Args:
            expression (str): The mathematical expression to evaluate
        Returns:
            dict: The result of the evaluation or error message
        """
        try:
            result = eval(expression)
            return {"result": result}
        except Exception as e:
            return {"error": f"Invalid mathematical expression: {str(e)}"}


# ============================================================================
# AGENT CLASS
# ============================================================================

class Agent:
    """A simple AI agent that can use tools to answer questions in a multi-turn conversation"""

    def __init__(self, tools=None):
        # Initialize Gemini client
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        self.model_name = "gemini-3-flash-preview"
        self.system_message = "You are a helpful assistant that breaks down problems into steps and solves them systematically."
        self.tools = tools or []
        self.tool_map = {tool.get_schema().name: tool for tool in self.tools}

        # Store chat history
        self.history = []

    def _get_tool_schemas(self):
        """Get tool schemas for all registered tools in Gemini format"""
        return [types.Tool(function_declarations=[tool.get_schema()]) for tool in self.tools]

    def chat(self, message):
        """Process a user message and return a response"""

        # Build config
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
            system_instruction=self.system_message,
        )

        # Add tools if available
        if self.tools:
            config.tools = self._get_tool_schemas()

        # Prepare contents - add history and new message
        contents = self.history.copy()

        # Handle different message types (string or Content object)
        if isinstance(message, str):
            contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
        else:
            contents.append(message)

        # Send message to Gemini
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )

        # Update history
        if isinstance(message, str):
            self.history.append(types.Content(role="user", parts=[types.Part(text=message)]))
        else:
            self.history.append(message)

        self.history.append(types.Content(role="model", parts=response.candidates[0].content.parts))

        return response


# ============================================================================
# AGENT LOOP FUNCTION
# ============================================================================

def run_agent(user_input, max_turns=10):
    """
    Run the agent with a user input, handling tool use in a loop.

    Args:
        user_input (str): The initial user message
        max_turns (int): Maximum number of iterations to prevent infinite loops

    Returns:
        str: The final agent response
    """
    calculator_tool = CalculatorTool()
    agent = Agent(tools=[calculator_tool])

    i = 0

    while i < max_turns:
        i += 1
        print(f"\nIteration {i}:")

        print(f"User input: {user_input}")
        response = agent.chat(user_input)

        # Check if response has text
        if response.text:
            print(f"Agent output: {response.text}")

        # Handle tool use if present
        if response.candidates[0].content.parts:
            has_function_calls = any(
                hasattr(part, 'function_call') and part.function_call
                for part in response.candidates[0].content.parts
            )

            if has_function_calls:
                # Process all function calls in the response
                function_responses = []

                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        tool_name = function_call.name
                        tool_id = function_call.id
                        tool_input = dict(function_call.args)

                        print(f"Using tool {tool_name} (id: {tool_id}) with input {tool_input}")

                        # Execute the tool
                        tool = agent.tool_map[tool_name]
                        tool_result = tool.execute(**tool_input)

                        print(f"Tool result: {tool_result}")

                        # Create function response with matching ID
                        function_responses.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=tool_id,
                                    name=tool_name,
                                    response=tool_result
                                )
                            )
                        )

                # Send function responses back to model
                user_input = types.Content(role="user", parts=function_responses)
            else:
                # No function calls, return the text response
                return response.text
        else:
            return "No response generated"

    return "Max iterations reached"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test 1: General question (no tool use)
    print("="*80)
    print("TEST 1: General Question")
    print("="*80)
    response = run_agent("I have 4 apples. How many do you have?")

    # Test 2: Tool Use
    print("\n" + "="*80)
    print("TEST 2: Simple Calculation")
    print("="*80)
    response = run_agent("What is 157.09 * 493.89?")

    # Test 3: Multi-step reasoning with tools
    print("\n" + "="*80)
    print("TEST 3: Complex Problem")
    print("="*80)
    response = run_agent("If my brother is 32 years younger than my mother and my mother is 30 years older than me and I am 20, how old is my brother?")
