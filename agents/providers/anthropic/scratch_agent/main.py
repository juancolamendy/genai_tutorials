import os
import json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

import anthropic

load_dotenv()


class BaseTool:
    """Base class for all agent tools"""
    
    def get_name(self) -> str:
        """Returns the tool name"""
        raise NotImplementedError
    
    def get_description(self) -> str:
        """Returns what the tool does"""
        raise NotImplementedError
    
    def get_parameters(self) -> Dict[str, Any]:
        """Returns the parameter schema for the tool"""
        raise NotImplementedError
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Executes the tool with given parameters"""
        raise NotImplementedError


class WeatherTool(BaseTool):
    """A tool for retrieving weather information"""
    
    def get_name(self) -> str:
        return "get_weather"
    
    def get_description(self) -> str:
        return "Retrieves current weather information for a specified city. Use this when users ask about weather conditions, temperature, or forecasts."
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name (e.g., 'New York', 'London', 'Tokyo')"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit preference"
                }
            },
            "required": ["city"]
        }
    
    def execute(self, city: str, unit: str = "celsius") -> Dict[str, Any]:
        """
        Simulates weather data retrieval.
        In production, this would call a real weather API.
        """
        # Simulated weather data
        weather_database = {
            "new york": {"temp": 22, "condition": "Partly cloudy", "humidity": 65},
            "london": {"temp": 15, "condition": "Rainy", "humidity": 80},
            "tokyo": {"temp": 28, "condition": "Sunny", "humidity": 55},
            "paris": {"temp": 18, "condition": "Overcast", "humidity": 70},
        }
        
        city_key = city.lower()
        
        if city_key not in weather_database:
            return {
                "error": f"Weather data not available for {city}",
                "available_cities": list(weather_database.keys())
            }
        
        data = weather_database[city_key]
        temp = data["temp"]
        
        # Convert temperature if needed
        if unit == "fahrenheit":
            temp = (temp * 9/5) + 32
        
        return {
            "city": city,
            "temperature": round(temp, 1),
            "unit": unit,
            "condition": data["condition"],
            "humidity": data["humidity"]
        }


def create_tool_schema(tool: BaseTool) -> Dict[str, Any]:
    """
    Converts a tool into Claude's expected schema format.
    
    Args:
        tool: The tool instance to convert
        
    Returns:
        A dictionary matching Claude's tool schema specification
    """
    return {
        "name": tool.get_name(),
        "description": tool.get_description(),
        "input_schema": tool.get_parameters()
    }

def register_tools(tools: List[BaseTool]) -> tuple[List[Dict], Dict[str, BaseTool]]:
    """
    Registers multiple tools and creates lookup structures.
    
    Args:
        tools: List of tool instances to register
        
    Returns:
        A tuple of (tool_schemas, tool_map) for agent use
    """
    tool_schemas = [create_tool_schema(tool) for tool in tools]
    tool_map = {tool.get_name(): tool for tool in tools}
    
    return tool_schemas, tool_map

class Agent:
    """
    An AI agent powered by Claude that can use tools to solve problems.
    """
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the agent with tools and configuration.
        
        Args:
            tools: List of tool instances the agent can use
            model: Claude model identifier
            system_prompt: Custom system instructions for the agent
        """
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Set default system prompt if none provided
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Register tools
        if tools:
            self.tool_schemas, self.tool_map = register_tools(tools)
        else:
            self.tool_schemas, self.tool_map = [], {}
    
    def _get_default_system_prompt(self) -> str:
        """Returns the default system prompt for the agent"""
        return """You are a helpful AI assistant that solves problems systematically.
        
When faced with complex questions, break them down into smaller steps.
Use available tools when you need to retrieve information or perform actions.
Think through problems logically and explain your reasoning.
Always verify your answers before presenting them to the user."""

    def send_message(self, user_message: str) -> anthropic.types.Message:
        """
        Sends a user message to Claude and returns the response.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Claude's response message object
        """
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "max_tokens": 4096,
            "system": self.system_prompt,
            "messages": self.conversation_history
        }
        
        # Add tools if available
        if self.tool_schemas:
            api_params["tools"] = self.tool_schemas
        
        # Call Claude API
        response = self.client.messages.create(**api_params)
        
        # Add assistant's response to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        return response


def get_text_response(response: anthropic.types.Message) -> str:
    """
    Extracts text content from Claude's response.
    
    Args:
        response: The message object from Claude
        
    Returns:
        Concatenated text from all text content blocks
    """
    text_parts = []
    
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
    
    return "\n".join(text_parts)


def extract_tool_uses(response: anthropic.types.Message) -> List[Dict[str, Any]]:
    """
    Extracts all tool use blocks from Claude's response.
    
    Args:
        response: The message object from Claude
        
    Returns:
        List of dictionaries containing tool use information
    """
    tool_uses = []
    
    for block in response.content:
        if block.type == "tool_use":
            tool_uses.append({
                "id": block.id,
                "name": block.name,
                "input": block.input
            })
    
    return tool_uses

def execute_tool_safely(
    tool: BaseTool,
    tool_name: str,
    tool_input: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes a tool with error handling and logging.
    
    Args:
        tool: The tool instance to execute
        tool_name: Name of the tool (for logging)
        tool_input: Parameters to pass to the tool
        
    Returns:
        The tool's execution result or error information
    """
    try:
        print(f"  → Executing {tool_name} with input: {tool_input}")
        result = tool.execute(**tool_input)
        print(f"  ✓ Tool execution successful")
        return result
        
    except Exception as e:
        error_msg = f"Tool execution failed: {str(e)}"
        print(f"  ✗ {error_msg}")
        return {"error": error_msg}

def create_tool_result_content(
    tool_uses: List[Dict[str, Any]],
    tool_map: Dict[str, BaseTool]
) -> List[Dict[str, Any]]:
    """
    Executes tools and formats results for Claude.
    
    Args:
        tool_uses: List of tool use requests from Claude
        tool_map: Mapping of tool names to tool instances
        
    Returns:
        List of tool result objects formatted for the API
    """
    tool_results = []
    
    for tool_use in tool_uses:
        tool_name = tool_use["name"]
        tool_id = tool_use["id"]
        tool_input = tool_use["input"]
        
        # Execute the tool
        tool = tool_map[tool_name]
        result = execute_tool_safely(tool, tool_name, tool_input)
        
        # Format result for Claude
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": json.dumps(result)
        })
    
    return tool_results

def run_agent_loop(
    agent: Agent,
    initial_message: str,
    max_iterations: int = 10,
    verbose: bool = True
) -> str:
    """
    Runs the agent in a loop, handling tool use until completion.
    
    Args:
        agent: The agent instance to run
        initial_message: The user's initial query
        max_iterations: Maximum iterations to prevent infinite loops
        verbose: Whether to print detailed execution logs
        
    Returns:
        The final text response from the agent
    """
    iteration = 0
    current_message = initial_message
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"AGENT EXECUTION START")
        print(f"{'='*80}")
        print(f"User Query: {initial_message}\n")
    
    while iteration < max_iterations:
        iteration += 1
        
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
        
        # Send message to Claude
        response = agent.send_message(current_message)
        
        # Check if Claude wants to use tools
        tool_uses = extract_tool_uses(response)
        
        if not tool_uses:
            # No tools needed - we have the final answer
            final_response = get_text_response(response)
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"AGENT EXECUTION COMPLETE")
                print(f"{'='*80}")
                print(f"Final Response: {final_response}\n")
            
            return final_response
        
        # Execute tools and prepare results
        if verbose:
            print(f"Claude wants to use {len(tool_uses)} tool(s):")
        
        tool_results = create_tool_result_content(tool_uses, agent.tool_map)
        
        # Send tool results back to Claude
        current_message = tool_results
    
    return "Maximum iterations reached without completion."

def create_weather_agent() -> Agent:
    """
    Factory function to create a weather-enabled agent.
    
    Returns:
        Configured agent instance with weather tool
    """
    weather_tool = WeatherTool()
    
    custom_system_prompt = """You are a helpful weather assistant.
    
When users ask about weather, use the get_weather tool to retrieve accurate information.
Present weather data in a clear, conversational manner.
If users don't specify a temperature unit, use celsius by default.
Always mention the weather condition along with the temperature."""
    
    return Agent(
        tools=[weather_tool],
        system_prompt=custom_system_prompt
    )

def main():
    """Main execution function demonstrating agent capabilities"""
    
    # Create agent with weather tool
    agent = create_weather_agent()
    
    # Test Case 1: Simple weather query
    print("\n" + "="*80)
    print("TEST 1: Simple Weather Query")
    print("="*80)
    response = run_agent_loop(
        agent,
        "What's the weather like in Tokyo?",
        verbose=True
    )
    print(response)
    
    # Test Case 2: Multi-city comparison
    print("\n" + "="*80)
    print("TEST 2: Multi-City Comparison")
    print("="*80)
    response = run_agent_loop(
        agent,
        "Compare the weather in London and Paris. Which city is warmer?",
        verbose=True
    )
    print(response)
    # Test Case 3: Temperature unit conversion
    print("\n" + "="*80)
    print("TEST 3: Temperature Unit Preference")
    print("="*80)
    response = run_agent_loop(
        agent,
        "What's the temperature in New York in Fahrenheit?",
        verbose=True
    )
    print(response)

if __name__ == "__main__":
    main()