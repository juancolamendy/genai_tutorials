import json
import os
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Literal, Optional, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, wrap_model_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import traceable

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = Path("langchain_agent_prompt.md").read_text(encoding="utf-8")


class WeatherResult(TypedDict):
    """Structured result returned by the check_weather tool."""

    status: Literal["ok", "error"]
    location: str
    condition: str
    temperature_c: float
    wind_speed_kmh: float
    source: str
    error_message: str


def create_config(
    thread_id: str,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Build the invocation config for the agent.

    Args:
        thread_id: Unique identifier for the conversation thread.
        tags: Optional list of LangSmith tags for the run.
        metadata: Optional custom metadata to merge into the config.

    Returns:
        A config dict with configurable thread_id, tags, and metadata.
    """
    merged_metadata = {
        "thread_id": thread_id,
        "workflow": "weather_assistant",
        "environment": os.getenv("ENVIRONMENT", "local"),
        "prompt_version": os.getenv("PROMPT_VERSION", "weather-agent-v1"),
        "app_version": os.getenv("APP_VERSION", "local"),
    }
    if metadata:
        merged_metadata.update(metadata)

    return {
        "configurable": {"thread_id": thread_id},
        "tags": tags or ["weather", "assistant", "local"],
        "metadata": merged_metadata,
    }


def check_weather(location: str) -> str:
    """
    Return a structured weather report for a location as a JSON string.

    This function queries the free Open-Meteo geocoding and forecast APIs.
    It first resolves the location name to latitude/longitude coordinates,
    then fetches the current weather for those coordinates.

    Args:
        location: A city or place name (e.g., "Miami", "Paris, France").

    Returns:
        A JSON string representing a WeatherResult. On success, status is "ok"
        and includes weather details. On failure, status is "error" and
        error_message contains the reason.
    """
    base_result: WeatherResult = {
        "status": "error",
        "location": location,
        "condition": "",
        "temperature_c": 0.0,
        "wind_speed_kmh": 0.0,
        "source": "Open-Meteo",
        "error_message": "",
    }

    try:
        # Geocode the location using Open-Meteo's free geocoding API.
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": location, "count": "1"})
        )
        with urllib.request.urlopen(geo_url, timeout=10) as response:
            geo_data = json.loads(response.read().decode("utf-8"))

        results = geo_data.get("results")
        if not results:
            base_result["error_message"] = (
                f"Could not find weather data for '{location}'."
            )
            return json.dumps(base_result)

        place = results[0]
        lat = place["latitude"]
        lon = place["longitude"]
        display_name = place.get("name", location)

        # Fetch current weather from Open-Meteo's free forecast API.
        forecast_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                }
            )
        )
        with urllib.request.urlopen(forecast_url, timeout=10) as response:
            weather_data = json.loads(response.read().decode("utf-8"))

        current = weather_data["current_weather"]
        temp = current["temperature"]
        wind = current["windspeed"]
        weather_code = current.get("weathercode", 0)

        # Interpret the WMO weather code.
        weather_descriptions = {
            0: "clear sky",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "fog",
            48: "depositing rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            71: "slight snow fall",
            73: "moderate snow fall",
            75: "heavy snow fall",
            77: "snow grains",
            80: "slight rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "slight snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with slight hail",
            99: "thunderstorm with heavy hail",
        }
        description = weather_descriptions.get(weather_code, "unknown conditions")

        ok_result: WeatherResult = {
            "status": "ok",
            "location": display_name,
            "condition": description,
            "temperature_c": temp,
            "wind_speed_kmh": wind,
            "source": "Open-Meteo",
            "error_message": "",
        }
        return json.dumps(ok_result)
    except Exception as exc:  # noqa: BLE001
        base_result["error_message"] = str(exc)
        return json.dumps(base_result)


@wrap_model_call
@traceable(run_type="chain")
def trim_tool_messages(request, handler):
    """
    Compress conversation context once it reaches 5 messages.

    Removes completed tool exchanges (AIMessage with tool_calls and ToolMessage)
    from history before the most recent user message, while preserving the
    current turn's context and all plain AI/user/system messages.
    """
    messages = request.messages
    if len(messages) >= 5:
        last_human_index = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if isinstance(messages[i], HumanMessage)
            ),
            -1,
        )

        kept_prefix = [
            msg
            for msg in messages[:last_human_index]
            if not isinstance(msg, ToolMessage)
            and not (isinstance(msg, AIMessage) and msg.tool_calls)
        ]
        kept_suffix = messages[last_human_index:]
        request = request.override(messages=kept_prefix + kept_suffix)

    return handler(request)


pii_guard = PIIMiddleware(
    "email",
    strategy="redact",
    apply_to_input=True,
    apply_to_output=True,
    apply_to_tool_results=True,
)

agent = create_agent(
    model="openai:gpt-5",
    tools=[check_weather],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
    middleware=[pii_guard, trim_tool_messages],
    #debug=True,
)

thread_id = str(uuid.uuid4())
print(f'thread_id: {thread_id}')
config = create_config(
    thread_id=thread_id,
    tags=["weather", "assistant", "production"],
    metadata={"user_key": thread_id[:8]},
)

print("Weather Assistant (type 'q' to quit)")
while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() == "q":
        break

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )

    print("\nLast message:")
    print(result["messages"][-1].content)
