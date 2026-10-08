import json
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Literal, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langgraph.checkpoint.memory import InMemorySaver

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
    middleware=[pii_guard],
    #debug=True,
)

thread_id = str(uuid.uuid4())
print(f'thread_id: {thread_id}')
config = {"configurable": {"thread_id": thread_id}}

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
