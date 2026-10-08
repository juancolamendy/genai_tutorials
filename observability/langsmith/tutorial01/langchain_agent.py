import json
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = Path("langchain_agent_prompt.md").read_text(encoding="utf-8")


def check_weather(location: str) -> str:
    """
    Return a brief weather report for the given location.

    This function queries the free Open-Meteo geocoding and forecast APIs.
    It first resolves the location name to latitude/longitude coordinates,
    then fetches the current weather for those coordinates.

    Args:
        location: A city or place name (e.g., "Miami", "Paris, France").

    Returns:
        A human-readable string describing the current weather, including
        temperature and wind speed, or an error message if the location
        cannot be resolved or the weather service is unavailable.
    """
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
            return f"Could not find weather data for '{location}'."

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

        return (
            f"The weather in {display_name} is {description} "
            f"with a temperature of {temp}°C and wind speed of {wind} km/h."
        )
    except Exception as exc:  # noqa: BLE001
        return f"Sorry, I couldn't retrieve the weather for '{location}': {exc}"


agent = create_agent(
    model="openai:gpt-5",
    tools=[check_weather],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
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
