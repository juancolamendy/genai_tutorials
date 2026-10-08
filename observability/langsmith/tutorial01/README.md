# Weather Assistant

A conversational travel assistant powered by LangChain/LangGraph that can answer weather-related questions using a real-time weather tool. It demonstrates agent tools, structured outputs, checkpoint memory, PII redaction, and custom context-compression middleware.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- An OpenAI API key

## Installation

1. Clone or navigate to the project directory.
2. Sync dependencies with uv:

```bash
uv sync
```

Or, if you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Environment Variables

Create a `.env` file in the project root with at least the following variable:

```bash
OPENAI_API_KEY=sk-...
```

Optional LangSmith tracing variables:

```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=tutorial01
LANGSMITH_API_KEY=...
```

## Running the Application

Start the assistant from the project root:

```bash
uv run python langchain_agent.py
```

Or, using the virtual environment directly:

```bash
.venv/bin/python langchain_agent.py
```

You will see a prompt like:

```
thread_id: <uuid>
Weather Assistant (type 'q' to quit)

You:
```

Type a weather question, for example:

```
What is the weather in Miami?
```

Type `q` to quit.

## Project Structure

- `langchain_agent.py` — Main application: agent setup, weather tool, and interactive loop.
- `langchain_agent_prompt.md` — System prompt loaded by the agent at runtime.
- `pyproject.toml` — Project metadata and dependencies.
- `.env` — Environment variables (not committed).

## Changes

The following changes were made to the original starter script:

1. **Real weather API integration** — `check_weather` now fetches live weather data from the free [Open-Meteo](https://open-meteo.com/) API. It uses the geocoding API to resolve a location name to coordinates and the forecast API to retrieve current conditions.

2. **Structured tool output with TypedDict** — `check_weather` returns a JSON string representing a `WeatherResult` with typed fields: `status`, `location`, `condition`, `temperature_c`, `wind_speed_kmh`, `source`, and `error_message`. Errors return `status="error"` with details in `error_message`.

3. **External system prompt** — The system prompt was moved from an inline string into `langchain_agent_prompt.md` and is loaded at runtime using `pathlib`.

4. **Checkpoint memory** — Added `InMemorySaver` from LangGraph so the agent maintains conversation state within a session. A unique `thread_id` is generated for each run using the `uuid` library.

5. **PII redaction middleware** — Added `PIIMiddleware` configured to detect email addresses and redact them from user input, model output, and tool results.

6. **Interactive loop** — Replaced the single-shot invocation with a `while True` loop that continuously accepts user input until the user enters `q`.

7. **Context-compression middleware** — Added a custom `wrap_model_call` middleware that compresses conversation context once it reaches five messages. It removes older `ToolMessage` and tool-call `AIMessage` pairs from history before the most recent user message, while preserving the current turn's tool context and all plain AI/user/system messages. The middleware is decorated with `@traceable` so it appears in LangSmith traces.
