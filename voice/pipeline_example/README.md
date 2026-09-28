# pipeline_example

A real-time voice AI phone agent built with Python, Twilio, Deepgram, and LangChain/Groq.

## Overview

This project implements an asynchronous web server that handles inbound phone calls via Twilio
Media Streams. It performs real-time speech-to-text with Deepgram, generates conversational
responses with a LangChain-powered LLM, and streams synthesized speech back to the caller using
Deepgram's text-to-speech API.

## Features

- **Twilio Media Stream integration** — accepts inbound calls and streams audio over WebSockets.
- **Real-time transcription** — streams caller audio to Deepgram for live speech-to-text.
- **Conversational AI** — generates context-aware responses using LangChain with Groq-hosted LLMs.
- **Streaming TTS** — converts assistant responses into audio with Deepgram Aura and sends them
  back to Twilio as chunked audio.
- **Backchanneling** — optionally emits short acknowledgments ("uh-huh", "I see") based on a
  separate LLM call and rate-limiting logic.
- **Call lifecycle management** — configurable silence timeout, maximum call duration, greeting,
  and goodbye messages.
- **Call summarization** — produces a concise summary of the conversation after the call ends.
- **Async throughout** — built with `aiohttp` and `asyncio` for concurrent, low-latency streaming.

## Architecture

The server exposes three HTTP/WebSocket endpoints:

- `GET /` — health/status check.
- `POST /call/start` — Twilio webhook that returns a `<Connect><Stream>` TwiML response,
  directing the call to the streaming WebSocket.
- `GET /call/stream/{called}/{callsid}` — WebSocket endpoint that receives Twilio media events
  and orchestrates the audio pipeline.

Inside a call, three concurrent workers run:

1. `process_in_msg_worker` — handles inbound Twilio events (`start`, `media`, `mark`, `stop`, etc.)
   and forwards audio to Deepgram.
2. `process_transcript_worker` — consumes Deepgram transcription events and triggers response
   generation + TTS streaming.
3. `monitor_conversation_worker` — enforces silence timeout and maximum call duration.

Key modules:

| Module | Responsibility |
|--------|----------------|
| `main.py` | Web handlers and worker orchestration |
| `config.py` | Environment configuration and `ExecConfig` |
| `models.py` | `ExecState` — thread-safe per-call state |
| `audio.py` | Twilio audio output, buffering, mark tracking |
| `voice.py` | Text-to-speech via Deepgram and sentence segmentation |
| `transcript.py` | Deepgram speech-to-text WebSocket handling |
| `conversation.py` | LLM conversation chain and summarization |
| `backchannel.py` | Optional backchannel generation and rate limiting |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- Accounts and API keys for:
  - [Deepgram](https://deepgram.com/) (STT + TTS)
  - [Groq](https://groq.com/) (LLM)
  - [Twilio](https://www.twilio.com/) (phone number + Programmable Voice)
- A publicly reachable domain (Twilio must be able to POST to `/call/start` and connect to
  `wss://your-domain/call/stream/{called}/{callsid}`)

## Installation

```bash
# Clone the repository and enter the project directory
git clone <repo-url>
cd voice/pipeline_example

# Install dependencies with uv
uv sync

# Or install with pip
pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```env
# Required
DEEPGRAM_API_KEY=your_deepgram_api_key
GROQ_API_KEY=your_groq_api_key
SERVER_DOMAIN=your-public-domain.com

# Optional
AGENT_NAME=Jess
AGENT_PROMPT_PATH=agent_prompt.txt
PORT=3005

CONVERSATION_LLM_PROVIDER=groq
CONVERSATION_LLM_MODEL_NAME=llama-3.1-8b-instant
BACKCHANNEL_LLM_MODEL_NAME=llama3-8b-8192

DEEPGRAM_TRANSCRIPT_MODEL_NAME=nova-2-phonecall
DEEPGRAM_ENDPOINTING=300

VOICE_PROVIDER=deepgram
VOICE_MODEL_NAME=aura-asteria-en

MAX_CHAT_HISTORY_MESSAGES=20
MAX_CALL_TIME=900
SILENCE_TIMEOUT=30
```

### Agent prompt

The `agent_prompt.txt` file defines the business context, conversation goal, flow, and example
utterances. It is concatenated into the LLM system prompt at runtime. The default file included
in the repository configures the agent for "Mary's Dental" demo office.

## Usage

### Run locally

```bash
# With uv
uv run python -m pipeline_example

# With a manually installed environment
python -m pipeline_example
```

The server listens on the configured `PORT` (default `3005`).

### Expose to the internet

For Twilio to reach your local server, use a tunnel such as [ngrok](https://ngrok.com/):

```bash
ngrok http 3005
```

Set `SERVER_DOMAIN` to the ngrok domain (without `https://`).

### Connect a Twilio phone number

1. Buy or configure a Twilio phone number.
2. Under **Voice & Fax > A call comes in**, set the webhook to:
   `https://<SERVER_DOMAIN>/call/start` using HTTP `POST`.
3. Call the number. The agent will answer.

## Development

### Run tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=pipeline_example --cov-report=term-missing
```

### Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

### Type check

```bash
uv run mypy src/pipeline_example
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPGRAM_API_KEY` | — | Deepgram API key (required) |
| `GROQ_API_KEY` | — | Groq API key (required) |
| `SERVER_DOMAIN` | — | Public domain for Twilio webhooks (required) |
| `AGENT_NAME` | `Jess` | Name used by the agent |
| `AGENT_PROMPT_PATH` | `agent_prompt.txt` | Path to the agent prompt file |
| `PORT` | `3005` | HTTP server port |
| `CONVERSATION_LLM_PROVIDER` | `groq` | LLM provider for conversation |
| `CONVERSATION_LLM_MODEL_NAME` | `llama-3.1-8b-instant` | Conversation model |
| `BACKCHANNEL_LLM_MODEL_NAME` | `llama3-8b-8192` | Model for backchannel responses |
| `DEEPGRAM_TRANSCRIPT_MODEL_NAME` | `nova-2-phonecall` | Deepgram STT model |
| `DEEPGRAM_ENDPOINTING` | `300` | Deepgram endpointing value (ms) |
| `VOICE_PROVIDER` | `deepgram` | TTS provider |
| `VOICE_MODEL_NAME` | `aura-asteria-en` | Deepgram TTS voice model |
| `MAX_CHAT_HISTORY_MESSAGES` | `20` | Rolling window of chat history |
| `MAX_CALL_TIME` | `900` | Maximum call duration in seconds |
| `SILENCE_TIMEOUT` | `30` | Seconds of silence before ending call |

## Project Structure

```
voice/pipeline_example/
├── agent_prompt.txt          # Agent behavior / business prompt
├── pyproject.toml            # Project metadata, dependencies, tool config
├── pytest.ini                # Pytest configuration
├── README.md                 # This file
├── src/pipeline_example/
│   ├── __init__.py
│   ├── __main__.py           # Entrypoint
│   ├── main.py               # Web server and worker orchestration
│   ├── config.py             # Settings and environment validation
│   ├── models.py             # Per-call state model
│   ├── audio.py              # Twilio audio output handling
│   ├── voice.py              # TTS and text segmentation
│   ├── transcript.py         # STT via Deepgram
│   ├── conversation.py       # LLM conversation + summarization
│   └── backchannel.py        # Backchannel generation and rate limits
└── tests/                    # Pytest test suite
```

## Notes

- The agent currently expects Twilio to stream inbound audio as `audio/x-mulaw` at 8 kHz.
- Responses are segmented at sentence boundaries so audio can stream incrementally.
- The `mark` event tracking in `AudioProcessor` is used to detect when Twilio has finished
  playing audio, supporting clear-audio-buffer behavior when the user interrupts.

## License

[MIT](LICENSE) — or specify the project's actual license here.
