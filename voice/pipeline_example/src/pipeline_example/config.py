import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("phoneagent")

# Load .env BEFORE any env read.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AGENT_NAME = os.getenv("AGENT_NAME", "Jess")
AGENT_PROMPT_PATH = Path(os.getenv("AGENT_PROMPT_PATH", PROJECT_ROOT / "agent_prompt.txt"))

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERVER_DOMAIN = os.getenv("SERVER_DOMAIN")
PORT = int(os.getenv("PORT", "3005"))

CONVERSATION_LLM_PROVIDER = os.getenv("CONVERSATION_LLM_PROVIDER", "groq")
CONVERSATION_LLM_MODEL_NAME = os.getenv("CONVERSATION_LLM_MODEL_NAME", "llama-3.1-8b-instant")
BACKCHANNEL_LLM_MODEL_NAME = os.getenv("BACKCHANNEL_LLM_MODEL_NAME", "llama3-8b-8192")
DEEPGRAM_TRANSCRIPT_MODEL_NAME = os.getenv("DEEPGRAM_TRANSCRIPT_MODEL_NAME", "nova-2-phonecall")
DEEPGRAM_ENDPOINTING = int(os.getenv("DEEPGRAM_ENDPOINTING", "300"))

VOICE_PROVIDER = os.getenv("VOICE_PROVIDER", "deepgram")
VOICE_MODEL_NAME = os.getenv("VOICE_MODEL_NAME", "aura-asteria-en")

MAX_CHAT_HISTORY_MESSAGES = int(os.getenv("MAX_CHAT_HISTORY_MESSAGES", "20"))
MAX_CALL_TIME = int(os.getenv("MAX_CALL_TIME", "900"))
SILENCE_TIMEOUT = int(os.getenv("SILENCE_TIMEOUT", "30"))


@dataclass(frozen=True)
class ExecConfig:
    agent_prompt: str = ""
    voice_provider: str = VOICE_PROVIDER
    voice_model_name: str = VOICE_MODEL_NAME
    backchannel_enabled: bool = False
    silence_timeout: int = SILENCE_TIMEOUT
    response_delay: int = 0
    start_call_message: str = "Hi, how can I help you today?"
    end_call_message: str = "Good bye. Have a good day!"
    max_call_time: int = MAX_CALL_TIME


def validate_environment() -> None:
    missing = [
        var
        for var in ("DEEPGRAM_API_KEY", "GROQ_API_KEY", "SERVER_DOMAIN")
        if not os.getenv(var)
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


def load_agent_prompt(path: Path | None = None) -> str:
    target = path or AGENT_PROMPT_PATH
    if not target.exists():
        raise FileNotFoundError(f"Agent prompt file not found: {target}")
    text = target.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Agent prompt file is empty: {target}")
    return text
