"""CLI entry point for bot2 (Agno-backed agent).

Usage::

    cd bot2
    uv run python main.py

On startup the user is prompted for a ``user_id`` and ``session_id``.
Within the REPL:

- ``/new``  — reset the session (generates a new timestamped session_id)
- ``/quit`` or ``/exit`` — terminate the bot
"""

import os
from datetime import datetime

from dotenv import load_dotenv

from agno.agent import Agent
from agno.tools.memory import MemoryTools

from engine.agents import AgentRegistry, AgentsToolkit
from constants import APPROVALS_FILE, MEMORY_DIR, SESSIONS_DIR
from engine.llm_config import ModelProvider, ModelSpec, load_model, register_model
from engine.memory_db import MarkdownMemoryDb
from engine.prompt import build_system_prompt
from engine.skills import SkillRegistry
from engine.storage import JsonlAgentDb
from engine.tools import BotToolkit

load_dotenv()

# Model registry — keys match agent markdown frontmatter ``model:`` values.
CLAUDE_SONNET = "claude-sonnet-4-6"
CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
GROQ_LLAMA_70B = "groq/llama-3.3-70b-versatile"

register_model(CLAUDE_SONNET,     ModelSpec(ModelProvider.ANTHROPIC, "claude-sonnet-4-6",          thinking=False))
register_model(CLAUDE_HAIKU,      ModelSpec(ModelProvider.ANTHROPIC, "claude-haiku-4-5-20251001",   thinking=False))
register_model(GEMINI_FLASH,      ModelSpec(ModelProvider.GOOGLE,    "gemini-2.5-flash",            thinking=False))
register_model(GEMINI_FLASH_LITE, ModelSpec(ModelProvider.GOOGLE,    "gemini-2.5-flash-lite",       thinking=False))
register_model(GROQ_LLAMA_70B,    ModelSpec(ModelProvider.GROQ,      "llama-3.3-70b-versatile",     thinking=False))

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

skill_registry = SkillRegistry()
agent_registry = AgentRegistry()


def build_agent() -> Agent:
    """Construct and return a fully-wired Agno Agent.

    Returns:
        Agent: Ready-to-use agent instance with storage, memory, and tools.
    """
    return Agent(
        model=load_model(CLAUDE_HAIKU, cache_system_prompt=True),
        system_message=build_system_prompt(skill_registry, agent_registry),
        tools=[
            BotToolkit(approvals_file=APPROVALS_FILE),
            AgentsToolkit(skill_registry=skill_registry, agent_registry=agent_registry, default_model_key=CLAUDE_HAIKU),
            MemoryTools(db=MarkdownMemoryDb(MEMORY_DIR)),
        ],
        db=JsonlAgentDb(sessions_dir=SESSIONS_DIR),
        add_history_to_context=True,
        num_history_runs=20,
        max_tool_calls_from_history=5,
        debug_mode=True,
    )


def main() -> None:
    """Run the interactive CLI REPL."""
    try:
        user_id = input('Enter your user ID: ').strip() or 'default'
        session_id = (
            input('Enter your session ID: ').strip()
            or datetime.now().strftime('%Y%m%d%H%M%S')
        )
    except EOFError:
        print('No TTY detected. Exiting.')
        return

    print(
        f"Session loaded for user '{user_id}', session '{session_id}'. "
        'Type /quit or /exit to quit. Type /new to reset the session.'
    )

    agent = build_agent()

    while True:
        try:
            text = input('You: ')
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break
        if text in ['/quit', '/exit']:
            print('Goodbye!')
            break
        if text == '/new':
            session_id = datetime.now().strftime('%Y%m%d%H%M%S')
            print(f'Session reset. New session ID: {session_id}')
            continue
        try:
            response = agent.run(text, user_id=user_id, session_id=session_id)
            print(f'Claude: {response.content or ""}')
        except KeyboardInterrupt:
            print('\nGoodbye!')
            break
        except Exception as e:  # noqa: BLE001
            print(f'Error: {e}')


if __name__ == '__main__':
    main()
