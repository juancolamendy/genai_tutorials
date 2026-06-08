import os
import re
from datetime import datetime, timedelta

from engine.agents import AgentRegistry, extract_frontmatter_body
from engine.skills import SkillRegistry

# Resolve relative to this file so the bot works from any working directory.
# Do NOT use "./workspace" — it breaks when invoked from outside bot2/.
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workspace")
CONTEXT_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "IDENTITY.md", "TOOLS.md"]


def parse_skill_frontmatter(content: str) -> dict:
    """Parse YAML-style frontmatter from a SKILL.md file."""
    parts = re.split(r"^---\s*$", content, maxsplit=2, flags=re.MULTILINE)
    if len(parts) < 3:
        return {}
    meta = {}
    for line in parts[1].splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def load_context_files() -> dict:
    """Load workspace context markdown files in CONTEXT_FILES order."""
    context = {}
    for filename in CONTEXT_FILES:
        path = os.path.join(WORKSPACE_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                context[filename] = f.read()
        except Exception:
            pass
    return context


def load_daily_memory() -> str:
    """Load today's and yesterday's daily memory logs from workspace/memory/."""
    entries = []
    for delta in [0, 1]:
        date_str = (datetime.now() - timedelta(days=delta)).strftime("%Y-%m-%d")
        path = os.path.join(WORKSPACE_DIR, "memory", f"{date_str}.md")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            entries.append(f"### Memory {date_str}\n\n{content}")
        except Exception:
            pass
    return "\n\n".join(entries)


def build_system_prompt(skill_registry: SkillRegistry, agent_registry: AgentRegistry) -> str:
    """Assemble the full system prompt from workspace files, skills index, and memory instructions.

    Args:
        skill_registry: Pre-loaded SkillRegistry singleton from main.
        agent_registry: Pre-loaded AgentRegistry singleton from main.
    """
    parts = []

    date_str = datetime.now().strftime("%A, %B %d, %Y")
    parts.append(f"## Current Date & Time\n\n{date_str}")

    for filename, content in load_context_files().items():
        if content:
            parts.append(f"## {filename}\n\n{content}")

    daily_mem = load_daily_memory()
    if daily_mem:
        parts.append(f"## Recent Memory\n\n{daily_mem}")

    skills = skill_registry.get_skills_index()
    if skills:
        parts.append(f"## Skills\n\n{skills}")

    agents_index = agent_registry.get_agents_index()
    if agents_index:
        parts.append(f"## Agents\n\n{agents_index}")

    return "\n\n".join(parts)


def build_subagent_system_prompt(
    agent_content: str,
    skill_registry: SkillRegistry | None,
    agent_registry: AgentRegistry | None,
) -> str:
    """Assemble a full system prompt for a subagent.

    Combines the agent body (frontmatter stripped) with current date,
    TOOLS.md context, skills index, and agents index.

    Args:
        agent_content: Raw markdown content of the agent's .md file (including frontmatter).
        skill_registry: Pre-loaded SkillRegistry singleton from main, or None to omit skills.
        agent_registry: Pre-loaded AgentRegistry singleton from main, or None to omit agents.
    """
    parts = []

    body = extract_frontmatter_body(agent_content)
    if body:
        parts.append(body)

    date_str = datetime.now().strftime("%A, %B %d, %Y")
    parts.append(f"## Current Date & Time\n\n{date_str}")

    if skill_registry is not None:
        skills = skill_registry.get_skills_index()
        if skills:
            parts.append(f"## Skills\n\n{skills}")

    if agent_registry is not None:
        agents_index = agent_registry.get_agents_index()
        if agents_index:
            parts.append(f"## Agents\n\n{agents_index}")

    return "\n\n".join(parts)
