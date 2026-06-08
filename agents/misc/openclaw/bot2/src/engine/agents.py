"""Sub-agent support for bot2 (Agno).

Provides:
    extract_frontmatter_body: strip YAML frontmatter, return body text
    AgentRegistry: scan workspace/agents/, cache entries, build XML index
    AgentsToolkit: Agno Toolkit with run_agent tool
"""

import html
import os
import re

from agno.agent import Agent
from agno.run import RunContext  # verified agno 2.5.9
from agno.tools import Toolkit

from engine.tools import BotToolkit
from constants import APPROVALS_FILE
from engine.llm_config import load_model

# Resolved via __file__ so the bot works from any working directory.
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "workspace")


def extract_frontmatter_body(content: str) -> str:
    """Return body text of a markdown file with YAML frontmatter stripped.

    Uses maxsplit=2 so --- horizontal rules inside the body are never split.
    If no closing --- is found, returns the full content stripped.
    """
    parts = re.split(r"^---\s*$", content, maxsplit=2, flags=re.MULTILINE)
    if len(parts) == 3:
        return parts[2].strip()
    return content.strip()


def _parse_frontmatter(content: str) -> dict:
    """Parse YAML-style frontmatter key-value pairs from markdown content."""
    parts = re.split(r"^---\s*$", content, maxsplit=2, flags=re.MULTILINE)
    if len(parts) < 3:
        return {}
    meta: dict[str, str] = {}
    for line in parts[1].splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


class AgentRegistry:
    def __init__(self, workspace_dir: str = WORKSPACE_DIR):
        self._workspace_dir = workspace_dir
        self._registry: dict[str, dict] | None = None
        self._agents_list: list[dict] | None = None

    def load_agents(self) -> list[dict]:
        """Read workspace/agents/, parse each agent .md, and cache the results."""
        if self._registry is not None:
            return self._agents_list  # type: ignore[return-value]

        self._registry = {}
        self._agents_list = []

        agents_dir = os.path.join(self._workspace_dir, "agents")
        try:
            entries = sorted(os.listdir(agents_dir))
        except OSError:
            return self._agents_list

        for name in entries:
            dir_path = os.path.join(agents_dir, name)
            if not os.path.isdir(dir_path):
                continue
            agent_file = os.path.join(dir_path, f"{name}.md")
            try:
                with open(agent_file, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue
            meta = _parse_frontmatter(content)
            agent_name = meta.get("name", "").strip()
            if not agent_name:
                continue
            description = meta.get("description", "")
            model = meta.get("model") or None
            self._registry[agent_name] = {"content": content, "model": model}
            self._agents_list.append({
                "name": agent_name,
                "description": description,
            })

        return self._agents_list

    def get(self, agent_name: str) -> dict | None:
        """Return registry entry for agent_name, or None if not found."""
        self.load_agents()
        return self._registry.get(agent_name)  # type: ignore[union-attr]

    def get_agents_index(self) -> str:
        """Return a formatted XML agents index string for use in the system prompt."""
        agents_list = self.load_agents()
        if not agents_list:
            return ""

        xml_entries = "\n".join(
            f"  <agent>\n"
            f"    <name>{html.escape(a['name'])}</name>\n"
            f"    <description>{html.escape(a['description'])}</description>\n"
            f"  </agent>"
            for a in agents_list
        )
        preamble = (
            "When a task is better handled by a specialist, use the `run_agent` tool "
            "with the agent's name and a clear task description.\n\n"
        )
        return preamble + f"<available_agents>\n{xml_entries}\n</available_agents>"


class AgentsToolkit(Toolkit):
    """Agno Toolkit providing the run_agent tool for sub-agent dispatch."""

    def __init__(
        self,
        skill_registry=None,
        agent_registry: AgentRegistry | None = None,
        default_model_key: str = "",
    ) -> None:
        self._skill_registry = skill_registry
        self._agent_registry = agent_registry
        self._default_model_key = default_model_key
        super().__init__(name="agent_tools", tools=[self.run_agent])

    def run_agent(self, run_context: RunContext, agent_name: str, task: str) -> str:
        """Dispatch a task to a specialized sub-agent and return its response.

        Args:
            agent_name: Name of the agent as listed in the agents index.
            task: The task or question to send to the agent.

        Note:
            run_context is injected automatically by Agno — do not pass it manually.
        """
        try:
            entry = self._agent_registry.get(agent_name) if self._agent_registry else None
            if entry is None:
                return f"Error: agent '{agent_name}' not found."

            from engine.prompt import build_subagent_system_prompt
            system_prompt = build_subagent_system_prompt(entry["content"], self._skill_registry, self._agent_registry)
            model_key = entry["model"] or self._default_model_key

            sub_agent = Agent(
                model=load_model(model_key),
                system_message=system_prompt,
                tools=[
                    BotToolkit(approvals_file=APPROVALS_FILE),
                    AgentsToolkit(skill_registry=self._skill_registry, agent_registry=self._agent_registry, default_model_key=self._default_model_key),
                ],
                debug_mode=True,
                # no db, no session_id — ephemeral one-shot agent
            )

            resp = sub_agent.run(task, session_state=run_context.session_state)
            return resp.content or ""

        except Exception as e:
            return f"Error running agent '{agent_name}': {e}"
