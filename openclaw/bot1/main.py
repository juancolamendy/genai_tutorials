import asyncio
import html
import inspect
import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from typing import get_type_hints

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

SESSIONS_DIR = "./sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

MEMORY_DIR = "./memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

SAFE_COMMANDS = {"ls", "cat", "head", "tail", "wc", "date", "whoami", "echo"}
DANGEROUS_PATTERNS = [r"\brm\b", r"\bsudo\b", r"\bchmod\b", r"\bcurl.*\|.*sh"]
APPROVALS_FILE = "./workspace/exec-approvals.json"

WORKSPACE_DIR = "./workspace"
_agents_registry: dict = {}
CONTEXT_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "IDENTITY.md", "TOOLS.md"]

def build_memory_prompt() -> str:
    return """## Memory Instructions
You have a long-term memory system.
- Use save_memory to store important information (user preferences, key facts, project details).
- Use memory_search at the start of conversations to recall context from previous sessions.
Memory files are stored in ./memory/ as markdown files."""

def build_subagent_system_prompt(agent_file_content: str) -> str:
    """Assemble the system prompt for a sub-agent.

    Combines the agent's own instructions (body of its .md file) with the
    current date, memory tool instructions, skills index, and agents index.
    """
    parts = []

    body = extract_frontmatter_body(agent_file_content)
    if body:
        parts.append(body)

    date_str = datetime.now().strftime("%A, %B %d, %Y")
    parts.append(f"## Current Date & Time\n\n{date_str}")

    skills = load_skills_index()
    if skills:
        parts.append(f"## Skills\n\n{skills}")

    agents = load_agents_index()
    if agents:
        parts.append(f"## Agents\n\n{agents}")

    return "\n\n".join(parts)


def build_system_prompt() -> str:
    """Assemble the system prompt from workspace files, skills index, and memory instructions."""
    parts = []

    # 1. Date (always present, date-only for prompt caching compatibility)
    date_str = datetime.now().strftime("%A, %B %d, %Y")
    parts.append(f"## Current Date & Time\n\n{date_str}")

    # 2–6. Context files (in CONTEXT_FILES order, silently skip missing)
    for filename, content in load_context_files().items():
        if content:
            parts.append(f"## {filename}\n\n{content}")

    # 7. Daily memory logs
    daily_mem = load_daily_memory()
    if daily_mem:
        parts.append(f"## Recent Memory\n\n{daily_mem}")

    # 8. Skills index
    skills = load_skills_index()
    if skills:
        parts.append(f"## Skills\n\n{skills}")

    # 9. Agents index
    agents = load_agents_index()
    if agents:
        parts.append(f"## Agents\n\n{agents}")

    # 10. Memory tool instructions (always present)
    parts.append(build_memory_prompt())

    return "\n\n".join(parts)

def parse_skill_frontmatter(content: str) -> dict:
    """Parse YAML-style frontmatter from a SKILL.md file.

    Returns a dict of key/value pairs from the frontmatter block,
    or {} if frontmatter is absent or malformed.
    """
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

def extract_frontmatter_body(content: str) -> str:
    """Return the body of a markdown file after stripping YAML frontmatter.

    Uses the same multiline regex as parse_skill_frontmatter so that
    '---' appearing mid-line or inside paragraphs is not treated as a
    delimiter. Returns the full content stripped if no frontmatter is found.
    """
    parts = re.split(r"^---\s*$", content, maxsplit=2, flags=re.MULTILINE)
    if len(parts) == 3:
        return parts[2].strip()
    return content.strip()

def load_context_files() -> dict:
    """Load workspace context markdown files.

    Returns a dict mapping filename to content, in CONTEXT_FILES order.
    Missing or unreadable files are silently skipped.
    """
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
    """Load today's and yesterday's daily memory logs from workspace/memory/.

    Returns a formatted string with each present file as a sub-section,
    today first. Returns "" if neither file exists or can be read.
    """
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

def load_skills_index() -> str:
    """Scan workspace/skills/ and build a compact XML skills index.

    Returns the preamble + <available_skills> XML block, or "" if no
    skills are found or the directory does not exist.
    """
    skills_dir = os.path.join(WORKSPACE_DIR, "skills")
    try:
        entries = sorted(os.listdir(skills_dir))
    except OSError:
        return ""

    skills = []
    for name in entries:
        dir_path = os.path.join(skills_dir, name)
        if not os.path.isdir(dir_path):
            continue
        skill_file = os.path.join(dir_path, "SKILL.md")
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        meta = parse_skill_frontmatter(content)
        skills.append({
            "name": meta.get("name", name),
            "description": meta.get("description", ""),
            "location": os.path.join(WORKSPACE_DIR, "skills", name, "SKILL.md"),
            "directory": os.path.join(WORKSPACE_DIR, "skills", name),
        })

    if not skills:
        return ""

    xml_entries = "\n".join(
        f"  <skill>\n"
        f"    <name>{html.escape(s['name'])}</name>\n"
        f"    <description>{html.escape(s['description'])}</description>\n"
        f"    <location>{html.escape(s['location'])}</location>\n"
        f"    <directory>{html.escape(s['directory'])}</directory>\n"
        f"  </skill>"
        for s in skills
    )
    return (
        "When a task matches one of the skills below, use the `read_file` tool to "
        "load the SKILL.md at the listed location for detailed instructions.\n\n"
        "All scripts and paths referenced inside a SKILL.md are relative to that "
        "skill's <directory>. For example, if a skill says `uv run ./scripts/foo.py`, "
        "the full path is <directory>/scripts/foo.py. Always prefix script paths with "
        "the skill's <directory> when calling run_command.\n\n"
        f"<available_skills>\n{xml_entries}\n</available_skills>"
    )

def load_agents_index() -> str:
    """Scan workspace/agents/ and build a compact XML agents index.

    Clears and repopulates _agents_registry on every call. Returns the
    preamble + <available_agents> XML block, or "" if no agents are found
    or the directory does not exist.
    """
    global _agents_registry
    _agents_registry.clear()

    agents_dir = os.path.join(WORKSPACE_DIR, "agents")
    try:
        entries = sorted(os.listdir(agents_dir))
    except OSError:
        return ""

    agents = []
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
        meta = parse_skill_frontmatter(content)
        agent_name = meta.get("name", "").strip()
        if not agent_name:
            continue
        model = meta.get("model", None) or None
        _agents_registry[agent_name] = {"file_path": agent_file, "model": model}
        agents.append({
            "name": agent_name,
            "description": meta.get("description", ""),
            "location": agent_file,
            "directory": dir_path,
        })

    if not agents:
        return ""

    xml_entries = "\n".join(
        f"  <agent>\n"
        f"    <name>{html.escape(a['name'])}</name>\n"
        f"    <description>{html.escape(a['description'])}</description>\n"
        f"    <location>{html.escape(a['location'])}</location>\n"
        f"    <directory>{html.escape(a['directory'])}</directory>\n"
        f"  </agent>"
        for a in agents
    )
    return (
        "When a task matches one of the agents below, use the `run_agent` tool "
        "to dispatch the task to that agent.\n\n"
        f"<available_agents>\n{xml_entries}\n</available_agents>"
    )


def load_approvals():
    if os.path.exists(APPROVALS_FILE):
        with open(APPROVALS_FILE) as f:
            return json.load(f)
    return {"allowed": [], "denied": []}

def save_approval(command, approved):
    approvals = load_approvals()
    key = "allowed" if approved else "denied"
    if command not in approvals[key]:
        approvals[key].append(command)
    with open(APPROVALS_FILE, "w") as f:
        json.dump(approvals, f, indent=2)

def check_command_safety(command):
    """Returns 'safe', 'approved', or 'needs_approval'."""
    base_cmd = command.strip().split()[0] if command.strip() else ""
    if base_cmd in SAFE_COMMANDS:
        return "safe"
    approvals = load_approvals()
    if command in approvals["allowed"]:
        return "approved"
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return "needs_approval"
    return "needs_approval"

# --- Tool registry ---

_PYTHON_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

def _parse_param_docs(docstring: str) -> dict:
    """Extract :param name: description entries from a docstring."""
    params = {}
    for match in re.finditer(r":param (\w+):\s*(.+)", docstring or ""):
        params[match.group(1)] = match.group(2).strip()
    return params

def _infer_schema(fn) -> dict:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    param_docs = _parse_param_docs(fn.__doc__)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        json_type = _PYTHON_TO_JSON_TYPE.get(hints.get(name), "string")
        prop = {"type": json_type}
        if name in param_docs:
            prop["description"] = param_docs[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}

class ToolRegistry:
    def __init__(self):
        self._tools: dict = {}

    def register(self, name: str, fn):
        description = (fn.__doc__ or "").strip().splitlines()[0]
        self._tools[name] = {
            "fn": fn,
            "schema": {
                "name": name,
                "description": description,
                "input_schema": _infer_schema(fn),
            }
        }

    def get_tool(self, name: str):
        return self._tools.get(name)

    def descriptions(self) -> list:
        return [entry["schema"] for entry in self._tools.values()]

# --- Tool functions ---

def tool_run_command(command: str) -> str:
    """Run a shell command on the user's computer.
    :param command: The shell command to run.
    """
    safety = check_command_safety(command)

    if safety == "needs_approval":
        print(f"  [approval needed] {command}")
        answer = input("  Allow this command? (y/n): ").strip().lower()
        approved = answer == "y"
        save_approval(command, approved)
        if not approved:
            return "Command denied by user."

    result = subprocess.run(
        command, shell=True,
        capture_output=True, text=True, timeout=30
    )
    return result.stdout + result.stderr

def tool_read_file(path: str) -> str:
    """Read a file from the filesystem.
    :param path: Path to the file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"

def tool_write_file(path: str, content: str) -> str:
    """Write content to a file.
    :param path: Path to the file.
    :param content: Content to write.
    """
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote to {path}"

def tool_web_search(query: str) -> str:
    """Search the web for information.
    :param query: Search query.
    """
    return f"Search results for: {query}"

def tool_save_memory(key: str, content: str) -> str:
    """Save important information to long-term memory. Use for user preferences, key facts, and anything worth remembering across sessions.
    :param key: Short label, e.g. 'user-preferences', 'project-notes'.
    :param content: The information to remember.
    """
    filepath = os.path.join(MEMORY_DIR, f"{key}.md")
    with open(filepath, "w") as f:
        f.write(content)
    return f"Saved to memory: {key}"

def tool_memory_search(query: str) -> str:
    """Search long-term memory for relevant information. Use at the start of conversations to recall context.
    :param query: What to search for.
    """
    q = query.lower()
    results = []
    for fname in os.listdir(MEMORY_DIR):
        if fname.endswith(".md"):
            with open(os.path.join(MEMORY_DIR, fname), "r") as f:
                content = f.read()
            if any(word in content.lower() for word in q.split()):
                results.append(f"--- {fname} ---\n{content}")
    return "\n\n".join(results) if results else "No matching memories found."

registry = ToolRegistry()
registry.register("run_command", tool_run_command)
registry.register("read_file", tool_read_file)
registry.register("write_file", tool_write_file)
registry.register("web_search", tool_web_search)
registry.register("save_memory", tool_save_memory)
registry.register("memory_search", tool_memory_search)

def _make_tool_run_agent(user_id: str, session_id: str):
    """Factory that returns a run_agent closure bound to a specific session."""
    def run_agent(agent_name: str, input: str) -> str:
        """Dispatch a task to a specialized sub-agent and return its response.
        :param agent_name: Name of the agent as listed in the agents index.
        :param input: The task or question to send to the agent.
        """
        entry = _agents_registry.get(agent_name)
        if entry is None:
            return f"Error: agent '{agent_name}' not found."

        try:
            with open(entry["file_path"], "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"Error running agent '{agent_name}': {e}"

        system_prompt = build_subagent_system_prompt(content)
        model = entry["model"] or "claude-sonnet-4-6"
        messages = load_session(user_id, session_id)
        messages = messages + [{"role": "user", "content": input}]

        try:
            # No tools= passed — sub-agent is stateless, prevents recursion
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=registry.descriptions(),
            )
            return response.content[0].text
        except Exception as e:
            return f"Error running agent '{agent_name}': {e}"

    return run_agent

def execute_tool(name, tool_input):
    tool = registry.get_tool(name)
    if tool is None:
        return f"Unknown tool: {name}"
    return tool["fn"](**tool_input)

def estimate_tokens(messages):
    """Rough token estimate: ~4 chars per token."""
    return sum(len(json.dumps(m)) for m in messages) // 4

def compact_session(user_id, session_id, messages):
    """Summarize old messages when context gets too long."""
    if estimate_tokens(messages) < 100_000:
        return messages

    split = len(messages) // 2
    old, recent = messages[:split], messages[split:]

    print("  [compacting session history...]")

    summary = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": (
                "Summarize this conversation concisely. Preserve:\n"
                "- Key facts about the user (name, preferences)\n"
                "- Important decisions made\n"
                "- Open tasks or TODOs\n\n"
                f"{json.dumps(old, indent=2)}"
            )
        }]
    )

    compacted = [{
        "role": "user",
        "content": f"[Previous conversation summary]\n{summary.content[0].text}"
    }] + recent

    save_session(user_id, session_id, compacted)
    return compacted

def serialize_content(content):
    serialized = []
    for block in content:
        if hasattr(block, "text"):
            serialized.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            serialized.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input
            })
    return serialized

def run_agent_turn(messages, system_prompt: str):
    """Run the agentic loop until the model stops calling tools.

    Args:
        messages: Conversation history (mutated in-place as turns are appended).
        system_prompt: Pre-built system prompt string (built once by handle_message).

    Returns:
        tuple[str, list]: (final text response, updated messages list)
    """
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=registry.descriptions(),
            messages=messages
        )

        content = serialize_content(response.content)

        if response.stop_reason == "end_turn":
            text = "".join(block.text for block in response.content if hasattr(block, "text"))
            messages.append({"role": "assistant", "content": content})
            return text, messages

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool] {block.name}({json.dumps(block.input)})")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result) or "(empty output)"
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Fallback for unexpected stop reasons (e.g. "max_tokens", "stop_sequence").
        # Return whatever text was generated rather than losing it or crashing.
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        messages.append({"role": "assistant", "content": content})
        return text, messages

def get_session_path(user_id, session_id):
    return os.path.join(SESSIONS_DIR, f"{user_id}_{session_id}.jsonl")

def load_session(user_id, session_id):
    path = get_session_path(user_id, session_id)
    messages = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    messages.append(json.loads(line))
    return messages

def save_session(user_id, session_id, messages):
    path = get_session_path(user_id, session_id)
    with open(path, "w") as f:
        for message in messages:
            f.write(json.dumps(message) + "\n")

async def handle_message(user_id: str, session_id: str, text: str):
    messages = load_session(user_id, session_id)
    messages = compact_session(user_id, session_id, messages)
    messages.append({"role": "user", "content": text})

    system_prompt = build_system_prompt()
    registry.register("run_agent", _make_tool_run_agent(user_id, session_id))
    response_text, messages = run_agent_turn(messages, system_prompt)

    save_session(user_id, session_id, messages)
    return response_text

async def main():
    user_id = input("Enter your user ID: ").strip() or "default"
    session_id = input("Enter your session ID: ").strip() or f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
    print(f"Session loaded for user '{user_id}', session '{session_id}'. Type /quit or /exit to quit. Type /new to reset the session.")

    while True:
        text = input("You: ")
        if text in ["/quit", "/exit"]:
            print("Goodbye!")
            break
        elif text == "/new":
            session_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            print(f"Session reset. New session ID: {session_id}")
            continue
        resp = await handle_message(user_id, session_id, text)
        print(f"Claude: {resp}")

if __name__ == "__main__":
    asyncio.run(main())
