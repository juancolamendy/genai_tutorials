import asyncio
import inspect
import json
import os
import re
import subprocess
from datetime import datetime
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

def load_soul(path: str = "workspace/SOUL.md") -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return ""

def build_memory_prompt() -> str:
    return """## Memory
You have a long-term memory system.
- Use save_memory to store important information (user preferences, key facts, project details).
- Use memory_search at the start of conversations to recall context from previous sessions.
Memory files are stored in ./memory/ as markdown files."""

def build_system_prompt() -> str:
    return load_soul() + "\n\n" + build_memory_prompt()

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
    with open(path, "r") as f:
        return f.read()

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

def run_agent_turn(messages):
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=build_system_prompt(),
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
                        "content": str(result)
                    })

            messages.append({"role": "user", "content": tool_results})

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

    response_text, messages = run_agent_turn(messages)

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

asyncio.run(main())
