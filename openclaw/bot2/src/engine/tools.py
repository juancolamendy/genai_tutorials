"""BotToolkit: shell/file tools for the Agno-based bot2.

Provides run_command, read_file, write_file, and web_search.
Does NOT include save_memory or memory_search — those are delegated
to MemoryTools.
"""

import json
import os
import re
import subprocess

from agno.tools import Toolkit

SAFE_COMMANDS = {"ls", "cat", "head", "tail", "wc", "date", "whoami", "echo"}
DANGEROUS_PATTERNS = [r"\brm\b", r"\bsudo\b", r"\bchmod\b", r"\bcurl.*\|.*sh"]
_DEFAULT_APPROVALS_FILE = "./workspace/exec-approvals.json"


class BotToolkit(Toolkit):
    """Tools for the bot: shell commands, file I/O, web search.

    Does NOT include save_memory or memory_search — those are in MemoryTools.
    """

    def __init__(self, approvals_file: str = _DEFAULT_APPROVALS_FILE):
        self.approvals_file = approvals_file
        super().__init__(
            name="bot_tools",
            tools=[self.run_command, self.read_file, self.write_file, self.web_search],
        )

    def _load_approvals(self) -> dict:
        """Load the approvals JSON file, returning empty structure if absent."""
        if os.path.exists(self.approvals_file):
            with open(self.approvals_file) as f:
                return json.load(f)
        return {"allowed": [], "denied": []}

    def _save_approval(self, command: str, approved: bool) -> None:
        """Persist an approval decision for a command."""
        approvals = self._load_approvals()
        key = "allowed" if approved else "denied"
        if command not in approvals[key]:
            approvals[key].append(command)
        os.makedirs(os.path.dirname(self.approvals_file) or ".", exist_ok=True)
        with open(self.approvals_file, "w") as f:
            json.dump(approvals, f, indent=2)

    def _check_command_safety(self, command: str) -> str:
        """Return 'safe', 'approved', or 'needs_approval' for the given command.

        Args:
            command: The shell command string to evaluate.

        Returns:
            'safe' if the base command is in SAFE_COMMANDS, 'approved' if the
            full command has been previously allowed, or 'needs_approval' otherwise.
        """
        base_cmd = command.strip().split()[0] if command.strip() else ""
        if base_cmd in SAFE_COMMANDS:
            return "safe"
        approvals = self._load_approvals()
        if command in approvals["allowed"]:
            return "approved"
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return "needs_approval"
        return "needs_approval"

    def run_command(self, command: str) -> str:
        """Run a shell command on the user's computer.

        Args:
            command: The shell command to run.

        Returns:
            The stdout and stderr of the command.
        """
        safety = self._check_command_safety(command)
        if safety == "needs_approval":
            print(f"  [approval needed] {command}")
            answer = input("  Allow this command? (y/n): ").strip().lower()
            approved = answer == "y"
            self._save_approval(command, approved)
            if not approved:
                return "Command denied by user."

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return f"Command timed out after 30 seconds: {command}"

    def read_file(self, path: str) -> str:
        """Read a file from the filesystem.

        Args:
            path: Path to the file.

        Returns:
            The file contents, or an error string if unreadable.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading {path}: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file.

        Args:
            path: Path to the file.
            content: Content to write.

        Returns:
            Confirmation message, or an error string if write failed.
        """
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote to {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"

    def web_search(self, query: str) -> str:
        """Search the web for information.

        Args:
            query: Search query.

        Returns:
            Search results string.
        """
        return f"Search results for: {query}"
