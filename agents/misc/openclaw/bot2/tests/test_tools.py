import json

import pytest

from tools import BotToolkit, SAFE_COMMANDS


@pytest.fixture
def toolkit(tmp_path):
    return BotToolkit(approvals_file=str(tmp_path / "approvals.json"))


class TestRunCommand:

    def test_safe_command_runs_without_approval(self, toolkit):
        result = toolkit.run_command("echo hello")
        assert "hello" in result

    def test_safe_command_set_includes_expected_commands(self):
        assert "ls" in SAFE_COMMANDS
        assert "cat" in SAFE_COMMANDS
        assert "echo" in SAFE_COMMANDS

    def test_denied_command_returns_denied_message(self, toolkit, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        result = toolkit.run_command("rm -rf /tmp/nonexistent")
        assert "denied" in result.lower()

    def test_approved_command_runs(self, toolkit, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        result = toolkit.run_command("date")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_timeout_returns_error_string(self, toolkit, monkeypatch):
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: (_ for _ in ()).throw(
                __import__("subprocess").TimeoutExpired(cmd="sleep 999", timeout=30)
            ),
        )
        result = toolkit.run_command("echo hello")
        assert "timed out" in result.lower()
        assert isinstance(result, str)

    def test_previously_approved_command_skips_prompt(self, tmp_path):
        approvals = {"allowed": ["pwd"], "denied": []}
        approvals_file = str(tmp_path / "approvals.json")
        with open(approvals_file, "w") as f:
            json.dump(approvals, f)
        tk = BotToolkit(approvals_file=approvals_file)
        result = tk.run_command("pwd")
        assert isinstance(result, str)
        assert len(result) > 0


class TestReadFile:

    def test_reads_existing_file(self, toolkit, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("hello", encoding="utf-8")
        result = toolkit.read_file(str(path))
        assert result == "hello"

    def test_missing_file_returns_error_string(self, toolkit, tmp_path):
        result = toolkit.read_file(str(tmp_path / "nonexistent.txt"))
        assert "Error reading" in result
        assert isinstance(result, str)

    def test_reads_utf8_with_emoji(self, toolkit, tmp_path):
        path = tmp_path / "test.md"
        path.write_text("Hello 🌍", encoding="utf-8")
        result = toolkit.read_file(str(path))
        assert result == "Hello 🌍"


class TestWriteFile:

    def test_writes_content_to_file(self, toolkit, tmp_path):
        path = tmp_path / "out.txt"
        result = toolkit.write_file(str(path), "content here")
        assert path.read_text(encoding="utf-8") == "content here"
        assert "Wrote" in result

    def test_overwrites_existing_file(self, toolkit, tmp_path):
        path = tmp_path / "out.txt"
        path.write_text("old", encoding="utf-8")
        toolkit.write_file(str(path), "new")
        assert path.read_text(encoding="utf-8") == "new"


class TestWebSearch:

    def test_returns_string(self, toolkit):
        result = toolkit.web_search("test query")
        assert isinstance(result, str)
        assert len(result) > 0


class TestToolkitRegistration:

    def test_all_four_tools_registered(self, toolkit):
        from agno.tools import Toolkit
        assert isinstance(toolkit, Toolkit)
        # toolkit.functions is an OrderedDict mapping tool name -> Function object
        tool_names = list(toolkit.functions.keys())
        assert "run_command" in tool_names
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "web_search" in tool_names

    def test_memory_tools_not_registered(self, toolkit):
        tool_names = list(toolkit.functions.keys())
        assert "save_memory" not in tool_names
        assert "memory_search" not in tool_names
