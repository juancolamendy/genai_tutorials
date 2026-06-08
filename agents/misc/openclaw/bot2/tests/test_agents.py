"""Tests for agents module.

Tests for extract_frontmatter_body, AgentRegistry, and AgentsToolkit.
"""

import shutil
from unittest.mock import MagicMock, patch

from agno.tools import Toolkit
import engine.agents as agents_module

CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
CLAUDE_SONNET = "claude-sonnet-4-6"


class TestExtractFrontmatterBody:

    def test_returns_body_when_frontmatter_present(self):
        content = "---\nname: test\n---\n\nThis is the body."
        assert agents_module.extract_frontmatter_body(content) == "This is the body."

    def test_returns_full_content_when_no_frontmatter(self):
        content = "No frontmatter here."
        assert agents_module.extract_frontmatter_body(content) == "No frontmatter here."

    def test_horizontal_rule_in_body_not_split(self):
        content = "---\nname: test\n---\n\nBody.\n\n---\n\nMore."
        result = agents_module.extract_frontmatter_body(content)
        assert result == "Body.\n\n---\n\nMore."

    def test_empty_content_returns_empty_string(self):
        assert agents_module.extract_frontmatter_body("") == ""

    def test_single_delimiter_returns_full_content_stripped(self):
        content = "---\nname: test\nno closing delimiter"
        assert agents_module.extract_frontmatter_body(content) == content.strip()


def _make_agent_file(agents_dir, dir_name, frontmatter: dict, body: str = "Do the task."):
    """Helper: create workspace/agents/<dir_name>/<dir_name>.md."""
    agent_dir = agents_dir / dir_name
    agent_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for k, v in frontmatter.items():
        lines.append(f"{k}: {v}")
    lines += ["---", "", body]
    (agent_dir / f"{dir_name}.md").write_text("\n".join(lines), encoding="utf-8")
    return agent_dir


class TestAgentRegistryMissingDir:

    def test_returns_empty_when_agents_dir_missing(self, tmp_path):
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        assert registry.get_agents_index() == ""

    def test_returns_empty_when_agents_dir_is_empty(self, tmp_path):
        (tmp_path / "agents").mkdir()
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        assert registry.get_agents_index() == ""

    def test_non_directory_entries_are_skipped(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "not_a_dir.md").write_text("content", encoding="utf-8")
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        assert registry.get_agents_index() == ""

    def test_subdir_with_no_matching_md_file_is_skipped(self, tmp_path):
        agent_dir = tmp_path / "agents" / "my_agent"
        agent_dir.mkdir(parents=True)
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        assert registry.get_agents_index() == ""


class TestAgentRegistryValid:

    def test_valid_agent_appears_in_xml(self, tmp_path):
        _make_agent_file(
            tmp_path / "agents", "summarizer_agent",
            {"name": "summarizer-agent", "description": "Summarize text."}
        )
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        result = registry.get_agents_index()
        assert "<name>summarizer-agent</name>" in result
        assert "<description>Summarize text.</description>" in result
        assert "<location>" not in result
        assert "<directory>" not in result

    def test_instructional_preamble_present(self, tmp_path):
        _make_agent_file(
            tmp_path / "agents", "summarizer_agent",
            {"name": "summarizer-agent", "description": "Summarize text."}
        )
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        result = registry.get_agents_index()
        assert "run_agent" in result
        assert result.index("run_agent") < result.index("<available_agents>")

    def test_registry_populated_with_content_and_model(self, tmp_path):
        _make_agent_file(
            tmp_path / "agents", "summarizer_agent",
            {"name": "summarizer-agent", "description": "Summarize.", "model": "claude-haiku-4-5-20251001"}
        )
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        registry.load_agents()
        entry = registry._registry["summarizer-agent"]
        assert "summarizer-agent" in entry["content"]
        assert entry["model"] == "claude-haiku-4-5-20251001"

    def test_model_absent_stored_as_none(self, tmp_path):
        _make_agent_file(
            tmp_path / "agents", "summarizer_agent",
            {"name": "summarizer-agent", "description": "Summarize."}
        )
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        registry.load_agents()
        assert registry._registry["summarizer-agent"]["model"] is None

    def test_agent_without_name_field_is_skipped(self, tmp_path):
        _make_agent_file(
            tmp_path / "agents", "nameless_agent",
            {"description": "Has no name field."}
        )
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        assert registry.get_agents_index() == ""
        assert registry._registry == {}


class TestAgentRegistryEdgeCases:

    def test_xml_special_chars_in_description_are_html_escaped(self, tmp_path):
        _make_agent_file(
            tmp_path / "agents", "my_agent",
            {"name": "my-agent", "description": "Load <data> & save"}
        )
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        result = registry.get_agents_index()
        assert "<data>" not in result
        assert "&lt;data&gt;" in result
        assert "&amp;" in result

    def test_multiple_agents_sorted_alphabetically(self, tmp_path):
        for dir_name, name in [
            ("weather_agent", "weather-agent"),
            ("alpha_agent", "alpha-agent"),
            ("beta_agent", "beta-agent"),
        ]:
            _make_agent_file(tmp_path / "agents", dir_name, {"name": name, "description": "desc"})
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        result = registry.get_agents_index()
        assert result.index("alpha-agent") < result.index("beta-agent") < result.index("weather-agent")

    def test_duplicate_name_last_alphabetical_dir_wins(self, tmp_path):
        _make_agent_file(tmp_path / "agents", "a_agent", {"name": "dup", "description": "first"})
        _make_agent_file(tmp_path / "agents", "z_agent", {"name": "dup", "description": "second"})
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        registry.load_agents()
        entry = registry._registry["dup"]
        assert "second" in entry["content"]  # z_agent's description wins

    def test_new_instance_rescans_after_agent_removed(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agent_dir = _make_agent_file(agents_dir, "my_agent", {"name": "my-agent", "description": "desc"})

        registry1 = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        registry1.load_agents()
        assert "my-agent" in registry1._registry

        shutil.rmtree(str(agent_dir))
        registry2 = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        registry2.load_agents()
        assert "my-agent" not in registry2._registry

    def test_cached_result_returned_on_second_call(self, tmp_path):
        _make_agent_file(tmp_path / "agents", "my_agent", {"name": "my-agent", "description": "desc"})
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        list1 = registry.load_agents()
        list2 = registry.load_agents()
        assert list1 is list2  # same object — cache hit

    def test_registry_matches_expected_keys_and_structure(self, tmp_path):
        _make_agent_file(
            tmp_path / "agents", "my_agent",
            {"name": "my-agent", "description": "desc", "model": "claude-opus-4-6"}
        )
        registry = agents_module.AgentRegistry(workspace_dir=str(tmp_path))
        registry.load_agents()
        assert set(registry._registry.keys()) == {"my-agent"}
        assert set(registry._registry["my-agent"].keys()) == {"content", "model"}
        assert registry._registry["my-agent"]["model"] == "claude-opus-4-6"


class TestAgentsToolkitRegistration:

    def test_is_agno_toolkit_instance(self):
        assert isinstance(agents_module.AgentsToolkit(), Toolkit)

    def test_run_agent_registered_in_functions(self):
        toolkit = agents_module.AgentsToolkit()
        assert "run_agent" in toolkit.functions

    def test_no_extra_tools_registered(self):
        toolkit = agents_module.AgentsToolkit()
        assert list(toolkit.functions.keys()) == ["run_agent"]


def _make_fake_ctx(session_state=None):
    """Create a minimal fake RunContext — avoids dependency on RunContext constructor API."""
    ctx = MagicMock()
    ctx.session_state = session_state
    return ctx


def _make_agent_registry(content: str, model=None) -> agents_module.AgentRegistry:
    """Return a pre-loaded AgentRegistry with a single cached agent entry."""
    registry = agents_module.AgentRegistry()
    registry._registry = {"my-agent": {"content": content, "model": model}}
    registry._agents_list = [{"name": "my-agent", "description": ""}]
    return registry


_AGENT_CONTENT = "---\nname: my-agent\n---\n\nYou are a specialist."


class TestRunAgentErrors:

    def test_agent_not_found_returns_error_string(self):
        registry = agents_module.AgentRegistry()
        registry._registry = {}
        registry._agents_list = []
        toolkit = agents_module.AgentsToolkit(agent_registry=registry)
        result = toolkit.run_agent(_make_fake_ctx(), "nonexistent-agent", "do something")
        assert "nonexistent-agent" in result
        assert result.startswith("Error")

    def test_agent_not_found_does_not_raise(self):
        registry = agents_module.AgentRegistry()
        registry._registry = {}
        registry._agents_list = []
        toolkit = agents_module.AgentsToolkit(agent_registry=registry)
        result = toolkit.run_agent(_make_fake_ctx(), "ghost", "task")
        assert isinstance(result, str)

    def test_api_exception_returns_error_string(self):
        registry = _make_agent_registry(_AGENT_CONTENT)
        toolkit = agents_module.AgentsToolkit(agent_registry=registry)
        with patch("engine.agents.Agent") as MockAgent:
            MockAgent.return_value.run.side_effect = RuntimeError("API down")
            result = toolkit.run_agent(_make_fake_ctx(), "my-agent", "task")
        assert "my-agent" in result
        assert result.startswith("Error")

    def test_none_response_content_returns_empty_string(self):
        registry = _make_agent_registry(_AGENT_CONTENT)
        toolkit = agents_module.AgentsToolkit(agent_registry=registry)
        with patch("engine.agents.Agent") as MockAgent:
            MockAgent.return_value.run.return_value = MagicMock(content=None)
            result = toolkit.run_agent(_make_fake_ctx(), "my-agent", "task")
        assert result == ""


class TestRunAgentHappyPath:

    def _setup(self, model=None):
        registry = _make_agent_registry(_AGENT_CONTENT, model=model)
        return agents_module.AgentsToolkit(agent_registry=registry, default_model_key=CLAUDE_HAIKU)

    def test_successful_call_returns_response_content(self):
        toolkit = self._setup()
        with patch("engine.agents.Agent") as MockAgent:
            MockAgent.return_value.run.return_value = MagicMock(content="summary result")
            result = toolkit.run_agent(_make_fake_ctx(), "my-agent", "summarize this")
        assert result == "summary result"

    def test_session_state_passed_to_sub_agent(self):
        toolkit = self._setup()
        state = {"user": "alice", "pref": "brief"}
        ctx = _make_fake_ctx(session_state=state)
        with patch("engine.agents.Agent") as MockAgent:
            MockAgent.return_value.run.return_value = MagicMock(content="ok")
            toolkit.run_agent(ctx, "my-agent", "task")
            call_kwargs = MockAgent.return_value.run.call_args
        assert call_kwargs.kwargs["session_state"] == state

    def test_none_session_state_passed_as_none(self):
        toolkit = self._setup()
        ctx = _make_fake_ctx(session_state=None)
        with patch("engine.agents.Agent") as MockAgent:
            MockAgent.return_value.run.return_value = MagicMock(content="ok")
            toolkit.run_agent(ctx, "my-agent", "task")
            call_kwargs = MockAgent.return_value.run.call_args
        assert "session_state" in call_kwargs.kwargs
        assert call_kwargs.kwargs["session_state"] is None

    def test_model_absent_defaults_to_haiku(self):
        toolkit = self._setup(model=None)
        with patch("engine.agents.Agent") as MockAgent, patch("engine.agents.load_model") as MockLoadModel:
            MockAgent.return_value.run.return_value = MagicMock(content="ok")
            MockLoadModel.return_value = MagicMock()
            toolkit.run_agent(_make_fake_ctx(), "my-agent", "task")
            MockLoadModel.assert_called_once_with(CLAUDE_HAIKU)

    def test_explicit_model_used_when_present(self):
        toolkit = self._setup(model="claude-haiku-4-5-20251001")
        with patch("engine.agents.Agent") as MockAgent, patch("engine.agents.load_model") as MockLoadModel:
            MockAgent.return_value.run.return_value = MagicMock(content="ok")
            MockLoadModel.return_value = MagicMock()
            toolkit.run_agent(_make_fake_ctx(), "my-agent", "task")
            MockLoadModel.assert_called_once_with("claude-haiku-4-5-20251001")

    def test_sub_agent_instantiated_without_db(self):
        toolkit = self._setup()
        with patch("engine.agents.Agent") as MockAgent:
            MockAgent.return_value.run.return_value = MagicMock(content="ok")
            toolkit.run_agent(_make_fake_ctx(), "my-agent", "task")
            call_kwargs = MockAgent.call_args.kwargs
        assert "db" not in call_kwargs
