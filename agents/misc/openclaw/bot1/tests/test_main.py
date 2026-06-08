import os
from datetime import datetime, timedelta

import pytest

import main


class TestParseSkillFrontmatter:

    def test_basic_frontmatter(self):
        content = "---\nname: github\ndescription: GitHub CLI.\n---\n\n# Body"
        result = main.parse_skill_frontmatter(content)
        assert result == {"name": "github", "description": "GitHub CLI."}

    def test_colon_in_value_preserved(self):
        content = "---\nname: web\ndescription: Load from https://example.com\n---"
        result = main.parse_skill_frontmatter(content)
        assert result["description"] == "Load from https://example.com"

    def test_no_frontmatter_returns_empty_dict(self):
        content = "# Just a heading\nSome content"
        result = main.parse_skill_frontmatter(content)
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = main.parse_skill_frontmatter("")
        assert result == {}

    def test_horizontal_rule_in_body_not_confused_with_frontmatter(self):
        content = "---\nname: test\n---\n# Body\n\n---\n\nMore content"
        result = main.parse_skill_frontmatter(content)
        assert result == {"name": "test"}

    def test_whitespace_stripped_from_key_and_value(self):
        content = "---\n  name :  github  \n---"
        result = main.parse_skill_frontmatter(content)
        assert result["name"] == "github"

    def test_emoji_in_value(self):
        content = "---\nname: github\nemoji: 🐙\n---"
        result = main.parse_skill_frontmatter(content)
        assert result["emoji"] == "🐙"

    def test_malformed_no_closing_delimiter(self):
        content = "---\nname: github\n"
        result = main.parse_skill_frontmatter(content)
        assert result == {}


class TestLoadContextFiles:

    def test_reads_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "SOUL.md").write_text("You are Jarvis.", encoding="utf-8")
        result = main.load_context_files()
        assert "SOUL.md" in result
        assert result["SOUL.md"] == "You are Jarvis."

    def test_missing_file_silently_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        # No files created
        result = main.load_context_files()
        assert result == {}

    def test_preserves_context_files_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        for fname in ["TOOLS.md", "AGENTS.md", "SOUL.md"]:
            (tmp_path / fname).write_text(f"# {fname}", encoding="utf-8")
        result = main.load_context_files()
        keys = list(result.keys())
        assert keys.index("AGENTS.md") < keys.index("SOUL.md")
        assert keys.index("SOUL.md") < keys.index("TOOLS.md")

    def test_unreadable_file_silently_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        soul = tmp_path / "SOUL.md"
        soul.write_text("content", encoding="utf-8")
        soul.chmod(0o000)
        try:
            result = main.load_context_files()
            assert "SOUL.md" not in result
        finally:
            soul.chmod(0o644)

    def test_reads_utf8_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "SOUL.md").write_text("Name: Jàrvis 🤖", encoding="utf-8")
        result = main.load_context_files()
        assert result["SOUL.md"] == "Name: Jàrvis 🤖"

    def test_only_context_files_loaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
        (tmp_path / "OTHER.md").write_text("other", encoding="utf-8")
        result = main.load_context_files()
        assert "OTHER.md" not in result
        assert "SOUL.md" in result


class TestLoadDailyMemory:

    def test_returns_empty_if_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        result = main.load_daily_memory()
        assert result == ""

    def test_returns_empty_if_memory_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        # No workspace/memory/ dir created
        result = main.load_daily_memory()
        assert result == ""

    def test_reads_todays_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        today = datetime.now().strftime("%Y-%m-%d")
        (memory_dir / f"{today}.md").write_text("Today content", encoding="utf-8")
        result = main.load_daily_memory()
        assert f"### Memory {today}" in result
        assert "Today content" in result

    def test_today_appears_before_yesterday(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        (memory_dir / f"{today}.md").write_text("Today", encoding="utf-8")
        (memory_dir / f"{yesterday}.md").write_text("Yesterday", encoding="utf-8")
        result = main.load_daily_memory()
        assert result.index(f"### Memory {today}") < result.index(f"### Memory {yesterday}")

    def test_missing_yesterday_returns_only_today(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        today = datetime.now().strftime("%Y-%m-%d")
        (memory_dir / f"{today}.md").write_text("Only today", encoding="utf-8")
        result = main.load_daily_memory()
        assert "Only today" in result
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        assert f"### Memory {yesterday}" not in result

    def test_entries_separated_by_double_newline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        (memory_dir / f"{today}.md").write_text("A", encoding="utf-8")
        (memory_dir / f"{yesterday}.md").write_text("B", encoding="utf-8")
        result = main.load_daily_memory()
        assert "\n\n" in result


class TestLoadSkillsIndex:

    def test_returns_empty_if_no_skills_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        result = main.load_skills_index()
        assert result == ""

    def test_returns_empty_if_no_skill_md(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "skills" / "empty-skill").mkdir(parents=True)
        result = main.load_skills_index()
        assert result == ""

    def test_returns_empty_if_only_plain_files_in_skills(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "README.md").write_text("readme", encoding="utf-8")
        result = main.load_skills_index()
        assert result == ""

    def test_basic_skill_appears_in_xml(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        skill_dir = tmp_path / "skills" / "github"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: github\ndescription: GitHub CLI.\n---\n# GitHub",
            encoding="utf-8",
        )
        result = main.load_skills_index()
        assert "<name>github</name>" in result
        assert "<description>GitHub CLI.</description>" in result
        assert "SKILL.md" in result
        assert "read_file" in result

    def test_skills_sorted_alphabetically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        for name in ["weather", "github", "calendar"]:
            d = tmp_path / "skills" / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"---\nname: {name}\n---", encoding="utf-8")
        result = main.load_skills_index()
        assert result.index("calendar") < result.index("github") < result.index("weather")

    def test_no_frontmatter_falls_back_to_dir_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# No frontmatter", encoding="utf-8")
        result = main.load_skills_index()
        assert "<name>my-skill</name>" in result
        assert "<description></description>" in result

    def test_location_path_relative_to_workspace(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        skill_dir = tmp_path / "skills" / "github"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: github\n---", encoding="utf-8")
        result = main.load_skills_index()
        assert "<location>" in result
        assert "SKILL.md" in result

    def test_no_empty_available_skills_block(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "skills").mkdir()
        result = main.load_skills_index()
        assert "<available_skills>" not in result

    def test_xml_special_chars_in_description_are_escaped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        skill_dir = tmp_path / "skills" / "myskill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Load <data> & save\n---",
            encoding="utf-8",
        )
        result = main.load_skills_index()
        assert "<data>" not in result
        assert "&lt;data&gt;" in result
        assert "&amp;" in result


class TestBuildMemoryPrompt:

    def test_header_is_memory_instructions(self):
        result = main.build_memory_prompt()
        assert result.startswith("## Memory Instructions")

    def test_no_old_header(self):
        result = main.build_memory_prompt()
        # Old header must not appear
        assert "## Memory\n" not in result

    def test_contains_save_memory_reference(self):
        result = main.build_memory_prompt()
        assert "save_memory" in result

    def test_contains_memory_search_reference(self):
        result = main.build_memory_prompt()
        assert "memory_search" in result


class TestBuildSystemPrompt:

    def test_date_section_always_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        result = main.build_system_prompt()
        assert result.startswith("## Current Date & Time")

    def test_date_is_date_only_no_time(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        result = main.build_system_prompt()
        # Should contain the day name (e.g. "Saturday") but NOT ":" (which appears in HH:MM)
        lines = result.splitlines()
        date_line = lines[2]  # third line after header and blank
        assert ":" not in date_line

    def test_no_empty_context_headers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        result = main.build_system_prompt()
        for fname in main.CONTEXT_FILES:
            assert f"## {fname}\n\n\n" not in result
            # If the file doesn't exist, the header must not appear
            assert f"## {fname}" not in result

    def test_soul_md_included_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "SOUL.md").write_text("You are Jarvis.", encoding="utf-8")
        result = main.build_system_prompt()
        assert "## SOUL.md" in result
        assert "You are Jarvis." in result

    def test_context_files_in_correct_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")
        (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
        result = main.build_system_prompt()
        assert result.index("## AGENTS.md") < result.index("## SOUL.md")

    def test_recent_memory_included_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        today = datetime.now().strftime("%Y-%m-%d")
        (memory_dir / f"{today}.md").write_text("Remember this.", encoding="utf-8")
        result = main.build_system_prompt()
        assert "## Recent Memory" in result
        assert "Remember this." in result

    def test_skills_section_included_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        skill_dir = tmp_path / "skills" / "github"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: github\ndescription: GitHub CLI.\n---", encoding="utf-8"
        )
        result = main.build_system_prompt()
        assert "## Skills" in result
        assert "github" in result

    def test_memory_instructions_always_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        result = main.build_system_prompt()
        assert "## Memory Instructions" in result

    def test_long_term_memory_dir_not_in_prompt(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        # The ./memory/ tool memory dir content must NOT appear in the system prompt
        # Only workspace/memory/ daily logs do
        result = main.build_system_prompt()
        # Just verify no section for the tool memory dir leaks in
        assert "user-preferences" not in result

    def test_agents_section_included_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setattr(main, "_agents_registry", {})
        d = tmp_path / "agents" / "my_agent"
        d.mkdir(parents=True)
        (d / "my_agent.md").write_text(
            "---\nname: my-agent\ndescription: does stuff.\n---\nbody",
            encoding="utf-8",
        )
        result = main.build_system_prompt()
        assert "## Agents" in result
        assert "my-agent" in result

    def test_agents_section_absent_when_no_agents(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setattr(main, "_agents_registry", {})
        result = main.build_system_prompt()
        assert "## Agents" not in result

    def test_agents_section_between_skills_and_memory_instructions(self, tmp_path, monkeypatch):
        """Three-way ordering: ## Skills < ## Agents < ## Memory Instructions."""
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setattr(main, "_agents_registry", {})
        # Create a skill
        skill_dir = tmp_path / "skills" / "myskill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: myskill\n---", encoding="utf-8")
        # Create an agent
        agent_dir = tmp_path / "agents" / "my_agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "my_agent.md").write_text(
            "---\nname: my-agent\ndescription: desc.\n---\nbody",
            encoding="utf-8",
        )
        result = main.build_system_prompt()
        assert "## Skills" in result
        assert "## Agents" in result
        assert "## Memory Instructions" in result
        skills_pos = result.index("## Skills")
        agents_pos = result.index("## Agents")
        memory_pos = result.index("## Memory Instructions")
        assert skills_pos < agents_pos
        assert agents_pos < memory_pos


class TestToolReadFile:

    def test_returns_error_string_for_missing_file(self, tmp_path):
        result = main.tool_read_file(str(tmp_path / "nonexistent.txt"))
        assert isinstance(result, str)
        assert "Error reading" in result

    def test_reads_utf8_with_emoji(self, tmp_path):
        path = tmp_path / "test.md"
        path.write_text("Hello 🌍", encoding="utf-8")
        result = main.tool_read_file(str(path))
        assert result == "Hello 🌍"

    def test_reads_normal_file(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("content here", encoding="utf-8")
        result = main.tool_read_file(str(path))
        assert result == "content here"

    def test_returns_error_string_not_exception(self, tmp_path):
        # Must not raise — must return a string
        result = main.tool_read_file("/definitely/does/not/exist/file.md")
        assert isinstance(result, str)


from unittest.mock import patch, MagicMock


class _FakeTextBlock:
    """Minimal text block that behaves like anthropic TextBlock for testing."""
    type = "text"

    def __init__(self, text):
        self.text = text


class TestRunAgentTurn:

    def _make_response(self, stop_reason, text="some text"):
        response = MagicMock()
        response.stop_reason = stop_reason
        response.content = [_FakeTextBlock(text)]
        return response

    def test_end_turn_returns_tuple(self):
        response = self._make_response("end_turn", "hello")
        with patch.object(main.client.messages, "create", return_value=response):
            result = main.run_agent_turn(
                [{"role": "user", "content": "hi"}], "system prompt"
            )
        assert isinstance(result, tuple)
        assert len(result) == 2
        text, messages = result
        assert text == "hello"
        assert isinstance(messages, list)

    def test_max_tokens_returns_tuple_not_none(self):
        """Bug fix: previously returned None on unexpected stop_reason."""
        response = self._make_response("max_tokens", "partial")
        with patch.object(main.client.messages, "create", return_value=response):
            result = main.run_agent_turn(
                [{"role": "user", "content": "hi"}], "system prompt"
            )
        assert result is not None
        text, messages = result
        assert isinstance(text, str)
        assert isinstance(messages, list)

    def test_stop_sequence_returns_tuple_not_none(self):
        response = self._make_response("stop_sequence", "stopped")
        with patch.object(main.client.messages, "create", return_value=response):
            result = main.run_agent_turn(
                [{"role": "user", "content": "hi"}], "system prompt"
            )
        assert result is not None
        assert isinstance(result, tuple)


class TestExtractFrontmatterBody:

    def test_returns_body_when_frontmatter_present(self):
        content = "---\nname: foo\n---\n\n## Role\nYou are an agent."
        result = main.extract_frontmatter_body(content)
        assert result == "## Role\nYou are an agent."

    def test_returns_full_content_when_no_frontmatter(self):
        content = "## Role\nYou are an agent."
        result = main.extract_frontmatter_body(content)
        assert result == "## Role\nYou are an agent."

    def test_body_with_horizontal_rule_not_split(self):
        content = "---\nname: foo\n---\n\n## Section\n\n---\n\nMore content"
        result = main.extract_frontmatter_body(content)
        assert "---" in result
        assert "More content" in result

    def test_strips_leading_trailing_whitespace_from_body(self):
        content = "---\nname: foo\n---\n\n\n  body content  \n\n"
        result = main.extract_frontmatter_body(content)
        assert result == "body content"

    def test_mid_line_dashes_not_treated_as_delimiter(self):
        content = "---\nname: foo\n---\nSome --- inline dashes"
        result = main.extract_frontmatter_body(content)
        assert result == "Some --- inline dashes"


class TestLoadAgentsIndex:

    @pytest.fixture(autouse=True)
    def reset_registry(self, monkeypatch):
        monkeypatch.setattr(main, "_agents_registry", {})

    def test_returns_empty_if_no_agents_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        assert main.load_agents_index() == ""

    def test_returns_empty_if_no_agent_subdirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "agents").mkdir()
        assert main.load_agents_index() == ""

    def test_returns_empty_if_agent_dir_has_no_md_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "agents" / "my_agent").mkdir(parents=True)
        assert main.load_agents_index() == ""

    def test_basic_agent_appears_in_xml(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        d = tmp_path / "agents" / "summarizer_agent"
        d.mkdir(parents=True)
        (d / "summarizer_agent.md").write_text(
            "---\nname: summarizer-agent\ndescription: Summarizes text.\n---\n\n## Role\nYou summarize.",
            encoding="utf-8",
        )
        result = main.load_agents_index()
        assert "<name>summarizer-agent</name>" in result
        assert "<description>Summarizes text.</description>" in result
        assert "<location>" in result
        assert "<directory>" in result
        assert "summarizer_agent.md" in result

    def test_no_model_field_stores_none_in_registry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        d = tmp_path / "agents" / "my_agent"
        d.mkdir(parents=True)
        (d / "my_agent.md").write_text(
            "---\nname: my-agent\ndescription: Does stuff.\n---\n\nbody",
            encoding="utf-8",
        )
        main.load_agents_index()
        assert main._agents_registry["my-agent"]["model"] is None

    def test_agent_without_name_skipped_silently(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        d = tmp_path / "agents" / "unnamed"
        d.mkdir(parents=True)
        (d / "unnamed.md").write_text(
            "---\ndescription: No name here.\n---\n\nbody",
            encoding="utf-8",
        )
        result = main.load_agents_index()
        assert result == ""
        assert main._agents_registry == {}

    def test_xml_special_chars_escaped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        d = tmp_path / "agents" / "my_agent"
        d.mkdir(parents=True)
        (d / "my_agent.md").write_text(
            "---\nname: my-agent\ndescription: Load <data> & save\n---\n\nbody",
            encoding="utf-8",
        )
        result = main.load_agents_index()
        assert "<data>" not in result
        assert "&lt;data&gt;" in result
        assert "&amp;" in result

    def test_multiple_agents_sorted_alphabetically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        for name in ["zebra_agent", "alpha_agent", "middle_agent"]:
            d = tmp_path / "agents" / name
            d.mkdir(parents=True)
            (d / f"{name}.md").write_text(
                f"---\nname: {name.replace('_', '-')}\ndescription: desc\n---\nbody",
                encoding="utf-8",
            )
        result = main.load_agents_index()
        assert result.index("alpha-agent") < result.index("middle-agent") < result.index("zebra-agent")

    def test_registry_populated_correctly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        d = tmp_path / "agents" / "my_agent"
        d.mkdir(parents=True)
        (d / "my_agent.md").write_text(
            "---\nname: my-agent\ndescription: Does stuff.\nmodel: claude-sonnet-4-6\n---\nbody",
            encoding="utf-8",
        )
        main.load_agents_index()
        entry = main._agents_registry["my-agent"]
        assert "file_path" in entry
        assert entry["model"] == "claude-sonnet-4-6"

    def test_stale_entry_absent_after_second_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        d = tmp_path / "agents" / "my_agent"
        d.mkdir(parents=True)
        md = d / "my_agent.md"
        md.write_text(
            "---\nname: my-agent\ndescription: desc.\n---\nbody",
            encoding="utf-8",
        )
        main.load_agents_index()
        assert "my-agent" in main._agents_registry

        # Remove the agent file and call again
        md.unlink()
        d.rmdir()
        main.load_agents_index()
        assert "my-agent" not in main._agents_registry


class TestToolRunAgent:

    @pytest.fixture(autouse=True)
    def reset_registry(self, monkeypatch):
        monkeypatch.setattr(main, "_agents_registry", {})

    def _make_fn(self, tmp_path):
        """Return a run_agent closure bound to an empty session."""
        return main._make_tool_run_agent(str(tmp_path), "test-session")

    def test_agent_not_found_returns_error_string(self, tmp_path):
        fn = self._make_fn(tmp_path)
        result = fn("nonexistent-agent", "hello")
        assert "Error" in result
        assert "nonexistent-agent" in result

    def test_agent_not_found_does_not_raise(self, tmp_path):
        fn = self._make_fn(tmp_path)
        result = fn("nonexistent-agent", "hello")
        assert isinstance(result, str)

    def test_unreadable_file_returns_error_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, "_agents_registry", {
            "my-agent": {"file_path": "/nonexistent/path.md", "model": None}
        })
        fn = self._make_fn(tmp_path)
        result = fn("my-agent", "hello")
        assert "Error" in result
        assert "my-agent" in result

    def test_api_exception_returns_error_string(self, tmp_path, monkeypatch):
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("---\nname: my-agent\n---\n\nYou are an agent.", encoding="utf-8")
        monkeypatch.setattr(main, "_agents_registry", {
            "my-agent": {"file_path": str(agent_file), "model": None}
        })
        with patch.object(main.client.messages, "create", side_effect=Exception("API down")):
            fn = self._make_fn(tmp_path)
            result = fn("my-agent", "hello")
        assert "Error" in result
        assert "my-agent" in result

    def test_successful_call_returns_response_text(self, tmp_path, monkeypatch):
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("---\nname: my-agent\n---\n\nYou are an agent.", encoding="utf-8")
        monkeypatch.setattr(main, "_agents_registry", {
            "my-agent": {"file_path": str(agent_file), "model": "claude-sonnet-4-6"}
        })
        mock_response = MagicMock()
        mock_response.content = [_FakeTextBlock("summarized result")]
        with patch.object(main.client.messages, "create", return_value=mock_response):
            fn = self._make_fn(tmp_path)
            result = fn("my-agent", "summarize this")
        assert result == "summarized result"

    def test_empty_session_appends_only_input(self, tmp_path, monkeypatch):
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("---\nname: my-agent\n---\n\nYou are an agent.", encoding="utf-8")
        monkeypatch.setattr(main, "_agents_registry", {
            "my-agent": {"file_path": str(agent_file), "model": None}
        })
        mock_response = MagicMock()
        mock_response.content = [_FakeTextBlock("ok")]
        captured = {}
        def capture_create(**kwargs):
            captured["messages"] = kwargs["messages"]
            return mock_response
        with patch.object(main.client.messages, "create", side_effect=capture_create):
            fn = self._make_fn(tmp_path)
            fn("my-agent", "my input")
        assert captured["messages"] == [{"role": "user", "content": "my input"}]

    def test_absent_model_uses_default(self, tmp_path, monkeypatch):
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("---\nname: my-agent\n---\n\nYou are an agent.", encoding="utf-8")
        monkeypatch.setattr(main, "_agents_registry", {
            "my-agent": {"file_path": str(agent_file), "model": None}
        })
        mock_response = MagicMock()
        mock_response.content = [_FakeTextBlock("ok")]
        captured = {}
        def capture_create(**kwargs):
            captured["model"] = kwargs["model"]
            return mock_response
        with patch.object(main.client.messages, "create", side_effect=capture_create):
            fn = self._make_fn(tmp_path)
            fn("my-agent", "hello")
        assert captured["model"] == "claude-sonnet-4-6"

    def test_no_tools_passed_to_sub_agent(self, tmp_path, monkeypatch):
        """Sub-agent must receive no tools — prevents recursion."""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text("---\nname: my-agent\n---\n\nYou are an agent.", encoding="utf-8")
        monkeypatch.setattr(main, "_agents_registry", {
            "my-agent": {"file_path": str(agent_file), "model": "claude-sonnet-4-6"}
        })
        mock_response = MagicMock()
        mock_response.content = [_FakeTextBlock("ok")]
        captured = {}
        def capture_create(**kwargs):
            captured["kwargs"] = kwargs
            return mock_response
        with patch.object(main.client.messages, "create", side_effect=capture_create):
            fn = self._make_fn(tmp_path)
            fn("my-agent", "hello")
        assert "tools" not in captured["kwargs"]

    def test_handle_message_registers_run_agent_sync(self, tmp_path, monkeypatch):
        """After handle_message is called, run_agent must be registered."""
        import asyncio
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setattr(main, "_agents_registry", {})
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [_FakeTextBlock("hello")]
        with patch.object(main.client.messages, "create", return_value=mock_response):
            asyncio.run(main.handle_message("u1", "s1", "hi"))
        assert main.registry.get_tool("run_agent") is not None
