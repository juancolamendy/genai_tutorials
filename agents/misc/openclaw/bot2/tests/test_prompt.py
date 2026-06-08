from datetime import datetime, timedelta

import agents as agents_module
import prompt  # bot2/prompt.py
from skills import SkillRegistry


class TestParseSkillFrontmatter:

    def test_basic_frontmatter(self):
        content = "---\nname: github\ndescription: GitHub CLI.\n---\n\n# Body"
        result = prompt.parse_skill_frontmatter(content)
        assert result == {"name": "github", "description": "GitHub CLI."}

    def test_colon_in_value_preserved(self):
        content = "---\nname: web\ndescription: Load from https://example.com\n---"
        result = prompt.parse_skill_frontmatter(content)
        assert result["description"] == "Load from https://example.com"

    def test_no_frontmatter_returns_empty_dict(self):
        content = "# Just a heading\nSome content"
        result = prompt.parse_skill_frontmatter(content)
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = prompt.parse_skill_frontmatter("")
        assert result == {}

    def test_horizontal_rule_in_body_not_confused_with_frontmatter(self):
        content = "---\nname: test\n---\n# Body\n\n---\n\nMore content"
        result = prompt.parse_skill_frontmatter(content)
        assert result == {"name": "test"}

    def test_whitespace_stripped_from_key_and_value(self):
        content = "---\n  name :  github  \n---"
        result = prompt.parse_skill_frontmatter(content)
        assert result["name"] == "github"

    def test_emoji_in_value(self):
        content = "---\nname: github\nemoji: 🐙\n---"
        result = prompt.parse_skill_frontmatter(content)
        assert result["emoji"] == "🐙"

    def test_malformed_no_closing_delimiter(self):
        content = "---\nname: github\n"
        result = prompt.parse_skill_frontmatter(content)
        assert result == {}


class TestLoadContextFiles:

    def test_reads_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "SOUL.md").write_text("You are Jarvis.", encoding="utf-8")
        result = prompt.load_context_files()
        assert "SOUL.md" in result
        assert result["SOUL.md"] == "You are Jarvis."

    def test_missing_file_silently_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        result = prompt.load_context_files()
        assert result == {}

    def test_preserves_context_files_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        for fname in ["TOOLS.md", "AGENTS.md", "SOUL.md"]:
            (tmp_path / fname).write_text(f"# {fname}", encoding="utf-8")
        result = prompt.load_context_files()
        keys = list(result.keys())
        assert keys.index("AGENTS.md") < keys.index("SOUL.md")
        assert keys.index("SOUL.md") < keys.index("TOOLS.md")

    def test_unreadable_file_silently_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        soul = tmp_path / "SOUL.md"
        soul.write_text("content", encoding="utf-8")
        soul.chmod(0o000)
        try:
            result = prompt.load_context_files()
            assert "SOUL.md" not in result
        finally:
            soul.chmod(0o644)

    def test_reads_utf8_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "SOUL.md").write_text("Name: Jàrvis 🤖", encoding="utf-8")
        result = prompt.load_context_files()
        assert result["SOUL.md"] == "Name: Jàrvis 🤖"

    def test_only_context_files_loaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
        (tmp_path / "OTHER.md").write_text("other", encoding="utf-8")
        result = prompt.load_context_files()
        assert "OTHER.md" not in result
        assert "SOUL.md" in result


class TestLoadDailyMemory:

    def test_returns_empty_if_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        result = prompt.load_daily_memory()
        assert result == ""

    def test_reads_todays_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        today = datetime.now().strftime("%Y-%m-%d")
        (memory_dir / f"{today}.md").write_text("Today content", encoding="utf-8")
        result = prompt.load_daily_memory()
        assert f"### Memory {today}" in result
        assert "Today content" in result

    def test_today_appears_before_yesterday(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        (memory_dir / f"{today}.md").write_text("Today", encoding="utf-8")
        (memory_dir / f"{yesterday}.md").write_text("Yesterday", encoding="utf-8")
        result = prompt.load_daily_memory()
        assert result.index(f"### Memory {today}") < result.index(f"### Memory {yesterday}")

    def test_missing_yesterday_returns_only_today(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        today = datetime.now().strftime("%Y-%m-%d")
        (memory_dir / f"{today}.md").write_text("Only today", encoding="utf-8")
        result = prompt.load_daily_memory()
        assert "Only today" in result
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        assert f"### Memory {yesterday}" not in result


class TestSkillRegistry:

    def test_returns_empty_if_no_skills_dir(self, tmp_path):
        result = SkillRegistry(workspace_dir=str(tmp_path)).get_skills_index()
        assert result == ""

    def test_basic_skill_appears_in_xml(self, tmp_path):
        skill_dir = tmp_path / "skills" / "github"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: github\ndescription: GitHub CLI.\n---\n# GitHub",
            encoding="utf-8",
        )
        result = SkillRegistry(workspace_dir=str(tmp_path)).get_skills_index()
        assert "<name>github</name>" in result
        assert "<description>GitHub CLI.</description>" in result
        assert "SKILL.md" in result
        assert "read_file" in result
        assert "<directory>" in result
        assert str(skill_dir) in result

    def test_skill_directory_path_instruction_present(self, tmp_path):
        skill_dir = tmp_path / "skills" / "weather"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: weather\n---", encoding="utf-8")
        result = SkillRegistry(workspace_dir=str(tmp_path)).get_skills_index()
        assert "relative to that" in result
        assert "run_command" in result

    def test_skills_sorted_alphabetically(self, tmp_path):
        for name in ["weather", "github", "calendar"]:
            d = tmp_path / "skills" / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"---\nname: {name}\n---", encoding="utf-8")
        result = SkillRegistry(workspace_dir=str(tmp_path)).get_skills_index()
        assert result.index("calendar") < result.index("github") < result.index("weather")

    def test_xml_special_chars_escaped(self, tmp_path):
        skill_dir = tmp_path / "skills" / "myskill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: Load <data> & save\n---",
            encoding="utf-8",
        )
        result = SkillRegistry(workspace_dir=str(tmp_path)).get_skills_index()
        assert "<data>" not in result
        assert "&lt;data&gt;" in result
        assert "&amp;" in result


def _make_registries(tmp_path):
    """Return (skill_registry, agent_registry) both pointing at tmp_path."""
    return (
        SkillRegistry(workspace_dir=str(tmp_path)),
        agents_module.AgentRegistry(workspace_dir=str(tmp_path)),
    )


class TestBuildSystemPrompt:

    def test_date_section_always_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        sr, ar = _make_registries(tmp_path)
        result = prompt.build_system_prompt(sr, ar)
        assert result.startswith("## Current Date & Time")

    def test_soul_md_included_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "SOUL.md").write_text("You are Jarvis.", encoding="utf-8")
        sr, ar = _make_registries(tmp_path)
        result = prompt.build_system_prompt(sr, ar)
        assert "## SOUL.md" in result
        assert "You are Jarvis." in result

    def test_skills_section_included_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        skill_dir = tmp_path / "skills" / "github"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: github\ndescription: GitHub CLI.\n---", encoding="utf-8"
        )
        sr, ar = _make_registries(tmp_path)
        result = prompt.build_system_prompt(sr, ar)
        assert "## Skills" in result
        assert "github" in result


def _make_agent_file(tmp_path, name="test-agent"):
    agents_dir = tmp_path / "agents" / "test_agent"
    agents_dir.mkdir(parents=True)
    (agents_dir / "test_agent.md").write_text(
        f"---\nname: {name}\ndescription: A test agent.\n---\nYou are a specialist.",
        encoding="utf-8",
    )


def _make_skill_file(tmp_path):
    skill_dir = tmp_path / "skills" / "test_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: A test skill.\n---\nBody.",
        encoding="utf-8",
    )


class TestBuildSystemPromptAgents:
    """Verify ## Agents section in build_system_prompt()."""

    def test_agents_section_present_when_agents_dir_has_agents(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        _make_agent_file(tmp_path)
        sr, ar = _make_registries(tmp_path)
        result = prompt.build_system_prompt(sr, ar)
        assert "## Agents" in result

    def test_agents_section_absent_when_agents_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        sr, ar = _make_registries(tmp_path)
        result = prompt.build_system_prompt(sr, ar)
        assert "## Agents" not in result

    def test_agents_section_absent_when_agents_dir_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        (tmp_path / "agents").mkdir()
        sr, ar = _make_registries(tmp_path)
        result = prompt.build_system_prompt(sr, ar)
        assert "## Agents" not in result

    def test_agents_section_after_skills_section(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prompt, "WORKSPACE_DIR", str(tmp_path))
        _make_skill_file(tmp_path)
        _make_agent_file(tmp_path)
        sr, ar = _make_registries(tmp_path)
        result = prompt.build_system_prompt(sr, ar)
        assert "## Skills" in result
        assert "## Agents" in result
        assert result.index("## Skills") < result.index("## Agents")
