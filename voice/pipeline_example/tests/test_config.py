from pathlib import Path

import pytest

from pipeline_example import config


class TestValidateEnvironment:
    def test_raises_when_required_vars_missing(self, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("SERVER_DOMAIN", raising=False)
        with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY.*GROQ_API_KEY.*SERVER_DOMAIN"):
            config.validate_environment()

    def test_passes_when_all_vars_present(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-key")
        monkeypatch.setenv("GROQ_API_KEY", "gq-key")
        monkeypatch.setenv("SERVER_DOMAIN", "example.com")
        config.validate_environment()


class TestLoadAgentPrompt:
    def test_raises_when_file_missing(self):
        with pytest.raises(FileNotFoundError):
            config.load_agent_prompt(Path("/does/not/exist.txt"))

    def test_raises_when_file_empty(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("   ")
        with pytest.raises(ValueError, match="empty"):
            config.load_agent_prompt(empty)

    def test_returns_stripped_content(self, tmp_path):
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("  Hello world  ")
        assert config.load_agent_prompt(prompt) == "Hello world"
