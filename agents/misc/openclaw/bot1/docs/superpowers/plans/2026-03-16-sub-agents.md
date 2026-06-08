# Sub-Agent Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sub-agent dispatch to `main.py` so the main LLM can call specialized agents defined as markdown files under `workspace/agents/`.

**Architecture:** Mirror the existing skills pattern — `load_agents_index()` scans `workspace/agents/`, populates a module-level `_agents_registry`, and injects an XML agents block into the system prompt. A closure factory `_make_tool_run_agent(user_id, session_id)` creates a registered `run_agent` tool per `handle_message` call. The inner closure loads the parent session as read-only context (no save), appends `input`, and calls `client.messages.create` **directly with no tools** — preventing recursion and keeping the sub-agent stateless. Default model fallback is `"claude-sonnet-4-6"`.

**Tech Stack:** Python 3.12, `anthropic` SDK, `pytest`, `unittest.mock`

---

## File Map

| File | Change |
|------|--------|
| `main.py` | Add `_agents_registry`, `extract_frontmatter_body`, `load_agents_index`, `_make_tool_run_agent`; update `build_system_prompt` and `handle_message` |
| `tests/test_main.py` | Add `TestExtractFrontmatterBody`, `TestLoadAgentsIndex`, `TestToolRunAgent`; extend `TestBuildSystemPrompt` |

---

## Chunk 1: `extract_frontmatter_body` helper

---

### Task 1: `extract_frontmatter_body` — tests + implementation

**Files:**
- Modify: `tests/test_main.py` (append new test class)
- Modify: `main.py:81` (insert after `parse_skill_frontmatter`)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_main.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/jcolamendy/ai_ml/agents_tutorials/openclaw/bot1
python -m pytest tests/test_main.py::TestExtractFrontmatterBody -v
```

Expected: `AttributeError: module 'main' has no attribute 'extract_frontmatter_body'`

- [ ] **Step 3: Implement `extract_frontmatter_body`**

In `main.py`, insert after `parse_skill_frontmatter` (after line 81, before `def load_context_files`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_main.py::TestExtractFrontmatterBody -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add extract_frontmatter_body helper"
```

---

## Chunk 2: `load_agents_index` and `_agents_registry`

---

### Task 2: `_agents_registry` module-level state

**Files:**
- Modify: `main.py` (add constant near other module-level state)

- [ ] **Step 1: Add `_agents_registry` to `main.py`**

In `main.py`, after the line `WORKSPACE_DIR = "./workspace"` (around line 28), add:

```python
_agents_registry: dict = {}
```

- [ ] **Step 2: Verify the file still imports cleanly**

```bash
python -c "import main; print('ok')"
```

Expected: `ok`

---

### Task 3: `load_agents_index` — tests + implementation

**Files:**
- Modify: `tests/test_main.py` (append new test class)
- Modify: `main.py` (insert `load_agents_index` after `load_skills_index`)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_main.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_main.py::TestLoadAgentsIndex -v
```

Expected: `AttributeError: module 'main' has no attribute 'load_agents_index'` (11 tests collected)

- [ ] **Step 3: Implement `load_agents_index`**

In `main.py`, insert after `load_skills_index` (after line ~168, before `def load_approvals`):

```python
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
        file_path = agent_file
        model = meta.get("model", None) or None
        _agents_registry[agent_name] = {"file_path": file_path, "model": model}
        agents.append({
            "name": agent_name,
            "description": meta.get("description", ""),
            "location": file_path,
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_main.py::TestLoadAgentsIndex -v
```

Expected: 11 passed

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
python -m pytest tests/ -v
```

Expected: all existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add _agents_registry and load_agents_index"
```

---

## Chunk 3: `build_system_prompt` agents section

---

### Task 4: Add `## Agents` section to `build_system_prompt` — tests + implementation

**Files:**
- Modify: `tests/test_main.py` (extend `TestBuildSystemPrompt`)
- Modify: `main.py:38-64` (`build_system_prompt`)

- [ ] **Step 1: Write failing tests**

Append to the existing `TestBuildSystemPrompt` class in `tests/test_main.py`:

```python
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
        # All three ordering relationships must hold
        assert skills_pos < agents_pos
        assert agents_pos < memory_pos
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_main.py::TestBuildSystemPrompt::test_agents_section_included_when_present tests/test_main.py::TestBuildSystemPrompt::test_agents_section_absent_when_no_agents tests/test_main.py::TestBuildSystemPrompt::test_agents_section_between_skills_and_memory_instructions -v
```

Expected: FAIL (agents section not in prompt yet)

- [ ] **Step 3: Update `build_system_prompt` in `main.py`**

Find the section in `build_system_prompt` that reads:

```python
    # 8. Skills index
    skills = load_skills_index()
    if skills:
        parts.append(f"## Skills\n\n{skills}")

    # 9. Memory tool instructions (always present)
    parts.append(build_memory_prompt())
```

Replace with:

```python
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
```

- [ ] **Step 4: Run the three new tests**

```bash
python -m pytest tests/test_main.py::TestBuildSystemPrompt -v
```

Expected: all pass

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add ## Agents section to build_system_prompt"
```

---

## Chunk 4: `_make_tool_run_agent` closure and `handle_message` update

---

### Task 5: `_make_tool_run_agent` — tests + implementation

**Files:**
- Modify: `tests/test_main.py` (append `TestToolRunAgent`)
- Modify: `main.py` (add `_make_tool_run_agent` before registry block; update `handle_message`)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_main.py`:

```python
class TestToolRunAgent:

    @pytest.fixture(autouse=True)
    def reset_registry(self, monkeypatch):
        monkeypatch.setattr(main, "_agents_registry", {})

    def _make_fn(self, tmp_path):
        """Return a tool_run_agent closure bound to an empty session."""
        return main._make_tool_run_agent(str(tmp_path), "test-session")

    def test_agent_not_found_returns_error_string(self, tmp_path):
        fn = self._make_fn(tmp_path)
        result = fn("nonexistent-agent", "hello")
        assert "Error" in result
        assert "nonexistent-agent" in result

    def test_agent_not_found_does_not_raise(self, tmp_path):
        fn = self._make_fn(tmp_path)
        # Must return a string, never raise
        result = fn("nonexistent-agent", "hello")
        assert isinstance(result, str)

    def test_unreadable_file_returns_error_string(self, tmp_path, monkeypatch):
        # Pre-populate registry with a path that doesn't exist
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_main.py::TestToolRunAgent -v
```

Expected: `AttributeError: module 'main' has no attribute '_make_tool_run_agent'`

- [ ] **Step 3: Implement `_make_tool_run_agent` in `main.py`**

In `main.py`, insert before the `registry = ToolRegistry()` line (around line 324).

Key design notes:
- The inner closure is named `run_agent` — this is the name the tool registry exposes to the LLM.
- `client.messages.create` is called **directly with no `tools=` argument** — the sub-agent is stateless and cannot recurse.
- The parent session is loaded **read-only** (no `save_session` call inside the closure).
- Default model is `"claude-sonnet-4-6"` when `entry["model"]` is `None` or empty.

```python
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

        body = extract_frontmatter_body(content)
        model = entry["model"] or "claude-sonnet-4-6"
        messages = load_session(user_id, session_id)
        messages = messages + [{"role": "user", "content": input}]

        try:
            # No tools= passed — sub-agent is stateless, prevents recursion
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=body,
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            return f"Error running agent '{agent_name}': {e}"

    return run_agent
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_main.py::TestToolRunAgent -v
```

Expected: 8 passed (7 original + 1 no-tools test)

- [ ] **Step 5: Update `handle_message` in `main.py`**

Find `handle_message`:

```python
async def handle_message(user_id: str, session_id: str, text: str):
    messages = load_session(user_id, session_id)
    messages = compact_session(user_id, session_id, messages)
    messages.append({"role": "user", "content": text})

    system_prompt = build_system_prompt()
    response_text, messages = run_agent_turn(messages, system_prompt)

    save_session(user_id, session_id, messages)
    return response_text
```

Replace with:

```python
async def handle_message(user_id: str, session_id: str, text: str):
    messages = load_session(user_id, session_id)
    messages = compact_session(user_id, session_id, messages)
    messages.append({"role": "user", "content": text})

    system_prompt = build_system_prompt()
    registry.register("run_agent", _make_tool_run_agent(user_id, session_id))
    response_text, messages = run_agent_turn(messages, system_prompt)

    save_session(user_id, session_id, messages)
    return response_text
```

- [ ] **Step 6: Add `handle_message` wiring test**

Append to `TestToolRunAgent` in `tests/test_main.py`:

```python
    @pytest.mark.asyncio
    async def test_handle_message_registers_run_agent(self, tmp_path, monkeypatch):
        """handle_message must register run_agent before run_agent_turn."""
        monkeypatch.setattr(main, "WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setattr(main, "_agents_registry", {})
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [_FakeTextBlock("hello")]
        with patch.object(main.client.messages, "create", return_value=mock_response):
            await main.handle_message("u1", "s1", "hi")
        assert main.registry.get_tool("run_agent") is not None
```

> Note: requires `pytest-asyncio`. Install if not present: `pip install pytest-asyncio`. Add `@pytest.mark.asyncio` import or configure `asyncio_mode = auto` in pytest.ini.

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add _make_tool_run_agent closure and wire into handle_message"
```

---

## Final Verification

- [ ] **Run full test suite one more time**

```bash
python -m pytest tests/ -v
```

Expected: all pass, no regressions

- [ ] **Smoke test the bot manually (optional)**

```bash
python main.py
# Enter user/session IDs, then ask: "Use the summarizer agent to summarize: The quick brown fox jumps over the lazy dog"
```

Expected: bot calls `run_agent("summarizer-agent", ...)` and returns a summary.
