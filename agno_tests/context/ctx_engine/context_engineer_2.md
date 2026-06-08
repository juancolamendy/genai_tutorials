# Technical Specification: context_enginer_2.py

## Overview

Generic Context-Aware Multi-Agent System built with [Agno](https://github.com/agno-agi/agno) and LanceDB. Implements a two-step workflow where a **Planner** agent produces a JSON execution plan and an **Executor** agent runs it by dispatching specialist subagents looked up from a central **AgentRegistry**.

---

## Architecture

```
run_context_engine(goal, style_hint)
        │
        ▼
┌─────────────────────────────────────┐
│         Workflow (Agno)             │
│  Step 1: Planner Step               │
│  Step 2: Executor Step              │
└─────────────────────────────────────┘
        │
        ▼ plan JSON embedded in prompt
┌─────────────────────────────────────┐
│         Executor Agent              │
│  Tools: SubagentRouterTools         │
│         (registry-backed lookup)    │
└─────────────────────────────────────┘
        │
        ├──► Librarian Agent  (LibrarianTools)
        │       └─ writes semantic_blueprint → session_state
        ├──► Researcher Agent (ResearcherTools)
        │       └─ writes research_results → session_state
        └──► Writer Agent     (WriterContextTools)
                └─ reads semantic_blueprint + research_results from session_state
```

**Key design principle:** each subagent is self-contained — it reads its own dependencies from `session_state` via its own tool. No placeholder substitution (`{{step_id}}`) is needed; the Executor LLM never resolves templates itself.

---

## Dependencies

| Package | Purpose |
|---|---|
| `agno` | Agent/Workflow framework |
| `lancedb` | Vector database for semantic search |
| `openai` | Embedding model (`text-embedding-3-small`) |
| `anthropic` (via agno) | LLM backbone (`claude-sonnet-4-20250514`) |
| `python-dotenv` | Load environment variables from `.env` |

### Required Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Used to generate embeddings via OpenAI API |
| `ANTHROPIC_API_KEY` | Used by Agno's `Claude` model for all agents |

---

## Module Sections

### 0. Embeddings + Chunking

**`embed_batch(texts, model)`**
- Calls OpenAI Embeddings API in batch.
- Default model: `text-embedding-3-small`.
- Returns `List[List[float]]`.

**`chunk_text(text, chunk_size, overlap)`**
- Splits raw text into overlapping character-level chunks.
- Defaults: `chunk_size=500`, `overlap=50`.

---

### 1. Data + LanceDB

**Vector Database:** `./context_enginer_1.db` (LanceDB on disk)

**Tables created at startup by `init_lancedb()`:**

| Table | Schema | Purpose |
|---|---|---|
| `context_library` | `id`, `description`, `blueprint` (JSON str), `vector` | Stores style blueprints; searched semantically by the Librarian |
| `knowledge_base` | `id`, `text`, `vector` | Stores chunked factual knowledge; searched semantically by the Researcher |

**Style Blueprints (seeded at startup):**

| ID | Description |
|---|---|
| `suspenseful_narrative` | Mysterious/tense narrative with hooks and reveals |
| `technical_explanation` | Precise, structured technical writing |
| `casual_summary` | Friendly, conversational summary for non-experts |

Each blueprint contains: `tone`, `style`, `structure`, `techniques[]`.

---

### 2. Subagent Toolkits

#### `LibrarianTools` (Toolkit)

Registered tool: **`semantic_blueprint_search(intent_query)`**
- Embeds `intent_query` and queries `context_library` with `limit=1`.
- Writes result into `session_state["semantic_blueprint"]` and `session_state["blueprint_found"]`.
- Returns JSON string of the matched blueprint (or a neutral fallback if no match).

#### `ResearcherTools` (Toolkit)

Registered tool: **`semantic_research(query, limit=5)`**
- Embeds `query` and queries `knowledge_base`.
- Writes result list into `session_state["research_results"]`.
- Returns JSON string of `[{id, text}, ...]`.

#### `WriterContextTools` (Toolkit)

Registered tool: **`get_writing_context()`**
- Reads `session_state["semantic_blueprint"]` and `session_state["research_results"]` (written by the Librarian and Researcher).
- Returns both as a single JSON object: `{"semantic_blueprint": {...}, "research_results": [...]}`.
- The Writer calls this before generating content, so it receives fully populated context from prior steps.

---

### 3. Subagents

All agents use `Claude(id="claude-sonnet-4-20250514")`.

| Agent | Role | Tools | session_state writes | Behavior |
|---|---|---|---|---|
| `librarian_agent` | Context Librarian | `LibrarianTools` | `semantic_blueprint`, `blueprint_found` | Calls `semantic_blueprint_search`; returns the matched blueprint |
| `researcher_agent` | Knowledge Researcher | `ResearcherTools` | `research_results` | Calls `semantic_research`; summarizes findings |
| `writer_agent` | Writer | `WriterContextTools` | — | Calls `get_writing_context()` to load context from session_state; generates final content following the blueprint |

---

### 4. Agent Registry

**`AgentRegistry`** — central registry mapping agent names to `Agent` instances and their descriptions.

```python
agent_registry = AgentRegistry()
agent_registry.register(librarian_agent, "...")
agent_registry.register(researcher_agent, "...")
agent_registry.register(writer_agent, "...")
```

**Methods:**

| Method | Description |
|---|---|
| `register(agent, description)` | Registers an agent by its `agent.name` (stored lowercase for lookup) |
| `get(name) → Agent \| None` | Case-insensitive exact lookup by name |
| `names() → List[str]` | Returns all registered agent names (display-cased) |
| `agent_list_for_prompt() → str` | Returns a formatted bullet list of `name: description` for use in system prompts |

**Extending the system:** registering a new specialist agent is a single `agent_registry.register(...)` call. The Planner's instructions and the router's dispatch table update automatically.

---

### 5. SubagentRouterTools (Toolkit)

**`SubagentRouterTools(registry)`** — takes an `AgentRegistry` at construction time.

Registered tool: **`call_subagent(agent_name, input_text)`**
- Looks up `agent_name` in the registry with `registry.get(agent_name)` (case-insensitive).
- Returns a clear error string if the name is not found (no silent fallbacks).
- Passes `session_state` into the subagent call so state is shared across agents.
- Returns the subagent's response content as a string.

```python
subagent_router_tools = SubagentRouterTools(agent_registry)
```

---

### 6. Planner Agent + Step

**`_build_planner_instructions(registry)`** — generates the planner's system prompt dynamically from the registry:
- Lists all registered agents with their descriptions.
- Builds the output schema example using actual registered agent names.
- Instructs the planner not to use `{{step_id}}` placeholders (each agent reads its own context from `session_state`).

**`planner_agent`** — an Agno `Agent` whose instructions are generated by `_build_planner_instructions(agent_registry)`.

**Output schema:**
```json
{
  "steps": [
    {"id": "step_librarian", "agent": "Librarian", "input_template": "..."},
    {"id": "step_researcher", "agent": "Researcher", "input_template": "..."},
    {"id": "step_writer",    "agent": "Writer",     "input_template": "<goal only>"}
  ]
}
```

- The Planner is instructed to return **only JSON** (no prose).
- The Writer's `input_template` states only the goal — it fetches its own context via `get_writing_context()`.

**`planner_step_fn(step_input, run_context)`**
1. Reads `user_goal` and `style_hint` from `session_state`.
2. Calls `planner_agent.run(prompt)`.
3. Parses JSON; on parse failure, falls back to a hardcoded 3-step plan.
4. Writes the plan into `session_state["plan"]`.
5. Returns the plan as a JSON string.

---

### 7. Executor Agent + Step

**`executor_agent`** — an Agno `Agent` equipped only with `SubagentRouterTools`.

The plan JSON is passed **directly in the prompt message** (not referenced from `session_state`), so the LLM can read it without any special session_state injection.

**Execution algorithm (defined in system instructions):**
1. Initialize `step_outputs = {}` in `session_state`.
2. For each step in `plan.steps` (in order):
   - a) Call `call_subagent(agent_name, input_text)` — the subagent reads its own dependencies from `session_state` via its tools.
   - b) Store result in `step_outputs[step.id]` and update `session_state["step_outputs"]`.
   - c) Append a trace entry to `session_state["trace_logs"]`.
3. Write `step_outputs[last_step_id]` into `session_state["final_output"]` and return it.

**`executor_step_fn(step_input, run_context)`**
1. Bootstraps `step_outputs = {}` and `trace_logs = []` in `session_state` (if not present).
2. Reads the plan from `session_state["plan"]` and serializes it to JSON.
3. Calls `executor_agent.run(f"Execute this plan...\n\n{plan_json}")` with the shared `session_state`.
4. Returns `session_state["final_output"]` (or falls back to `resp.content`).

---

### 8. Workflow

```python
context_engine_workflow = Workflow(
    name="Generic Context Engine",
    steps=[planner_step, executor_step],
    session_state={},
    db=SqliteDb(
        session_table="generic_context_engine_sessions",
        db_file="context_engine_2.db",
    ),
)
```

- Session state is persisted to SQLite for replay/resumption via `session_id`.
- Steps execute sequentially; `session_state` is the shared communication channel.

---

### 9. Public API

```python
def run_context_engine(
    goal: str,
    style_hint: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `goal` | `str` | The user's content generation goal |
| `style_hint` | `str \| None` | Natural language hint for tone/style selection |
| `session_id` | `str \| None` | Resume a prior session (SQLite-backed) |

**Returns:**

```python
{
    "final_output": str,       # Generated content from the Writer agent
    "plan":         dict,      # The JSON plan produced by the Planner
    "trace_logs":   list,      # Per-step execution trace entries
    "session_id":   str,       # Session ID (new or resumed)
}
```

---

## Session State Schema

| Key | Set by | Type | Description |
|---|---|---|---|
| `user_goal` | `run_context_engine` | `str` | User's original goal |
| `style_hint` | `run_context_engine` | `str \| None` | Style hint for the Planner |
| `plan` | `planner_step_fn` | `dict` | JSON execution plan |
| `semantic_blueprint` | `LibrarianTools` | `dict` | Best-match style blueprint |
| `blueprint_found` | `LibrarianTools` | `bool` | Whether a semantic match was found |
| `research_results` | `ResearcherTools` | `list` | Retrieved knowledge chunks |
| `step_outputs` | `executor_agent` | `dict[str, str]` | Per-step output keyed by step ID |
| `trace_logs` | `executor_agent` | `list[dict]` | Execution trace entries |
| `final_output` | `executor_agent` | `str` | Final generated content |

---

## Data Flow

```
run_context_engine(goal, style_hint)
  │
  ├─ session_state["user_goal"] = goal
  ├─ session_state["style_hint"] = style_hint
  │
  ▼ Planner Step
  planner_agent.run(prompt)         ← instructions built from agent_registry
  ├─ session_state["plan"] = { steps: [...] }
  │
  ▼ Executor Step
  executor_agent.run(plan_json)     ← plan passed directly in message
  │
  ├─ Step: Librarian
  │   SubagentRouterTools → registry.get("Librarian") → librarian_agent
  │   LibrarianTools.semantic_blueprint_search(intent_query)
  │   └─ session_state["semantic_blueprint"] = { tone, style, ... }
  │
  ├─ Step: Researcher
  │   SubagentRouterTools → registry.get("Researcher") → researcher_agent
  │   ResearcherTools.semantic_research(query)
  │   └─ session_state["research_results"] = [{ id, text }, ...]
  │
  └─ Step: Writer
      SubagentRouterTools → registry.get("Writer") → writer_agent
      WriterContextTools.get_writing_context()
      └─ reads semantic_blueprint + research_results from session_state
      └─ session_state["final_output"] = <generated content>
```

---

## Adding a New Specialist Agent

1. Create a `Toolkit` subclass with the agent's tool(s).
2. Define the `Agent` instance.
3. Register it:
   ```python
   agent_registry.register(my_agent, "Description of what it does and what it writes to session_state.")
   ```

The Planner's system prompt and `SubagentRouterTools` dispatch table update automatically — no other changes needed.

---

## Example Usage

```python
result = run_context_engine(
    goal="Explain retrieval-augmented generation to a non-technical founder.",
    style_hint="casual summary with friendly tone",
)

print(result["final_output"])
print(result["plan"])
print(result["trace_logs"])
```
