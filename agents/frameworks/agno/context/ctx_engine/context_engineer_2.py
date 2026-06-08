"""
Generic Context-Aware Multi-Agent System (Agno + LanceDB)

- Workflow:
    Step 1: PlannerAgent → writes JSON plan into session_state["plan"]
    Step 2: ExecutorAgent → reads plan (passed directly in message), calls subagents

- Execution model:
    * Planner and Executor are *agents* used as workflow steps.
    * Each subagent reads its own dependencies from session_state via its own tools:
      - Librarian writes semantic_blueprint → session_state
      - Researcher writes research_results → session_state
      - Writer calls get_writing_context() to read both from session_state
    * No {{step_id}} placeholder substitution needed.
"""

import os
import json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

import lancedb
from openai import OpenAI as OpenAIClient

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import Toolkit
from agno.run import RunContext
from agno.workflow.workflow import Workflow
from agno.workflow.step import Step
from agno.db.sqlite import SqliteDb

## Load env vars
load_dotenv()

# ============================================================
# 0. EMBEDDINGS + CHUNKING
# ============================================================

openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

def embed_batch(texts: List[str],
                model: str = "text-embedding-3-small") -> List[List[float]]:
    resp = openai_client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in resp.data]

def chunk_text(text: str,
               chunk_size: int = 500,
               overlap: int = 50) -> List[str]:
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += chunk_size - overlap
    return chunks

# ============================================================
# 1. DATA + LANCEDB
# ============================================================

BLUEPRINTS = [
    {
        "id": "suspenseful_narrative",
        "description": "Creative writing with suspenseful narrative and vivid imagery",
        "blueprint": {
            "tone": "mysterious and tense",
            "style": "narrative",
            "structure": "Hook → rising tension → twist → resolution",
            "techniques": [
                "End paragraphs on questions or hints",
                "Use sensory detail",
                "Reveal information gradually"
            ]
        }
    },
    {
        "id": "technical_explanation",
        "description": "Technical explanation with precise terminology and clear structure",
        "blueprint": {
            "tone": "professional and precise",
            "style": "technical",
            "structure": "Overview → concepts → examples → recap",
            "techniques": [
                "Define key terms",
                "Use numbered steps",
                "Avoid metaphors unless explicitly requested"
            ]
        }
    },
    {
        "id": "casual_summary",
        "description": "Casual friendly summary for non-experts",
        "blueprint": {
            "tone": "friendly and relaxed",
            "style": "conversational",
            "structure": "Main idea → simple breakdown → takeaway",
            "techniques": [
                "Use analogies",
                "Short paragraphs",
                "Minimal jargon"
            ]
        }
    },
]

RAW_KNOWLEDGE = """
Artificial Intelligence (AI) refers to computer systems that perform tasks usually requiring human intelligence.
Machine learning is a subset of AI that learns patterns from data.
Deep learning uses multi-layer neural networks to model complex relationships.
Retrieval-Augmented Generation (RAG) combines retrieval over a knowledge base with a generator model for better factuality.
Vector databases store dense embeddings and support semantic similarity search.
Multi-agent systems split responsibilities across agents like Librarian, Researcher, and Writer coordinated by an orchestrator.
Context engineering designs prompts, blueprints, and workflows to control how agents use tools and knowledge.
"""

db = lancedb.connect("./context_engineer_library.db")

def init_lancedb():
    existing = set(db.table_names())

    # Context Library
    if "context_library" in existing:
        ctx_table = db.open_table("context_library")
        print(f"Context library: reusing existing table ({ctx_table.count_rows()} rows)")
    else:
        descs = [bp["description"] for bp in BLUEPRINTS]
        desc_emb = embed_batch(descs)
        ctx_rows = []
        for bp, e in zip(BLUEPRINTS, desc_emb):
            ctx_rows.append({
                "id": bp["id"],
                "description": bp["description"],
                "blueprint": json.dumps(bp["blueprint"]),
                "vector": e,
            })
        ctx_table = db.create_table("context_library", data=ctx_rows)
        print(f"Context library: created with {len(ctx_rows)} rows")

    # Knowledge Base
    if "knowledge_base" in existing:
        kb_table = db.open_table("knowledge_base")
        print(f"Knowledge base: reusing existing table ({kb_table.count_rows()} rows)")
    else:
        chunks = chunk_text(RAW_KNOWLEDGE, chunk_size=260, overlap=40)
        ch_emb = embed_batch(chunks)
        kb_rows = []
        for i, (txt, e) in enumerate(zip(chunks, ch_emb)):
            kb_rows.append({
                "id": f"chunk_{i}",
                "text": txt,
                "vector": e,
            })
        kb_table = db.create_table("knowledge_base", data=kb_rows)
        print(f"Knowledge base: created with {len(kb_rows)} rows")

    return ctx_table, kb_table

context_table, knowledge_table = init_lancedb()

# ============================================================
# 2. SUBAGENT TOOLKITS
# ============================================================

class LibrarianTools(Toolkit):
    name = "librarian_tools"

    def __init__(self):
        super().__init__(name=self.name)
        self.register(self.semantic_blueprint_search)

    def semantic_blueprint_search(self,
                                 run_context: RunContext,
                                 intent_query: Optional[str] = None) -> str:
        """Procedural RAG over context_library."""
        if intent_query is None:
            intent_query = "Default neutral blueprint for a casual summary"
        emb = embed_batch([intent_query])[0]
        df = context_table.search(emb).limit(1).to_pandas()
        if len(df) == 0:
            blueprint = {
                "id": "default_neutral",
                "description": "Neutral fallback blueprint",
                "blueprint": {
                    "tone": "neutral",
                    "style": "plain",
                    "structure": "Intro → body → conclusion",
                    "techniques": ["Explain clearly and concisely"]
                }
            }
            found = False
        else:
            bp = json.loads(df.iloc[0]["blueprint"])
            blueprint = {
                "id": df.iloc[0]["id"],
                "description": df.iloc[0]["description"],
                "blueprint": bp,
            }
            found = True

        if run_context.session_state is not None:
            run_context.session_state["semantic_blueprint"] = blueprint
            run_context.session_state["blueprint_found"] = found
        return json.dumps(blueprint)


class ResearcherTools(Toolkit):
    name = "researcher_tools"

    def __init__(self):
        super().__init__(name=self.name)
        self.register(self.semantic_research)

    def semantic_research(self,
                          run_context: RunContext,
                          query: Optional[str] = None,
                          limit: int = 5) -> str:
        """Factual RAG over knowledge_base."""
        if query is None:
            query = "No query provided"
        emb = embed_batch([query])[0]
        df = knowledge_table.search(emb).limit(limit).to_pandas()
        results = [{"id": r["id"], "text": r["text"]} for _, r in df.iterrows()]

        if run_context.session_state is not None:
            run_context.session_state["research_results"] = results
        return json.dumps(results)


class WriterContextTools(Toolkit):
    name = "writer_context_tools"

    def __init__(self):
        super().__init__(name=self.name)
        self.register(self.get_writing_context)

    def get_writing_context(self, run_context: RunContext) -> str:
        """Read semantic_blueprint and research_results stored by prior agents."""
        blueprint = run_context.session_state.get("semantic_blueprint", {}) if run_context.session_state else {}
        research = run_context.session_state.get("research_results", []) if run_context.session_state else []
        return json.dumps({"semantic_blueprint": blueprint, "research_results": research})

# ============================================================
# 3. SUBAGENTS (Librarian, Researcher, Writer)
# ============================================================

model = Claude(id="claude-sonnet-4-20250514")

librarian_agent = Agent(
    name="Librarian",
    role="Context Librarian",
    model=model,
    tools=[LibrarianTools()],
    instructions=(
        "You fetch semantic style blueprints from the Context Library.\n"
        "Always call semantic_blueprint_search(intent_query=...) first.\n"
        "Keep your own text short; the important data is in the JSON tool output."
    ),
    markdown=True,
    debug_mode=True,
)

researcher_agent = Agent(
    name="Researcher",
    role="Knowledge Researcher",
    model=model,
    tools=[ResearcherTools()],
    instructions=(
        "You fetch factual context from the Knowledge Base.\n"
        "Always call semantic_research(query=...) first.\n"
        "Then summarise briefly what you found."
    ),
    markdown=True,
    debug_mode=True,
)

writer_agent = Agent(
    name="Writer",
    role="Writer",
    model=model,
    tools=[WriterContextTools()],
    instructions=(
        "You generate final content.\n"
        "Always call get_writing_context() first to load the style blueprint and research facts.\n"
        "Follow blueprint.tone/style/structure/techniques exactly.\n"
        "Do not mention agents, tools, or the internal process."
    ),
    markdown=True,
    debug_mode=True,
)

# ============================================================
# 4. AGENT REGISTRY
# ============================================================

class AgentRegistry:
    """Maps agent names to Agent instances and their descriptions."""

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._descriptions: Dict[str, str] = {}

    def register(self, agent: Agent, description: str) -> None:
        self._agents[agent.name.lower()] = agent
        self._descriptions[agent.name] = description

    def get(self, name: str) -> Optional[Agent]:
        return self._agents.get(name.lower().strip())

    def names(self) -> List[str]:
        return list(self._descriptions.keys())

    def agent_list_for_prompt(self) -> str:
        return "\n".join(
            f"- {name}: {desc}" for name, desc in self._descriptions.items()
        )


agent_registry = AgentRegistry()
agent_registry.register(
    librarian_agent,
    "Fetches style blueprints via semantic search; writes semantic_blueprint to session_state.",
)
agent_registry.register(
    researcher_agent,
    "Fetches factual context via semantic search; writes research_results to session_state.",
)
agent_registry.register(
    writer_agent,
    "Generates final content; reads semantic_blueprint and research_results from session_state via its own tool.",
)

# ============================================================
# 5. EXECUTOR TOOLS – CALL SUBAGENTS (GENERIC)
# ============================================================

class SubagentRouterTools(Toolkit):
    """Generic router tool so the Executor can call subagents by name."""
    name = "subagent_router_tools"

    def __init__(self, registry: AgentRegistry):
        super().__init__(name=self.name)
        self._registry = registry
        self.register(self.call_subagent)

    def call_subagent(self,
                      run_context: RunContext,
                      agent_name: str,
                      input_text: str) -> str:
        """
        Call a registered subagent by name.
        agent_name must exactly match a name in the agent registry
        (case-insensitive): use the value from the plan step's 'agent' field.
        """
        agent = self._registry.get(agent_name)
        if agent is None:
            available = self._registry.names()
            return f"Error: unknown agent '{agent_name}'. Available agents: {available}"
        resp = agent.run(input_text, session_state=run_context.session_state)
        return resp.content

subagent_router_tools = SubagentRouterTools(agent_registry)

# ============================================================
# 6. PLANNER AGENT (OPTIONAL – CAN BE BYPASSED)
# ============================================================

def _build_planner_instructions(registry: AgentRegistry) -> str:
    agent_lines = registry.agent_list_for_prompt()
    names_example = registry.names()
    steps_example = "\n".join(
        f'    {{"id": "step{i+1}", "agent": "{name}", "input_template": "..."}}'
        for i, name in enumerate(names_example)
    )
    return (
        "You create a JSON execution plan for the Context Engine.\n"
        "Input: user_goal and optional style_hint.\n\n"
        "Available agents:\n"
        f"{agent_lines}\n\n"
        "Each agent reads its own dependencies from session_state via its own tools.\n"
        "For the final content agent, input_template should only state the goal.\n\n"
        "Call the subagents in the order that makes the most sense for the plan goals.\n"
        "Output schema (use the exact agent names listed above):\n"
        "{\n"
        "  \"steps\": [\n"
        f"{steps_example}\n"
        "  ]\n"
        "}\n"
        "Return ONLY JSON."
    )


planner_agent = Agent(
    name="Planner",
    role="Planner",
    model=model,
    instructions=_build_planner_instructions(agent_registry),
    markdown=False,
    debug_mode=True,
)

def planner_step_fn(step_input, run_context: RunContext):
    goal = run_context.session_state.get("user_goal", step_input.input)
    style_hint = run_context.session_state.get("style_hint", "Librarian can infer the style from this hint.")
    prompt = f"""
User goal:
{goal}

Style hint:
{style_hint}

Create the JSON plan now.
"""
    resp = planner_agent.run(prompt)
    try:
        plan = json.loads(resp.content)
    except Exception:
        # simple fallback plan
        plan = {
            "steps": [
                {
                    "id": "step_librarian",
                    "agent": "Librarian",
                    "input_template": f"Find the best writing style blueprint for: {style_hint}",
                },
                {
                    "id": "step_researcher",
                    "agent": "Researcher",
                    "input_template": f"Gather factual context about: {goal}",
                },
                {
                    "id": "step_writer",
                    "agent": "Writer",
                    "input_template": f"Write content for goal: {goal}",
                },
            ]
        }

    if run_context.session_state is not None:
        run_context.session_state["plan"] = plan
    return json.dumps(plan)

# ============================================================
# 7. EXECUTOR AGENT (GENERIC – USES TOOLS & SESSION STATE)
# ============================================================

executor_agent = Agent(
    name="Executor",
    role="Executor",
    model=model,
    tools=[subagent_router_tools],
    instructions=(
        "You are the Executor of a Context Engine.\n"
        "You receive a JSON plan directly in the message with a list of steps.\n"
        "Each step contains: id, agent, input_template.\n\n"
        "Algorithm:\n"
        "1) Initialize step_outputs = {} in session_state.\n"
        "2) For each step in plan.steps (in order):\n"
        "   a) Call subagent_router_tools.call_subagent(agent_name=step.agent,\n"
        "      input_text=step.input_template) to execute the subagent.\n"
        "      Each subagent reads its own dependencies from session_state via its tools.\n"
        "   b) Store the returned text in step_outputs[step.id] and update\n"
        "      session_state['step_outputs'].\n"
        "   c) Append a trace entry to session_state['trace_logs'].\n"
        "3) After all steps, write step_outputs[last_step_id] into\n"
        "   session_state['final_output'] and return it.\n\n"
        "Execute the above algorithm carefully and deterministically."
    ),
    markdown=True,
    debug_mode=True,
)

def executor_step_fn(step_input, run_context: RunContext):
    if run_context.session_state is not None:
        run_context.session_state.setdefault("step_outputs", {})
        run_context.session_state.setdefault("trace_logs", [])

    plan = run_context.session_state.get("plan", {"steps": []})
    plan_json = json.dumps(plan, indent=2)

    resp = executor_agent.run(
        f"Execute this plan step by step as described in your system instructions:\n\n{plan_json}\n\nReturn ONLY the final output.",
        session_state=run_context.session_state,
    )
    return run_context.session_state.get("final_output", resp.content)

# ============================================================
# 8. WORKFLOW WITH TWO STEPS
# ============================================================

planner_step = Step(
    name="Planner Step",
    executor=planner_step_fn,   # function that calls Planner agent and writes plan
    description="Planner agent produces JSON plan and stores it in session_state['plan'].",
)

executor_step = Step(
    name="Executor Step",
    executor=executor_step_fn,  # function that calls Executor agent
    description="Executor agent receives plan in prompt and calls subagents; each subagent reads its own context from session_state.",
)

context_engine_workflow = Workflow(
    name="Generic Context Engine",
    description="Two-step workflow: Planner → Executor; Executor uses team of subagents.",
    steps=[planner_step, executor_step],
    session_state={},
    db=SqliteDb(
        session_table="generic_context_engine_sessions",
        db_file="context_engineer_sessions.db",
    ),
    debug_mode=True,
)

# ============================================================
# 9. PUBLIC API
# ============================================================

def run_context_engine(
    goal: str,
    style_hint: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    context_engine_workflow.session_state = {
        "user_goal": goal,
        "style_hint": style_hint,
    }
    result = context_engine_workflow.run(input=goal, session_id=session_id)
    state = context_engine_workflow.get_session_state()
    return {
        "final_output": state.get("final_output", result.content),
        "plan": state.get("plan"),
        "trace_logs": state.get("trace_logs", []),
        "session_id": result.session_id,
    }

# ============================================================
# 10. EXAMPLE
# ============================================================

if __name__ == "__main__":
    print("Context Engine ready. Type your goal and press Enter. Type /exit to quit.")
    while True:
        try:
            user_input = input("\nGoal> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input == "/exit":
            print("Exiting.")
            break

        if not user_input:
            continue

        res = run_context_engine(goal=user_input)
        print("\n=== FINAL OUTPUT ===\n")
        print(res["final_output"])
        print("\n=== PLAN ===\n")
        print(json.dumps(res["plan"], indent=2))
        print("\n=== TRACE LOGS ===\n")
        for log in res["trace_logs"]:
            print(log)

