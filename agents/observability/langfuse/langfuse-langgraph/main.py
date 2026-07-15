"""Terminal chat built on LangGraph with checkpointed memory.
All turns are traced under one Langfuse session.

Run:  uv run main.py   (type 'exit' to quit)
"""

import uuid

from dotenv import load_dotenv

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

# --- Langfuse setup ---------------------------------------------------------
langfuse = get_client()
if not langfuse.auth_check():
    raise RuntimeError("Langfuse auth failed — check .env keys and base URL")

langfuse_handler = CallbackHandler()

# One id reused for both LangGraph's thread and Langfuse's session
session_id = f"langgraph-chat-{uuid.uuid4().hex[:8]}"
print(f"Langfuse session: {session_id}\n")

# --- Build the graph ----------------------------------------------------------
llm = ChatAnthropic(model="claude-haiku-4-5")


def chatbot(state: MessagesState):
    """Single node: call the model with the full message history."""
    response = llm.invoke(
        [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")

# The checkpointer IS the memory: it stores state per thread_id,
# so each invoke automatically continues the same conversation.
graph = builder.compile(checkpointer=InMemorySaver())

# --- Terminal loop ------------------------------------------------------------
config = {
    "callbacks": [langfuse_handler],
    "configurable": {"thread_id": session_id},        # LangGraph memory key
    "metadata": {
        "langfuse_session_id": session_id,            # Langfuse session grouping
        "langfuse_user_id": "terminal-user",
        "langfuse_tags": ["langgraph-chat"],
    },
}

print("Chat started. Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if user_input.lower() in {"exit", "quit"}:
        break
    if not user_input:
        continue

    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )
    print(f"AI: {result['messages'][-1].content}\n")

langfuse.flush()
print("Bye! Traces are under session:", session_id)
