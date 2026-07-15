"""LangGraph terminal chat — checkpointed memory + LangSmith thread grouping."""

import uuid

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

session_id = f"langgraph-chat-{uuid.uuid4().hex[:8]}"
print(f"LangSmith thread: {session_id}\n")

llm = ChatAnthropic(model="claude-haiku-4-5")


def chatbot(state: MessagesState):
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
graph = builder.compile(checkpointer=InMemorySaver())

config = {
    "configurable": {"thread_id": session_id},   # LangGraph memory
    "metadata": {"session_id": session_id},      # LangSmith thread grouping
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

print("Bye! Thread:", session_id)
