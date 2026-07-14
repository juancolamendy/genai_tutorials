"""Terminal chat with memory — all turns traced under one Langfuse session.

Run with:  uv run chat_session.py
Type 'exit' or 'quit' to stop.
"""

import uuid

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Langfuse setup ---------------------------------------------------------
langfuse = get_client()
if not langfuse.auth_check():
    raise RuntimeError("Langfuse auth failed — check .env keys and base URL")

langfuse_handler = CallbackHandler()

# One session id for the whole run: every trace groups under it in the UI
session_id = f"terminal-chat-{uuid.uuid4().hex[:8]}"
print(f"Langfuse session: {session_id}\n")

# --- Chain with memory ------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Keep answers concise."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)
llm = ChatAnthropic(model="claude-haiku-4-5")
chain = prompt | llm | StrOutputParser()

# In-memory store: one chat history per session id
_histories: dict[str, InMemoryChatMessageHistory] = {}


def get_history(sid: str) -> InMemoryChatMessageHistory:
    if sid not in _histories:
        _histories[sid] = InMemoryChatMessageHistory()
    return _histories[sid]


chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- Terminal loop ------------------------------------------------------------
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

    reply = chat.invoke(
        {"input": user_input},
        config={
            "callbacks": [langfuse_handler],
            # Used by RunnableWithMessageHistory to pick the right memory
            "configurable": {"session_id": session_id},
            # Used by Langfuse to group all traces under one session
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_user_id": "terminal-user",
                "langfuse_tags": ["terminal-chat"],
            },
        },
    )
    print(f"AI: {reply}\n")

# Make sure all queued traces reach Langfuse before the process exits
langfuse.flush()
print("Bye! Traces are under session:", session_id)
