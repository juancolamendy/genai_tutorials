"""Terminal chat with memory — all turns grouped as one LangSmith thread."""

import uuid

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

session_id = f"terminal-chat-{uuid.uuid4().hex[:8]}"
print(f"LangSmith thread: {session_id}\n")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Keep answers concise."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)
llm = ChatAnthropic(model="claude-haiku-4-5")
chain = prompt | llm | StrOutputParser()

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
            # LangChain memory key
            "configurable": {"session_id": session_id},
            # LangSmith thread grouping: session_id in metadata
            "metadata": {"session_id": session_id},
        },
    )
    print(f"AI: {reply}\n")

print("Bye! Thread:", session_id)
