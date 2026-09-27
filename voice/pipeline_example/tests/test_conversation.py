from langchain_core.messages import HumanMessage

from pipeline_example.conversation import BoundedChatMessageHistory, ConversationProcessor


def test_bounded_history_does_not_grow_unbounded():
    history = BoundedChatMessageHistory(max_messages=3)
    for i in range(5):
        history.add_message(HumanMessage(content=f"msg {i}"))
    assert len(history.messages) == 3
    assert history.messages[-1].content == "msg 4"


def test_build_system_prompt_receives_agent_prompt(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    processor = ConversationProcessor(
        model_name="llama-3.1-8b-instant",
        model_provider="groq",
        agent_prompt="BUSINESS CONTEXT",
    )
    prompt = processor._build_system_prompt(agent_prompt="BUSINESS CONTEXT")
    assert "BUSINESS CONTEXT" in prompt
    assert "You are Jess" in prompt
