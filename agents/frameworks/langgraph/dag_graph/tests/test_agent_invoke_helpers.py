"""Tests for invoke_agent / ainvoke_agent / astream_agent helpers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.engine import chains


def _register_fake_agent(name: str, agent: MagicMock) -> None:
    chains._agent_registry[name] = agent


def _clear_agent(name: str) -> None:
    chains._agent_registry.pop(name, None)


def test_invoke_agent_calls_sync_invoke():
    agent = MagicMock()
    agent.invoke.return_value = {"messages": [], "ok": True}
    _register_fake_agent("test_invoke_agent", agent)
    try:
        result = chains.invoke_agent("test_invoke_agent", "hello")
        assert result["ok"] is True
        agent.invoke.assert_called_once()
        args = agent.invoke.call_args[0][0]
        assert args["messages"][0].content == "hello"
    finally:
        _clear_agent("test_invoke_agent")


@pytest.mark.asyncio
async def test_ainvoke_agent_calls_async_ainvoke():
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"messages": [], "ok": True})
    _register_fake_agent("test_ainvoke_agent", agent)
    try:
        result = await chains.ainvoke_agent("test_ainvoke_agent", "hello")
        assert result["ok"] is True
        agent.ainvoke.assert_awaited_once()
    finally:
        _clear_agent("test_ainvoke_agent")


@pytest.mark.asyncio
async def test_astream_agent_yields_tokens_and_result():
    async def _fake_astream(*_args, **_kwargs):
        yield ("messages", (MagicMock(content="Hi"), {"langgraph_node": "model"}))
        yield ("values", {"messages": [], "done": True})

    agent = MagicMock()
    agent.astream = _fake_astream
    _register_fake_agent("test_astream_agent", agent)
    try:
        chunks = []
        async for chunk in chains.astream_agent("test_astream_agent", "hello"):
            chunks.append(chunk)
        assert chunks[0] == {"type": "token", "text": "Hi", "node": "model"}
        assert chunks[-1] == {"type": "result", "state": {"messages": [], "done": True}}
    finally:
        _clear_agent("test_astream_agent")
