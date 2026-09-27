import pytest
from aiohttp.test_utils import TestClient, TestServer

from pipeline_example.main import init_web_app


@pytest.fixture
async def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-key")
    monkeypatch.setenv("GROQ_API_KEY", "gq-key")
    monkeypatch.setenv("SERVER_DOMAIN", "example.com")
    prompt = tmp_path / "agent_prompt.txt"
    prompt.write_text("test prompt")
    app = init_web_app(prompt.read_text())
    test_client = TestClient(TestServer(app))
    await test_client.start_server()
    yield test_client
    await test_client.close()


@pytest.mark.asyncio
async def test_index_route(client):
    resp = await client.request("GET", "/")
    assert resp.status == 200
    body = await resp.json()
    assert body["message"] == "nlpengine"


@pytest.mark.asyncio
async def test_call_start_returns_twiml(client):
    resp = await client.post(
        "/call/start",
        data={"CallSid": "CA123", "Called": "+15551234567"},
    )
    assert resp.status == 200
    text = await resp.text()
    assert "wss://example.com/call/stream/15551234567/CA123" in text
    assert "<Stream" in text
