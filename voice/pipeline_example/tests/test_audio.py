from unittest.mock import AsyncMock

import pytest

from pipeline_example.audio import AudioProcessor, MarkCache


class TestMarkCache:
    @pytest.mark.asyncio
    async def test_put_delete_count_exists(self):
        cache = MarkCache()
        await cache.put("a")
        assert await cache.count() == 1
        assert await cache.exists("a")
        await cache.delete("a")
        assert await cache.count() == 0
        assert not await cache.exists("a")


class TestAudioProcessor:
    @pytest.fixture
    def mock_ws(self):
        ws = AsyncMock()
        ws.closed = False
        return ws

    @pytest.mark.asyncio
    async def test_send_audio_forwards_media_and_mark(self, mock_ws):
        processor = AudioProcessor(mock_ws)
        processor.streamsid = "MX123"

        await processor.asend_audio_chunk(b"audio")

        assert mock_ws.send_json.call_count == 2
        media_call = mock_ws.send_json.call_args_list[0].args[0]
        mark_call = mock_ws.send_json.call_args_list[1].args[0]
        assert media_call["event"] == "media"
        assert media_call["streamSid"] == "MX123"
        assert "payload" in media_call["media"]
        assert mark_call["event"] == "mark"
        assert await processor.mark_cache.count() == 1

    @pytest.mark.asyncio
    async def test_no_send_when_ws_closed(self, mock_ws):
        mock_ws.closed = True
        processor = AudioProcessor(mock_ws)
        processor.streamsid = "MX123"
        await processor.asend_audio_chunk(b"audio")
        mock_ws.send_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_out_of_order_buffering(self, mock_ws):
        processor = AudioProcessor(mock_ws)
        processor.streamsid = "MX123"

        await processor.asend_audio_chunk(b"second", index=2)
        assert len(processor._audio_buffer) == 1
        await processor.asend_audio_chunk(b"first", index=1)
        assert len(processor._audio_buffer) == 0
        assert mock_ws.send_json.call_count == 4
