from unittest.mock import AsyncMock, Mock

import pytest

from pipeline_example.transcript import TranscriptProcessor, aprocess_media_event


def test_extract_transcript_results():
    processor = TranscriptProcessor(state=None)
    event = {
        "type": "Results",
        "channel": {"alternatives": [{"transcript": "hello world"}]},
    }
    assert processor._extract_transcript(event) == "hello world"


@pytest.mark.asyncio
async def test_on_transcript_fired_on_speech_final():
    callback = AsyncMock()
    processor = TranscriptProcessor(state=None)
    processor.on_transcript_afn = callback

    msg = AsyncMock()
    msg.json = Mock(return_value={
        "type": "Results",
        "is_final": True,
        "speech_final": True,
        "channel": {"alternatives": [{"transcript": "hello"}]},
    })
    await processor.aprocess_transcript_event(msg)

    callback.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_aprocess_media_event_skips_closed_ws():
    ws = AsyncMock()
    ws.closed = True
    await aprocess_media_event({"media": {"payload": "aGVsbG8="}}, ws)
    ws.send_bytes.assert_not_awaited()
