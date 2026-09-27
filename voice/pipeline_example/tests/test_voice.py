import pytest

from pipeline_example.voice import VoiceProcessor, asegment_text_gen


@pytest.mark.asyncio
async def test_segment_text_splits_on_bullet_and_sentence():
    text = "Hello world. How are you? I am fine• thanks"
    chunks = [chunk async for chunk in asegment_text_gen(text)]
    assert len(chunks) == 4
    assert "Hello world." in chunks
    assert "How are you?" in chunks
    assert "I am fine•" in chunks
    assert "thanks" in chunks


@pytest.mark.asyncio
async def test_voice_processor_provider_lookup():
    vp = VoiceProcessor()
    assert "deepgram" in vp.providers
    with pytest.raises(ValueError, match="not supported"):
        async for _ in vp.apost_text_voice_gen("hi", "model", provider="unknown"):
            pass
