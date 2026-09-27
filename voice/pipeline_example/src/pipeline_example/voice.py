import logging
import re
import time

from aiohttp import ClientSession

from pipeline_example import config

logger = logging.getLogger("phoneagent")


class VoiceProcessor:
    def __init__(self):
        self.providers = {
            "deepgram": self._apost_text_voice_deepgram_gen,
        }

    async def _apost_text_voice_deepgram_gen(
        self,
        text: str,
        model: str,
        encoding: str = "mulaw",
        sample_rate: int = 8000,
        chunk_size: int = 1024,
        container: str = "none",
    ):
        url = (
            f"https://api.deepgram.com/v1/speak?model={model}"
            f"&encoding={encoding}&sample_rate={sample_rate}&container={container}"
        )
        headers = {
            "Authorization": f"Token {config.DEEPGRAM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        async with ClientSession() as session:
            try:
                first_byte = False
                start_time = time.time()
                async with session.post(url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for chunk in response.content.iter_chunked(chunk_size):
                        if not first_byte:
                            first_byte = True
                            elapsed = int((time.time() - start_time) * 1000)
                            logger.debug(
                                "--- text2audio first-byte latency (%sms)", elapsed
                            )
                            if elapsed > 500:
                                logger.debug(
                                    "--- ********** --- text2audio latency above 500ms"
                                )
                        yield chunk
            except Exception as e:  # noqa: BLE001
                logger.error("An error occurred: %s", e)

    async def apost_text_voice_gen(
        self,
        text: str,
        model: str,
        provider: str = "deepgram",
        **kwargs,
    ):
        provider_fn = self.providers.get(provider)
        if not provider_fn:
            raise ValueError(f"Provider {provider} not supported.")
        async for chunk in provider_fn(text, model, **kwargs):
            yield chunk


async def asegment_text_gen(
    text: str,
    boundary: str = r"(?<=[•.!?])\s+",
):
    def is_valid_text_chunk(chunk: str) -> bool:
        return chunk != "•"

    matches = list(re.finditer(boundary, text))
    boundaries_indices = [m.start() for m in matches]

    start = 0
    for boundary_index in boundaries_indices:
        chunk = text[start : boundary_index + 1].strip()
        if is_valid_text_chunk(chunk):
            yield chunk
        start = boundary_index + 1

    remaining = text[start:].strip()
    if is_valid_text_chunk(remaining):
        yield remaining
