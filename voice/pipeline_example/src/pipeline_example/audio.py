import asyncio
import base64
import logging
import uuid

from aiohttp import web

logger = logging.getLogger("phoneagent")


async def _safe_send_json(ws: web.WebSocketResponse | None, payload: dict) -> None:
    if ws is None or getattr(ws, "closed", False):
        logger.warning("Skipping send: websocket is closed")
        return
    await ws.send_json(payload)


class MarkCache:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._marks: dict[str, bool] = {}

    async def put(self, key: str):
        async with self._lock:
            self._marks[key] = True

    async def delete(self, key: str):
        async with self._lock:
            self._marks.pop(key, None)

    async def count(self) -> int:
        async with self._lock:
            return len(self._marks)

    async def exists(self, key: str) -> bool:
        async with self._lock:
            return key in self._marks


class AudioProcessor:
    def __init__(self, twilio_ws: web.WebSocketResponse):
        self.streamsid: str | None = None
        self.twilio_ws = twilio_ws
        self.mark_cache = MarkCache()
        self._audio_index = 1
        self._audio_buffer: dict[int, bytes] = {}

    async def _aforward_audio_chunk(self, chunk: bytes):
        if not self.streamsid:
            return

        encoded_chunk = base64.b64encode(chunk).decode("utf-8")
        await _safe_send_json(
            self.twilio_ws,
            {
                "event": "media",
                "streamSid": self.streamsid,
                "media": {"payload": encoded_chunk},
            },
        )

        mark_label = str(uuid.uuid4())
        await _safe_send_json(
            self.twilio_ws,
            {
                "event": "mark",
                "streamSid": self.streamsid,
                "mark": {"name": mark_label},
            },
        )
        await self.mark_cache.put(mark_label)

    async def asend_audio_chunk(self, audio: bytes, index: int | None = None):
        if not self.streamsid:
            return
        if index is None or index == self._audio_index:
            await self._aforward_audio_chunk(audio)
            self._audio_index += 1
            while self._audio_index in self._audio_buffer:
                buffered = self._audio_buffer.pop(self._audio_index)
                await self._aforward_audio_chunk(buffered)
                self._audio_index += 1
        else:
            self._audio_buffer[index] = audio

    async def asend_clear(self):
        if not self.streamsid:
            return
        await _safe_send_json(
            self.twilio_ws,
            {"event": "clear", "streamSid": self.streamsid},
        )

    async def aclose(self):
        if self.twilio_ws and not self.twilio_ws.closed:
            await self.twilio_ws.close()
