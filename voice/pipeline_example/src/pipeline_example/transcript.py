import base64
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import ClientSession, ClientWebSocketResponse, WSMessage

from pipeline_example import config
from pipeline_example.models import ExecState

logger = logging.getLogger("phoneagent")

OnTranscriptFn = Callable[[str], Awaitable[None]]
OnUtteranceFn = Callable[[str], Awaitable[None]]


class TranscriptProcessor:
    def __init__(self, state: ExecState | None):
        self._transcription_buffer: list[str] = []
        self.state = state
        self.deepgram_ws: ClientWebSocketResponse | None = None
        self.on_transcript_afn: OnTranscriptFn | None = None
        self.on_utterance_afn: OnUtteranceFn | None = None

    def _extract_transcript(self, event: dict) -> str:
        if event.get("type") != "Results":
            return ""
        try:
            transcript = event["channel"]["alternatives"][0]["transcript"]
            return transcript if isinstance(transcript, str) else ""
        except (KeyError, IndexError):
            return ""

    def _get_full_transcript(self) -> str:
        full_transcript = " ".join(self._transcription_buffer).strip()
        self._transcription_buffer.clear()
        return full_transcript

    def open_stream(
        self,
        model: str,
        encoding: str = "mulaw",
        sample_rate: int = 8000,
        endpointing: int = 200,
        utterance_end_ms: int = 1000,
        punctuate: str = "true",
        interim_results: str = "true",
        no_delay: str = "true",
        channels: int = 1,
        language: str = "en",
        detect_language: str = "false",
        profanity_filter: str = "true",
    ) -> tuple[ClientSession, Any]:
        session = ClientSession()
        headers = {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}
        params: dict[str, str | int] = {
            "model": model,
            "encoding": encoding,
            "sample_rate": sample_rate,
            "endpointing": endpointing,
            "utterance_end_ms": utterance_end_ms,
            "punctuate": punctuate,
            "interim_results": interim_results,
            "no_delay": no_delay,
            "channels": channels,
            "language": language,
            "detect_language": detect_language,
            "profanity_filter": profanity_filter,
        }
        logger.debug("--- deepgram conn params: %s", params)
        dg_connection = session.ws_connect(
            "wss://api.deepgram.com/v1/listen",
            headers=headers,
            params=params,
        )
        return session, dg_connection

    async def aprocess_transcript_event(self, message: WSMessage):
        event = message.json()
        event_type = event.get("type", "")

        if event_type == "UtteranceEnd":
            if self._transcription_buffer:
                full_transcript = self._get_full_transcript()
                if self.on_transcript_afn is not None:
                    await self.on_transcript_afn(full_transcript)
            return

        if event_type != "Results":
            return

        transcript = self._extract_transcript(event)
        if not transcript:
            return

        speech_final = event.get("speech_final", False)
        is_final = event.get("is_final", False)

        if self.state:
            await self.state.set_last_activity(time.time())

        if is_final:
            self._transcription_buffer.append(transcript)
            if speech_final and self.on_transcript_afn is not None:
                await self.on_transcript_afn(self._get_full_transcript())
            return

        if self.on_utterance_afn is not None:
            await self.on_utterance_afn(transcript)


async def aprocess_media_event(
    event: dict,
    deepgram_ws: ClientWebSocketResponse | None,
):
    payload = event.get("media", {}).get("payload")
    if not payload or deepgram_ws is None or deepgram_ws.closed:
        return
    decoded = base64.b64decode(payload)
    if isinstance(decoded, bytes):
        await deepgram_ws.send_bytes(decoded)
    elif isinstance(decoded, str):
        await deepgram_ws.send_str(decoded)
    else:
        logger.warning("Got unsupported message datatype from Twilio stream.")
