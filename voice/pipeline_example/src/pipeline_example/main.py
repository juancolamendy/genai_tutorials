import asyncio
import logging
import traceback
from collections.abc import Mapping
from typing import Any

from aiohttp import WSMsgType, web
from twilio.twiml.voice_response import VoiceResponse

from pipeline_example import config
from pipeline_example.audio import AudioProcessor
from pipeline_example.backchannel import BackchannelProcessor, apick_backchannel
from pipeline_example.conversation import ConversationProcessor, asummarize_conversation
from pipeline_example.models import ExecState
from pipeline_example.transcript import TranscriptProcessor, aprocess_media_event
from pipeline_example.voice import VoiceProcessor, asegment_text_gen

logger = logging.getLogger("phoneagent")


# ---- Twilio event parsers ----

def get_streamsid_start_event(event: dict) -> str | None:
    start_event = event.get("start")
    if not isinstance(start_event, dict):
        return None
    stream_sid = start_event.get("streamSid")
    return stream_sid if isinstance(stream_sid, str) else None


def get_callsid_start_event(event: dict) -> str | None:
    start_event = event.get("start")
    if not isinstance(start_event, dict):
        return None
    call_sid = start_event.get("callSid")
    return call_sid if isinstance(call_sid, str) else None


def get_encoding_start_event(event: dict) -> str | None:
    start_event = event.get("start")
    if not isinstance(start_event, dict):
        return None
    fmt = start_event.get("mediaFormat")
    if not isinstance(fmt, dict):
        return None
    encoding = fmt.get("encoding")
    return encoding if isinstance(encoding, str) else None


def get_samplerate_start_event(event: dict) -> int | None:
    start_event = event.get("start")
    if not isinstance(start_event, dict):
        return None
    fmt = start_event.get("mediaFormat")
    if not isinstance(fmt, dict):
        return None
    sample_rate = fmt.get("sampleRate")
    return sample_rate if isinstance(sample_rate, int) else None


def get_media_payload(data: dict) -> str | None:
    if not isinstance(data, dict):
        return None
    media = data.get("media")
    if not isinstance(media, dict):
        return None
    payload = media.get("payload")
    return payload if isinstance(payload, str) else None


def get_mark_name(data: dict) -> str | None:
    if not isinstance(data, dict):
        return None
    mark = data.get("mark")
    if not isinstance(mark, dict):
        return None
    name = mark.get("name")
    return name if isinstance(name, str) else None


def get_called(body: Mapping[str, Any]) -> str:
    called = body.get("Called")
    if isinstance(called, str) and len(called) > 1:
        return called[1:]
    return "none"


# ---- Call lifecycle helpers ----

async def aprocess_start_event(
    state: ExecState,
    exec_config: config.ExecConfig,
    event: dict,
    audio_processor: AudioProcessor,
    voice_processor: VoiceProcessor,
):
    logger.debug("--- start event: %s", event)
    encoding = get_encoding_start_event(event)
    logger.debug("--- encoding: %s", encoding)
    if encoding is None or encoding != "audio/x-mulaw":
        raise ValueError("encoding error")
    sample_rate = get_samplerate_start_event(event)
    logger.debug("--- sample rate: %s", sample_rate)
    if sample_rate is None or sample_rate != 8000:
        raise ValueError("sample error")

    streamsid = get_streamsid_start_event(event)
    logger.debug("--- streamSid: %s", streamsid)
    if streamsid is None:
        raise ValueError("streamSid error")

    callsid = get_callsid_start_event(event)
    logger.debug("--- callsid: %s", callsid)
    expected_callsid = await state.get_callsid()
    if callsid is None or callsid != expected_callsid:
        raise ValueError("callsid error")

    audio_processor.streamsid = streamsid
    await asend_text_as_audio(
        exec_config.start_call_message,
        audio_processor,
        voice_processor,
        exec_config,
    )


async def process_mark_event(
    event: dict,
    transcript_processor: TranscriptProcessor,
    audio_processor: AudioProcessor,
):
    if transcript_processor.state is None:
        return
    mark_name = get_mark_name(event)
    if mark_name:
        await audio_processor.mark_cache.delete(mark_name)
        await transcript_processor.state.set_last_activity(asyncio.get_event_loop().time())


# ---- Response pipeline ----

async def asend_text_as_audio(
    text: str,
    audio_processor: AudioProcessor,
    voice_processor: VoiceProcessor,
    exec_config: config.ExecConfig,
):
    async for audio_chunk in voice_processor.apost_text_voice_gen(
        text,
        exec_config.voice_model_name,
        exec_config.voice_provider,
    ):
        await audio_processor.asend_audio_chunk(audio_chunk)


async def aprocess_transcript(
    transcript: str,
    conv_processor: ConversationProcessor,
    audio_processor: AudioProcessor,
    backchannel_processor: BackchannelProcessor,
    voice_processor: VoiceProcessor,
    exec_config: config.ExecConfig,
):
    if exec_config.response_delay > 0:
        await asyncio.sleep(exec_config.response_delay)

    if exec_config.backchannel_enabled:
        should_backchannel = backchannel_processor.can_backchannel()
        logger.debug("--- should_backchannel: %s", should_backchannel)
        if should_backchannel:
            backchannel = await apick_backchannel(transcript)
            if backchannel:
                backchannel += " ... ..."
                logger.debug("--- sst backchannel: %s", backchannel)
                await asend_text_as_audio(
                    backchannel,
                    audio_processor,
                    voice_processor,
                    exec_config,
                )
                backchannel_processor.record_backchannel()

    async for text in conv_processor.aprocess_gen(transcript):
        async for text_chunk in asegment_text_gen(text):
            logger.debug("--- *** text chunk: *[%s]*", text_chunk)
            await asend_text_as_audio(
                text_chunk,
                audio_processor,
                voice_processor,
                exec_config,
            )


# ---- Workers ----

async def process_in_msg_worker(
    state: ExecState,
    exec_config: config.ExecConfig,
    transcript_processor: TranscriptProcessor,
    audio_processor: AudioProcessor,
    voice_processor: VoiceProcessor,
    shutdown_event: asyncio.Event,
):
    logger.debug("--- *** start process_in_msg_worker")
    try:
        async for message in audio_processor.twilio_ws:
            if shutdown_event.is_set():
                break
            match message.type:
                case WSMsgType.TEXT:
                    event = message.json()
                    event_type = event.get("event")
                    if event_type is None:
                        continue
                    match event_type:
                        case "start":
                            await aprocess_start_event(
                                state,
                                exec_config,
                                event,
                                audio_processor,
                                voice_processor,
                            )
                        case "connected":
                            logger.debug("--- connected event: %s", event)
                        case "mark":
                            await process_mark_event(
                                event,
                                transcript_processor,
                                audio_processor,
                            )
                        case "media":
                            await aprocess_media_event(
                                event,
                                transcript_processor.deepgram_ws,
                            )
                        case "stop":
                            logger.debug("--- stop event: %s", event)
                            break
                case WSMsgType.CLOSE:
                    logger.debug("--- twilio websocket closed")
                    break
                case _:
                    logger.warning("Got unsupported message type from Twilio stream!")
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error in process_in_msg_worker: %s", e)
        logger.debug(traceback.format_exc())
    finally:
        shutdown_event.set()
        logger.debug("--- *** end process_in_msg_worker")


async def process_transcript_worker(
    transcript_processor: TranscriptProcessor,
    conv_processor: ConversationProcessor,
    audio_processor: AudioProcessor,
    backchannel_processor: BackchannelProcessor,
    voice_processor: VoiceProcessor,
    exec_config: config.ExecConfig,
    shutdown_event: asyncio.Event,
):
    if transcript_processor.deepgram_ws is None:
        return

    async def handle_transcript(transcript: str):
        await aprocess_transcript(
            transcript,
            conv_processor,
            audio_processor,
            backchannel_processor,
            voice_processor,
            exec_config,
        )

    async def handle_utterance(
        transcript: str,
        cap: int = 10,
    ):
        mark_count = await audio_processor.mark_cache.count()
        if mark_count > 0 and len(transcript) > cap:
            logger.debug(
                "--- ********** --- sending clear audio buffer: mark_cache: %d - len transcript: %d",
                mark_count,
                len(transcript),
            )
            await audio_processor.asend_clear()

    transcript_processor.on_transcript_afn = handle_transcript
    transcript_processor.on_utterance_afn = handle_utterance

    logger.debug("--- *** start process_transcript_worker")
    try:
        async for message in transcript_processor.deepgram_ws:
            if shutdown_event.is_set():
                break
            match message.type:
                case WSMsgType.TEXT:
                    await transcript_processor.aprocess_transcript_event(message)
                case WSMsgType.CLOSE:
                    logger.debug("--- process_transcript_worker - close event")
                    break
                case _:
                    logger.warning("Got unsupported message type from Deepgram!")
                    continue
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error in process_transcript_worker: %s", e)
        logger.debug(traceback.format_exc())
    finally:
        shutdown_event.set()
        logger.debug("--- *** end process_transcript_worker")


# ---- Timeout checkers ----

async def check_silence_timeout(
    transcript_processor: TranscriptProcessor,
    audio_processor: AudioProcessor,
    exec_config: config.ExecConfig,
) -> bool:
    if transcript_processor.state is None:
        return False
    mark_count = await audio_processor.mark_cache.count()
    if mark_count > 0:
        return False
    last_activity = await transcript_processor.state.get_last_activity()
    if last_activity is None:
        return False
    elapsed_time = asyncio.get_event_loop().time() - last_activity
    if elapsed_time > exec_config.silence_timeout:
        logger.debug("--- silence timeout reached, ending call.")
        return True
    return False


async def check_max_call_time(
    transcript_processor: TranscriptProcessor,
    exec_config: config.ExecConfig,
) -> bool:
    if transcript_processor.state is None:
        return False
    total_secs = await transcript_processor.state.get_total_duration()
    if total_secs > exec_config.max_call_time:
        logger.debug("--- max call time reached, ending call")
        return True
    return False


async def monitor_conversation_worker(
    transcript_processor: TranscriptProcessor,
    audio_processor: AudioProcessor,
    voice_processor: VoiceProcessor,
    exec_config: config.ExecConfig,
    shutdown_event: asyncio.Event,
    sleep_time: int = 1,
    endcall_message_sleep_time: int = 5,
):
    if transcript_processor.state is None:
        return
    try:
        while not shutdown_event.is_set():
            await transcript_processor.state.increment_total_duration()
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=sleep_time,
                )
            except TimeoutError:
                pass
            if shutdown_event.is_set():
                break
            silence_timeout = await check_silence_timeout(
                transcript_processor,
                audio_processor,
                exec_config,
            )
            max_call_time = await check_max_call_time(
                transcript_processor,
                exec_config,
            )
            if silence_timeout or max_call_time:
                if exec_config.end_call_message:
                    await asend_text_as_audio(
                        exec_config.end_call_message,
                        audio_processor,
                        voice_processor,
                        exec_config,
                    )
                    await asyncio.sleep(endcall_message_sleep_time)
                break
    finally:
        shutdown_event.set()


# ---- Orchestration ----

async def aorchestrate_workers(
    deepgram_ws,
    transcript_processor: TranscriptProcessor,
    conv_processor: ConversationProcessor,
    audio_processor: AudioProcessor,
    backchannel_processor: BackchannelProcessor,
    voice_processor: VoiceProcessor,
    state: ExecState,
    exec_config: config.ExecConfig,
):
    logger.info("Start async tasks / coroutines / workers")
    shutdown_event = asyncio.Event()
    transcript_processor.deepgram_ws = deepgram_ws

    tasks = [
        asyncio.create_task(
            process_in_msg_worker(
                state,
                exec_config,
                transcript_processor,
                audio_processor,
                voice_processor,
                shutdown_event,
            ),
            name="process_in_msg_worker",
        ),
        asyncio.create_task(
            process_transcript_worker(
                transcript_processor,
                conv_processor,
                audio_processor,
                backchannel_processor,
                voice_processor,
                exec_config,
                shutdown_event,
            ),
            name="process_transcript_worker",
        ),
        asyncio.create_task(
            monitor_conversation_worker(
                transcript_processor,
                audio_processor,
                voice_processor,
                exec_config,
                shutdown_event,
            ),
            name="monitor_conversation_worker",
        ),
    ]

    try:
        done, _pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            task.result()
    except Exception:
        logger.exception("A worker terminated with an error")
    finally:
        shutdown_event.set()
        await audio_processor.aclose()
        if deepgram_ws and not deepgram_ws.closed:
            await deepgram_ws.close()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All workers have been completed or cancelled")


# ---- Web handlers ----

async def ahandle_call_stream(request: web.Request) -> web.WebSocketResponse:
    logger.debug("--- *** start /call/stream")
    callsid = request.match_info.get("callsid")
    logger.debug("--- callsid: %s", callsid)
    called = request.match_info.get("called")
    logger.debug("--- called: %s", called)

    twilio_ws = web.WebSocketResponse()
    await twilio_ws.prepare(request)
    logger.info("Opened connection to Twilio")

    exec_config = config.ExecConfig(
        agent_prompt=request.app["agent_prompt"],
    )

    state = ExecState()
    transcript_processor = TranscriptProcessor(state)
    conv_processor = ConversationProcessor(
        model_name=config.CONVERSATION_LLM_MODEL_NAME,
        model_provider=config.CONVERSATION_LLM_PROVIDER,
        agent_prompt=exec_config.agent_prompt,
    )
    audio_processor = AudioProcessor(twilio_ws)
    backchannel_processor = BackchannelProcessor()
    voice_processor = VoiceProcessor()

    await state.set_callsid(callsid)
    await state.set_called(called)

    try:
        session, dg_connection = transcript_processor.open_stream(
            model=config.DEEPGRAM_TRANSCRIPT_MODEL_NAME,
            endpointing=config.DEEPGRAM_ENDPOINTING,
        )
        async with session, dg_connection as deepgram_ws:
            await aorchestrate_workers(
                deepgram_ws,
                transcript_processor,
                conv_processor,
                audio_processor,
                backchannel_processor,
                voice_processor,
                state,
                exec_config,
            )
    except Exception as e:  # noqa: BLE001
        logger.error("Error in ahandle_call_stream: %s", e)
        logger.debug(traceback.format_exc())
    finally:
        await audio_processor.aclose()

    try:
        logger.debug(
            "--- *** Finish conversation - total seconds %s",
            await state.get_total_duration(),
        )
        logger.debug("--- *** summarizing")
        summary = await asummarize_conversation(conv_processor.chat_history)
        logger.debug("--- summary: %s", summary)
        logger.debug("-- *** end /call/stream")
    except Exception as e:  # noqa: BLE001
        logger.error("Error in ahandle_call_stream - finalize conversation: %s", e)
        logger.debug(traceback.format_exc())

    return twilio_ws


async def ahandle_call_start(request: web.Request) -> web.Response:
    twilio_response = VoiceResponse()
    body = await request.post()
    logger.debug("--- *** /call/start: %s", body)
    raw_callsid = body.get("CallSid")
    logger.debug("--- CallSid: %s", raw_callsid)
    called = get_called(body)
    logger.debug("--- Called: %s", called)

    if isinstance(raw_callsid, str):
        host = config.SERVER_DOMAIN
        assert isinstance(host, str), "SERVER_DOMAIN must be set"
        stream_url = f"wss://{host}/call/stream/{called}/{raw_callsid}"
        logger.info("Connect to websocket URL: %s", stream_url)
        twilio_response.connect().stream(
            url=stream_url,
            track="inbound_track",
        )
    else:
        logger.error("Expected payload from Twilio with a CallSid value!")
        twilio_response.say("Something went wrong! Please try again later.")

    response = web.Response(text=str(twilio_response))
    response.content_type = "text/html"
    return response


async def ahandle_index(request: web.Request) -> web.Response:
    data = await request.post()
    logger.debug("hit index: %s", data)
    return web.json_response({"message": "nlpengine"})


def init_web_app(agent_prompt: str) -> web.Application:
    app = web.Application()
    app["agent_prompt"] = agent_prompt

    routes = [
        web.get("/", ahandle_index),
        web.post("/call/start", ahandle_call_start),
        web.get("/call/stream/{called}/{callsid}", ahandle_call_stream),
    ]
    logger.debug("--- routes: %s", routes)
    app.add_routes(routes)

    remote_url = f"https://{config.SERVER_DOMAIN}"
    logger.debug("--- Remote URL: %s", remote_url)

    return app


# ---- Entrypoint ----

def main():
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.DEBUG)

    config.validate_environment()
    agent_prompt = config.load_agent_prompt()

    web.run_app(init_web_app(agent_prompt), port=config.PORT)
