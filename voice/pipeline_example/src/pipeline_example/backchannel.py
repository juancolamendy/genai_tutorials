import logging
import random
from datetime import UTC, datetime, timedelta

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from pipeline_example import config

logger = logging.getLogger("phoneagent")


class BackchannelProcessor:
    def __init__(
        self,
        min_interval_seconds: int = 5,
        max_frequency_per_minute: int = 10,
        chance_factor: float = 0.8,
    ):
        self._min_interval = timedelta(seconds=min_interval_seconds)
        self._max_frequency = max_frequency_per_minute
        self._chance_factor = chance_factor
        self._backchannel_timestamps: list[datetime] = []
        self._last_backchannel_time: datetime | None = None

    def can_backchannel(self) -> bool:
        now = datetime.now(tz=UTC)
        if (
            self._last_backchannel_time
            and now - self._last_backchannel_time < self._min_interval
        ):
            return False

        self._backchannel_timestamps = [
            ts
            for ts in self._backchannel_timestamps
            if now - ts < timedelta(minutes=1)
        ]
        if len(self._backchannel_timestamps) >= self._max_frequency:
            return False

        if self._last_backchannel_time:
            time_since_last = (now - self._last_backchannel_time).total_seconds()
            min_interval_seconds = self._min_interval.total_seconds()
            if min_interval_seconds > 0:
                probability = min(
                    1.0,
                    time_since_last / (min_interval_seconds * 2),
                )
                probability *= self._chance_factor
            else:
                probability = self._chance_factor
        else:
            probability = self._chance_factor

        return random.random() < probability

    def record_backchannel(self):
        now = datetime.now(tz=UTC)
        self._last_backchannel_time = now
        self._backchannel_timestamps.append(now)


def build_backchannel_chain(
    model_name: str = config.BACKCHANNEL_LLM_MODEL_NAME,
):
    system_prompt = """You are a backchanneling response generator.
Your task is to assess whether it makes sense to include a backchanneling response for a given input text and context.
Backchanneling is a conversational response that indicates a listener's engagement with the speaker.
It can involve short verbal responses or non-verbal cues, such as "uh-huh", "I see", or "go on".

% INSTRUCTIONS:
- Evaluate whether to generate a backchanneling response based on the input and context.
- Provide a 'yes' or 'no' score to indicate your evaluation.
- If the score is 'yes', determine the best short backchanneling response (max 5 words).
- If the score is 'no', leave the backchanneling response as an empty string.
- Assign a confidence score between 0 and 1 for your decision.
- Respond with a JSON object containing 'score', 'backchannel', and 'confidence' fields.

% INPUT: {input}

% OUTPUT:
"""
    llm = ChatGroq(
        temperature=0,
        model_name=model_name,
        groq_api_key=config.GROQ_API_KEY,
    )  # type: ignore[call-arg]
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    return prompt | llm | JsonOutputParser()


async def apick_backchannel(text: str) -> str:
    chain = build_backchannel_chain()
    start_time = datetime.now(tz=UTC).timestamp()
    try:
        json_resp = await chain.ainvoke({"input": text})
    except Exception as e:  # noqa: BLE001
        logger.warning("Backchannel chain failed: %s", e)
        return ""
    elapsed = int((datetime.now(tz=UTC).timestamp() - start_time) * 1000)
    logger.debug("--- llm latency: (%sms)", elapsed)
    logger.debug(
        "--- backchanneling: text: [%s] - backchannel_response: %s",
        text,
        json_resp,
    )
    if isinstance(json_resp, dict) and json_resp.get("backchannel"):
        return str(json_resp.get("backchannel"))
    return ""
