import logging
import time
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages.utils import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import ConfigDict

from pipeline_example import config

logger = logging.getLogger("phoneagent")


class BoundedChatMessageHistory(InMemoryChatMessageHistory):
    model_config = ConfigDict(extra="allow")

    def __init__(self, max_messages: int = config.MAX_CHAT_HISTORY_MESSAGES):
        super().__init__()
        self.max_messages = max_messages

    def add_message(self, message):
        super().add_message(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]


class ConversationProcessor:
    def __init__(
        self,
        model_name: str,
        model_provider: str,
        agent_prompt: str,
        temperature: int = 0,
        max_tokens: int = 300,
    ):
        self.session_id = str(uuid.uuid4())
        self.chat_history = BoundedChatMessageHistory()

        system_prompt = self._build_system_prompt(agent_prompt=agent_prompt)

        llm = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "\n%QUESTION:\n{input}"),
        ])

        base_chain = prompt | llm | StrOutputParser()

        self.conversation_chain = RunnableWithMessageHistory(
            base_chain,
            lambda _session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def _build_system_prompt(
        self,
        agent_name: str = config.AGENT_NAME,
        agent_prompt: str = "",
    ) -> str:
        return f"""% ROLE:
You are {agent_name}, an AI Phone Agent/Voice Assistant for a business (see BUSINESS INFORMATION below).

% TASK:
Your task is to manage phone conversations with callers maintaining a [friendly/professional/helpful] tone.

% INSTRUCTIONS:
- Be concise: Respond succinctly, addressing one topic at most
- Embrace variety: Use diverse language and rephrasing to enhance clarity without repeating content
- Be conversational: Use everyday language
- Be proactive: Lead the conversation, often wrapping up with a question or next-step suggestion
- Avoid multiple questions in a single response
- Get clarity: If the caller only partially answers a question, or if the answer is unclear, keep asking to get clarity
- If caller asks something you do not know, check the CONTEXT and CHAT_HISTORY first. If you still do not know, let them know you don't have the answer and do not make up answers
- Add '•' symbol every 15-20 words at natural pauses for text-to-speech conversion
- Execute the conversation guided by CONVERSATION GOAL, CONVERSATION FLOW and CONVERSATION EXAMPLES
- Execute only one step at a time in CONVERSATION FLOW. One step is a numbered line, for example, 1. 2. 3.
- [CONDITION] It is a conditional block. Use it to guide the conversation based on the callers' intent
- <variable> It is a variable block. It should always be substituted by the information the caller has provided

{agent_prompt}

% CHAT_HISTORY:
"""

    async def aprocess_gen(self, text: str):
        logger.debug("--- *** llm input: *[%s]*", text)
        start_time = time.time()
        output = await self.conversation_chain.ainvoke(
            {"input": text},
            config={"configurable": {"session_id": self.session_id}},
        )
        output = output.strip()
        elapsed = int((time.time() - start_time) * 1000)
        logger.debug("--- llm latency: (%sms)", elapsed)
        if elapsed > 900:
            logger.debug("--- ********** --- llm latency above 900ms")
        logger.debug("--- *** llm output: *[%s]*", output)
        yield output


def build_summary_chain(
    model_provider: str = config.CONVERSATION_LLM_PROVIDER,
    model_name: str = config.CONVERSATION_LLM_MODEL_NAME,
):
    system_prompt = """You are an expert in summarizing phone calls.
% TASK:
Your task is to generate a concise summary based on the provided phone call conversation as a context.
Follow the instructions.

% INSTRUCTIONS:
- Understand the key takeaways of the conversation
- Identify key entities such as names, addresses, phone numbers, dates, and times
- Capture essential conversation details, outcomes, and any action items or next steps
- Generate a summary in 3-4 sentences using the takeaways and entities

% CONTEXT:
{context}

% SUMMARY:
"""
    llm = init_chat_model(
        model=model_name,
        model_provider=model_provider,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    return prompt | llm | StrOutputParser()


async def asummarize_conversation(chat_history: BaseChatMessageHistory) -> str:
    context = get_buffer_string(chat_history.messages)
    logger.debug("--- context: %s", context)
    chain = build_summary_chain()
    start_time = time.time()
    resp = await chain.ainvoke({"context": context})
    elapsed = int((time.time() - start_time) * 1000)
    logger.debug("--- llm latency: (%sms)", elapsed)
    return str(resp)
