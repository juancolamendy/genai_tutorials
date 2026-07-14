from langfuse import get_client
from langfuse.langchain import CallbackHandler

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatAnthropic(model="claude-haiku-4-5")
chain = prompt | llm | StrOutputParser()

result = chain.invoke(
    {"topic": "cats"},
    config={"callbacks": [langfuse_handler]},
)
print(result)

langfuse.flush()
