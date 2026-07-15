from dotenv import load_dotenv

load_dotenv()  # LANGSMITH_* vars must be set before chains run

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatAnthropic(model="claude-haiku-4-5")
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "cats"})
print(result)
