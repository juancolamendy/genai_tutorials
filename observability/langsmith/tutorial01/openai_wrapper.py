from openai import OpenAI

from dotenv import load_dotenv

from langsmith import traceable
from langsmith.wrappers import wrap_openai


load_dotenv()


client = wrap_openai(OpenAI())


@traceable
def count_words(text: str) -> int:
    return len(text.split())


@traceable
def describe(text: str) -> str:
    word_count = count_words(text)

    response = client.chat.completions.create(
        model='gpt-5.4',
        messages=[{'role': 'user', 'content': f'Is this text long? It has {word_count} words.'}]
    )

    return response.choices[0].message.content


print(describe('This is an awesome crash course because you will truly learn important skills.'))
