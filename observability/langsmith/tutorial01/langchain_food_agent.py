from langchain.agents import create_agent

from dotenv import load_dotenv


load_dotenv()


def get_favorite_food(name: str) -> str:
    """Get the favorite food of a person based on their name."""

    favorite_foods = {
        'Mike': 'Pizza',
        'Sara': 'Pasta',
        'Bob': 'Steak'
    }

    if name in favorite_foods:
        return favorite_foods[name]
    else:
        return 'Not found.'


agent = create_agent(model='openai:gpt-5.4', tools=[get_favorite_food])

response = agent.invoke({'messages': [{'role': 'user', 'content': "What is Bob's favorite food?"}]})

print(response['messages'][-1].content)
