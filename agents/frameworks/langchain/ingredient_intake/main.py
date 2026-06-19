# imports
from enum import Enum

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


# models
class UnitType(str, Enum):
    EMPTY = ''
    PIECES = 'pieces'
    TBSP = 'tbsp'
    CLOVES = 'cloves'
    TSP = 'tsp'


class Ingredient(BaseModel):
    name: str = Field(description="Ingredient name")
    quantity: int = Field(description="Ingredient quantity. default = 1")
    unit: UnitType = Field(description="Ingredient unit. default empty string")

class Recipe(BaseModel):
    ingredients: list[Ingredient] = Field(description="List of ingredients")


# variables
prompt_template = """
# ROLE:
You are a chef expert on extracting ingredients and its parts
# GOAL:
Your sole task is to extract the ingredients from a recipe text and format into json structure.
Follow the instructions below:
# INSTRUCTIONS:
- understand the input text
- parse the input into Recipe structure and a list of Ingredient(name, quantity, unit)
- normalize ingredient name to the singular format ('chicken breasts' -> 'chicken breast')
- provide the ingredient name without the transformation ('lemon juice' -> 'lemon')
- if the quantity is not specified, by default is 1
- if the unit is not specified, by default is empty string
- given the ingredient interpret and provide UnitType using the rules below
- UnitType: if solid small item (dried oregano), then tsp
- UnitType: if liquid items (olive oil), then tbsp 
- UnitType: if leaf-based items (garlic), then cloves
- UnitType: if solid bigger items (chicken breast), then pieces
- UnitType: if not identified, then empty string
- provide output as json in the following output format instructions
# FORMAT INSTRUCTIONS
{format_instructions}
# INPUT:
{text}
"""


# functions
# chain
def get_ingredients(text: str, model = 'llama3') -> dict:
    parser = PydanticOutputParser(pydantic_object=Recipe)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | ChatOllama(model=model) | parser
    parsed_model = chain.invoke({'text': text})
    return parsed_model.model_dump_json()


# entry point
if __name__ == "__main__":
    text = 'Season 2 chicken breasts with salt and pepper, then sear them in 2 tbsp olive oil until golden.  Lower the heat, add 3 minced garlic cloves, and deglaze with the juice of 1 lemon.  Finish with 1 tsp dried oregano and simmer for 8 minutes.'
    result = get_ingredients(text)
    print(result)
