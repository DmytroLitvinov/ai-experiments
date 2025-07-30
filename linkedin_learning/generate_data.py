from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from experiments.constants import MODEL_ENGINE

load_dotenv()

class Person(BaseModel):
    first_name: str = Field(description="first name")
    last_name: str = Field(description="last name")
    dob: str = Field(description="date of birth")


class PeopleList(BaseModel):
    people: list[Person] = Field(description="A list of people")


model = ChatOpenAI(model=MODEL_ENGINE)
people_data = model.invoke(
    "Generate a list of 10 fake peoples information. Only return the list. Each person should have a first name, last name and date of birth.")

parser = PydanticOutputParser(pydantic_object=PeopleList)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query=people_data)

model = ChatOpenAI()
output = model.invoke(_input.to_string())

parsed = parser.parse(output.content)
print(parsed)
print(parsed.people[0].last_name)