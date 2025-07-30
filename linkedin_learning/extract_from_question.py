from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()

class Furniture(BaseModel):
    type: str = Field(description="the type of furniture")
    style: str = Field(description="style of the furniture")
    colour: str = Field(description="colour")

furniture_request = "I'd like a blue mid century chair"

parser = PydanticOutputParser(pydantic_object=Furniture)

prompt = PromptTemplate(
    template="You are a helpful assistant extracting structured data from text. Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()},
)

_input = prompt.format_prompt(query=furniture_request)


model = ChatOpenAI()
output = model.invoke(_input.to_string())

parsed = parser.parse(output.content)
print(parsed.colour)