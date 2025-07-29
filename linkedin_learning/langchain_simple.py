import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import StrOutputParser

from experiments.constants import MODEL_ENGINE

from dotenv import load_dotenv

load_dotenv()

system_prompt = "You are a helpful assistant that answers generals inquiries and assist with technical issues"

str_parser = StrOutputParser()

# basic example of how to get started with the OpenAI Chat models
# The above cell assumes that your OpenAI API key is set in your environment variables.
model = ChatOpenAI(model=MODEL_ENGINE, temperature=0.3)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


def main():
    user_input = "I want to return a pair of shoes"

    # prompt value
    prompt_value = chat_prompt.invoke({"question": user_input})
    # print(prompt_value.to_string())

    # model response
    messages = chat_prompt.format_prompt(question=user_input).to_messages()
    response = model.invoke(messages)
    print(response)

    # string output parser
    content = str_parser.invoke(response)
    print(content)

    # LCEL makes it easy to build complex chains from basic components, and supports out of the box functionality such as streaming, parallelism, and logging.


if __name__ == "__main__":
    main()
