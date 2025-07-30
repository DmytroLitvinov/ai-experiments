from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from experiments.constants import MODEL_ENGINE

load_dotenv()

system_template = "Translate the following from English into {language}"

model = init_chat_model(MODEL_ENGINE, model_provider="openai")

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)


def main():
    prompt = prompt_template.invoke({"language": "French", "text": "hi!"})
    print(prompt)

    response = model.invoke(prompt)
    print(response.content)


if __name__ == "__main__":
    main()
