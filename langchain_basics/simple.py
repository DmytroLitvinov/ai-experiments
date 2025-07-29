from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from experiments.constants import MODEL_ENGINE

load_dotenv()

model = init_chat_model(MODEL_ENGINE, model_provider="openai")

def main():
    messages = [
        SystemMessage("Translate the following from English into Italian"),
        HumanMessage("hi!"),
    ]

    resp = model.invoke(messages)
    print(resp.content)


if __name__ == "__main__":
    main()
