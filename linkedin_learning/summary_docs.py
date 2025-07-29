from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage
from llama_index.core.readers import SimpleDirectoryReader

from experiments.constants import MODEL_ENGINE

load_dotenv()

application_prompt = """You are a professional technical writer. Summarize the following internal handbook into a concise guide with the key takeaways for a new team member.

    DOCUMENT:
"""

llm = OpenAILike(
    is_chat_model=True,
    temperature=0.7,
    model=MODEL_ENGINE
)

documents = SimpleDirectoryReader("./linkedin_learning/handbook").load_data()

# documents will be a list of Documents
fulltext = "\n\n".join([doc.text for doc in documents])
textlen = len(fulltext)
print(f"Document text size is {textlen}")
if textlen > 100000:
    print("Too much text to fit in context window")
    exit()

messages = [
    ChatMessage(role="system", content=application_prompt),
    ChatMessage(role="user", content=fulltext),
]
results = llm.chat(messages)

with open("linkedin_learning/summary_docs.txt", "w") as f:
    f.write(results.message.content)