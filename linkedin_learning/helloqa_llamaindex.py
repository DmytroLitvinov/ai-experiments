from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType

load_dotenv()

# # changing the global default
Settings.embed_model = OpenAIEmbedding(
    model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
)

documents = [
    Document(text="Abraham Lincoln was the 16th president of the United States."),
    Document(text="Abraham Shakespeare was a Florida lottery winner in 2006."),
    Document(text="William Shakespeare married Anne Hathaway."),
]

index = VectorStoreIndex(documents)
query_engine = index.as_query_engine()
response1 = query_engine.query("Who was Shakespeare's wife?")
print(response1)

response2 = query_engine.query("Did William Shakespeare win the lottery?")
print(response2)
