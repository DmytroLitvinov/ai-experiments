from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client.http import models as rest
from pydantic import BaseModel, Field


import csv
from typing import Dict, List, Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document

load_dotenv()

loader = CSVLoader(
    file_path="./linkedin_learning/docs/dataset_small.csv", source_column="title")

data = loader.load()

# Store data in vector database
qdrant = Qdrant.from_documents(
    documents=data,
    embedding=OpenAIEmbeddings(),
    location=":memory:",
    collection_name="linkedin_learning_docs",
)

# Create a retriever
retriever = qdrant.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True
)

while True:
    user_input = input("Hi im an AI librarian what can I help you with?\n")

    book_request = "You are a librarian. Help the user answer their question. Do not provide the ISBN." +\
        f"\nUser:{user_input}"
    result = qa_chain.invoke({"query": book_request})
    print(len(result['source_documents']))
    print(result["result"])