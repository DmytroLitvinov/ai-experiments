from datetime import datetime, timedelta
from typing import List
import math
import faiss
import os

import logging

from dotenv import load_dotenv
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from termcolor import colored
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

load_dotenv()


USER_NAME = "Future.Agent"  # The name you want to use when interviewing the agent.

LLM = ChatOpenAI(max_tokens=1500)  # Can be any LLM you want.


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )

alexis_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
)

# Defining the Generative Agent: Alexis
alexis = GenerativeAgent(
    name="Alexis",
    age=30,
    traits="curious, creative writer, world traveler",  # Persistent traits of Alexis
    status="exploring the intersection of technology and storytelling",  # Current status of Alexis
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=alexis_memory,
)

# The current "Summary" of a character can't be made because the agent hasn't made
# any observations yet.
print(alexis.get_summary())


# We can add memories directly to the memory object

alexis_observations = [
    "Alexis recalls her morning walk in the park",
    "Alexis feels excited about the new book she started reading",
    "Alexis remembers her conversation with a close friend",
    "Alexis thinks about the painting she saw at the art gallery",
    "Alexis is planning to learn a new recipe for dinner",
    "Alexis is looking forward to her weekend trip",
    "Alexis contemplates her goals for the month."
]

for observation in alexis_observations:
    alexis.memory.add_memory(observation)



# We will see how this summary updates after more observations to create a more rich description.
print(alexis.get_summary(force_refresh=True))


def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]

print(interview_agent(alexis, "What do you like to do?"))


# Let's give Alexa a series of observations to reflect on her day
# Adding observations to Alexis' memory
alexis_observations_day = [
    "Alexis starts her day with a refreshing yoga session.",
    "Alexis spends time writing in her journal.",
    "Alexis experiments with a new recipe she found online.",
    "Alexis gets lost in her thoughts while gardening.",
    "Alexis decides to call her grandmother for a heartfelt chat.",
    "Alexis relaxes in the evening by playing her favorite piano pieces.",
]

for observation in alexis_observations_day:
    alexis.memory.add_memory(observation)


# Let's observe how Alexis's day influences her memory and character
for i, observation in enumerate(alexis_observations_day):
    _, reaction = alexis.generate_reaction(observation)
    print(colored(observation, "green"), reaction)
    if ((i + 1) % len(alexis_observations_day)) == 0:
        print("*" * 40)
        print(
            colored(
                f"After these observations, Alexis's summary is:\n{alexis.get_summary(force_refresh=True)}",
                "blue",
            )
        )
        print("*" * 40)


def calculateDaysBetweenDates(start_date, end_date):
    delta = end_date - start_date
    return delta.days