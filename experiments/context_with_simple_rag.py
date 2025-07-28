import click
import json

from experiments.constants import embedding_model, model
from experiments.openai_client import client
from experiments.system_prompts import useful_assistant_prompt
from experiments.utils import (
    preprocess_text,
    get_embedding,
    cosine_similarity,
    download_nltk_data,
)

context_window = 5
history_file_path = "context.txt"
full_history = []

global_context = [{"role": "system", "content": useful_assistant_prompt}]

download_nltk_data()

# Pre-create/clear content of file
with open(history_file_path, "w") as file:
    pass


def save_history_to_file(history):
    with open(history_file_path, "w") as f:
        f.write(json.dumps(history))


def load_history_from_file():
    with open(history_file_path, "r") as f:
        try:
            history = json.loads(f.read())
            return history
        except json.JSONDecodeError:
            return []


def sort_history(history, prompt, context_window):
    sorted_history = []
    for segment in history:
        content = segment["content"]
        preprocessed_content = preprocess_text(content)
        preprocessed_prompt = preprocess_text(prompt)

        embedding_content = get_embedding(preprocessed_content, embedding_model)
        embedding_prompt = get_embedding(preprocessed_prompt, embedding_model)

        similarity = cosine_similarity(embedding_content, embedding_prompt)

        sorted_history.append((segment, similarity))

    sorted_history = sorted(sorted_history, key=lambda x: x[1], reverse=True)

    sorted_history = [x[0] for x in sorted_history]

    return sorted_history[:context_window]


while True:
    request: str = input(
        click.style('Input: (enter "exit"/"quit" to exit the program): ', fg="green")
    )

    if request.lower() in ("exit", "quit"):
        break

    user_prompt = {"role": "user", "content": request}

    full_history = load_history_from_file()

    sorted_history = sort_history(full_history, request, context_window)
    sorted_history.append(user_prompt)

    messages = global_context + sorted_history

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=1,
    )

    # Debug output
    click.echo(
        click.style("History: ", fg="blue") + str(json.dumps(messages, indent=4))
    )

    content = response.choices[0].message.content.strip()

    # Output to user
    click.echo(click.style("Output: ", fg="yellow") + content)

    full_history.append({"role": "assistant", "content": content})

    save_history_to_file(full_history)
