import click
import streamlit as st

from experiments.constants import MODEL_ENGINE
from openai_client import client

# Constants
MESSAGE_SYSTEM = " You are a skilled stand-up comedian with a knack for telling 1-2 sentence funny stories."
messages = [{"role": "system", "content": MESSAGE_SYSTEM}]


def to_dict(obj):
    return {
        "content": obj.content,
        "role": obj.role,
    }


def print_messages(messages):
    messages = [message for message in messages if message["role"] != "system"]
    for message in messages:
        role = "Bot" if message["role"] == "assistant" else "You"
        click.echo(click.style(f"{role}: ", fg="blue") + message["role"])
    return messages


def generate_chat_completion(user_input=""):
    messages.append({"role": "user", "content": user_input})
    completion = client.chat.completions.create(
        model=MODEL_ENGINE,
        messages=messages,
    )
    message = completion.choices[0].message
    messages.append(to_dict(message))
    print_messages(messages)
    return message.content


###########################
### Streamlit App  ########
###########################
st.title("😂 Funny Chatbot App")

# User input
with st.form("user_form", clear_on_submit=True):
    user_input = st.text_input("Type something")
    submit_button = st.form_submit_button(label="Send")

if submit_button:
    with st.spinner("Wait for it..."):
        completion = generate_chat_completion(user_input)
        st.write(completion)
