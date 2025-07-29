import streamlit as st

from experiments.constants import MODEL_ENGINE
from openai_client import client


def moderate(user_input):
    response = client.moderations.create(input=user_input)
    return response.results[0].flagged


def generate_chat_completion(user_input, messages):
    flagged = moderate(user_input)
    print(f"Flagged: {flagged}")
    if flagged:
        return ":red[Your comment has been flagged as inappropriate.]"
    completion = client.chat.completions.create(
        model=MODEL_ENGINE,
        messages=messages,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return completion.choices[0].message.content


st.title("🤖 Chatbot App")
chat_placeholder = st.empty()


def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]


def start_chat():
    # Display chat messages from history on app rerun
    with chat_placeholder.container():
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from Chat models
        response = generate_chat_completion(prompt, st.session_state.messages)

        # message_placeholder.markdown(response)
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    init_chat_history()
    start_chat()
