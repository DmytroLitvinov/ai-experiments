from experiments.constants import embedding_model
from experiments.utils import preprocess_text, get_embedding, cosine_similarity

context_window = 5


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
