from openai_client import client
import numpy as np
import nltk


def cosine_similarity(a, b):
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return numerator / denominator


def download_nltk_data():
    resources = {
        "punkt_tab": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
    }

    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


def get_embedding(text, model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(text)

    tokens = [word.lower() for word in tokens]

    words = [word for word in tokens if word.isalpha()]

    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]

    return " ".join(stemmed_words)
