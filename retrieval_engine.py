import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import re

# -----------------------------
# 1. Load USE model (once)
# -----------------------------


class USEEmbedder:
    def __init__(
        self, model_url="https://tfhub.dev/google/universal-sentence-encoder/4"
    ):
        print("Loading Universal Sentence Encoder...")
        self.model = hub.load(model_url)

    def embed(self, texts):
        # texts: list[str]
        return self.model(texts).numpy()


# -----------------------------
# 2. Sentence Splitter
# -----------------------------


def split_sentences(text):
    """
    Very simple sentence splitter.
    Splits on period/question/exclamation, strips whitespace.
    """
    raw = text.replace("\n", " ")
    out = []
    buf = ""

    for ch in raw:
        buf += ch
        if ch in [".", "?", "!"]:
            sent = buf.strip()
            if len(sent) > 0:
                out.append(sent)
            buf = ""

    # catch trailing text without punctuation
    last = buf.strip()
    if len(last) > 0:
        out.append(last)

    return out


# -----------------------------
# 3. Cosine Similarity
# -----------------------------


def cosine_similarity(query_vec, doc_matrix):
    """
    query_vec: shape (d,)
    doc_matrix: shape (n, d)
    return: similarities shape (n,)
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    d = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-9)
    return np.dot(d, q)


# -----------------------------
# 4. Index Builder
# -----------------------------


class RetrievalIndex:
    def __init__(self, sentences, embeddings):
        """
        sentences: list[str]
        embeddings: np.ndarray shape (n, 512)
        """
        self.sentences = sentences
        self.embeddings = embeddings

    def search(self, query, embedder, top_k=3):
        """
        query: string
        embedder: USEEmbedder instance
        """
        q_vec = embedder.embed([query])[0]  # shape (512,)
        sims = cosine_similarity(q_vec, self.embeddings)
        idx = np.argsort(sims)[::-1][:top_k]  # top-k descending

        results = []
        for i in idx:
            results.append(
                {"sentence": self.sentences[i], "similarity": float(sims[i])}
            )
        return results


# -----------------------------
# 5. High-Level Build Function
# -----------------------------


def build_index(text, embedder):
    """
    text: multi-paragraph raw text
    embedder: USEEmbedder
    """
    sentences = split_sentences(text)
    embeddings = embedder.embed(sentences)
    return RetrievalIndex(sentences, embeddings)

