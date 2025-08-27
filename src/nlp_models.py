"""
nlp_models.py
-------------
Wrappers for Hugging Face transformer models used in the project:
- Sentiment analysis (BERT-5star)
- Summarization (BART-large-CNN)
- Embeddings (SentenceTransformers)
"""

from transformers import pipeline
from sentence_transformers import SentenceTransformer


class SentimentAnalyzer:
    """
    Wrapper for Amazon 1–5 star sentiment analysis.
    Model: nlptown/bert-base-multilingual-uncased-sentiment
    """

    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        self.pipe = pipeline("sentiment-analysis", model=model_name)

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a given review text.

        Args:
            text (str): Review text.

        Returns:
            dict: {"label": str, "score": float}
        """
        if not text.strip():
            return {"label": "unknown", "score": 0.0}
        return self.pipe(text)[0]


class Summarizer:
    """
    Wrapper for abstractive summarization of review sets.
    Model: facebook/bart-large-cnn
    """

    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_input_tokens: int = 800):
        self.pipe = pipeline("summarization", model=model_name)
        self.max_input_tokens = max_input_tokens  # safe buffer under 1024

    def summarize(self, text: str, max_len: int = 130, min_len: int = 30) -> str:
        """
        Summarize long text safely by chunking.
        """
        if not text.strip():
            return ""

        # Split into chunks (very simple split by words)
        words = text.split()
        chunks = [
            " ".join(words[i:i + self.max_input_tokens])
            for i in range(0, len(words), self.max_input_tokens)
        ]

        # Summarize each chunk separately
        summaries = []
        for chunk in chunks:
            result = self.pipe(chunk, max_length=max_len, min_length=min_len, do_sample=False)
            summaries.append(result[0]["summary_text"])

        # Optionally, run a "final summary of summaries"
        if len(summaries) > 1:
            joined = " ".join(summaries)
            result = self.pipe(joined, max_length=max_len, min_length=min_len, do_sample=False)
            return result[0]["summary_text"]

        return summaries[0]


class Embedder:
    """
    Wrapper for sentence embeddings.
    Model: multi-qa-MiniLM-L6-cos-v1
    """

    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (list[str]): Input sentences/reviews.

        Returns:
            list[list[float]]: Embedding vectors.
        """
        if not texts:
            return []
        return self.model.encode(texts, convert_to_numpy=True).tolist()


if __name__ == "__main__":
    # Example usage
    sa = SentimentAnalyzer()
    sm = Summarizer()
    em = Embedder()

    test_text = "The battery life is amazing, but the screen is too dim."

    print("Sentiment:", sa.predict(test_text))
    print("Summary:", sm.summarize(test_text))
    print("Embedding vector (dim):", len(em.embed([test_text])[0]))
