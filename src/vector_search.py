"""
vector_search.py
----------------
Builds and queries a FAISS vector index for review embeddings.
"""

import faiss
import numpy as np
import pickle
from typing import List, Tuple
from nlp_models import Embedder


class ReviewVectorSearch:
    """
    Wrapper for FAISS index to search over review embeddings.
    """

    def __init__(self, embedder: Embedder, dim: int = 384):
        """
        Args:
            embedder (Embedder): SentenceTransformer wrapper.
            dim (int): Embedding dimension (default 384 for MiniLM).
        """
        self.embedder = embedder
        self.index = faiss.IndexFlatL2(dim)
        self.reviews: List[str] = []  # store original texts

    def build_index(self, texts: List[str]):
        """
        Build FAISS index from list of review texts.

        Args:
            texts (List[str]): Reviews to embed and index.
        """
        self.reviews = texts
        embeddings = np.array(self.embedder.embed(texts)).astype("float32")
        self.index.add(embeddings)

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Query FAISS index with a search string.

        Args:
            query_text (str): User question or phrase.
            top_k (int): Number of nearest reviews to return.

        Returns:
            List[Tuple[str, float]]: [(review_text, distance), ...]
        """
        query_vec = np.array(self.embedder.embed([query_text])).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.reviews):
                results.append((self.reviews[idx], float(dist)))
        return results

    def save_index(self, path: str):
        """
        Save FAISS index and associated reviews.

        Args:
            path (str): File path prefix (without extension).
        """
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.reviews, f)

    def load_index(self, path: str):
        """
        Load FAISS index and associated reviews.

        Args:
            path (str): File path prefix (without extension).
        """
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", "rb") as f:
            self.reviews = pickle.load(f)


if __name__ == "__main__":
    # Example usage
    embedder = Embedder()
    vs = ReviewVectorSearch(embedder, dim=384)

    sample_reviews = [
        "The battery lasts forever, super impressed!",
        "Screen brightness is too low outdoors.",
        "Great sound quality but a bit overpriced.",
        "Fast shipping and amazing packaging.",
    ]

    print("Building index...")
    vs.build_index(sample_reviews)

    q = "What do people dislike about the screen?"
    results = vs.query(q, top_k=3)
    for r in results:
        print(r)

    # Save + reload demo
    vs.save_index("demo_index")
    vs.load_index("demo_index")
