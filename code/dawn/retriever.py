from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DawnRetriever:
    def __init__(self, model_name: str) -> None:
        self.embedder = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []

    def fit(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        embeddings = self.embedder.encode(
            [chunk["text"] for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, question: str, top_k: int) -> list[dict]:
        if self.index is None:
            raise RuntimeError("L'index FAISS n'est pas initialise.")

        query_embedding = self.embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": float(score),
                }
            )
        return results
