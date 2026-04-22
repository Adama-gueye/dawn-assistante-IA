from __future__ import annotations

import unicodedata

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

        candidate_k = min(max(top_k * 5, 15), len(self.chunks))
        scores, indices = self.index.search(query_embedding, candidate_k)
        query_tokens = self._tokenize(question)

        ranked_results: list[dict] = []
        seen_keys: set[tuple[str, int, int]] = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            lexical_score = self._lexical_overlap_score(query_tokens, chunk)
            combined_score = float(score) + lexical_score
            ranked_results.append(self._build_result(chunk, combined_score))
            seen_keys.add((chunk["source_path"], chunk["page"], chunk["chunk_id"]))

        lexical_candidates = sorted(
            self.chunks,
            key=lambda chunk: self._lexical_overlap_score(query_tokens, chunk),
            reverse=True,
        )[:candidate_k]

        for chunk in lexical_candidates:
            key = (chunk["source_path"], chunk["page"], chunk["chunk_id"])
            if key in seen_keys:
                continue
            lexical_score = self._lexical_overlap_score(query_tokens, chunk)
            if lexical_score <= 0:
                continue
            ranked_results.append(self._build_result(chunk, lexical_score))

        ranked_results.sort(key=lambda item: item["score"], reverse=True)
        return ranked_results[:top_k]

    def _lexical_overlap_score(self, query_tokens: set[str], chunk: dict) -> float:
        combined_text = f"{chunk['source_name']} {chunk['source_path']} {chunk['text']}"
        chunk_tokens = self._tokenize(combined_text)
        if not query_tokens or not chunk_tokens:
            return 0.0

        overlap = len(query_tokens & chunk_tokens)
        if overlap == 0:
            return 0.0

        exact_phrase_bonus = 0.0
        normalized_text = self._normalize_text(combined_text)
        for token in query_tokens:
            if token in normalized_text:
                exact_phrase_bonus += 0.03

        path_bonus = self._path_specialty_bonus(query_tokens, chunk["source_path"])
        general_penalty = self._general_source_penalty(query_tokens, chunk["source_path"])

        return overlap * 0.12 + exact_phrase_bonus + path_bonus - general_penalty

    def _build_result(self, chunk: dict, score: float) -> dict:
        return {
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source_name": chunk["source_name"],
            "source_path": chunk["source_path"],
            "score": float(score),
        }

    def _tokenize(self, text: str) -> set[str]:
        ascii_text = self._normalize_text(text)
        cleaned = []
        for char in ascii_text:
            cleaned.append(char if char.isalnum() else " ")
        return {token for token in "".join(cleaned).split() if len(token) >= 4}

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        return normalized.encode("ascii", "ignore").decode("ascii")

    def _path_specialty_bonus(self, query_tokens: set[str], source_path: str) -> float:
        normalized_path = self._normalize_text(source_path)
        bonus = 0.0

        specialty_keywords = {
            "anemie": ["anemie"],
            "paludisme": ["paludisme"],
            "accouchement": ["accouchement", "travail"],
            "grossesse": ["consultation_prenatale", "gyneco_obstetrique"],
            "preeclampsie": ["hta_preeclampsie_eclampsie"],
            "eclampsie": ["hta_preeclampsie_eclampsie"],
            "hemorragie": ["hemorragies_obstetricales"],
            "respiratoire": ["detresse_respiratoire"],
            "deshydratation": ["diarrhee_deshydratation"],
            "malnutrition": ["nutrition_malnutrition"],
        }

        for token, path_markers in specialty_keywords.items():
            if token in query_tokens and any(marker in normalized_path for marker in path_markers):
                bonus += 0.45

        return bonus

    def _general_source_penalty(self, query_tokens: set[str], source_path: str) -> float:
        normalized_path = self._normalize_text(source_path)
        if "communs" not in normalized_path:
            return 0.0

        specialty_query_tokens = {
            "anemie",
            "paludisme",
            "accouchement",
            "grossesse",
            "preeclampsie",
            "eclampsie",
            "hemorragie",
            "respiratoire",
            "deshydratation",
            "malnutrition",
        }
        if query_tokens & specialty_query_tokens:
            return 0.18
        return 0.0
