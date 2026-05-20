from __future__ import annotations

import unicodedata

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


TOPIC_RULES = {
    "anemie": {
        "query_markers": {"anemie", "anemies"},
        "content_markers": {
            "anemie",
            "anemies",
            "hemoglobine",
            "hematies",
            "hemoly",
            "thalassem",
            "drapan",
            "ferritine",
            "fer",
        },
    },
    "paludisme": {
        "query_markers": {"paludisme", "impalude", "palustre"},
        "content_markers": {"paludisme", "impalude", "palustre", "parasitemie"},
    },
    "accouchement": {
        "query_markers": {"accouchement", "travail"},
        "content_markers": {"accouchement", "travail", "partogramme", "delivrance"},
    },
    "grossesse": {
        "query_markers": {"grossesse", "enceinte", "prenatal", "prenatale"},
        "content_markers": {"grossesse", "enceinte", "prenatal", "prenatale"},
    },
    "preeclampsie": {
        "query_markers": {"preeclampsie", "eclampsie"},
        "content_markers": {"preeclampsie", "eclampsie", "hta", "convulsion"},
    },
    "hemorragie": {
        "query_markers": {"hemorragie", "saignement"},
        "content_markers": {"hemorragie", "saignement", "metrorragie"},
    },
    "respiratoire": {
        "query_markers": {"respiratoire", "detresse", "asthme", "pneumonie"},
        "content_markers": {"respiratoire", "detresse", "asthme", "pneumonie", "toux"},
    },
    "deshydratation": {
        "query_markers": {"deshydratation", "diarrhee"},
        "content_markers": {"deshydratation", "diarrhee", "selles", "rehydratation"},
    },
    "malnutrition": {
        "query_markers": {"malnutrition", "nutrition"},
        "content_markers": {"malnutrition", "nutrition", "amaigrissement", "oedeme"},
    },
}

FOCUS_RULES = {
    "severity_signs": {
        "query_markers": {"gravite", "grave", "signes"},
        "content_markers": {
            "grave",
            "gravite",
            "signes de gravite",
            "convulsion",
            "coma",
            "prostration",
            "detresse",
            "respiratoire",
            "choc",
            "anemie severe",
            "hypoglycemie",
            "acidose",
            "insuffisance renale",
            "saignement",
            "ictre",
            "trouble de conscience",
        },
        "preferred_source_markers": {
            "grave",
            "protocole",
            "transcription",
            "signes de gravite",
            "manifestations cliniques",
            "manifestations biologiques",
        },
        "negative_markers": {
            "moustiquaires",
            "rideaux",
            "pyrrethrinoide",
            "pyrethrinoide",
            "pulverisation",
            "insecticide",
            "intermittent",
            "prevention",
            "lutte antivectorielle",
            "zones avec transmission active",
        },
    },
    "malaria_diagnosis": {
        "query_markers": {"paludisme", "tdr", "goutte epaisse", "diagnostic"},
        "content_markers": {
            "paludisme",
            "diagnostic",
            "tdr",
            "test diagnostique",
            "goutte epaisse",
            "frottis",
            "parasitemie",
            "plasmodium",
            "fievre",
            "enfant",
        },
        "preferred_source_markers": {
            "conduite a tenir",
            "prise en charge",
            "traitement",
            "diagnostic",
            "tdr",
            "test diagnostique rapide",
            "goutte epaisse",
        },
        "negative_markers": {
            "moustiquaires",
            "rideaux",
            "pyrrethrinoide",
            "pyrethrinoide",
            "pulverisation",
            "insecticide",
            "prevention",
            "lutte antivectorielle",
            "intermittent",
            "chimioprophylaxie",
            "g6pd",
            "primaquine",
            "radical",
            "ovale",
            "cynomolgi",
        },
    },
}

GENERIC_QUERY_TOKENS = {
    "chez",
    "enfant",
    "enfants",
    "presentant",
    "presence",
    "persistante",
    "persistant",
    "quels",
    "quelles",
    "critere",
    "criteres",
    "permettant",
    "diagnostiquer",
    "diagnostic",
    "doivent",
    "demandes",
    "demandes",
    "examens",
    "complementaires",
    "prise",
    "charge",
    "urgente",
    "urgent",
    "necessaire",
    "necessaires",
    "doit",
    "dans",
    "leurs",
    "avec",
    "pour",
    "trois",
    "ans",
    "quel",
    "quelle",
}


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
        active_topics = self._detect_active_topics(query_tokens)
        active_focuses = self._detect_active_focuses(question, query_tokens)

        ranked_results: list[dict] = []
        seen_keys: set[tuple[str, int, int]] = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            lexical_score = self._lexical_overlap_score(query_tokens, chunk, active_topics, active_focuses)
            combined_score = float(score) + lexical_score
            ranked_results.append(
                self._build_result(
                    chunk,
                    combined_score,
                    semantic_score=float(score),
                    lexical_score=lexical_score,
                )
            )
            seen_keys.add((chunk["source_path"], chunk["page"], chunk["chunk_id"]))

        lexical_candidates = sorted(
            self.chunks,
            key=lambda chunk: self._lexical_overlap_score(query_tokens, chunk, active_topics, active_focuses),
            reverse=True,
        )[:candidate_k]

        for chunk in lexical_candidates:
            key = (chunk["source_path"], chunk["page"], chunk["chunk_id"])
            if key in seen_keys:
                continue
            lexical_score = self._lexical_overlap_score(query_tokens, chunk, active_topics, active_focuses)
            if lexical_score <= 0:
                continue
            ranked_results.append(
                self._build_result(
                    chunk,
                    lexical_score,
                    semantic_score=0.0,
                    lexical_score=lexical_score,
                )
            )

        ranked_results.sort(key=lambda item: item["score"], reverse=True)
        if not active_topics:
            return self._prioritize_focus_results(ranked_results, active_focuses, top_k)

        topic_matched = self._collect_topic_matched_results(ranked_results, active_topics)
        topic_matched = self._prioritize_focus_results(topic_matched, active_focuses, top_k)
        if len(topic_matched) >= top_k:
            return topic_matched[:top_k]

        fallback_results = [
            item
            for item in ranked_results
            if not self._chunk_matches_active_topics(item, active_topics)
        ]
        fallback_results = self._prioritize_focus_results(fallback_results, active_focuses, top_k)
        return (topic_matched + fallback_results)[:top_k]

    def _collect_topic_matched_results(self, ranked_results: list[dict], active_topics: set[str]) -> list[dict]:
        strict_topic_matches = [
            item
            for item in ranked_results
            if self._chunk_matches_active_topics(item, active_topics)
            and not self._chunk_mentions_competing_topics(item, active_topics)
        ]
        if strict_topic_matches:
            return strict_topic_matches

        return [
            item for item in ranked_results if self._chunk_matches_active_topics(item, active_topics)
        ]

    def _prioritize_focus_results(
        self,
        ranked_results: list[dict],
        active_focuses: set[str],
        top_k: int,
    ) -> list[dict]:
        if not active_focuses:
            return ranked_results[:top_k]

        strong_focus_results = [
            item for item in ranked_results if self._chunk_strongly_matches_focus(item, active_focuses)
        ]
        if len(strong_focus_results) >= top_k:
            return strong_focus_results[:top_k]

        neutral_or_positive_results = [
            item for item in ranked_results if not self._chunk_has_negative_focus_markers(item, active_focuses)
        ]
        if neutral_or_positive_results:
            return neutral_or_positive_results[:top_k]

        return ranked_results[:top_k]

    def _lexical_overlap_score(
        self,
        query_tokens: set[str],
        chunk: dict,
        active_topics: set[str],
        active_focuses: set[str],
    ) -> float:
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
        topic_bonus = self._topic_alignment_bonus(chunk, active_topics)
        overlap_bonus = self._focus_overlap_bonus(query_tokens, chunk_tokens)
        language_bonus = self._language_alignment_bonus(query_tokens, chunk)
        focus_bonus = self._focus_alignment_bonus(chunk, active_focuses)
        local_bonus = self._local_protocol_bonus(chunk)

        return (
            overlap * 0.08
            + overlap_bonus
            + exact_phrase_bonus
            + path_bonus
            + topic_bonus
            + language_bonus
            + focus_bonus
            + local_bonus
            - general_penalty
        )

    def _build_result(
        self,
        chunk: dict,
        score: float,
        semantic_score: float,
        lexical_score: float,
    ) -> dict:
        positive_score = max(0.0, score)
        relevance_score = positive_score / (positive_score + 0.5) if positive_score else 0.0
        return {
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source_name": chunk["source_name"],
            "source_path": chunk["source_path"],
            "storage_folder": chunk.get("storage_folder", ""),
            "document_language": chunk.get("document_language", "unknown"),
            "score": float(score),
            "semantic_score": float(semantic_score),
            "lexical_score": float(lexical_score),
            "relevance_score": relevance_score,
        }

    def _tokenize(self, text: str) -> set[str]:
        ascii_text = self._normalize_text(text)
        cleaned = []
        for char in ascii_text:
            cleaned.append(char if char.isalnum() else " ")
        return {
            token
            for token in "".join(cleaned).split()
            if len(token) >= 4 and token not in GENERIC_QUERY_TOKENS
        }

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        return normalized.encode("ascii", "ignore").decode("ascii")

    def _question_is_probably_french(self, query_tokens: set[str]) -> bool:
        french_query_markers = {
            "quelle",
            "quelles",
            "quels",
            "comment",
            "pourquoi",
            "enfant",
            "fievre",
            "prise",
            "charge",
            "grossesse",
            "paludisme",
            "anemie",
        }
        return bool(query_tokens & french_query_markers)

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
            return 0.45
        return 0.0

    def _detect_active_topics(self, query_tokens: set[str]) -> set[str]:
        active_topics: set[str] = set()
        for topic_name, rule in TOPIC_RULES.items():
            if any(marker in token for token in query_tokens for marker in rule["query_markers"]):
                active_topics.add(topic_name)
        return active_topics

    def _detect_active_focuses(self, question: str, query_tokens: set[str]) -> set[str]:
        normalized_question = self._normalize_text(question)
        active_focuses: set[str] = set()
        for focus_name, rule in FOCUS_RULES.items():
            if any(marker in normalized_question for marker in rule["query_markers"]) or any(
                marker in query_tokens for marker in rule["query_markers"]
            ):
                active_focuses.add(focus_name)
        return active_focuses

    def _focus_overlap_bonus(self, query_tokens: set[str], chunk_tokens: set[str]) -> float:
        overlap_tokens = query_tokens & chunk_tokens
        if not overlap_tokens:
            return 0.0

        bonus = 0.0
        for token in overlap_tokens:
            if len(token) >= 7:
                bonus += 0.16
            else:
                bonus += 0.08
        return bonus

    def _topic_alignment_bonus(self, chunk: dict, active_topics: set[str]) -> float:
        if not active_topics:
            return 0.0
        if self._chunk_matches_active_topics(chunk, active_topics):
            if self._chunk_mentions_competing_topics(chunk, active_topics):
                return -0.55
            return 1.1
        return -0.9

    def _language_alignment_bonus(self, query_tokens: set[str], chunk: dict) -> float:
        if not self._question_is_probably_french(query_tokens):
            return 0.0

        chunk_language = chunk.get("document_language", "unknown")
        if chunk_language == "fr":
            return 0.2
        if chunk_language == "en":
            return -0.12
        return 0.0

    def _local_protocol_bonus(self, chunk: dict) -> float:
        """Bonus for local protocol files (markdown) that are structured clinical guides."""
        combined_text = self._normalize_text(
            f"{chunk['source_name']} {chunk['source_path']}"
        )
        bonus = 0.0
        if chunk.get("source_name", "").endswith(".md"):
            if "fiche_dawn" in combined_text:
                bonus += 1.6
            if any(marker in combined_text for marker in ("transcription", "protocole", "guide", "conduite")):
                bonus += 0.8
            if any(folder in combined_text for folder in ("paludisme", "anemie", "detresse")):
                bonus += 0.4
        return bonus

    def _focus_alignment_bonus(self, chunk: dict, active_focuses: set[str]) -> float:
        if not active_focuses:
            return 0.0

        combined_text = self._normalize_text(
            f"{chunk['source_name']} {chunk['source_path']} {chunk['text']}"
        )
        bonus = 0.0
        for focus_name in active_focuses:
            rule = FOCUS_RULES[focus_name]
            if any(marker in combined_text for marker in rule["content_markers"]):
                bonus += 1.25
            if any(marker in combined_text for marker in rule["preferred_source_markers"]):
                bonus += 1.1
            if any(marker in combined_text for marker in rule["negative_markers"]):
                bonus -= 2.4
        return bonus

    def _chunk_strongly_matches_focus(self, chunk: dict, active_focuses: set[str]) -> bool:
        combined_text = self._normalize_text(
            f"{chunk['source_name']} {chunk['source_path']} {chunk['text']}"
        )
        for focus_name in active_focuses:
            rule = FOCUS_RULES[focus_name]
            has_positive = any(marker in combined_text for marker in rule["content_markers"]) or any(
                marker in combined_text for marker in rule["preferred_source_markers"]
            )
            has_negative = any(marker in combined_text for marker in rule["negative_markers"])
            if has_positive and not has_negative:
                return True
        return False

    def _chunk_has_negative_focus_markers(self, chunk: dict, active_focuses: set[str]) -> bool:
        combined_text = self._normalize_text(
            f"{chunk['source_name']} {chunk['source_path']} {chunk['text']}"
        )
        source_text = self._normalize_text(f"{chunk['source_name']} {chunk['source_path']}")
        if "fiche_dawn" in source_text:
            return False
        for focus_name in active_focuses:
            rule = FOCUS_RULES[focus_name]
            if any(marker in combined_text for marker in rule["negative_markers"]):
                return True
        return False

    def _chunk_matches_active_topics(self, chunk: dict, active_topics: set[str]) -> bool:
        combined_text = self._normalize_text(
            f"{chunk['source_name']} {chunk['source_path']} {chunk['text']}"
        )
        for topic_name in active_topics:
            content_markers = TOPIC_RULES[topic_name]["content_markers"]
            if any(marker in combined_text for marker in content_markers):
                return True
        return False

    def _chunk_mentions_competing_topics(self, chunk: dict, active_topics: set[str]) -> bool:
        combined_text = self._normalize_text(
            f"{chunk['source_name']} {chunk['source_path']} {chunk['text']}"
        )
        for topic_name, rule in TOPIC_RULES.items():
            if topic_name in active_topics:
                continue
            if any(marker in combined_text for marker in rule["content_markers"]):
                return True
        return False
