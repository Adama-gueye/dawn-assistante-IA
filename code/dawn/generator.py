from __future__ import annotations

import os
from pathlib import Path

from dawn.chunking import build_chunks
from dawn.config import DawnConfig
from dawn.pdf_loader import load_knowledge_pages
from dawn.prompts import build_prompt, detect_question_kind, format_context
from dawn.retriever import DawnRetriever


class DawnAssistant:
    def __init__(self, config: DawnConfig) -> None:
        self.config = config
        self.retriever = DawnRetriever(config.embedding_model)
        self.loaded_documents: list[dict] = []
        self.skipped_documents: list[dict] = []
        self._build_knowledge_base()

    def _build_knowledge_base(self) -> None:
        pages, loaded_documents, skipped_documents = load_knowledge_pages(
            Path(self.config.knowledge_path),
            enable_ocr_fallback=self.config.enable_ocr_fallback,
            ocr_language=self.config.ocr_language,
        )
        self.loaded_documents = loaded_documents
        self.skipped_documents = skipped_documents
        chunks = build_chunks(
            pages,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.retriever.fit(chunks)

    def answer(self, question: str, mode: str | None = None) -> dict:
        selected_mode = mode or self.config.default_mode
        retrieved_chunks = self.retriever.search(question, top_k=self.config.top_k)
        context = format_context(retrieved_chunks)
        question_kind = detect_question_kind(question)
        prompt = build_prompt(question, context, selected_mode)
        answer = self._generate_answer(prompt)

        return {
            "question": question,
            "mode": selected_mode,
            "question_kind": question_kind,
            "answer": answer,
            "sources": [
                {
                    "page": item["page"],
                    "score": item["score"],
                    "source_name": item["source_name"],
                    "source_path": item["source_path"],
                    "storage_folder": item.get("storage_folder", ""),
                    "relevance_score": item.get("relevance_score", 0.0),
                    "semantic_score": item.get("semantic_score", 0.0),
                    "lexical_score": item.get("lexical_score", 0.0),
                }
                for item in retrieved_chunks
            ],
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "prompt": prompt,
        }

    def _generate_answer(self, prompt: str) -> str:
        provider = self.config.provider.lower()

        if provider == "ollama":
            return self._generate_with_ollama(prompt)
        if provider == "anthropic":
            return self._generate_with_anthropic(prompt)

        raise ValueError(f"Provider non supporte: {self.config.provider}")

    def _generate_with_ollama(self, prompt: str) -> str:
        from openai import NotFoundError, OpenAI

        client = OpenAI(base_url=f"{self.config.ollama_base_url}/v1", api_key="ollama")
        try:
            response = client.chat.completions.create(
                model=self.config.generation_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es DAWN, un assistant d'aide a la decision medicale. "
                            "Tu dois toujours repondre en francais, de facon sobre, clinique et fidele au contexte fourni."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
        except NotFoundError as exc:
            available_models = self._list_ollama_models(client)
            model_hint = (
                f" Modeles disponibles: {', '.join(available_models)}."
                if available_models
                else " Aucun modele disponible n'a pu etre liste."
            )
            raise RuntimeError(
                "Le modele Ollama configure est introuvable: "
                f"{self.config.generation_model}. "
                f"Installe-le avec `ollama pull {self.config.generation_model}` "
                "ou change DAWN_GENERATION_MODEL / le champ Modele de generation."
                f"{model_hint}"
            ) from exc
        return response.choices[0].message.content.strip()

    @staticmethod
    def _list_ollama_models(client) -> list[str]:
        try:
            models = client.models.list()
        except Exception:
            return []
        return [model.id for model in models.data]

    def _generate_with_anthropic(self, prompt: str) -> str:
        from anthropic import Anthropic

        api_key = os.getenv(self.config.anthropic_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"La variable d'environnement {self.config.anthropic_api_key_env} est introuvable."
            )

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=self.config.generation_model,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            system=(
                "Tu es DAWN, un assistant d'aide a la decision medicale. "
                "Tu dois toujours repondre en francais, de facon sobre, clinique et fidele au contexte fourni."
            ),
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        texts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "\n".join(texts).strip()
