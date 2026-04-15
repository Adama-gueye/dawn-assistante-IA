from __future__ import annotations

import os
from pathlib import Path

from dawn.chunking import build_chunks
from dawn.config import DawnConfig
from dawn.pdf_loader import load_pdf_pages
from dawn.prompts import build_medical_prompt, format_context
from dawn.retriever import DawnRetriever


class DawnAssistant:
    def __init__(self, config: DawnConfig) -> None:
        self.config = config
        self.retriever = DawnRetriever(config.embedding_model)
        self._build_knowledge_base()

    def _build_knowledge_base(self) -> None:
        pages = load_pdf_pages(Path(self.config.pdf_path))
        chunks = build_chunks(
            pages,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.retriever.fit(chunks)

    def answer(self, question: str) -> dict:
        retrieved_chunks = self.retriever.search(question, top_k=self.config.top_k)
        context = format_context(retrieved_chunks)
        prompt = build_medical_prompt(question, context)
        answer = self._generate_answer(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": [{"page": item["page"], "score": item["score"]} for item in retrieved_chunks],
        }

    def _generate_answer(self, prompt: str) -> str:
        provider = self.config.provider.lower()

        if provider == "ollama":
            return self._generate_with_ollama(prompt)
        if provider == "anthropic":
            return self._generate_with_anthropic(prompt)

        raise ValueError(f"Provider non supporte: {self.config.provider}")

    def _generate_with_ollama(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=f"{self.config.ollama_base_url}/v1", api_key="ollama")
        response = client.chat.completions.create(
            model=self.config.generation_model,
            messages=[
                {"role": "system", "content": "Tu es DAWN, un assistant d'aide a la decision medicale."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content.strip()

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
            system="Tu es DAWN, un assistant d'aide a la decision medicale.",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        texts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        return "\n".join(texts).strip()
