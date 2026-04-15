from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DawnConfig:
    pdf_path: Path
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    provider: str = "ollama"
    generation_model: str = "qwen2.5:3b"
    ollama_base_url: str = "http://localhost:11434"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    chunk_size: int = 220
    chunk_overlap: int = 40
    top_k: int = 3
    max_new_tokens: int = 280
    temperature: float = 0.1
    use_4bit: bool = True
