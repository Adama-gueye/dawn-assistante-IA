from dataclasses import dataclass
import os
from pathlib import Path

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PROVIDER = "ollama"
DEFAULT_GENERATION_MODEL = "qwen2.5:14b"
DEFAULT_MODE = "medecin"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_CHUNK_SIZE = 450
DEFAULT_CHUNK_OVERLAP = 90
DEFAULT_TOP_K = 5
DEFAULT_MAX_NEW_TOKENS = 800
DEFAULT_TEMPERATURE = 0.1
DEFAULT_USE_4BIT = True
DEFAULT_ENABLE_OCR_FALLBACK = True
DEFAULT_OCR_LANGUAGE = "fra"


@dataclass(slots=True)
class DawnConfig:
    knowledge_path: Path
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    provider: str = DEFAULT_PROVIDER
    generation_model: str = DEFAULT_GENERATION_MODEL
    default_mode: str = DEFAULT_MODE
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    anthropic_api_key_env: str = DEFAULT_ANTHROPIC_API_KEY_ENV
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    use_4bit: bool = DEFAULT_USE_4BIT
    enable_ocr_fallback: bool = DEFAULT_ENABLE_OCR_FALLBACK
    ocr_language: str = DEFAULT_OCR_LANGUAGE

    @classmethod
    def from_env(cls, knowledge_path: Path) -> "DawnConfig":
        return cls(
            knowledge_path=knowledge_path,
            embedding_model=os.getenv("DAWN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            provider=os.getenv("DAWN_PROVIDER", DEFAULT_PROVIDER),
            generation_model=os.getenv("DAWN_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
            default_mode=os.getenv("DAWN_DEFAULT_MODE", DEFAULT_MODE),
            ollama_base_url=os.getenv("DAWN_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            anthropic_api_key_env=os.getenv("DAWN_ANTHROPIC_API_KEY_ENV", DEFAULT_ANTHROPIC_API_KEY_ENV),
            chunk_size=_env_int("DAWN_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
            chunk_overlap=_env_int("DAWN_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
            top_k=_env_int("DAWN_TOP_K", DEFAULT_TOP_K),
            max_new_tokens=_env_int("DAWN_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS),
            temperature=_env_float("DAWN_TEMPERATURE", DEFAULT_TEMPERATURE),
            use_4bit=_env_bool("DAWN_USE_4BIT", DEFAULT_USE_4BIT),
            enable_ocr_fallback=_env_bool("DAWN_ENABLE_OCR_FALLBACK", DEFAULT_ENABLE_OCR_FALLBACK),
            ocr_language=os.getenv("DAWN_OCR_LANGUAGE", DEFAULT_OCR_LANGUAGE),
        )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "oui"}
