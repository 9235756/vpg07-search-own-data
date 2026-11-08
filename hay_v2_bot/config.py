"""Загрузка настроек из .env.local и проверка обязательных переменных."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.local"

REQUIRED_VARS = (
    "TELEGRAM_BOT_TOKEN",
    "OPENAI_API_KEY",
    # Все запросы к OpenAI идут через прокси, поэтому адрес обязателен.
    "OPENAI_BASE_URL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
)

# Размерность векторов для моделей эмбеддингов OpenAI. Индекс Pinecone создаётся под неё.
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class ConfigError(RuntimeError):
    """Не хватает переменной окружения или она задана неверно."""


@dataclass(frozen=True)
class Config:
    telegram_token: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_model: str
    embedding_dimension: int
    pinecone_api_key: str
    pinecone_index: str
    namespace_docs: str
    namespace_chat: str
    max_file_size_mb: int
    docs_top_k: int
    memory_top_k: int
    docs_min_score: float
    summary_chunks: int

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, "").strip() or default


def _get_number(name: str, default, cast):
    raw = _get(name)
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} должна быть числом, сейчас: {raw!r}.") from exc
    if value <= 0:
        raise ConfigError(f"{name} должна быть больше нуля, сейчас: {raw!r}.")
    return value


def load_config() -> Config:
    """Читает .env.local из корня проекта и возвращает настройки."""
    load_dotenv(ENV_FILE)

    missing = [name for name in REQUIRED_VARS if not _get(name)]
    if missing:
        raise ConfigError(
            "Не заданы обязательные переменные окружения: "
            + ", ".join(missing)
            + f". Заполните их в файле {ENV_FILE} (образец: .env.local.example)."
        )

    base_url = _get("OPENAI_BASE_URL").rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigError(f"OPENAI_BASE_URL должен начинаться с http:// или https://, сейчас: {base_url!r}.")

    embedding_model = _get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    if embedding_model not in EMBEDDING_DIMENSIONS:
        raise ConfigError(
            f"Неизвестная модель эмбеддингов {embedding_model!r}. "
            f"Поддерживаются: {', '.join(EMBEDDING_DIMENSIONS)}."
        )

    namespace_docs = _get("PINECONE_NAMESPACE_DOCS", "docs")
    namespace_chat = _get("PINECONE_NAMESPACE_CHAT", "chat")
    if namespace_docs == namespace_chat:
        raise ConfigError(
            "PINECONE_NAMESPACE_DOCS и PINECONE_NAMESPACE_CHAT должны различаться: "
            "документы и память диалога хранятся раздельно."
        )

    return Config(
        telegram_token=_get("TELEGRAM_BOT_TOKEN"),
        openai_api_key=_get("OPENAI_API_KEY"),
        openai_base_url=base_url,
        openai_model=_get("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_model=embedding_model,
        embedding_dimension=EMBEDDING_DIMENSIONS[embedding_model],
        pinecone_api_key=_get("PINECONE_API_KEY"),
        pinecone_index=_get("PINECONE_INDEX"),
        namespace_docs=namespace_docs,
        namespace_chat=namespace_chat,
        max_file_size_mb=_get_number("MAX_FILE_SIZE_MB", 20, int),
        docs_top_k=_get_number("DOCS_TOP_K", 5, int),
        memory_top_k=_get_number("MEMORY_TOP_K", 5, int),
        docs_min_score=_get_number("DOCS_MIN_SCORE", 0.3, float),
        summary_chunks=_get_number("SUMMARY_CHUNKS", 5, int),
    )


def openai_kwargs(config: Config, model: str | None = None) -> dict:
    """Общие параметры для всех компонентов OpenAI: ключ, модель и адрес прокси.

    В Haystack адрес передаётся параметром api_base_url (не base_url).
    """
    from haystack.utils import Secret

    return {
        "api_key": Secret.from_env_var("OPENAI_API_KEY"),
        "model": model or config.openai_model,
        "api_base_url": config.openai_base_url,
    }
