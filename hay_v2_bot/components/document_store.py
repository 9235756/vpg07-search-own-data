"""Фабрика PineconeDocumentStore и фильтры Haystack для документов пользователя."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PINECONE_SPEC = {"serverless": {"cloud": "aws", "region": "us-east-1"}}


def create_document_store(config, namespace: str):
    """Хранилище в общем индексе Pinecone. Namespace разделяет документы и память диалога."""
    from haystack.utils import Secret
    from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

    store = PineconeDocumentStore(
        api_key=Secret.from_env_var("PINECONE_API_KEY"),
        index=config.pinecone_index,
        namespace=namespace,
        dimension=config.embedding_dimension,
        metric="cosine",
        spec=PINECONE_SPEC,
    )
    logger.info(
        "Pinecone: индекс=%s, namespace=%s, dimension=%s, metric=cosine",
        config.pinecone_index,
        namespace,
        config.embedding_dimension,
    )
    return store


def user_filter(user_id: int | str) -> dict:
    """Фильтр Haystack: документы только этого пользователя."""
    return {"field": "meta.user_id", "operator": "==", "value": str(user_id)}


def user_file_filter(user_id: int | str, file_name: str) -> dict:
    """Фильтр Haystack: чанки одного файла этого пользователя."""
    return {
        "operator": "AND",
        "conditions": [
            user_filter(user_id),
            {"field": "meta.file_name", "operator": "==", "value": file_name},
        ],
    }


def delete_by_filter(store, filters: dict) -> int:
    """Удаляет документы по фильтру. Возвращает число удалённых."""
    if hasattr(store, "delete_by_filter"):
        return store.delete_by_filter(filters) or 0
    # Старые версии pinecone-haystack: ищем документы и удаляем по id порциями.
    deleted = 0
    for _ in range(100):
        ids = [doc.id for doc in store.filter_documents(filters=filters)]
        if not ids:
            break
        store.delete_documents(ids)
        deleted += len(ids)
    return deleted
