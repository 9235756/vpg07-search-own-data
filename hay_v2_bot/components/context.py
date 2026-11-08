"""Память диалога в Pinecone (namespace chat): только текст сообщений пользователя."""

from __future__ import annotations

import logging
import time
from typing import Any

from hay_v2_bot.components.document_store import delete_by_filter, user_filter

logger = logging.getLogger(__name__)


def should_save_message(message: Any) -> bool:
    """Решает, нужно ли сохранять сообщение Telegram в память.

    Сохраняется только непустой текст от живого пользователя.
    Команды, нетекстовые сообщения, пустые строки и сообщения от ботов не сохраняются.
    """
    if getattr(message, "content_type", None) != "text":
        return False
    text = getattr(message, "text", None)
    if not isinstance(text, str) or not text.strip():
        return False
    if text.strip().startswith("/"):
        return False
    sender = getattr(message, "from_user", None)
    if sender is None or getattr(sender, "is_bot", False):
        return False
    return True


class ChatMemory:
    """Сохраняет сообщения пользователя в namespace chat и очищает их.

    Поиск по памяти выполняет generation_pipeline (memory_retriever), а сохранение
    переиспользует эмбеддинг запроса из того же пайплайна, чтобы не платить за него дважды.
    """

    def __init__(self, document_store):
        self.store = document_store

    def save_user_message(self, message: Any, embedding: list[float] | None) -> bool:
        """Сохраняет сообщение, если оно проходит фильтр. Возвращает True при сохранении."""
        if not should_save_message(message):
            logger.debug("Сообщение не сохраняется в память (команда, пустое или не текст)")
            return False
        if not embedding:
            logger.warning("Нет эмбеддинга для сообщения, в память не сохраняю")
            return False

        from haystack import Document
        from haystack.document_stores.types import DuplicatePolicy

        user_id = str(message.from_user.id)
        chat_id = str(message.chat.id)
        document = Document(
            id=f"{chat_id}-{message.message_id}",
            content=message.text.strip(),
            embedding=embedding,
            meta={
                "user_id": user_id,
                "chat_id": chat_id,
                "timestamp": int(getattr(message, "date", None) or time.time()),
                "role": "user",
                "source": "chat",
            },
        )
        written = self.store.write_documents([document], policy=DuplicatePolicy.OVERWRITE)
        logger.info(
            "Память: записано в Pinecone user_id=%s, chat_id=%s, id=%s, документов=%s",
            user_id,
            chat_id,
            document.id,
            written,
        )
        return True

    def clear(self, user_id: int | str) -> int:
        """Удаляет всю память диалога пользователя. Документы не трогает."""
        deleted = delete_by_filter(self.store, user_filter(user_id))
        logger.info("Память диалога очищена: user_id=%s, удалено=%s", user_id, deleted)
        return deleted
