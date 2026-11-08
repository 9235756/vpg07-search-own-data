"""Точка входа: загрузка конфига, сборка пайплайнов и запуск бота.

Запуск из корня проекта:  python -m hay_v2_bot.main
"""

from __future__ import annotations

import logging
import sys

from hay_v2_bot.config import PROJECT_ROOT, ConfigError, load_config

LOG_DIR = PROJECT_ROOT / "logs"


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
        ],
    )
    # Сетевые библиотеки слишком болтливы на уровне INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "pinecone"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_services(config):
    # Тяжёлые импорты (Haystack, Docling) — только после проверки конфига.
    from hay_v2_bot.bot.telegram_bot import BotServices
    from hay_v2_bot.components.context import ChatMemory
    from hay_v2_bot.components.document_store import create_document_store
    from hay_v2_bot.pipelines.generation_pipeline import Generator, build_generation_pipeline
    from hay_v2_bot.pipelines.ingestion_pipeline import DocumentIngestor, build_ingestion_pipeline
    from hay_v2_bot.pipelines.summary_pipeline import Summarizer, build_summary_pipeline

    docs_store = create_document_store(config, config.namespace_docs)
    chat_store = create_document_store(config, config.namespace_chat)
    return BotServices(
        generator=Generator(build_generation_pipeline(config, docs_store, chat_store)),
        memory=ChatMemory(chat_store),
        ingestor=DocumentIngestor(build_ingestion_pipeline(config, docs_store), docs_store),
        summarizer=Summarizer(build_summary_pipeline(config), max_chunks=config.summary_chunks),
        docs_store=docs_store,
    )


def main() -> None:
    setup_logging()
    logger = logging.getLogger("hay_v2_bot")
    try:
        config = load_config()
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    from hay_v2_bot.bot.telegram_bot import create_bot

    services = build_services(config)
    bot = create_bot(config, services)
    logger.info("Бот запущен, ожидаю сообщения")
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()
