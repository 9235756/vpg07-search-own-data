"""Хендлеры Telegram: /start, /help, /clear, /docs, /cleardocs, текст и документы."""

from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from hay_v2_bot.bot.files import SUPPORTED_EXTENSIONS, validate_upload
from hay_v2_bot.components.chunk_meta import safe_file_name, summarize_files
from hay_v2_bot.components.document_store import delete_by_filter, user_filter
from hay_v2_bot.config import PROJECT_ROOT
from hay_v2_bot.pipelines.ingestion_pipeline import IngestionError

logger = logging.getLogger(__name__)

TMP_DIR = PROJECT_ROOT / "tmp"
TELEGRAM_LIMIT = 4096

FORMATS = ", ".join(sorted(ext.lstrip(".").upper() for ext in SUPPORTED_EXTENSIONS - {".htm"}))

START_TEXT = f"""Привет! Я персональный ассистент. Вот что я умею:

• изучаю ваши файлы ({FORMATS}) и отвечаю на вопросы по ним со ссылкой на файл и страницу;
• запоминаю, что вы мне пишете, и учитываю это в ответах;
• рассказываю факты о кошках;
• показываю случайную собаку и определяю её породу;
• подсказываю погоду в любом городе.

Команды:
/help — подсказка
/docs — какие файлы я изучил
/cleardocs — удалить все ваши файлы
/clear — забыть, что вы мне писали (файлы останутся)"""

HELP_TEXT = f"""Примеры запросов:
• пришлите файл {FORMATS} — изучу его и пришлю резюме;
• «О чём говорится в разделе про оплату?» — отвечу по вашим файлам;
• «Меня зовут Иван, я люблю горы» — запомню;
• «Расскажи факт о кошках»;
• «Покажи собаку и определи породу»;
• «Какая погода в Москве?».

/docs — список файлов
/cleardocs — удалить файлы
/clear — удалить память диалога"""

RECEIVED_TEXT = "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…"
DONE_TEXT = "Готово. Я изучил этот файл, теперь можем его обсудить."


@dataclass
class BotServices:
    generator: object  # pipelines.generation_pipeline.Generator
    memory: object  # components.context.ChatMemory
    ingestor: object  # pipelines.ingestion_pipeline.DocumentIngestor
    summarizer: object  # pipelines.summary_pipeline.Summarizer
    docs_store: object  # PineconeDocumentStore, namespace docs


def split_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [""]


def format_docs_list(files: list[dict]) -> str:
    if not files:
        return "Вы ещё не присылали файлов. Пришлите документ, и я его изучу."
    lines = ["Изученные файлы:"]
    for index, item in enumerate(files, start=1):
        date = item["uploaded_at"][:10] if item["uploaded_at"] else "—"
        lines.append(f"{index}. {item['file_name']} — фрагментов: {item['chunks']}, загружен {date}")
    return "\n".join(lines)


def create_bot(config, services: BotServices):
    import telebot

    bot = telebot.TeleBot(config.telegram_token, parse_mode=None)
    # Docling тяжёлый: файлы обрабатываются по одному в отдельном потоке, бот не блокируется.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingestion")
    TMP_DIR.mkdir(exist_ok=True)

    @bot.message_handler(commands=["start"])
    def handle_start(message):
        bot.send_message(message.chat.id, START_TEXT)

    @bot.message_handler(commands=["help"])
    def handle_help(message):
        bot.send_message(message.chat.id, HELP_TEXT)

    @bot.message_handler(commands=["clear"])
    def handle_clear(message):
        try:
            deleted = services.memory.clear(message.from_user.id)
        except Exception:
            logger.exception("Не удалось очистить память user_id=%s", message.from_user.id)
            bot.send_message(message.chat.id, "Не получилось очистить память. Попробуйте позже.")
            return
        bot.send_message(message.chat.id, f"Память диалога очищена. Удалено записей: {deleted}. Файлы не тронуты.")

    @bot.message_handler(commands=["docs"])
    def handle_docs(message):
        try:
            chunks = services.docs_store.filter_documents(filters=user_filter(message.from_user.id))
        except Exception:
            logger.exception("Не удалось получить список файлов user_id=%s", message.from_user.id)
            bot.send_message(message.chat.id, "Не получилось получить список файлов. Попробуйте позже.")
            return
        files = summarize_files(chunks)
        logger.info("/docs: user_id=%s, файлов=%s, чанков=%s", message.from_user.id, len(files), len(chunks))
        for part in split_text(format_docs_list(files)):
            bot.send_message(message.chat.id, part)

    @bot.message_handler(commands=["cleardocs"])
    def handle_cleardocs(message):
        try:
            deleted = delete_by_filter(services.docs_store, user_filter(message.from_user.id))
        except Exception:
            logger.exception("Не удалось удалить файлы user_id=%s", message.from_user.id)
            bot.send_message(message.chat.id, "Не получилось удалить файлы. Попробуйте позже.")
            return
        logger.info("/cleardocs: user_id=%s, удалено чанков=%s", message.from_user.id, deleted)
        bot.send_message(message.chat.id, f"Файлы удалены. Удалено фрагментов: {deleted}.")

    @bot.message_handler(content_types=["text"])
    def handle_text(message):
        user_id = message.from_user.id
        text = (message.text or "").strip()
        if not text:
            return
        if text.startswith("/"):
            bot.send_message(message.chat.id, "Не знаю такой команды. Список команд: /help")
            return

        bot.send_chat_action(message.chat.id, "typing")
        try:
            reply = services.generator.answer(user_id, text)
        except Exception:
            logger.exception("Ошибка generation_pipeline для user_id=%s", user_id)
            bot.send_message(message.chat.id, "Что-то пошло не так. Попробуйте ещё раз чуть позже.")
            return

        # Память сохраняется после поиска: текущее сообщение не попадает в свой же контекст.
        try:
            services.memory.save_user_message(message, reply.query_embedding)
        except Exception:
            logger.exception("Ошибка сохранения памяти в Pinecone для user_id=%s", user_id)

        for url in reply.image_urls:
            try:
                bot.send_photo(message.chat.id, url)
            except Exception:
                logger.exception("Не удалось отправить фото %s", url)
                bot.send_message(message.chat.id, url)

        for chunk in split_text(reply.text):
            bot.send_message(message.chat.id, chunk)

    @bot.message_handler(content_types=["document"])
    def handle_document(message):
        document = message.document
        error = validate_upload(document.file_name, document.file_size, config.max_file_size_mb)
        if error:
            logger.info("Файл отклонён: %s (%s байт): %s", document.file_name, document.file_size, error)
            bot.send_message(message.chat.id, error)
            return
        bot.send_message(message.chat.id, RECEIVED_TEXT)
        executor.submit(process_document, message)

    def process_document(message):
        document = message.document
        file_name = safe_file_name(document.file_name)
        user_id, chat_id = message.from_user.id, message.chat.id
        try:
            with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmp:
                path = Path(tmp) / file_name
                file_info = bot.get_file(document.file_id)
                path.write_bytes(bot.download_file(file_info.file_path))
                logger.info("Файл скачан: %s, %s байт", path, path.stat().st_size)

                result = services.ingestor.ingest(path, file_name=file_name, user_id=user_id, chat_id=chat_id)
                bot.send_message(chat_id, DONE_TEXT)

                summary = services.summarizer.summarize(file_name, result.chunks)
                bot.send_message(chat_id, summary)
            # Выход из with удаляет временную папку вместе с файлом.
        except IngestionError as exc:
            logger.warning("Файл %s не обработан: %s", file_name, exc)
            bot.send_message(chat_id, str(exc))
        except Exception:
            logger.exception("Ошибка обработки файла %s для user_id=%s", file_name, user_id)
            bot.send_message(
                chat_id,
                "Не получилось обработать файл. Проверьте, что он открывается, и попробуйте ещё раз.",
            )

    @bot.message_handler(content_types=["photo", "sticker", "voice", "audio", "video", "video_note", "location", "contact"])
    def handle_other(message):
        # Нетекстовые сообщения (кроме документов) не сохраняются и не отправляются агенту.
        bot.send_message(message.chat.id, f"Я понимаю текстовые сообщения и файлы {FORMATS}.")

    return bot
