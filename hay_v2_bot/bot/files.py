"""Проверка присланного файла до скачивания: расширение и размер."""

from __future__ import annotations

from pathlib import PurePath

# Форматы, которые Docling разбирает из коробки.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md"}
# Telegram Bot API отдаёт ботам файлы не больше 20 МБ.
TELEGRAM_DOWNLOAD_LIMIT_MB = 20


def file_extension(file_name: str | None) -> str:
    return PurePath(file_name or "").suffix.lower()


def validate_upload(file_name: str | None, file_size: int | None, max_size_mb: int) -> str | None:
    """Возвращает текст ошибки для пользователя или None, если файл подходит."""
    extension = file_extension(file_name)
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(ext.lstrip(".").upper() for ext in SUPPORTED_EXTENSIONS - {".htm"}))
        shown = extension.lstrip(".").upper() or "без расширения"
        return f"Формат {shown} не поддерживается. Пришлите файл в одном из форматов: {allowed}."

    limit_mb = min(max_size_mb, TELEGRAM_DOWNLOAD_LIMIT_MB)
    if file_size is None:
        return None
    if file_size <= 0:
        return "Файл пустой. Пришлите другой файл."
    if file_size > limit_mb * 1024 * 1024:
        size_mb = file_size / 1024 / 1024
        return f"Файл слишком большой: {size_mb:.1f} МБ. Максимум — {limit_mb} МБ."
    return None
