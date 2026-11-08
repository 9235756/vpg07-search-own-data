"""Проверка присланных файлов: формат и размер."""

import pytest

from hay_v2_bot.bot.files import validate_upload

MB = 1024 * 1024


@pytest.mark.parametrize("name", ["a.pdf", "b.DOCX", "c.pptx", "d.html", "e.htm", "f.md"])
def test_supported_formats_pass(name):
    assert validate_upload(name, 1 * MB, max_size_mb=20) is None


@pytest.mark.parametrize("name", ["virus.exe", "notes.txt", "table.xlsx", "photo.jpg", "archive.zip", "noext", "", None])
def test_unsupported_format_is_rejected(name):
    error = validate_upload(name, 1 * MB, max_size_mb=20)

    assert error is not None
    assert "не поддерживается" in error
    assert "PDF" in error and "DOCX" in error


def test_too_big_file_is_rejected():
    error = validate_upload("big.pdf", 25 * MB, max_size_mb=50)

    # Telegram не отдаёт ботам файлы больше 20 МБ, даже если в конфиге больше.
    assert "слишком большой" in error
    assert "20 МБ" in error


def test_config_limit_is_respected():
    assert "5 МБ" in validate_upload("a.pdf", 6 * MB, max_size_mb=5)
    assert validate_upload("a.pdf", 5 * MB, max_size_mb=5) is None


def test_empty_file_is_rejected():
    assert "пустой" in validate_upload("a.pdf", 0, max_size_mb=20)
