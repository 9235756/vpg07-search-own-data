"""Приведение текста резюме к одному предложению. Без зависимостей от Haystack."""

from __future__ import annotations

import re

FALLBACK_SUMMARY = "Не удалось составить резюме файла."

# Граница предложения: знак конца, пробел, затем заглавная буква, цифра или открывающая кавычка.
# Сокращения вроде «т. е.» не режутся, потому что после них идёт строчная буква.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9«\"(])")
_PREFIX = re.compile(r"^\s*(резюме|summary|итог)\s*[:\-—]\s*", re.IGNORECASE)


def to_one_sentence(text: str) -> str:
    """Возвращает первое предложение текста, оканчивающееся знаком препинания."""
    cleaned = " ".join((text or "").replace("\n", " ").split())
    cleaned = _PREFIX.sub("", cleaned).strip(" \"'«»")
    if not cleaned:
        return FALLBACK_SUMMARY
    sentence = _SENTENCE_END.split(cleaned, maxsplit=1)[0].strip()
    if sentence[-1] not in ".!?…":
        sentence = sentence.rstrip(",;:—- ")
        if not sentence:
            return FALLBACK_SUMMARY
        sentence += "."
    return sentence
