"""Резюме файла — ровно одно предложение. LLM замокан."""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hay_v2_bot.pipelines.summary_pipeline import Summarizer
from hay_v2_bot.pipelines.summary_text import FALLBACK_SUMMARY, to_one_sentence


def count_sentences(text: str) -> int:
    return len(re.findall(r"[.!?…](?=\s+[A-ZА-ЯЁ0-9«]|$)", text))


@pytest.mark.parametrize(
    "llm_text, expected",
    [
        ("Документ описывает условия договора поставки.", "Документ описывает условия договора поставки."),
        (
            "Документ описывает договор. Он состоит из пяти разделов. Подписан в 2026 году.",
            "Документ описывает договор.",
        ),
        ("Резюме: отчёт о продажах за третий квартал", "отчёт о продажах за третий квартал."),
        ("Отчёт о продажах,\nпо регионам и т. е. по филиалам.\n\nВторое.", "Отчёт о продажах, по регионам и т. е. по филиалам."),
        ("«Руководство пользователя CRM.»", "Руководство пользователя CRM."),
        ("The paper introduces Docling! It is fast.", "The paper introduces Docling!"),
    ],
)
def test_to_one_sentence(llm_text, expected):
    result = to_one_sentence(llm_text)

    assert result == expected
    assert count_sentences(result) == 1


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "—"])
def test_empty_text_gives_fallback(text):
    assert to_one_sentence(text) == FALLBACK_SUMMARY


def test_summarizer_uses_first_chunks():
    pipeline = MagicMock()
    pipeline.run.return_value = {"one_sentence": {"summary": "Документ о договоре."}}
    chunks = [SimpleNamespace(content=f"фрагмент {i}") for i in range(8)]

    summary = Summarizer(pipeline, max_chunks=2).summarize("contract.docx", chunks)

    assert summary == "Документ о договоре."
    data = pipeline.run.call_args.kwargs["data"]["prompt_builder"]
    assert data["file_name"] == "contract.docx"
    assert data["text"] == "фрагмент 0\n\nфрагмент 1"


def test_summarizer_without_text_skips_llm():
    pipeline = MagicMock()

    summary = Summarizer(pipeline).summarize("empty.pdf", [SimpleNamespace(content="  ")])

    assert summary == FALLBACK_SUMMARY
    pipeline.run.assert_not_called()


def test_one_sentence_component():
    pytest.importorskip("haystack")
    from haystack.dataclasses import ChatMessage

    from hay_v2_bot.components.processors import OneSentence

    replies = [ChatMessage.from_assistant("Первое предложение. Второе предложение.")]

    assert OneSentence().run(replies=replies) == {"summary": "Первое предложение."}
