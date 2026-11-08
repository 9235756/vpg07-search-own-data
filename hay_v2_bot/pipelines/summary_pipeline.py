"""Резюме файла ровно одним предложением по первым N чанкам.

    prompt_builder (ChatPromptBuilder)
        └─ prompt → llm.messages
    llm (OpenAIChatGenerator через прокси)
        └─ replies → one_sentence.replies
    one_sentence (OneSentence) → summary
"""

from __future__ import annotations

import logging

from hay_v2_bot.components.chunk_meta import summary_source_text
from hay_v2_bot.pipelines.summary_text import FALLBACK_SUMMARY

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM = (
    "Ты пишешь резюме документов. Ответь ровно одним предложением на русском языке, "
    "не длиннее 40 слов: о чём документ и для чего он нужен. Без вступлений и списков."
)
SUMMARY_USER = """Файл: {{ file_name }}

Начало документа:
{{ text }}"""


def build_summary_pipeline(config):
    from haystack import Pipeline
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    from hay_v2_bot.components.processors import OneSentence
    from hay_v2_bot.config import openai_kwargs

    pipeline = Pipeline()
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=[ChatMessage.from_system(SUMMARY_SYSTEM), ChatMessage.from_user(SUMMARY_USER)],
            required_variables=["file_name", "text"],
        ),
    )
    pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            **openai_kwargs(config),
            generation_kwargs={"temperature": 0.2, "max_tokens": 120},
        ),
    )
    pipeline.add_component("one_sentence", OneSentence())
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "one_sentence.replies")
    return pipeline


class Summarizer:
    def __init__(self, pipeline, max_chunks: int = 5):
        self.pipeline = pipeline
        self.max_chunks = max_chunks

    def summarize(self, file_name: str, chunks: list) -> str:
        text = summary_source_text(chunks, self.max_chunks)
        if not text:
            return FALLBACK_SUMMARY
        result = self.pipeline.run(data={"prompt_builder": {"file_name": file_name, "text": text}})
        summary = result.get("one_sentence", {}).get("summary") or FALLBACK_SUMMARY
        logger.info("Резюме file=%s по %s чанкам: %s", file_name, min(len(chunks), self.max_chunks), summary)
        return summary
