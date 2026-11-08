"""Ответ на текстовое сообщение: поиск по документам и памяти → промпт → LLM с инструментами.

    text_embedder (OpenAITextEmbedder)
        ├─ embedding → docs_retriever.query_embedding    (namespace docs, фильтр user_id)
        └─ embedding → memory_retriever.query_embedding  (namespace chat, фильтр user_id)
    docs_retriever.documents → docs_threshold.documents  (отсечение по score)
    docs_threshold.documents → prompt_builder.documents
    memory_retriever.documents → prompt_builder.memories
    prompt_builder.prompt → llm.messages

LLM-узел — Haystack Agent поверх OpenAIChatGenerator: так сохраняются инструменты
старого бота (кошки, собаки, погода), а вызовы инструментов агент делает сам.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from hay_v2_bot.components.document_store import user_filter
from hay_v2_bot.components.tools import DOG_IMAGE_TOOL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — дружелюбный персональный ассистент в Telegram. Всегда отвечай на русском языке, коротко и по делу.

У тебя есть инструменты:
1. cat_fact — случайный факт о кошках. Вызывай, когда пользователь просит факт о кошках или что-то интересное про котов. Факт приходит на английском: переведи его на русский.
2. dog_image — ссылка на случайную фотографию собаки. Вызывай, когда пользователь просит показать собаку. Бот сам отправит фото, ссылку в ответе не повторяй.
3. dog_image_analyzer — определяет породу собаки по ссылке на фото и кратко рассказывает историю породы. Если пользователь просит показать собаку и определить породу, сначала вызови dog_image, затем dog_image_analyzer с полученной ссылкой.
4. weather — текущая погода в городе. Вызывай, когда спрашивают о погоде, температуре, ветре или как одеться. Если город не назван, спроси его или возьми из контекста памяти, если пользователь там упоминал свой город.

Память:
- В сообщении может быть блок «Что пользователь писал раньше». Это его прошлые сообщения из долговременной памяти.
- Используй их, чтобы отвечать на вопросы о пользователе: как его зовут, что он любит, что говорил раньше.
- Если в памяти нет ответа, честно скажи, что не знаешь, и не выдумывай факты о пользователе.

Документы пользователя:
- В сообщении может быть блок «Фрагменты загруженных документов» с пометкой источника у каждого фрагмента.
- Если вопрос про содержимое документов, отвечай только по этим фрагментам и в конце укажи источник: «Источник: <имя файла>, стр. <номер>» (страницу — если она указана).
- Фрагменты могут быть на другом языке: отвечай всё равно по-русски, термины можно оставить в оригинале.
- Если во фрагментах ответа нет, так и скажи и не придумывай содержание документа. Не ссылайся на файл, если ответ взят не из него.

Не вызывай инструменты без необходимости. Если инструмент вернул сообщение об ошибке, объясни это пользователю простыми словами."""

USER_TEMPLATE = """{% if memories %}Что пользователь писал раньше:
{% for memory in memories %}- {{ memory.content }}
{% endfor %}
{% endif %}{% if documents %}Фрагменты загруженных документов:
{% for doc in documents %}[Источник: {{ doc.meta.file_name }}{% if doc.meta.page %}, стр. {{ doc.meta.page }}{% endif %}]
{{ doc.content }}

{% endfor %}{% endif %}Новое сообщение пользователя:
{{ question }}"""

URL_RE = re.compile(r"https?://\S+")


@dataclass
class GenerationReply:
    text: str
    image_urls: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    query_embedding: list[float] | None = None


def build_generation_pipeline(config, docs_store, chat_store):
    from haystack import Pipeline
    from haystack.components.agents import Agent
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.embedders import OpenAITextEmbedder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

    from hay_v2_bot.components.processors import ScoreThreshold
    from hay_v2_bot.components.tools import create_tools
    from hay_v2_bot.config import openai_kwargs

    def make_generator():
        return OpenAIChatGenerator(**openai_kwargs(config))

    tools = create_tools(make_generator)
    agent = Agent(
        chat_generator=make_generator(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        max_agent_steps=10,
    )

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", OpenAITextEmbedder(**openai_kwargs(config, model=config.embedding_model)))
    pipeline.add_component(
        "docs_retriever", PineconeEmbeddingRetriever(document_store=docs_store, top_k=config.docs_top_k)
    )
    pipeline.add_component("docs_threshold", ScoreThreshold(min_score=config.docs_min_score))
    pipeline.add_component(
        "memory_retriever", PineconeEmbeddingRetriever(document_store=chat_store, top_k=config.memory_top_k)
    )
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(template=[ChatMessage.from_user(USER_TEMPLATE)], required_variables=["question"]),
    )
    pipeline.add_component("llm", agent)

    pipeline.connect("text_embedder.embedding", "docs_retriever.query_embedding")
    pipeline.connect("text_embedder.embedding", "memory_retriever.query_embedding")
    pipeline.connect("docs_retriever.documents", "docs_threshold.documents")
    pipeline.connect("docs_threshold.documents", "prompt_builder.documents")
    pipeline.connect("memory_retriever.documents", "prompt_builder.memories")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    logger.info(
        "generation_pipeline готов: модель=%s, base_url=%s, инструменты=%s",
        config.openai_model,
        config.openai_base_url,
        [tool.name for tool in tools],
    )
    return pipeline


def extract_dog_image_urls(messages: list) -> list[str]:
    """Достаёт ссылки на картинки из результатов инструмента dog_image."""
    urls = []
    for message in messages:
        tool_result = getattr(message, "tool_call_result", None)
        if tool_result is None:
            continue
        origin = getattr(tool_result, "origin", None)
        if getattr(origin, "tool_name", None) != DOG_IMAGE_TOOL:
            continue
        if getattr(tool_result, "error", False):
            continue
        urls.extend(URL_RE.findall(str(tool_result.result)))
    return urls


class Generator:
    """Запускает generation_pipeline для сообщения пользователя."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def answer(self, user_id: int | str, text: str) -> GenerationReply:
        filters = user_filter(user_id)
        result = self.pipeline.run(
            data={
                "text_embedder": {"text": text},
                "docs_retriever": {"filters": filters},
                "memory_retriever": {"filters": filters},
                "prompt_builder": {"question": text},
            },
            include_outputs_from={"text_embedder", "docs_threshold", "memory_retriever"},
        )

        memories = result.get("memory_retriever", {}).get("documents", [])
        documents = result.get("docs_threshold", {}).get("documents", [])
        logger.info(
            "Поиск в Pinecone: user_id=%s, память=%s (score=%s), документы=%s",
            user_id,
            len(memories),
            [round(doc.score, 3) for doc in memories if doc.score is not None],
            len(documents),
        )

        messages = result.get("llm", {}).get("messages", [])
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                logger.info("Агент вызвал инструмент %s с аргументами %s", call.tool_name, call.arguments)

        last = result.get("llm", {}).get("last_message") or (messages[-1] if messages else None)
        answer = (getattr(last, "text", None) or "").strip()
        return GenerationReply(
            text=answer or "Не получилось сформулировать ответ. Попробуйте переспросить.",
            image_urls=extract_dog_image_urls(messages),
            sources=[{"file_name": d.meta.get("file_name"), "page": d.meta.get("page")} for d in documents],
            query_embedding=result.get("text_embedder", {}).get("embedding"),
        )
