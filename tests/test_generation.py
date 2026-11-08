"""Generator: фильтры по пользователю, ссылки на собак, источники. Пайплайн замокан."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from hay_v2_bot.components.document_store import user_filter
from hay_v2_bot.pipelines.generation_pipeline import USER_TEMPLATE, Generator, extract_dog_image_urls


def tool_message(tool_name, result, error=False):
    return SimpleNamespace(
        tool_calls=None,
        tool_call_result=SimpleNamespace(origin=SimpleNamespace(tool_name=tool_name), result=result, error=error),
    )


def make_result(messages, documents=(), memories=()):
    return {
        "text_embedder": {"embedding": [0.1, 0.2]},
        "docs_threshold": {"documents": list(documents)},
        "memory_retriever": {"documents": list(memories)},
        "llm": {"messages": messages, "last_message": messages[-1]},
    }


def test_filters_by_user_in_both_retrievers():
    pipeline = MagicMock()
    pipeline.run.return_value = make_result([SimpleNamespace(text="Ответ", tool_calls=None)])

    reply = Generator(pipeline).answer(111, "О чём файл?")

    data = pipeline.run.call_args.kwargs["data"]
    assert data["docs_retriever"]["filters"] == user_filter(111)
    assert data["memory_retriever"]["filters"] == user_filter(111)
    assert data["text_embedder"]["text"] == "О чём файл?"
    assert data["prompt_builder"]["question"] == "О чём файл?"
    assert reply.text == "Ответ"
    # Эмбеддинг запроса переиспользуется для записи в память.
    assert reply.query_embedding == [0.1, 0.2]


def test_sources_are_returned():
    pipeline = MagicMock()
    doc = SimpleNamespace(meta={"file_name": "paper.pdf", "page": 3}, score=0.8)
    pipeline.run.return_value = make_result([SimpleNamespace(text="Ответ", tool_calls=None)], documents=[doc])

    reply = Generator(pipeline).answer(1, "q")

    assert reply.sources == [{"file_name": "paper.pdf", "page": 3}]


def test_empty_answer_has_fallback():
    pipeline = MagicMock()
    pipeline.run.return_value = make_result([SimpleNamespace(text="  ", tool_calls=None)])

    assert "Не получилось" in Generator(pipeline).answer(1, "q").text


def test_dog_urls_extracted_only_from_dog_tool():
    messages = [
        tool_message("dog_image", "https://images.dog.ceo/breeds/husky/1.jpg"),
        tool_message("weather", "https://example.com/not-a-dog"),
        tool_message("dog_image", "https://images.dog.ceo/err.jpg", error=True),
        SimpleNamespace(text="Вот собака"),
    ]

    assert extract_dog_image_urls(messages) == ["https://images.dog.ceo/breeds/husky/1.jpg"]


def test_template_renders_sources():
    jinja2 = __import__("pytest").importorskip("jinja2")
    doc = SimpleNamespace(content="Docling converts PDF.", meta={"file_name": "paper.pdf", "page": 2})
    no_page = SimpleNamespace(content="Hello", meta={"file_name": "notes.md"})
    memory = SimpleNamespace(content="Меня зовут Иван")

    text = jinja2.Template(USER_TEMPLATE).render(documents=[doc, no_page], memories=[memory], question="Что делает Docling?")

    assert "[Источник: paper.pdf, стр. 2]" in text
    assert "[Источник: notes.md]" in text
    assert "- Меня зовут Иван" in text
    assert text.rstrip().endswith("Что делает Docling?")
