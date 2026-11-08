"""Ограничения задания: только эмбеддинги OpenAI и только через прокси."""

import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "hay_v2_bot"
SOURCES = {path: path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py")}
OPENAI_COMPONENT = re.compile(r"\b(OpenAIChatGenerator|OpenAITextEmbedder|OpenAIDocumentEmbedder)\(")


def test_no_default_openai_endpoint():
    for path, text in SOURCES.items():
        assert "api.openai.com" not in text, path


def test_no_local_embeddings():
    for path, text in SOURCES.items():
        assert "sentence_transformers" not in text, path
        assert "SentenceTransformers" not in text, path


def test_every_openai_component_goes_through_proxy_kwargs():
    for path, text in SOURCES.items():
        for match in OPENAI_COMPONENT.finditer(text):
            call = text[match.end() : match.end() + 80]
            assert "**openai_kwargs(config" in call, f"{path}: {match.group(0)} без api_base_url"


def test_requirements_have_no_local_embedders():
    requirements = (PACKAGE.parent / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "sentence-transformers" not in requirements
    for package in ("haystack-ai", "docling", "docling-haystack", "pinecone-haystack", "pytelegrambotapi", "python-dotenv", "requests"):
        assert package in requirements
