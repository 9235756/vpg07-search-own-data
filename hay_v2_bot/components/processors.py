"""Собственные компоненты Haystack, которые встраиваются в пайплайны."""

from __future__ import annotations

import logging

from haystack import Document, component
from haystack.dataclasses import ChatMessage

from hay_v2_bot.components.chunk_meta import build_chunk_meta, chunk_id
from hay_v2_bot.pipelines.summary_text import to_one_sentence

logger = logging.getLogger(__name__)


@component
class ChunkMetaEnricher:
    """После Docling: нумерует чанки, выставляет плоскую meta и детерминированные id."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        enriched = []
        for doc in documents:
            if not (doc.content or "").strip():
                continue
            meta = build_chunk_meta(doc.meta or {}, chunk_index=len(enriched))
            enriched.append(
                Document(
                    id=chunk_id(str(meta.get("user_id", "")), str(meta.get("file_name", "")), meta["chunk_index"]),
                    content=doc.content,
                    meta=meta,
                )
            )
        pages = sorted({doc.meta["page"] for doc in enriched if "page" in doc.meta})
        logger.info(
            "Ingestion: Docling вернул чанков=%s, непустых=%s, страницы=%s",
            len(documents),
            len(enriched),
            f"{pages[0]}–{pages[-1]}" if pages else "нет",
        )
        return {"documents": enriched}


@component
class ScoreThreshold:
    """Отбрасывает найденные чанки с низким косинусным сходством, чтобы бот не ссылался на нерелевантное."""

    def __init__(self, min_score: float = 0.3):
        self.min_score = min_score

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        kept = [doc for doc in documents if doc.score is None or doc.score >= self.min_score]
        logger.info(
            "Поиск по документам: найдено=%s, после порога %.2f осталось=%s, источники=%s",
            len(documents),
            self.min_score,
            len(kept),
            [(doc.meta.get("file_name"), doc.meta.get("page"), round(doc.score or 0, 3)) for doc in kept],
        )
        return {"documents": kept}


@component
class OneSentence:
    """Превращает ответ LLM в резюме ровно из одного предложения."""

    @component.output_types(summary=str)
    def run(self, replies: list[ChatMessage]):
        text = replies[0].text if replies else ""
        return {"summary": to_one_sentence(text or "")}
