"""Ingestion: Docling → чанки → эмбеддинги OpenAI → Pinecone (namespace docs).

    converter (DoclingConverter, DOC_CHUNKS + HybridChunker)
        └─ documents → meta_enricher.documents
    meta_enricher (ChunkMetaEnricher)
        └─ documents → embedder.documents
    embedder (OpenAIDocumentEmbedder, text-embedding-3-small через прокси)
        └─ documents → writer.documents
    writer (DocumentWriter, DuplicatePolicy.OVERWRITE)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from hay_v2_bot.components.chunk_meta import utc_now_iso
from hay_v2_bot.components.document_store import delete_by_filter, user_file_filter

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Ошибка обработки файла, текст которой можно показать пользователю."""


@dataclass
class IngestionResult:
    file_name: str
    chunks: list
    written: int
    seconds: float


def build_ingestion_pipeline(config, docs_store):
    from docling.chunking import HybridChunker
    from haystack import Pipeline
    from haystack.components.embedders import OpenAIDocumentEmbedder
    from haystack.components.writers import DocumentWriter
    from haystack.document_stores.types import DuplicatePolicy
    from haystack_integrations.components.converters.docling import DoclingConverter, ExportType

    from hay_v2_bot.components.processors import ChunkMetaEnricher
    from hay_v2_bot.config import openai_kwargs

    pipeline = Pipeline()
    pipeline.add_component(
        "converter",
        DoclingConverter(export_type=ExportType.DOC_CHUNKS, chunker=HybridChunker()),
    )
    pipeline.add_component("meta_enricher", ChunkMetaEnricher())
    pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            **openai_kwargs(config, model=config.embedding_model),
            meta_fields_to_embed=["file_name", "headings"],
            progress_bar=False,
        ),
    )
    pipeline.add_component(
        "writer",
        DocumentWriter(document_store=docs_store, policy=DuplicatePolicy.OVERWRITE),
    )
    pipeline.connect("converter.documents", "meta_enricher.documents")
    pipeline.connect("meta_enricher.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline


class DocumentIngestor:
    """Загружает файл пользователя в Pinecone через ingestion_pipeline."""

    def __init__(self, pipeline, docs_store):
        self.pipeline = pipeline
        self.store = docs_store

    def ingest(self, path: Path, file_name: str, user_id: int | str, chat_id: int | str) -> IngestionResult:
        started = time.monotonic()
        user_id, chat_id = str(user_id), str(chat_id)
        logger.info("Ingestion: старт file=%s, user_id=%s, путь=%s", file_name, user_id, path)

        # Повторная загрузка того же файла заменяет старые чанки, а не дублирует их.
        removed = delete_by_filter(self.store, user_file_filter(user_id, file_name))
        if removed:
            logger.info("Ingestion: удалены прежние чанки file=%s, удалено=%s", file_name, removed)

        base_meta = {
            "file_name": file_name,
            "user_id": user_id,
            "chat_id": chat_id,
            "uploaded_at": utc_now_iso(),
            "source": "document",
        }
        result = self.pipeline.run(
            data={"converter": {"sources": [str(path)], "meta": base_meta}},
            include_outputs_from={"meta_enricher"},
        )
        chunks = result.get("meta_enricher", {}).get("documents", [])
        written = result.get("writer", {}).get("documents_written", 0)
        seconds = time.monotonic() - started
        logger.info(
            "Ingestion: запись в Pinecone file=%s, чанков=%s, записано=%s, время=%.1f с",
            file_name,
            len(chunks),
            written,
            seconds,
        )
        if not chunks:
            raise IngestionError(
                "В файле не нашлось текста. Если это скан, проверьте, что OCR в Docling включён, "
                "или пришлите файл с текстовым слоем."
            )
        return IngestionResult(file_name=file_name, chunks=chunks, written=written, seconds=seconds)
