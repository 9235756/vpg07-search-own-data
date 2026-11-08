"""DocumentIngestor: что уходит в Docling и Pinecone. Пайплайн и хранилище замоканы."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hay_v2_bot.components.document_store import user_file_filter
from hay_v2_bot.pipelines.ingestion_pipeline import DocumentIngestor, IngestionError


def make_ingestor(chunks, written=None):
    pipeline = MagicMock()
    pipeline.run.return_value = {
        "meta_enricher": {"documents": chunks},
        "writer": {"documents_written": len(chunks) if written is None else written},
    }
    store = MagicMock()
    store.delete_by_filter.return_value = 0
    return DocumentIngestor(pipeline, store), pipeline, store


def test_meta_passed_to_converter():
    chunks = [SimpleNamespace(content="text", meta={})]
    ingestor, pipeline, _ = make_ingestor(chunks)

    result = ingestor.ingest(Path("/tmp/x/report.pdf"), file_name="report.pdf", user_id=111, chat_id=222)

    data = pipeline.run.call_args.kwargs["data"]["converter"]
    assert data["sources"] == ["/tmp/x/report.pdf"]
    meta = data["meta"]
    assert meta["file_name"] == "report.pdf"
    assert meta["user_id"] == "111" and meta["chat_id"] == "222"
    assert meta["source"] == "document"
    assert meta["uploaded_at"].endswith("+00:00")
    assert "meta_enricher" in pipeline.run.call_args.kwargs["include_outputs_from"]
    assert result.written == 1 and result.chunks == chunks


def test_old_chunks_of_same_file_are_replaced():
    ingestor, _, store = make_ingestor([SimpleNamespace(content="text", meta={})])

    ingestor.ingest(Path("a.pdf"), file_name="a.pdf", user_id=1, chat_id=1)

    store.delete_by_filter.assert_called_once_with(user_file_filter("1", "a.pdf"))


def test_empty_document_raises_user_error():
    ingestor, _, _ = make_ingestor([])

    with pytest.raises(IngestionError, match="не нашлось текста"):
        ingestor.ingest(Path("scan.pdf"), file_name="scan.pdf", user_id=1, chat_id=1)


def test_meta_enricher_component():
    pytest.importorskip("haystack")
    from haystack import Document

    from hay_v2_bot.components.processors import ChunkMetaEnricher

    base = {"file_name": "a.pdf", "user_id": "1", "chat_id": "2", "uploaded_at": "t", "source": "document"}
    docs = [
        Document(content="первый", meta={**base, "page_number": 1, "dl_meta": {"headings": ["Введение"]}}),
        Document(content="   ", meta=dict(base)),
        Document(content="второй", meta={**base, "page_number": 2}),
    ]

    result = ChunkMetaEnricher().run(documents=docs)["documents"]

    assert [d.content for d in result] == ["первый", "второй"]
    assert [d.meta["chunk_index"] for d in result] == [0, 1]
    assert [d.meta["page"] for d in result] == [1, 2]
    assert result[0].meta["headings"] == "Введение"
    assert all("dl_meta" not in d.meta for d in result)
    assert len({d.id for d in result}) == 2
