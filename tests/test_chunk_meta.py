"""Meta чанков Docling: плоские поля для Pinecone, страница, id, список файлов."""

from types import SimpleNamespace

import pytest

from hay_v2_bot.components.chunk_meta import (
    build_chunk_meta,
    chunk_id,
    extract_page,
    safe_file_name,
    summarize_files,
    summary_source_text,
)

BASE = {
    "file_name": "report.pdf",
    "user_id": "111",
    "chat_id": "222",
    "uploaded_at": "2026-11-08T10:00:00+00:00",
    "source": "document",
}


def docling_meta(**extra):
    """Похоже на то, что кладёт DoclingConverter в режиме DOC_CHUNKS."""
    return {
        "dl_meta": {
            "schema_name": "docling_core.transforms.chunker.DocMeta",
            "doc_items": [
                {"self_ref": "#/texts/3", "prov": [{"page_no": 4, "bbox": {"l": 1.0}}]},
                {"self_ref": "#/texts/4", "prov": [{"page_no": 3}]},
            ],
            "headings": ["2. Оплата", "2.1 Сроки"],
            "origin": {"filename": "tmp123.pdf", "binary_hash": "123"},
        },
        "page_number": 3,
        "split_id": 0,
        "split_idx_start": 0,
        **BASE,
        **extra,
    }


def test_meta_is_filled():
    meta = build_chunk_meta(docling_meta(), chunk_index=7)

    assert meta == {
        **BASE,
        "chunk_index": 7,
        "page": 3,
        "headings": "2. Оплата / 2.1 Сроки",
    }


def test_meta_is_pinecone_compatible():
    meta = build_chunk_meta(docling_meta(extra_none=None, nested={"a": 1}, ints=[1, 2], tags=["a"]), 0)

    assert "dl_meta" not in meta and "nested" not in meta and "ints" not in meta and "extra_none" not in meta
    assert meta["tags"] == ["a"]
    for value in meta.values():
        assert isinstance(value, (str, int, float, bool, list))


def test_source_is_always_document():
    assert build_chunk_meta({**BASE, "source": "chat"}, 0)["source"] == "document"


def test_page_falls_back_to_dl_meta():
    raw = docling_meta()
    raw.pop("page_number")

    assert extract_page(raw) == 3


def test_page_from_nested_meta_layout():
    raw = {"dl_meta": {"meta": {"doc_items": [{"prov": [{"page_no": 9}]}]}}}

    assert extract_page(raw) == 9


def test_no_page_for_docx():
    raw = {**BASE, "dl_meta": {"doc_items": [{"prov": []}]}}

    meta = build_chunk_meta(raw, 0)

    assert "page" not in meta
    assert meta["chunk_index"] == 0


def test_chunk_id_is_deterministic_ascii():
    first = chunk_id("111", "Отчёт за год.pdf", 3)

    assert first == chunk_id("111", "Отчёт за год.pdf", 3)
    assert first != chunk_id("222", "Отчёт за год.pdf", 3)
    assert first != chunk_id("111", "Отчёт за год.pdf", 4)
    assert first.isascii()


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Отчёт 2026.pdf", "Отчёт 2026.pdf"),
        ("../../etc/passwd.md", "passwd.md"),
        ("C:\\Users\\a\\file.docx", "file.docx"),
        ("bad:name?.pptx", "bad_name_.pptx"),
        (None, "document"),
    ],
)
def test_safe_file_name(name, expected):
    assert safe_file_name(name) == expected


def test_summarize_files_groups_chunks():
    docs = [
        SimpleNamespace(meta={"file_name": "a.pdf", "uploaded_at": "2026-11-01T00:00:00+00:00"}),
        SimpleNamespace(meta={"file_name": "a.pdf", "uploaded_at": "2026-11-01T00:00:00+00:00"}),
        SimpleNamespace(meta={"file_name": "b.docx", "uploaded_at": "2026-11-05T00:00:00+00:00"}),
        SimpleNamespace(meta={}),
    ]

    files = summarize_files(docs)

    assert [(f["file_name"], f["chunks"]) for f in files] == [("b.docx", 1), ("a.pdf", 2)]


def test_summary_source_uses_first_chunks_only():
    docs = [SimpleNamespace(content=f"chunk {i}") for i in range(10)]

    text = summary_source_text(docs, limit=3)

    assert text == "chunk 0\n\nchunk 1\n\nchunk 2"
