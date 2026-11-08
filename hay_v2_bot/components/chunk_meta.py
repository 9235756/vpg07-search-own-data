"""Подготовка meta чанков Docling к записи в Pinecone. Без зависимостей от Haystack.

DoclingConverter кладёт в meta вложенный словарь dl_meta. Pinecone принимает только
str, int, float, bool и список строк, остальное отбрасывает. Поэтому из dl_meta
достаём нужное (страница, заголовки) в плоские поля, а сам dl_meta убираем.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

# Служебные поля Docling, которые не нужны в индексе.
DROP_FIELDS = ("dl_meta", "split_id", "split_idx_start")
MAX_HEADINGS_LENGTH = 300
SUMMARY_CHAR_LIMIT = 6000


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iter_doc_items(dl_meta: dict) -> list:
    # export_json_dict() кладёт doc_items либо на верхний уровень, либо под "meta".
    items = dl_meta.get("doc_items")
    if items is None:
        items = (dl_meta.get("meta") or {}).get("doc_items")
    return items if isinstance(items, list) else []


def extract_page(raw_meta: dict) -> int | None:
    """Номер первой страницы чанка. У DOCX, HTML и MD страниц может не быть."""
    page = raw_meta.get("page_number")
    if isinstance(page, int) and page > 0:
        return page
    dl_meta = raw_meta.get("dl_meta")
    if not isinstance(dl_meta, dict):
        return None
    pages = []
    for item in _iter_doc_items(dl_meta):
        for prov in (item or {}).get("prov") or []:
            page_no = (prov or {}).get("page_no")
            if isinstance(page_no, int) and page_no > 0:
                pages.append(page_no)
    return min(pages) if pages else None


def extract_headings(raw_meta: dict) -> str | None:
    """Заголовки раздела, к которому относится чанк, одной строкой."""
    dl_meta = raw_meta.get("dl_meta")
    if not isinstance(dl_meta, dict):
        return None
    headings = dl_meta.get("headings")
    if headings is None:
        headings = (dl_meta.get("meta") or {}).get("headings")
    if not headings:
        return None
    text = " / ".join(str(h).strip() for h in headings if str(h).strip())
    return text[:MAX_HEADINGS_LENGTH] or None


def build_chunk_meta(raw_meta: dict, chunk_index: int) -> dict[str, Any]:
    """Плоская meta чанка для Pinecone.

    В raw_meta уже лежат поля, переданные в DoclingConverter.run(meta=...):
    file_name, user_id, chat_id, uploaded_at, source.
    """
    meta = {key: value for key, value in raw_meta.items() if key not in DROP_FIELDS}
    meta.pop("page_number", None)
    meta["chunk_index"] = chunk_index
    meta["source"] = "document"
    page = extract_page(raw_meta)
    if page is not None:
        meta["page"] = page
    headings = extract_headings(raw_meta)
    if headings:
        meta["headings"] = headings
    # None и вложенные структуры Pinecone не хранит: убираем заранее, чтобы не было предупреждений.
    return {
        key: value
        for key, value in meta.items()
        if isinstance(value, (str, int, float, bool))
        or (isinstance(value, list) and all(isinstance(v, str) for v in value))
    }


def chunk_id(user_id: str, file_name: str, chunk_index: int) -> str:
    """Детерминированный ASCII-id: повторная загрузка файла перезаписывает те же векторы."""
    digest = hashlib.sha256(f"{user_id}\x00{file_name}".encode("utf-8")).hexdigest()[:24]
    return f"doc-{digest}-{chunk_index:05d}"


def safe_file_name(name: str | None) -> str:
    """Имя файла без путей и опасных символов, пригодное для временной папки."""
    base = (name or "").replace("\\", "/").split("/")[-1].strip()
    base = re.sub(r"[^\w.\- ]+", "_", base, flags=re.UNICODE)
    return base.strip(" .") or "document"


def summarize_files(documents: list) -> list[dict]:
    """Группирует чанки из Pinecone по файлам для команды /docs."""
    files: dict[str, dict] = {}
    for doc in documents:
        meta = getattr(doc, "meta", None) or {}
        name = meta.get("file_name")
        if not name:
            continue
        entry = files.setdefault(name, {"file_name": name, "chunks": 0, "uploaded_at": ""})
        entry["chunks"] += 1
        uploaded = str(meta.get("uploaded_at") or "")
        entry["uploaded_at"] = max(entry["uploaded_at"], uploaded)
    return sorted(files.values(), key=lambda item: item["uploaded_at"], reverse=True)


def summary_source_text(documents: list, limit: int, char_limit: int = SUMMARY_CHAR_LIMIT) -> str:
    """Текст первых limit чанков для резюме, обрезанный по длине."""
    parts = []
    for doc in documents[:limit]:
        content = (getattr(doc, "content", None) or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)[:char_limit]
