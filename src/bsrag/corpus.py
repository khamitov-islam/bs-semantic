"""Подготовка корпуса: разбор страниц и кэш результата на диске.

Модуль намеренно лёгкий — не тянет torch, Qdrant и onnxruntime, чтобы разбор
разметки запускался за доли секунды.
"""

from __future__ import annotations

from pathlib import Path

from bsrag.config import Settings, get_settings
from bsrag.parsing.loader import Page, load_pages, read_pages, save_pages


def pages_file(settings: Settings) -> Path:
    """Кэш страниц зависит от ``table_mode``: rows пересобирает markdown таблиц.

    Старый файл ``pages.jsonl`` остаётся кэшем режима markdown, чтобы бейслайн
    не пересчитывался.
    """
    processed = settings.resolve(settings.paths.processed)
    mode = settings.chunking.table_mode
    if mode == "markdown":
        legacy = processed / "pages.jsonl"
        if legacy.exists():
            return legacy
        return processed / "pages_markdown.jsonl"
    return processed / f"pages_{mode}.jsonl"


def prepare_corpus(settings: Settings | None = None) -> list[Page]:
    settings = settings or get_settings()
    pages = load_pages(settings)
    save_pages(pages, pages_file(settings))
    return pages


def load_or_prepare_corpus(settings: Settings | None = None) -> list[Page]:
    settings = settings or get_settings()
    path = pages_file(settings)
    return read_pages(path) if path.exists() else prepare_corpus(settings)
