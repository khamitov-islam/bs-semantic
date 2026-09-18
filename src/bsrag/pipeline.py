"""Сборка RAG-пайплайна: подготовка корпуса, индексация, ответ на вопрос."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from bsrag.chunking import Chunk, chunk_pages
from bsrag.config import Settings, get_settings
from bsrag.corpus import load_or_prepare_corpus, pages_file, prepare_corpus
from bsrag.generation import Answer, generate, stream
from bsrag.index import build_index
from bsrag.parsing.loader import Page
from bsrag.retrieval import Hit, Retriever, SearchResult, deduplicate_by_page

__all__ = [
    "RagPipeline",
    "RagResponse",
    "build",
    "load_or_prepare_corpus",
    "pages_file",
    "prepare_corpus",
]


@dataclass
class RagResponse:
    question: str
    answer: Answer
    hits: list[Hit]
    search: SearchResult

    @property
    def sources(self) -> list[Hit]:
        """Источники в порядке нумерации, использованной в ответе."""
        return self.hits


def build(settings: Settings | None = None, pages: list[Page] | None = None) -> list[Chunk]:
    settings = settings or get_settings()
    pages = pages or load_or_prepare_corpus(settings)
    chunks = chunk_pages(pages, settings.chunking, settings.embedding.model_name)
    build_index(chunks, settings)
    return chunks


class RagPipeline:
    """Поиск по справке + ответ локальной LLM со ссылками на источники."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.retriever = Retriever(self.settings)

    def search(self, question: str, top_k: int | None = None) -> SearchResult:
        return self.retriever.search(question, top_k=top_k)

    def _context_hits(self, question: str, top_k: int | None = None) -> tuple[SearchResult, list[Hit]]:
        top_k = top_k or self.settings.retrieval.top_k
        # Рерanker сортирует широкий пул, затем оставляем по одному фрагменту
        # на страницу — иначе в контекст LLM попадают три соседних куска одного
        # раздела вместо трёх разных источников.
        result = self.retriever.search(
            question,
            top_k=max(top_k * 3, self.settings.retrieval.candidates),
        )
        return result, deduplicate_by_page(result.hits, top_k)

    def ask(self, question: str, top_k: int | None = None) -> RagResponse:
        result, hits = self._context_hits(question, top_k)
        answer = generate(question, hits, self.settings.generation)
        return RagResponse(question=question, answer=answer, hits=hits, search=result)

    def ask_stream(self, question: str, top_k: int | None = None) -> tuple[list[Hit], Iterator[str]]:
        result, hits = self._context_hits(question, top_k)
        return hits, stream(question, hits, self.settings.generation)
