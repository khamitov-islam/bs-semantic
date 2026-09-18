"""Поиск по индексу: гибридный retrieval + кросс-энкодерный реранкинг."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from langchain_qdrant import QdrantVectorStore

from bsrag.chunking import Chunk
from bsrag.config import Settings
from bsrag.embeddings import resolve_device
from bsrag.index import document_to_chunk, open_index


@dataclass
class Hit:
    """Найденный фрагмент справки."""

    chunk: Chunk
    score: float
    rank: int
    retriever_score: float | None = None

    @property
    def source_label(self) -> str:
        return self.chunk.citation


@dataclass
class SearchResult:
    query: str
    hits: list[Hit]
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.rerank_ms


@lru_cache(maxsize=3)
def get_reranker(model_name: str, device: str = "auto"):
    """Кросс-энкодер для переупорядочивания кандидатов.

    На Apple Silicon половинная точность ускоряет модель примерно вчетверо при
    неразличимом качестве, а длина входа ограничена 512 токенами — ровно столько
    занимает чанк вместе с запросом.
    """
    from sentence_transformers import CrossEncoder

    resolved = resolve_device(device)
    kwargs: dict[str, object] = {"device": resolved, "max_length": 512}
    if resolved in ("mps", "cuda"):
        kwargs["model_kwargs"] = {"torch_dtype": "float16"}
    return CrossEncoder(model_name, **kwargs)


class Retriever:
    """Достаёт кандидатов из Qdrant и при необходимости переупорядочивает их.

    Гибридный поиск берёт широкий пул кандидатов (dense + BM25, слияние RRF),
    а кросс-энкодер уже прицельно сортирует их по релевантности запросу.
    """

    def __init__(self, settings: Settings, store: QdrantVectorStore | None = None) -> None:
        self.settings = settings
        self.store = store or open_index(settings)
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker(
                self.settings.retrieval.reranker_model, self.settings.embedding.device
            )
        return self._reranker

    def warmup(self) -> None:
        self.search("проверка готовности системы", top_k=1)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        candidates: int | None = None,
        use_reranker: bool | None = None,
    ) -> SearchResult:
        cfg = self.settings.retrieval
        top_k = top_k or cfg.top_k
        use_reranker = cfg.use_reranker if use_reranker is None else use_reranker
        candidates = candidates or (cfg.candidates if use_reranker else top_k)
        candidates = max(candidates, top_k)

        started = time.perf_counter()
        scored = self.store.similarity_search_with_score(query, k=candidates)
        retrieval_ms = (time.perf_counter() - started) * 1000

        hits = [
            Hit(chunk=document_to_chunk(doc), score=float(score), rank=i, retriever_score=float(score))
            for i, (doc, score) in enumerate(scored)
        ]

        rerank_ms = 0.0
        if use_reranker and hits:
            started = time.perf_counter()
            hits = self._rerank(query, hits)
            rerank_ms = (time.perf_counter() - started) * 1000

        hits = hits[:top_k]
        if cfg.score_threshold is not None:
            hits = [h for h in hits if h.score >= cfg.score_threshold] or hits[:1]
        for i, hit in enumerate(hits):
            hit.rank = i

        return SearchResult(query=query, hits=hits, retrieval_ms=retrieval_ms, rerank_ms=rerank_ms)

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        pairs = [(query, hit.chunk.embed_text) for hit in hits]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        for hit, score in zip(hits, scores):
            hit.score = float(score)
        return sorted(hits, key=lambda h: h.score, reverse=True)


def deduplicate_by_page(hits: list[Hit], limit: int) -> list[Hit]:
    """Оставляет по одному лучшему фрагменту на страницу.

    Нужно для честного подсчёта попаданий по страницам и чтобы в контекст LLM не
    попадали три соседних куска одного раздела вместо трёх разных источников.
    """
    seen: set[str] = set()
    unique: list[Hit] = []
    for hit in hits:
        if hit.chunk.page_id in seen:
            continue
        seen.add(hit.chunk.page_id)
        unique.append(hit)
        if len(unique) >= limit:
            break
    return unique
