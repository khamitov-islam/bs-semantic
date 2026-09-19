"""Поиск по индексу: гибридный retrieval + кросс-энкодерный реранкинг."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache

from langchain_qdrant import QdrantVectorStore

from bsrag.chunking import Chunk
from bsrag.config import Settings
from bsrag.embeddings import resolve_device
from bsrag.index import document_to_chunk, open_index

# Короткие подсказки лексики справки. Не LLM: не добавляют секунд к первому ответу.
_QUERY_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"включа|настроит|где .*опц|как включ", re.I), "опция настройка пункт меню"),
    (re.compile(r"клавиш|ctrl\+|shift\+|alt\+|сочетани", re.I), "горячая клавиша сочетание клавиш"),
    (re.compile(r"прав[ао].*доступ|доступ.*пользовател", re.I), "окно Права доступа пользователи"),
    (re.compile(r"ветк|основн(ую|ой) модел|перенес", re.I), "слияние ветка основная модель"),
    (re.compile(r"\bole\b|\bodata\b", re.I), "OLE OData протокол"),
]


def expand_queries(question: str) -> list[str]:
    """Исходный вопрос плюс одна-две формулировки языком справки."""
    variants = [question]
    seen = {question.casefold()}
    for pattern, hint in _QUERY_HINTS:
        if pattern.search(question):
            extra = f"{question} {hint}"
            key = extra.casefold()
            if key not in seen:
                variants.append(extra)
                seen.add(key)
    return variants[:3]


def rrf_merge(ranked: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Сливает несколько списков кандидатов по Reciprocal Rank Fusion."""
    return weighted_rrf([(hits, 1.0) for hits in ranked], k=k)


def weighted_rrf(ranked: list[tuple[list[Hit], float]], k: int = 60) -> list[Hit]:
    """RRF с весом канала: короткие запросы усиливают BM25, длинные — вектор."""
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits, weight in ranked:
        if not hits or weight <= 0:
            continue
        for rank, hit in enumerate(hits):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
            stored = best.get(cid)
            if stored is None or (hit.retriever_score or 0) >= (stored.retriever_score or 0):
                best[cid] = hit
    ordered = sorted(best.values(), key=lambda h: scores[h.chunk.chunk_id], reverse=True)
    for hit in ordered:
        fused = scores[hit.chunk.chunk_id]
        hit.retriever_score = fused
        hit.score = fused
    return ordered


_LEXICAL_QUERY = re.compile(
    r"\b(ctrl|alt|shift|ins|del|esc|tab|enter|f\d{1,2})\b|\b\d{3,5}\b|клавиш|сочетани|горяч",
    re.I,
)


def is_lexical_query(question: str) -> bool:
    return bool(_LEXICAL_QUERY.search(question))


def fusion_weights(question: str) -> dict[str, float]:
    """Веса не учатся на goldset: эвристика по виду запроса."""
    if is_lexical_query(question):
        return {"dense": 0.5, "sparse": 2.0, "tables": 2.5}
    return {"dense": 2.0, "sparse": 0.5, "tables": 1.0}


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
    """Достаёт кандидатов и при необходимости переупорядочивает их.

    Базовый путь — hybrid RRF Qdrant. При ``weighted_fusion`` / ``table_index``
    dense, BM25 и строки таблиц ищутся раздельно и сливаются с весами по запросу.
    """

    def __init__(self, settings: Settings, store: QdrantVectorStore | None = None) -> None:
        self.settings = settings
        self.store = store or open_index(settings)
        self._reranker = None
        self._dense = None
        self._sparse = None
        self._tables = None
        if settings.retrieval.weighted_fusion:
            self._dense = open_index(settings, mode="dense")
            self._sparse = open_index(settings, mode="sparse")
        if settings.retrieval.table_index:
            self._tables = open_index(settings, suffix="_tables", mode="hybrid")

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker(
                self.settings.retrieval.reranker_model, self.settings.embedding.device
            )
        return self._reranker

    def warmup(self) -> None:
        if self.settings.retrieval.use_reranker:
            _ = self.reranker
        self.search("проверка готовности системы", top_k=1, use_reranker=self.settings.retrieval.use_reranker)

    def _search_store(self, store: QdrantVectorStore, query: str, candidates: int) -> list[Hit]:
        scored = store.similarity_search_with_score(query, k=candidates)
        return [
            Hit(chunk=document_to_chunk(doc), score=float(score), rank=i, retriever_score=float(score))
            for i, (doc, score) in enumerate(scored)
        ]

    def _channel(self, store: QdrantVectorStore, variants: list[str], candidates: int) -> list[Hit]:
        lists = [self._search_store(store, variant, candidates) for variant in variants]
        return rrf_merge(lists, k=self.settings.retrieval.rrf_k) if len(lists) > 1 else lists[0]

    def search(
        self,
        query: str,
        top_k: int | None = None,
        candidates: int | None = None,
        use_reranker: bool | None = None,
        query_expand: bool | None = None,
    ) -> SearchResult:
        cfg = self.settings.retrieval
        top_k = top_k or cfg.top_k
        use_reranker = cfg.use_reranker if use_reranker is None else use_reranker
        query_expand = cfg.query_expand if query_expand is None else query_expand
        candidates = candidates or (cfg.candidates if use_reranker else top_k)
        candidates = max(candidates, top_k)

        variants = expand_queries(query) if query_expand else [query]
        started = time.perf_counter()
        if cfg.weighted_fusion or cfg.table_index:
            hits = self._fused_search(query, variants, candidates)
        elif len(variants) == 1:
            hits = self._search_store(self.store, variants[0], candidates)
        else:
            hits = rrf_merge(
                [self._search_store(self.store, variant, candidates) for variant in variants],
                k=cfg.rrf_k,
            )
        retrieval_ms = (time.perf_counter() - started) * 1000

        rerank_ms = 0.0
        if use_reranker and hits:
            started = time.perf_counter()
            hits = self._rerank(query, hits[:candidates])
            rerank_ms = (time.perf_counter() - started) * 1000

        hits = hits[:top_k]
        if cfg.score_threshold is not None:
            hits = [h for h in hits if h.score >= cfg.score_threshold] or hits[:1]
        for i, hit in enumerate(hits):
            hit.rank = i

        return SearchResult(query=query, hits=hits, retrieval_ms=retrieval_ms, rerank_ms=rerank_ms)

    def _fused_search(self, query: str, variants: list[str], candidates: int) -> list[Hit]:
        cfg = self.settings.retrieval
        weights = (
            fusion_weights(query)
            if cfg.weighted_fusion
            else {"dense": 1.0, "sparse": 1.0, "tables": 1.0}
        )
        channels: list[tuple[list[Hit], float]] = []
        if cfg.weighted_fusion:
            channels.append((self._channel(self._dense, variants, candidates), weights["dense"]))
            channels.append((self._channel(self._sparse, variants, candidates), weights["sparse"]))
        else:
            channels.append((self._channel(self.store, variants, candidates), 1.0))
        if self._tables is not None:
            channels.append((self._channel(self._tables, variants, candidates), weights["tables"]))
        return weighted_rrf(channels, k=cfg.rrf_k)

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
