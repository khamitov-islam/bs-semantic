"""Метрики качества поиска (считаются библиотекой ranx)."""

from __future__ import annotations

from dataclasses import dataclass

from ranx import Qrels, Run, evaluate

from bsrag.evaluation.goldset import QAItem
from bsrag.retrieval import Hit

DEFAULT_METRICS = [
    "hit_rate@1",
    "hit_rate@3",
    "hit_rate@5",
    "recall@5",
    "recall@10",
    "mrr@10",
    "ndcg@10",
    "precision@1",
]


@dataclass
class QueryOutcome:
    """Результат одного запроса: что нашли и за сколько."""

    item: QAItem
    hits: list[Hit]
    retrieval_ms: float
    rerank_ms: float

    @property
    def page_scores(self) -> dict[str, float]:
        """Страница получает лучший скор среди своих чанков.

        Оценка ведётся на уровне страниц: только так метрики сопоставимы между
        конфигурациями с разным размером чанка.
        """
        scores: dict[str, float] = {}
        for rank, hit in enumerate(self.hits):
            # Скор подменяем на позицию: у RRF и кросс-энкодера разные шкалы,
            # а ранжирование сравнивать нужно одинаково.
            value = 1.0 / (rank + 1)
            page = hit.chunk.page_id
            scores[page] = max(scores.get(page, 0.0), value)
        return scores

    @property
    def found_gold_chunk(self) -> bool:
        if not self.item.gold_chunk_id:
            return False
        return any(h.chunk.chunk_id == self.item.gold_chunk_id for h in self.hits)


def compute_metrics(
    outcomes: list[QueryOutcome], metrics: list[str] | None = None
) -> dict[str, float]:
    outcomes = [o for o in outcomes if o.item.gold_pages and o.item.kind != "trap"]
    if not outcomes:
        return {}

    qrels = Qrels({o.item.id: {page: 1 for page in o.item.gold_pages} for o in outcomes})
    run = Run({o.item.id: o.page_scores or {"__empty__": 0.0} for o in outcomes})
    scores = evaluate(qrels, run, metrics or DEFAULT_METRICS, make_comparable=True)
    if isinstance(scores, float):
        scores = {(metrics or DEFAULT_METRICS)[0]: scores}

    latencies = sorted(o.retrieval_ms + o.rerank_ms for o in outcomes)
    scores = {k: float(v) for k, v in scores.items()}
    scores["latency_p50_ms"] = latencies[len(latencies) // 2]
    scores["latency_p95_ms"] = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    scores["retrieval_ms_mean"] = sum(o.retrieval_ms for o in outcomes) / len(outcomes)
    scores["rerank_ms_mean"] = sum(o.rerank_ms for o in outcomes) / len(outcomes)

    with_gold_chunk = [o for o in outcomes if o.item.gold_chunk_id]
    if with_gold_chunk:
        found = sum(o.found_gold_chunk for o in with_gold_chunk)
        scores["gold_chunk_recall"] = found / len(with_gold_chunk)
    return scores


def metrics_by_kind(outcomes: list[QueryOutcome]) -> dict[str, dict[str, float]]:
    """Отдельно по авто-вопросам и по ручным: они меряют разное."""
    result: dict[str, dict[str, float]] = {}
    for kind in ("auto", "manual"):
        subset = [o for o in outcomes if o.item.kind == kind]
        if subset:
            result[kind] = compute_metrics(subset)
    return result
