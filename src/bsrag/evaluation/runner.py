"""Прогон бенчмарков: чанкинг, эмбеддинги, схема поиска, LLM.

Полный перебор всех комбинаций стоил бы десятки часов, поэтому эксперименты
разбиты на этапы: на каждом варьируется одна группа параметров, а остальные
зафиксированы на лучшем известном значении. Результаты копятся в reports/.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from bsrag.config import Settings
from bsrag.corpus import load_or_prepare_corpus
from bsrag.evaluation.goldset import GoldSet, QAItem, goldset_path, load_goldset
from bsrag.evaluation.metrics import QueryOutcome, compute_metrics, metrics_by_kind

RESULTS_FILE = "benchmarks.csv"

# Опорная конфигурация для этапов, где параметр не варьируется.
BASELINE_EMBEDDING = "intfloat/multilingual-e5-base"

CHUNK_SIZES = [256, 384, 512]
OVERLAP_RATIOS = [0.0, 0.15, 0.3]
EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "sergeyzh/BERTA",
    "ai-forever/ru-en-RoSBERTa",
    "deepvk/USER-bge-m3",
    "BAAI/bge-m3",
]
LLM_MODELS = ["qwen3:8b", "gemma3:12b-it-qat", "vikhr-nemo-12b"]


@dataclass
class ExperimentRow:
    stage: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def flat(self) -> dict[str, Any]:
        return {"stage": self.stage, "name": self.name, **self.params, **self.metrics}


def results_path(settings: Settings) -> Path:
    return settings.resolve(settings.paths.reports) / RESULTS_FILE


def save_rows(rows: list[ExperimentRow], settings: Settings) -> Path:
    path = results_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row.flat() for row in rows])
    if path.exists():
        previous = pd.read_csv(path)
        frame = pd.concat([previous, frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["stage", "name"], keep="last")
    frame.to_csv(path, index=False)
    return path


def _log_mlflow(row: ExperimentRow) -> None:
    try:
        import mlflow

        mlflow.set_experiment("bs-semantic")
        with mlflow.start_run(run_name=f"{row.stage}/{row.name}"):
            mlflow.log_params(row.params)
            mlflow.log_metrics({k: v for k, v in row.metrics.items() if isinstance(v, (int, float))})
    except Exception:  # трекинг не должен ронять эксперимент
        pass


def ensure_index(settings: Settings, pages, console=None) -> tuple[int, float]:
    """Строит индекс, если коллекции для такой конфигурации ещё нет."""
    from bsrag.chunking import chunk_pages
    from bsrag.index import build_index, collection_name, index_exists, read_manifest

    if index_exists(settings):
        manifest = read_manifest(settings)
        return int(manifest.get("n_chunks", 0)), 0.0

    if console:
        console.print(f"[dim]строю индекс {collection_name(settings)}[/]")
    started = time.perf_counter()
    chunks = chunk_pages(pages, settings.chunking, settings.embedding.model_name)
    build_index(chunks, settings)
    return len(chunks), time.perf_counter() - started


def run_queries(
    settings: Settings, items: Iterable[QAItem], top_k: int = 10, console=None
) -> list[QueryOutcome]:
    from bsrag.retrieval import Retriever

    retriever = Retriever(settings)
    outcomes: list[QueryOutcome] = []
    items = list(items)
    for i, item in enumerate(items):
        result = retriever.search(item.question, top_k=top_k)
        outcomes.append(
            QueryOutcome(
                item=item,
                hits=result.hits,
                retrieval_ms=result.retrieval_ms,
                rerank_ms=result.rerank_ms,
            )
        )
        if console and (i + 1) % 20 == 0:
            console.print(f"[dim]  {i + 1}/{len(items)} запросов[/]")
    return outcomes


def evaluate_settings(
    stage: str,
    name: str,
    settings: Settings,
    goldset: GoldSet,
    pages,
    params: dict[str, Any],
    console=None,
) -> ExperimentRow:
    n_chunks, build_time = ensure_index(settings, pages, console=console)
    outcomes = run_queries(settings, goldset.answerable, top_k=10)

    metrics = compute_metrics(outcomes)
    for kind, values in metrics_by_kind(outcomes).items():
        for key in ("hit_rate@3", "ndcg@10", "mrr@10"):
            if key in values:
                metrics[f"{kind}_{key}"] = values[key]
    metrics["n_chunks"] = n_chunks
    metrics["index_build_s"] = round(build_time, 1)

    row = ExperimentRow(stage=stage, name=name, params=params, metrics=metrics)
    _log_mlflow(row)
    if console:
        console.print(
            f"[bold]{name}[/]: nDCG@10 {metrics.get('ndcg@10', 0):.3f} · "
            f"hit@3 {metrics.get('hit_rate@3', 0):.3f} · "
            f"чанков {n_chunks} · {metrics.get('latency_p50_ms', 0):.0f} мс"
        )
    return row


def _variant(settings: Settings, **changes: Any) -> Settings:
    """Копия настроек с изменёнными полями вида ``chunking__chunk_size``."""
    data = settings.model_dump()
    for key, value in changes.items():
        section, _, field_name = key.partition("__")
        data[section][field_name] = value
    return Settings.model_validate(data)


def stage_chunking(settings: Settings, goldset: GoldSet, pages, console=None) -> list[ExperimentRow]:
    """Этап A: размер чанка, оверлап и способ нарезки.

    Считается на быстрой модели e5-base — выводы о нарезке переносятся между
    моделями, а прогон выходит втрое дешевле.
    """
    # Реранкер сюда не подключаем: он сглаживает разницу между нарезками,
    # а этап как раз должен показать вклад чанкинга самого по себе.
    base = _variant(
        settings,
        embedding__model_name=BASELINE_EMBEDDING,
        retrieval__use_reranker=False,
    )
    rows: list[ExperimentRow] = []

    for size in CHUNK_SIZES:
        for overlap in OVERLAP_RATIOS:
            variant = _variant(base, chunking__chunk_size=size, chunking__chunk_overlap_ratio=overlap)
            rows.append(
                evaluate_settings(
                    "chunking",
                    f"header_recursive/{size}/{int(overlap * 100)}%",
                    variant,
                    goldset,
                    pages,
                    {
                        "strategy": "header_recursive",
                        "chunk_size": size,
                        "overlap": overlap,
                        "table_mode": variant.chunking.table_mode,
                        "breadcrumb": variant.chunking.prepend_breadcrumb,
                        "embedding_model": BASELINE_EMBEDDING,
                    },
                    console=console,
                )
            )

    scored = [
        r
        for r in rows
        if not r.params.get("ablation")
    ]
    best = max(
        scored,
        key=lambda r: (
            r.metrics.get("manual_ndcg@10") or r.metrics.get("ndcg@10", 0),
            r.metrics.get("manual_hit_rate@3") or r.metrics.get("hit_rate@3", 0),
            -r.metrics.get("n_chunks", 0),
        ),
    )
    best_size = best.params["chunk_size"]
    best_overlap = best.params["overlap"]
    tuned = _variant(
        base, chunking__chunk_size=best_size, chunking__chunk_overlap_ratio=best_overlap
    )

    ablations = {
        "fixed (без учёта заголовков)": {"chunking__strategy": "fixed"},
        "parent-document": {"chunking__strategy": "parent"},
        "таблицы построчно": {"chunking__table_mode": "rows"},
        "без хлебных крошек": {"chunking__prepend_breadcrumb": False},
    }
    for label, changes in ablations.items():
        variant = _variant(tuned, **changes)
        rows.append(
            evaluate_settings(
                "chunking",
                f"{label} @{best_size}/{int(best_overlap * 100)}%",
                variant,
                goldset,
                pages,
                {
                    "strategy": variant.chunking.strategy,
                    "chunk_size": best_size,
                    "overlap": best_overlap,
                    "table_mode": variant.chunking.table_mode,
                    "breadcrumb": variant.chunking.prepend_breadcrumb,
                    "embedding_model": BASELINE_EMBEDDING,
                    "ablation": label,
                },
                console=console,
            )
        )
    return rows


def stage_embeddings(
    settings: Settings, goldset: GoldSet, pages, console=None
) -> list[ExperimentRow]:
    """Этап B: сравнение эмбеддеров на лучшей нарезке."""
    rows: list[ExperimentRow] = []
    for model in EMBEDDING_MODELS:
        variant = _variant(
            settings,
            embedding__model_name=model,
            retrieval__use_reranker=False,
        )
        try:
            rows.append(
                evaluate_settings(
                    "embeddings",
                    model,
                    variant,
                    goldset,
                    pages,
                    {
                        "embedding_model": model,
                        "chunk_size": variant.chunking.chunk_size,
                        "overlap": variant.chunking.chunk_overlap_ratio,
                        "strategy": variant.chunking.strategy,
                    },
                    console=console,
                )
            )
        except Exception as error:  # модель может не скачаться или не влезть в память
            if console:
                console.print(f"[red]{model}: {error}[/]")
    return rows


def stage_retrieval(
    settings: Settings, goldset: GoldSet, pages, console=None
) -> list[ExperimentRow]:
    """Этап C: плотный / BM25 / гибрид и вклад реранкера."""
    rows: list[ExperimentRow] = []
    for mode in ("dense", "sparse", "hybrid"):
        for rerank in (False, True):
            variant = _variant(settings, retrieval__mode=mode, retrieval__use_reranker=rerank)
            label = f"{mode}{' + реранкер' if rerank else ''}"
            rows.append(
                evaluate_settings(
                    "retrieval",
                    label,
                    variant,
                    goldset,
                    pages,
                    {
                        "mode": mode,
                        "reranker": rerank,
                        "embedding_model": variant.embedding.model_name,
                        "candidates": variant.retrieval.candidates,
                    },
                    console=console,
                )
            )
    return rows


def stage_llm(settings: Settings, goldset: GoldSet, pages, console=None) -> list[ExperimentRow]:
    """Этап D: качество и скорость ответов локальных LLM."""
    from bsrag.evaluation.answers import evaluate_llm

    ensure_index(settings, pages, console=console)
    # Полный набор ручных + ловушки: авто-вопросы для LLM не информативны,
    # а судья удваивает число вызовов. Берём все ловушки и первые 12 каверзных.
    goldset = GoldSet(goldset.of_kind("manual")[:12] + goldset.of_kind("trap"))
    rows: list[ExperimentRow] = []
    for model in LLM_MODELS:
        variant = _variant(settings, generation__model=model)
        try:
            metrics = evaluate_llm(variant, goldset, console=console)
        except Exception as error:
            if console:
                console.print(f"[red]{model}: {error}[/]")
            continue
        row = ExperimentRow("llm", model, {"llm": model}, metrics)
        _log_mlflow(row)
        rows.append(row)
    return rows


STAGES = {
    "chunking": stage_chunking,
    "embeddings": stage_embeddings,
    "retrieval": stage_retrieval,
    "llm": stage_llm,
}


def run_stage(stage: str, settings: Settings, console=None, limit: int | None = None) -> Path:
    path = goldset_path(settings)
    if not path.exists():
        raise FileNotFoundError("Золотой набор не найден. Сначала выполните `bsrag goldset`.")
    goldset = load_goldset(path)
    if limit:
        goldset = GoldSet(goldset.items[:limit])
    pages = load_or_prepare_corpus(settings)

    names = list(STAGES) if stage == "all" else [stage]
    rows: list[ExperimentRow] = []
    for name in names:
        if console:
            console.rule(f"Этап: {name}")
        rows.extend(STAGES[name](copy.deepcopy(settings), goldset, pages, console=console))
        save_rows(rows, settings)

    out = save_rows(rows, settings)
    if console:
        console.print(f"Результаты: [cyan]{out}[/]")
    return out
