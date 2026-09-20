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


def ensure_index(settings: Settings, pages=None, console=None, force: bool = False) -> tuple[int, float]:
    """Строит индекс, если коллекции для такой конфигурации ещё нет.

    Страницы читаются заново из кэша, зависящего от ``table_mode``: иначе абляция
    «таблицы построчно» молча ищет по markdown-корпусу бейслайна.
    """
    from bsrag.chunking import chunk_pages
    from bsrag.corpus import load_or_prepare_corpus
    from bsrag.index import build_index, collection_name, index_exists, read_manifest

    started = time.perf_counter()
    pages = load_or_prepare_corpus(settings)
    if not force and index_exists(settings):
        n_chunks = int(read_manifest(settings).get("n_chunks", 0))
        build_s = 0.0
    else:
        if console:
            verb = "пересобираю" if force else "строю"
            console.print(f"[dim]{verb} индекс {collection_name(settings)}[/]")
        chunks = chunk_pages(pages, settings.chunking, settings.embedding.model_name)
        build_index(chunks, settings)
        n_chunks = len(chunks)
        build_s = time.perf_counter() - started
    table_n, table_s = ensure_table_index(settings, pages, console=console, force=force)
    return n_chunks + table_n, build_s + table_s


def ensure_table_index(settings: Settings, pages, console=None, force: bool = False) -> tuple[int, float]:
    if not settings.retrieval.table_index:
        return 0, 0.0
    from bsrag.chunking import chunk_table_rows
    from bsrag.corpus import load_or_prepare_corpus
    from bsrag.index import build_index, collection_name, index_exists, read_manifest

    if not force and index_exists(settings, suffix="_tables"):
        return int(read_manifest(settings, suffix="_tables").get("n_chunks", 0)), 0.0
    pages = load_or_prepare_corpus(settings)
    if console:
        console.print(f"[dim]строю индекс таблиц {collection_name(settings, '_tables')}[/]")
    started = time.perf_counter()
    chunks = chunk_table_rows(pages, settings.chunking, settings.embedding.model_name)
    build_index(chunks, settings, suffix="_tables")
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
    force_index: bool = False,
) -> ExperimentRow:
    from bsrag.index import collection_name

    n_chunks, build_time = ensure_index(settings, pages, console=console, force=force_index)
    outcomes = run_queries(settings, goldset.answerable, top_k=10)

    metrics = compute_metrics(outcomes)
    for kind, values in metrics_by_kind(outcomes).items():
        for key in ("hit_rate@3", "ndcg@10", "mrr@10"):
            if key in values:
                metrics[f"{kind}_{key}"] = values[key]
    metrics["n_chunks"] = n_chunks
    metrics["index_build_s"] = round(build_time, 1)
    metrics["n_questions"] = float(len(outcomes))
    metrics["n_manual"] = float(sum(1 for o in outcomes if o.item.kind == "manual"))
    params = {**params, "collection": collection_name(settings)}

    row = ExperimentRow(stage=stage, name=name, params=params, metrics=metrics)
    _log_mlflow(row)
    if console:
        console.print(
            f"[bold]{name}[/]: nDCG@10 {metrics.get('ndcg@10', 0):.3f} · "
            f"hit@1 {metrics.get('hit_rate@1', 0):.3f} · "
            f"hit@3 {metrics.get('hit_rate@3', 0):.3f} · "
            f"n={int(metrics.get('n_questions', 0))} · "
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


def _manuals(goldset: GoldSet) -> GoldSet:
    """A–E и сравнение профилей — только ручные вопросы, без синтетики из чанка."""
    return GoldSet(goldset.of_kind("manual"))


def _factor_base(settings: Settings, **changes: Any) -> Settings:
    """Опора этапов A–C: без улучшений E, иначе факторы смешаются.

    Крошки включены, индекс таблиц и раскрытие запроса выключены. Чанкинг
    512/15% markdown — то, что ушло в замороженное решение.
    """
    return _variant(
        settings,
        chunking__strategy="header_recursive",
        chunking__chunk_size=512,
        chunking__chunk_overlap_ratio=0.15,
        chunking__table_mode="markdown",
        chunking__prepend_breadcrumb=True,
        retrieval__table_index=False,
        retrieval__query_expand=False,
        retrieval__weighted_fusion=False,
        retrieval__candidates=16,
        **changes,
    )


def stage_chunking(settings: Settings, goldset: GoldSet, pages, console=None) -> list[ExperimentRow]:
    """Этап A: сетка размер × оверлап. Абляции — отдельный этап ``ablations``.

    Считается на быстрой модели e5-base, без реранкера: иначе он сглаживает
    разницу между нарезками. Улучшения E выключены.
    """
    goldset = _manuals(goldset)
    base = _factor_base(
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
    return rows


def stage_embeddings(
    settings: Settings, goldset: GoldSet, pages, console=None
) -> list[ExperimentRow]:
    """Этап B: сравнение эмбеддеров на зафиксированной нарезке 512/15%."""
    goldset = _manuals(goldset)
    rows: list[ExperimentRow] = []
    for model in EMBEDDING_MODELS:
        variant = _factor_base(
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
    """Этап C: плотный / BM25 / гибрид и вклад реранкера на победителе B."""
    goldset = _manuals(goldset)
    rows: list[ExperimentRow] = []
    for mode in ("dense", "sparse", "hybrid"):
        for rerank in (False, True):
            variant = _factor_base(
                settings,
                retrieval__mode=mode,
                retrieval__use_reranker=rerank,
            )
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
    # Все ручные + ловушки. Авто из чанка для LLM не информативны.
    goldset = GoldSet(goldset.of_kind("manual") + goldset.of_kind("trap"))
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


def stage_ablations(settings: Settings, goldset: GoldSet, pages, console=None) -> list[ExperimentRow]:
    """Абляции нарезки на 512/15% — на том размере, который ушёл в решение.

    Отдельные коллекции, ``force_index``: иначе варианты читают чужой индекс.
    """
    goldset = _manuals(goldset)
    base = _factor_base(
        settings,
        embedding__model_name=BASELINE_EMBEDDING,
        retrieval__use_reranker=False,
    )
    rows: list[ExperimentRow] = []
    ablations = {
        "fixed (без учёта заголовков)": {"chunking__strategy": "fixed"},
        "parent-document": {"chunking__strategy": "parent"},
        "таблицы построчно": {"chunking__table_mode": "rows"},
        "no breadcrumbs": {"chunking__prepend_breadcrumb": False},
    }
    for label, changes in ablations.items():
        variant = _variant(base, **changes)
        rows.append(
            evaluate_settings(
                "chunking",
                f"{label} @512/15%",
                variant,
                goldset,
                pages,
                {
                    "strategy": variant.chunking.strategy,
                    "chunk_size": 512,
                    "overlap": 0.15,
                    "table_mode": variant.chunking.table_mode,
                    "breadcrumb": variant.chunking.prepend_breadcrumb,
                    "embedding_model": BASELINE_EMBEDDING,
                    "ablation": label,
                },
                console=console,
                force_index=True,
            )
        )
    return rows


def stage_improve(settings: Settings, goldset: GoldSet, pages, console=None) -> list[ExperimentRow]:
    """Этап E: один фактор поверх победителя C, затем собранный улучшенный профиль."""
    goldset = _manuals(goldset)
    control = _factor_base(
        settings,
        retrieval__mode="hybrid",
        retrieval__use_reranker=True,
    )
    rows: list[ExperimentRow] = []
    trials: list[tuple[str, Settings]] = [
        ("опора C: hybrid+реранкер", control),
        (
            "+ лексическое раскрытие",
            _variant(control, retrieval__query_expand=True),
        ),
        (
            "+ без крошек",
            _variant(control, chunking__prepend_breadcrumb=False),
        ),
        (
            "+ индекс таблиц",
            _variant(control, retrieval__table_index=True),
        ),
        (
            "+ 32 кандидата",
            _variant(control, retrieval__candidates=32),
        ),
        (
            "взвешенный RRF",
            _variant(control, retrieval__weighted_fusion=True),
        ),
        (
            "весь корпус rows",
            _variant(control, chunking__table_mode="rows"),
        ),
        ("всё вместе (улучшенный)", settings),
    ]
    for name, variant in trials:
        rows.append(
            evaluate_settings(
                "improve",
                name,
                variant,
                goldset,
                pages,
                {
                    "mode": variant.retrieval.mode,
                    "reranker": variant.retrieval.use_reranker,
                    "query_expand": variant.retrieval.query_expand,
                    "table_index": variant.retrieval.table_index,
                    "weighted_fusion": variant.retrieval.weighted_fusion,
                    "table_mode": variant.chunking.table_mode,
                    "breadcrumb": variant.chunking.prepend_breadcrumb,
                    "embedding_model": variant.embedding.model_name,
                    "candidates": variant.retrieval.candidates,
                },
                console=console,
            )
        )
    return rows


def stage_warmup(settings: Settings, goldset: GoldSet, pages, console=None) -> list[ExperimentRow]:
    """Холодный vs тёплый первый поиск: обоснование прогрева в UI."""
    from bsrag.embeddings import get_embeddings
    from bsrag.retrieval import Retriever, get_reranker

    ensure_index(settings, pages, console=console)
    get_embeddings.cache_clear()
    get_reranker.cache_clear()

    retriever = Retriever(settings)
    question = goldset.answerable[0].question if goldset.answerable else "проверка"

    cold = retriever.search(question, top_k=5)
    warm = retriever.search(question, top_k=5)
    rows = [
        ExperimentRow(
            "warmup",
            "холодный первый поиск",
            {"wave": "cold"},
            {
                "latency_p50_ms": cold.total_ms,
                "retrieval_ms_mean": cold.retrieval_ms,
                "rerank_ms_mean": cold.rerank_ms,
            },
        ),
        ExperimentRow(
            "warmup",
            "повторный поиск (после прогрева)",
            {"wave": "warm"},
            {
                "latency_p50_ms": warm.total_ms,
                "retrieval_ms_mean": warm.retrieval_ms,
                "rerank_ms_mean": warm.rerank_ms,
            },
        ),
    ]
    if console:
        console.print(
            f"[bold]холодный[/]: {cold.total_ms:.0f} мс · "
            f"[bold]тёплый[/]: {warm.total_ms:.0f} мс"
        )
    return rows


def stage_goldset_expand(settings: Settings, goldset: GoldSet, pages, console=None) -> list[ExperimentRow]:
    """Бейслайн vs улучшенный только на ручных вопросах текущего goldset."""
    from bsrag.config import PROJECT_ROOT, load_settings

    manuals = GoldSet(goldset.of_kind("manual"))
    baseline = load_settings(PROJECT_ROOT / "configs" / "baseline.yaml")
    rows: list[ExperimentRow] = []
    for name, variant in (("улучшенный профиль", settings), ("бейслайн", baseline)):
        rows.append(
            evaluate_settings(
                "goldset_expand",
                name,
                variant,
                manuals,
                pages,
                {
                    "profile": name,
                    "n_manual": len(manuals.of_kind("manual")),
                    "embedding_model": variant.embedding.model_name,
                    "reranker": variant.retrieval.use_reranker,
                    "table_index": variant.retrieval.table_index,
                    "query_expand": variant.retrieval.query_expand,
                    "breadcrumb": variant.chunking.prepend_breadcrumb,
                },
                console=console,
            )
        )
    return rows


STAGES = {
    "chunking": stage_chunking,
    "ablations": stage_ablations,
    "embeddings": stage_embeddings,
    "retrieval": stage_retrieval,
    "improve": stage_improve,
    "warmup": stage_warmup,
    "llm": stage_llm,
    "goldset_expand": stage_goldset_expand,
}


def run_stage(stage: str, settings: Settings, console=None, limit: int | None = None) -> Path:
    path = goldset_path(settings)
    if not path.exists():
        raise FileNotFoundError("Золотой набор не найден. Сначала выполните `bsrag goldset`.")
    goldset = load_goldset(path)
    if limit:
        goldset = GoldSet(goldset.items[:limit])
    if console:
        console.print(
            f"Goldset: ручных {len(goldset.of_kind('manual'))}, "
            f"ловушек {len(goldset.of_kind('trap'))}, "
            f"авто {len(goldset.of_kind('auto'))} "
            f"(A–E ищут только по ручным)"
        )
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
