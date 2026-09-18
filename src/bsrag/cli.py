"""Консольный интерфейс: разбор корпуса, индексация, поиск, ответы, бенчмарки."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from bsrag.config import PROJECT_ROOT, load_settings

app = typer.Typer(
    add_completion=False,
    help="Локальный семантический поиск по справке Business Studio.",
    no_args_is_help=True,
)
console = Console()

ConfigOption = typer.Option(None, "--config", "-c", help="Путь к YAML-конфигу.")


def _settings(config: Optional[str], **overrides):
    return load_settings(config, **overrides)


def _overrides(**kwargs) -> dict:
    """Собирает частичный конфиг из непустых опций командной строки."""
    sections: dict[str, dict] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        section, _, field = key.partition("__")
        sections.setdefault(section, {})[field] = value
    return sections


@app.command()
def ingest(config: Optional[str] = ConfigOption) -> None:
    """Очистить разметку DokuWiki и сохранить страницы в data/processed."""
    # Импорт внутри команды: тянуть torch и onnxruntime ради разбора текста —
    # это лишние десятки секунд на старте.
    from bsrag.corpus import pages_file, prepare_corpus

    settings = _settings(config)
    with console.status("Разбираю разметку DokuWiki..."):
        pages = prepare_corpus(settings)

    chars = sum(p.n_chars for p in pages)
    table = Table(title="Корпус справки после очистки", show_header=False)
    table.add_row("Страниц", str(len(pages)))
    table.add_row("Символов", f"{chars:,}".replace(",", " "))
    table.add_row("Таблиц", str(sum(p.tables for p in pages)))
    table.add_row("Иллюстраций", str(sum(len(p.images) for p in pages)))
    table.add_row("Сохранено в", str(pages_file(settings)))
    console.print(table)

    by_ns = {}
    for page in pages:
        root = page.page_id.split("/")[1] if page.page_id.count("/") >= 1 else page.page_id
        by_ns[root] = by_ns.get(root, 0) + 1
    console.print("Разделы: " + ", ".join(f"{k} ({v})" for k, v in sorted(by_ns.items(), key=lambda x: -x[1])))


@app.command()
def index(
    config: Optional[str] = ConfigOption,
    model: Optional[str] = typer.Option(None, help="Модель эмбеддингов."),
    chunk_size: Optional[int] = typer.Option(None, help="Размер чанка в токенах."),
    strategy: Optional[str] = typer.Option(None, help="header_recursive | fixed | parent."),
    mode: Optional[str] = typer.Option(None, help="hybrid | dense | sparse."),
) -> None:
    """Нарезать корпус на чанки и построить индекс Qdrant."""
    from bsrag.index import collection_name
    from bsrag.pipeline import build

    settings = _settings(
        config,
        **_overrides(
            embedding__model_name=model,
            chunking__chunk_size=chunk_size,
            chunking__strategy=strategy,
            retrieval__mode=mode,
        ),
    )
    console.print(f"Модель: [bold]{settings.embedding.model_name}[/], "
                  f"чанк: {settings.chunking.chunk_size} токенов, режим: {settings.retrieval.mode}")
    with console.status("Индексирую... (первый запуск скачивает модель)"):
        chunks = build(settings)

    tokens = [c.n_tokens for c in chunks]
    table = Table(title="Индекс построен", show_header=False)
    table.add_row("Коллекция", collection_name(settings))
    table.add_row("Чанков", str(len(chunks)))
    table.add_row("Страниц", str(len({c.page_id for c in chunks})))
    table.add_row("Токенов в чанке (среднее)", f"{sum(tokens) / max(len(tokens), 1):.0f}")
    table.add_row("Токенов в чанке (макс.)", str(max(tokens, default=0)))
    console.print(table)


def _print_sources(hits, show_text: bool = False) -> None:
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        location = " > ".join(x for x in (chunk.breadcrumb, chunk.title, chunk.headings) if x)
        body = f"\n\n{chunk.text[:600]}" if show_text else ""
        console.print(
            Panel(
                f"[bold]{location}[/]\n"
                f"Файл: [cyan]{chunk.source_path}[/]\n"
                f"Страница: {chunk.page_id}  ·  Релевантность: {hit.score:.3f}"
                f"{body}",
                title=f"[{i}]",
                border_style="dim",
            )
        )


@app.command()
def search(
    query: str = typer.Argument(..., help="Поисковый запрос."),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Без кросс-энкодера."),
    show_text: bool = typer.Option(False, "--text", help="Показать текст фрагментов."),
    config: Optional[str] = ConfigOption,
) -> None:
    """Найти фрагменты справки без генерации ответа."""
    from bsrag.retrieval import Retriever

    settings = _settings(config)
    retriever = Retriever(settings)
    result = retriever.search(query, top_k=top_k, use_reranker=not no_rerank)
    _print_sources(result.hits, show_text=show_text)
    console.print(
        f"[dim]поиск {result.retrieval_ms:.0f} мс · реранкинг {result.rerank_ms:.0f} мс[/]"
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Вопрос по функционалу Business Studio."),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k"),
    llm: Optional[str] = typer.Option(None, help="Модель Ollama."),
    config: Optional[str] = ConfigOption,
) -> None:
    """Ответить на вопрос по справке со ссылками на источники."""
    from bsrag.pipeline import RagPipeline

    settings = _settings(config, **_overrides(generation__model=llm))
    pipeline = RagPipeline(settings)

    hits, tokens = pipeline.ask_stream(question, top_k=top_k)
    console.print(f"[dim]Модель: {settings.generation.model}[/]\n")
    answer = ""
    for token in tokens:
        answer += token
        console.print(token, end="")
    console.print("\n")
    console.print("[bold]Источники:[/]")
    _print_sources(hits)


@app.command()
def chat(config: Optional[str] = ConfigOption) -> None:
    """Диалоговый режим в терминале."""
    from bsrag.pipeline import RagPipeline

    settings = _settings(config)
    pipeline = RagPipeline(settings)
    console.print("[dim]Вопрос по справке Business Studio (пустая строка — выход)[/]\n")
    while True:
        try:
            question = console.input("[bold cyan]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        hits, tokens = pipeline.ask_stream(question)
        for token in tokens:
            console.print(token, end="")
        console.print()
        for i, hit in enumerate(hits, start=1):
            console.print(f"[dim][{i}] {hit.chunk.citation} — {hit.chunk.source_path}[/]")
        console.print()


@app.command()
def goldset(
    n_auto: int = typer.Option(60, help="Сколько вопросов сгенерировать локальной LLM."),
    llm: Optional[str] = typer.Option(None, help="Модель-генератор вопросов."),
    config: Optional[str] = ConfigOption,
) -> None:
    """Сгенерировать синтетические вопросы и собрать общий золотой набор."""
    from bsrag.evaluation.goldset import build_goldset

    settings = _settings(config, **_overrides(generation__model=llm))
    path = build_goldset(settings, n_auto=n_auto, console=console)
    console.print(f"Золотой набор сохранён: [cyan]{path}[/]")


@app.command()
def bench(
    stage: str = typer.Argument("all", help="chunking | embeddings | retrieval | llm | all"),
    config: Optional[str] = ConfigOption,
    limit: Optional[int] = typer.Option(None, help="Ограничить число вопросов (для отладки)."),
) -> None:
    """Прогнать бенчмарки и сохранить метрики в reports/."""
    from bsrag.evaluation.runner import run_stage

    settings = _settings(config)
    run_stage(stage, settings, console=console, limit=limit)


@app.command()
def ui(port: int = typer.Option(8501), config: Optional[str] = ConfigOption) -> None:
    """Запустить веб-интерфейс на Streamlit."""
    app_path = PROJECT_ROOT / "app" / "Поиск.py"
    env_config = ["--", "--config", config] if config else []
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
        + env_config,
        cwd=PROJECT_ROOT,
        check=False,
    )


@app.command()
def doctor(config: Optional[str] = ConfigOption) -> None:
    """Проверить готовность окружения: корпус, индекс, Ollama, устройство."""
    from bsrag.embeddings import resolve_device
    from bsrag.generation import available_models
    from bsrag.index import collection_name, index_path

    settings = _settings(config)
    table = Table(title="Проверка окружения")
    table.add_column("Компонент")
    table.add_column("Статус")

    pages_dir = settings.resolve(settings.corpus.pages_dir)
    n_pages = len(list(pages_dir.rglob("*.txt"))) if pages_dir.exists() else 0
    table.add_row("Корпус справки", f"[green]{n_pages} файлов[/]" if n_pages else "[red]не найден[/]")

    manifest = index_path(settings) / f"{collection_name(settings)}.manifest.json"
    table.add_row("Индекс", f"[green]{manifest.name}[/]" if manifest.exists() else "[yellow]не построен[/]")

    table.add_row("Устройство", resolve_device(settings.embedding.device))

    models = available_models(settings.generation.base_url)
    if models:
        mark = "green" if settings.generation.model in models else "yellow"
        table.add_row("Ollama", f"[{mark}]{len(models)} моделей: {', '.join(models)}[/]")
    else:
        table.add_row("Ollama", "[red]недоступен на " + settings.generation.base_url + "[/]")

    console.print(table)


@app.command()
def show(page_id: str, config: Optional[str] = ConfigOption) -> None:
    """Показать очищенный Markdown страницы справки."""
    from bsrag.corpus import load_or_prepare_corpus

    settings = _settings(config)
    pages = {p.page_id: p for p in load_or_prepare_corpus(settings)}
    page = pages.get(page_id.lower().strip("/"))
    if not page:
        matches = [pid for pid in pages if page_id.lower() in pid][:10]
        console.print(f"[red]Страница не найдена.[/] Похожие: {', '.join(matches) or '—'}")
        raise typer.Exit(1)
    console.print(Markdown(page.markdown))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
