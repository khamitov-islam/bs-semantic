"""Загрузка корпуса справки: обход файлов DokuWiki и конвертация в документы."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bsrag.config import Settings
from bsrag.parsing.dokuwiki import DokuWikiConverter, extract_title


@dataclass
class Page:
    """Страница справки после очистки разметки."""

    page_id: str
    title: str
    markdown: str
    source_path: str
    namespace: str
    breadcrumb: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    figures: list[list[str]] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    tables: int = 0
    n_chars: int = 0
    url: str = ""

    @property
    def breadcrumb_text(self) -> str:
        return " / ".join(self.breadcrumb)


def iter_page_files(
    pages_dir: Path,
    namespaces: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Path]:
    files = sorted(p for p in pages_dir.rglob("*") if p.suffix.lower() == ".txt" and p.is_file())
    allowed = tuple(namespaces) if namespaces else None
    skipped = tuple(x.lower() for x in exclude) if exclude else ()
    return [
        f
        for f in files
        if (allowed is None or f.relative_to(pages_dir).as_posix().startswith(allowed))
        and not page_id_of(f, pages_dir).startswith(skipped)
    ]


def page_id_of(path: Path, pages_dir: Path) -> str:
    return path.relative_to(pages_dir).with_suffix("").as_posix().lower()


def build_title_map(files: list[Path], pages_dir: Path) -> dict[str, str]:
    """Первый проход: id страницы -> её заголовок.

    Нужен, чтобы ссылки без подписи превращались в осмысленный текст и чтобы
    собрать «хлебные крошки» из заголовков родительских разделов.
    """
    titles: dict[str, str] = {}
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        pid = page_id_of(path, pages_dir)
        title = extract_title(raw)
        if title:
            titles[pid] = title
    return titles


def _breadcrumb_for(page_id: str, titles: dict[str, str]) -> list[str]:
    """Собирает путь по разделам из заголовков родительских страниц.

    В DokuWiki раздел `ru/manual/interface` описывается либо страницей
    `ru/manual/interface`, либо `ru/manual/interface/interface` — проверяем оба.
    """
    parts = page_id.split("/")
    crumbs: list[str] = []
    for depth in range(1, len(parts)):
        prefix = "/".join(parts[:depth])
        title = titles.get(prefix) or titles.get(f"{prefix}/{parts[depth - 1]}")
        if title and title not in crumbs:
            crumbs.append(title)
    return crumbs


def load_pages(settings: Settings, keep_all: bool = False) -> list[Page]:
    pages_dir = settings.resolve(settings.corpus.pages_dir)
    if not pages_dir.exists():
        raise FileNotFoundError(
            f"Каталог со страницами справки не найден: {pages_dir}. "
            "Укажите путь в configs/default.yaml -> corpus.pages_dir"
        )

    files = iter_page_files(pages_dir, settings.corpus.namespaces, settings.corpus.exclude)
    titles = build_title_map(files, pages_dir)
    converter = DokuWikiConverter(
        title_map=titles,
        table_mode=settings.chunking.table_mode,  # type: ignore[arg-type]
    )

    pages: list[Page] = []
    for path in files:
        pid = page_id_of(path, pages_dir)
        raw = path.read_text(encoding="utf-8", errors="replace")
        parsed = converter.convert(raw)

        if not keep_all:
            if len(parsed.markdown) < settings.corpus.min_chars:
                continue
            if settings.corpus.drop_index_pages and parsed.is_index_page:
                continue

        pages.append(
            Page(
                page_id=pid,
                title=parsed.title or titles.get(pid, pid.rsplit("/", 1)[-1]),
                markdown=parsed.markdown,
                source_path=path.relative_to(settings.resolve(Path("."))).as_posix()
                if path.is_relative_to(settings.resolve(Path(".")))
                else str(path),
                namespace=pid.rsplit("/", 1)[0] if "/" in pid else "",
                breadcrumb=_breadcrumb_for(pid, titles),
                images=parsed.images,
                figures=[[caption, path] for caption, path in parsed.figures],
                links=parsed.links,
                tables=parsed.tables,
                n_chars=len(parsed.markdown),
                url=settings.corpus.doc_url_template.format(page_id=pid),
            )
        )
    return pages


def save_pages(pages: list[Page], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for page in pages:
            fh.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
    return path


def read_pages(path: Path) -> list[Page]:
    with path.open(encoding="utf-8") as fh:
        return [Page(**json.loads(line)) for line in fh if line.strip()]
