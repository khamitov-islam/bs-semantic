"""Нарезка страниц справки на чанки.

Стратегии реализованы поверх готовых сплиттеров LangChain. Своё здесь только то,
чего в библиотеках нет: починка разорванных markdown-таблиц и обогащение чанка
контекстом («хлебные крошки» + путь по заголовкам).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from bsrag.config import ChunkingConfig
from bsrag.parsing.loader import Page

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
_RE_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_RE_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_TOKENIZER_FALLBACK = "intfloat/multilingual-e5-base"
CHUNK_NAMESPACE = uuid.UUID("6f9b1d2e-2f2a-4b4c-9a3b-0f8e1c2d3a4b")


@dataclass
class Chunk:
    """Единица индексации."""

    chunk_id: str
    page_id: str
    title: str
    breadcrumb: str
    headings: str
    text: str
    url: str
    source_path: str
    chunk_index: int = 0
    parent_text: str = ""
    images: list[str] = field(default_factory=list)
    n_chars: int = 0
    n_tokens: int = 0

    @property
    def embed_text(self) -> str:
        """Текст, который реально уходит в эмбеддер.

        Заголовок раздела и путь по справке добавляются в тело чанка: без них
        куски из середины длинной страницы теряют тему («нажмите ОК» — где?).
        """
        header = " > ".join(x for x in (self.breadcrumb, self.headings) if x)
        return f"{header}\n\n{self.text}" if header else self.text

    @property
    def context_text(self) -> str:
        """Текст, который уходит в LLM (для parent-document — расширенный)."""
        return self.parent_text or self.text

    @property
    def citation(self) -> str:
        return f"{self.title} — {self.headings}" if self.headings else self.title


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return AutoTokenizer.from_pretrained(_TOKENIZER_FALLBACK)


def _headings_path(metadata: dict[str, str]) -> str:
    return " > ".join(metadata[key] for key in ("h1", "h2", "h3", "h4") if metadata.get(key))


def _table_header(lines: Sequence[str]) -> list[str] | None:
    """Ищет шапку markdown-таблицы (строка заголовков + разделитель)."""
    for i, line in enumerate(lines[:-1]):
        if _RE_TABLE_ROW.match(line) and _RE_TABLE_SEP.match(lines[i + 1]):
            return [line, lines[i + 1]]
    return None


def repair_table_fragments(fragments: list[str], section: str) -> list[str]:
    """Возвращает шапку таблицы фрагментам, которые начались с её середины.

    Таблицы в справке Business Studio описывают кнопки и пункты меню и легко
    превышают размер чанка. После нарезки второй фрагмент начинается строками
    вида `| {{icon}} | Новый (Ins) | ...` — без шапки такие строки не понять ни
    эмбеддеру, ни LLM.
    """
    header = _table_header(section.split("\n"))
    if not header:
        return fragments

    repaired: list[str] = []
    for fragment in fragments:
        lines = fragment.split("\n")
        first = next((line for line in lines if line.strip()), "")
        starts_mid_table = _RE_TABLE_ROW.match(first) and not _table_header(lines)
        repaired.append("\n".join(header + lines) if starts_mid_table else fragment)
    return repaired


def _merge_tiny(chunks: list[str], min_chars: int, max_chars: int) -> list[str]:
    """Приклеивает слишком короткие куски к соседу, чтобы не плодить мусор.

    Слияние не должно выводить чанк за лимит длины, иначе эмбеддер обрежет хвост.
    """
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars and len(merged[-1]) + len(chunk) <= max_chars:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)
    if len(merged) > 1 and len(merged[0]) < min_chars and len(merged[0]) + len(merged[1]) <= max_chars:
        merged[1] = merged[0] + "\n\n" + merged[1]
        merged.pop(0)
    return merged


class Chunker:
    """Нарезает страницы согласно выбранной стратегии."""

    def __init__(self, config: ChunkingConfig, tokenizer_model: str | None = None) -> None:
        self.config = config
        self.tokenizer = get_tokenizer(tokenizer_model or _TOKENIZER_FALLBACK)
        self.overlap = int(config.chunk_size * config.chunk_overlap_ratio)
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self._splitters: dict[int, RecursiveCharacterTextSplitter] = {}

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _splitter(self, budget: int) -> RecursiveCharacterTextSplitter:
        """Сплиттер под конкретный бюджет длины.

        Перед текстом чанка в эмбеддер уходит контекстная шапка («хлебные крошки»
        и путь по заголовкам), и её длина тоже расходует лимит модели. Бюджет
        считается на страницу и округляется, чтобы сплиттеров было немного.
        """
        budget = max(64, budget)
        if budget not in self._splitters:
            self._splitters[budget] = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                self.tokenizer,
                chunk_size=budget,
                chunk_overlap=min(self.overlap, budget // 4),
                separators=["\n\n", "\n", ". ", "; ", " ", ""],
            )
        return self._splitters[budget]

    def split_page(self, page: Page) -> list[Chunk]:
        strategy = self.config.strategy
        if strategy == "fixed":
            sections = [(page.markdown, "")]
        elif strategy in ("header_recursive", "parent"):
            sections = [
                (doc.page_content, _headings_path(doc.metadata))
                for doc in self.header_splitter.split_text(page.markdown)
            ]
        else:
            raise ValueError(f"Неизвестная стратегия чанкинга: {strategy}")

        breadcrumb = page.breadcrumb_text if self.config.prepend_breadcrumb else ""
        chunks: list[Chunk] = []
        for section_text, headings in sections:
            if not section_text.strip():
                continue
            header = " > ".join(x for x in (breadcrumb, headings) if x)
            budget = self.config.chunk_size - (self.n_tokens(header) + 4 if header else 0)
            if self.n_tokens(section_text) <= budget:
                fragments = [section_text]
            else:
                fragments = self._splitter(budget // 16 * 16).split_text(section_text)
                fragments = repair_table_fragments(fragments, section_text)
            fragments = _merge_tiny(fragments, min_chars=120, max_chars=budget * 3)
            fragments = self._enforce_budget(fragments, budget)

            for fragment in fragments:
                chunks.append(
                    self._make_chunk(
                        page=page,
                        text=fragment.strip(),
                        headings=headings,
                        index=len(chunks),
                        parent=section_text if strategy == "parent" else "",
                    )
                )
        return chunks

    def _enforce_budget(self, fragments: list[str], budget: int) -> list[str]:
        """Дорезает куски, вышедшие за лимит после возврата шапки таблицы и слияний.

        Всё, что длиннее лимита модели, эмбеддер молча обрежет по хвосту, поэтому
        лучше разделить такой кусок явно.
        """
        result: list[str] = []
        for fragment in fragments:
            if self.n_tokens(fragment) <= budget:
                result.append(fragment)
            else:
                result.extend(self._splitter(budget // 16 * 16 - 16).split_text(fragment))
        return result

    def _make_chunk(self, page: Page, text: str, headings: str, index: int, parent: str) -> Chunk:
        # Qdrant принимает в качестве идентификатора только UUID или целое число,
        # поэтому детерминированный ключ чанка сворачиваем в uuid5.
        digest = str(uuid.uuid5(CHUNK_NAMESPACE, f"{page.page_id}:{index}:{text[:64]}"))
        chunk = Chunk(
            chunk_id=digest,
            page_id=page.page_id,
            title=page.title,
            breadcrumb=page.breadcrumb_text if self.config.prepend_breadcrumb else "",
            headings=headings,
            text=text,
            url=page.url,
            source_path=page.source_path,
            chunk_index=index,
            parent_text=parent.strip() if parent and parent.strip() != text else "",
            # К чанку прикрепляем только те иллюстрации, чьи подписи в него попали.
            images=[path for caption, path in page.figures if caption and caption in text],
            n_chars=len(text),
        )
        # Считаем длину того, что реально уходит в модель, — вместе с шапкой.
        chunk.n_tokens = self.n_tokens(chunk.embed_text)
        return chunk

    def split(self, pages: Iterable[Page]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(self.split_page(page))
        return chunks


def chunk_pages(
    pages: Iterable[Page], config: ChunkingConfig, tokenizer_model: str | None = None
) -> list[Chunk]:
    return Chunker(config, tokenizer_model).split(pages)


def _cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


def iter_markdown_tables(markdown: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """Таблицы markdown вместе с путём заголовков над ними."""
    headings: list[tuple[int, str]] = []
    tables: list[tuple[str, list[str], list[list[str]]]] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        heading = re.match(r"^(#{1,4})\s+(.*)$", lines[i])
        if heading:
            level = len(heading.group(1))
            headings = [item for item in headings if item[0] < level]
            headings.append((level, heading.group(2).strip()))
            i += 1
            continue
        if i + 1 < len(lines) and _RE_TABLE_ROW.match(lines[i]) and _RE_TABLE_SEP.match(lines[i + 1]):
            header = _cells(lines[i])
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and _RE_TABLE_ROW.match(lines[i]) and not _RE_TABLE_SEP.match(lines[i]):
                rows.append(_cells(lines[i]))
                i += 1
            path = " > ".join(title for _, title in headings)
            tables.append((path, header, rows))
            continue
        i += 1
    return tables


def chunk_table_rows(
    pages: Iterable[Page], config: ChunkingConfig, tokenizer_model: str | None = None
) -> list[Chunk]:
    """Одна строка таблицы — один чанк: «Клавиша: Ctrl+R; Действие: …».

    Основной индекс остаётся header-aware markdown. Сюда попадают только строки
    таблиц, чтобы BM25 видел короткие токены целиком.
    """
    chunker = Chunker(config, tokenizer_model)
    chunks: list[Chunk] = []
    for page in pages:
        for headings, header, rows in iter_markdown_tables(page.markdown):
            for row in rows:
                pairs = [
                    f"{header[j]}: {cell}" if j < len(header) and header[j] else cell
                    for j, cell in enumerate(row)
                    if cell
                ]
                if not pairs:
                    continue
                text = "- " + "; ".join(pairs)
                chunks.append(
                    chunker._make_chunk(
                        page=page,
                        text=text,
                        headings=headings,
                        index=len(chunks),
                        parent="",
                    )
                )
    return chunks
