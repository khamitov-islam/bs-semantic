"""Конвертер разметки DokuWiki (с плагинами Business Studio) в Markdown.

Готового парсера DokuWiki под Python, понимающего плагины Business Studio
(``bslink``, ``startTableBox``, ``contextnavigator``), не существует, поэтому
конвертер написан здесь. Всё остальное в проекте построено на готовых библиотеках.

Разбор поблочный: сначала из текста вырезаются защищённые участки (код, html,
``%%``-экранирование), затем построчно разбираются блочные конструкции
(заголовки, таблицы, списки), и уже внутри строк работают инлайновые замены.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

TableMode = Literal["markdown", "rows"]

# --- защищённые блоки -------------------------------------------------------

_RE_CODE = re.compile(r"<code(?:\s+[^>]*)?>(.*?)</code>", re.DOTALL | re.IGNORECASE)
_RE_FILE = re.compile(r"<file(?:\s+[^>]*)?>(.*?)</file>", re.DOTALL | re.IGNORECASE)
_RE_HTML = re.compile(r"<html>(.*?)</html>", re.DOTALL | re.IGNORECASE)
_RE_NOWIKI = re.compile(r"%%(.*?)%%", re.DOTALL)
_RE_IFRAME_SRC = re.compile(r'src="([^"]+)"')

# --- блочные конструкции ----------------------------------------------------

_RE_HEADING = re.compile(r"^\s*(={2,6})\s*(.*?)\s*\1\s*$")
_RE_TABLE_BOX_START = re.compile(r"^\s*<startTableBox[^>]*>\s*$", re.IGNORECASE)
_RE_TABLE_BOX_END = re.compile(r"^\s*<endTableBox(?:\|(.*?))?>\s*$", re.IGNORECASE)
_RE_LIST_ITEM = re.compile(r"^(\s+)([*-])\s+(.*)$")
_RE_NOTE_OPEN = re.compile(r"<note(?:\s+\w+)?>", re.IGNORECASE)
_RE_NOTE_CLOSE = re.compile(r"</note>", re.IGNORECASE)
_RE_PLUGIN_MACRO = re.compile(r"\[<[^\]]*>\]")
_RE_HR = re.compile(r"^\s*-{4,}\s*$")

# --- инлайновые конструкции -------------------------------------------------

_RE_FIGURE = re.compile(r"\[\{\{([^{}]*)\}\}\]")
# Цель ссылки может содержать одиночные скобки (`?s[]=...` в ссылках с поиском),
# поэтому останавливаемся на первом `]]`, а не запрещаем скобки внутри.
_RE_LINK = re.compile(r"\[\[(.*?)\]\]", re.DOTALL)
_RE_MULTILINE_CONSTRUCT = re.compile(r"\{\{[^{}]*?\}\}|\[\[.*?\]\]", re.DOTALL)
_RE_BSLINK = re.compile(r"\{\{bslink>(.*?)\}\}", re.DOTALL)
_RE_MEDIA = re.compile(r"\{\{([^{}]*)\}\}")
_RE_FOOTNOTE = re.compile(r"\(\(([^()]*(?:\([^()]*\)[^()]*)*)\)\)")
_RE_ITALIC = re.compile(r"(?<![:/])//(?=\S)(.+?)(?<=\S)//")
_RE_MONO = re.compile(r"''(.+?)''")
_RE_UNDERLINE = re.compile(r"__(.+?)__")
_RE_LINEBREAK = re.compile(r"\\\\\s*")
_RE_IMAGE_EXT = re.compile(r"\.(png|jpe?g|gif|svg|webp)\b", re.IGNORECASE)
_RE_MULTI_BLANK = re.compile(r"\n{3,}")
_RE_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


@dataclass
class ParsedPage:
    """Результат конвертации одной страницы DokuWiki."""

    markdown: str
    title: str
    images: list[str] = field(default_factory=list)
    # Подпись рисунка -> путь к файлу: подпись остаётся в тексте, поэтому по ней
    # интерфейс находит и показывает нужный скриншот рядом с фрагментом.
    figures: list[tuple[str, str]] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    tables: int = 0
    is_index_page: bool = False


def normalize_page_id(target: str) -> str:
    """Приводит ссылку DokuWiki к каноническому id страницы.

    В корпусе встречаются варианты записи ``ru/manual/manage_model//manage_model``,
    ``ru:manual:terms`` и ссылки с якорями ``...#сортировка_строк``.
    """
    target = target.split("#", 1)[0].strip()
    target = target.replace(":", "/")
    target = re.sub(r"/{2,}", "/", target)
    return target.strip("/").lower()


def _anchor_of(target: str) -> str:
    return target.split("#", 1)[1].strip() if "#" in target else ""


class DokuWikiConverter:
    """Конвертирует текст страницы DokuWiki в Markdown.

    ``title_map`` сопоставляет id страницы её заголовку: он нужен, чтобы ссылки
    без подписи (``[[ru/manual/terms]]``) превращались в читаемый текст, а не в
    технический путь. Страницы-оглавления состоят почти целиком из таких ссылок.
    """

    def __init__(
        self,
        title_map: dict[str, str] | None = None,
        table_mode: TableMode = "markdown",
        keep_code: bool = True,
    ) -> None:
        self.title_map = title_map or {}
        self.table_mode = table_mode
        self.keep_code = keep_code

    # -- публичный API -------------------------------------------------------

    def convert(self, raw: str) -> ParsedPage:
        self._images: list[str] = []
        self._figures: list[tuple[str, str]] = []
        self._links: list[str] = []
        self._tables = 0
        self._protected: list[str] = []

        # BOM стоит у 51 страницы из 653 и без удаления ломает заголовок первой
        # строки: `\ufeff====== Права пользователя ======` перестаёт быть заголовком.
        text = raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00a0", " ").replace("\u200b", "")
        text = self._protect(text)
        text = _RE_PLUGIN_MACRO.sub("", text)
        # Ссылки и картинки в справке иногда разорваны переносом строки, а
        # блочный разбор идёт построчно — сначала склеиваем такие конструкции.
        text = _RE_MULTILINE_CONSTRUCT.sub(lambda m: " ".join(m.group(0).split()), text)

        body, link_only, content_lines = self._convert_blocks(text.split("\n"))
        body = self._restore(body)
        body = _RE_TRAILING_WS.sub("", body)
        body = _RE_MULTI_BLANK.sub("\n\n", body).strip()

        title = self._extract_title(body)
        is_index = content_lines > 0 and link_only / content_lines > 0.7

        return ParsedPage(
            markdown=body,
            title=title,
            images=self._images,
            figures=self._figures,
            links=self._links,
            tables=self._tables,
            is_index_page=is_index,
        )

    # -- защищённые блоки ----------------------------------------------------

    def _protect(self, text: str) -> str:
        def store(payload: str) -> str:
            self._protected.append(payload)
            return f"\x00PROTECTED{len(self._protected) - 1}\x00"

        def on_code(m: re.Match[str]) -> str:
            if not self.keep_code:
                return ""
            return store("```\n" + m.group(1).strip("\n") + "\n```")

        def on_html(m: re.Match[str]) -> str:
            # В справке <html> используется только для встроенных видео с YouTube.
            src = _RE_IFRAME_SRC.search(m.group(1))
            return store(f"Видеоролик: {src.group(1)}") if src else ""

        def on_nowiki(m: re.Match[str]) -> str:
            return store(m.group(1))

        text = _RE_CODE.sub(on_code, text)
        text = _RE_FILE.sub(on_code, text)
        text = _RE_HTML.sub(on_html, text)
        return _RE_NOWIKI.sub(on_nowiki, text)

    def _restore(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            return self._protected[int(m.group(1))]

        return re.sub(r"\x00PROTECTED(\d+)\x00", repl, text)

    # -- блочный разбор ------------------------------------------------------

    def _convert_blocks(self, lines: list[str]) -> tuple[str, int, int]:
        out: list[str] = []
        link_only = 0
        content = 0
        i = 0

        while i < len(lines):
            line = lines[i]

            if _RE_TABLE_BOX_START.match(line):
                i, caption, rows = self._collect_table_box(lines, i)
                out.extend(self._render_table(rows, caption))
                continue

            if self._is_table_line(line):
                start = i
                while i < len(lines) and self._is_table_line(lines[i]):
                    i += 1
                out.extend(self._render_table(lines[start:i], None))
                continue

            heading = _RE_HEADING.match(line)
            if heading:
                marks, text = heading.groups()
                if text:  # `== ==` — пустой заголовок-разделитель, он ничего не значит
                    level = 7 - len(marks)
                    out.append(f"{'#' * level} {self._inline(text)}")
                    out.append("")
                i += 1
                continue

            if _RE_HR.match(line):
                out.append("---")
                i += 1
                continue

            item = _RE_LIST_ITEM.match(line)
            if item:
                indent, marker, text = item.groups()
                level = max(0, (len(indent.expandtabs(2)) - 2) // 2)
                bullet = "-" if marker == "*" else "1."
                rendered = self._inline(text)
                out.append(f"{'  ' * level}{bullet} {rendered}")
                content += 1
                if self._is_link_only(text):
                    link_only += 1
                i += 1
                continue

            rendered = self._inline(line)
            rendered = _RE_NOTE_OPEN.sub("> **Примечание.**\n> ", rendered)
            rendered = _RE_NOTE_CLOSE.sub("", rendered)
            out.append(rendered)
            if rendered.strip():
                content += 1
                if self._is_link_only(line):
                    link_only += 1
            i += 1

        return "\n".join(out), link_only, content

    def _collect_table_box(self, lines: list[str], i: int) -> tuple[int, str | None, list[str]]:
        i += 1
        rows: list[str] = []
        caption: str | None = None
        while i < len(lines):
            end = _RE_TABLE_BOX_END.match(lines[i])
            if end:
                caption = (end.group(1) or "").strip() or None
                i += 1
                break
            rows.append(lines[i])
            i += 1
        return i, caption, rows

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        return len(stripped) > 1 and stripped[0] in "|^" and stripped[-1] in "|^"

    def _render_table(self, raw_rows: list[str], caption: str | None) -> list[str]:
        parsed: list[tuple[bool, list[str]]] = []
        for raw in raw_rows:
            stripped = raw.strip()
            if not self._is_table_line(stripped):
                continue
            is_header = stripped[0] == "^"
            # Инлайновые замены выполняются до разбиения на ячейки: и `bslink`,
            # и обычные ссылки содержат внутри себя `|`, поэтому разбиение
            # «сначала по разделителям» разрывает такие конструкции пополам.
            converted = self._inline(stripped, in_table=True).strip()
            cells = [c.strip() for c in re.split(r"[|^]", converted)[1:-1]]
            parsed.append((is_header, cells))

        parsed = self._expand_rowspans(parsed)

        if not parsed:
            return [self._inline(r) for r in raw_rows]

        parsed = self._drop_empty_columns(parsed)
        self._tables += 1
        # Подпись в DokuWiki-плагине Business Studio стоит после таблицы; поднимаем
        # её наверх, иначе при нарезке на чанки таблица теряет заголовок.
        out = [""]
        if caption:
            out.append(f"**{caption}**")
            out.append("")

        header_cells = parsed[0][1] if parsed[0][0] else []
        if self.table_mode == "rows" and header_cells:
            out.extend(self._render_table_as_rows(header_cells, parsed[1:]))
        else:
            out.extend(self._render_table_as_markdown(parsed))
        out.append("")
        return out

    @staticmethod
    def _expand_rowspans(parsed: list[tuple[bool, list[str]]]) -> list[tuple[bool, list[str]]]:
        """Раскрывает объединение строк: `:::` означает «значение как в строке выше».

        В таблице горячих клавиш одна комбинация расписана на десяток строк через
        `:::`. Если просто удалить эти метки, у большинства строк не останется ни
        клавиши, ни действия — и на вопрос «чем обновить список» ответить нечем.
        """
        expanded: list[tuple[bool, list[str]]] = []
        previous: list[str] = []
        for is_header, cells in parsed:
            filled = [
                previous[i] if cell == ":::" and i < len(previous) else cell
                for i, cell in enumerate(cells)
            ]
            filled = ["" if cell == ":::" else cell for cell in filled]
            expanded.append((is_header, filled))
            previous = filled
        return expanded

    @staticmethod
    def _drop_empty_columns(parsed: list[tuple[bool, list[str]]]) -> list[tuple[bool, list[str]]]:
        """Убирает колонки, целиком состоявшие из иконок (после очистки — пустые).

        Смотрим только на строки данных: у колонки с иконками заголовок обычно
        осмысленный («Кнопка»), а все ячейки под ним пустые.
        """
        width = max(len(cells) for _, cells in parsed)
        body = [cells for is_header, cells in parsed if not is_header] or [c for _, c in parsed]
        keep = [col for col in range(width) if any(col < len(c) and c[col].strip() for c in body)]
        if len(keep) == width:
            return parsed
        return [
            (is_header, [cells[col] if col < len(cells) else "" for col in keep])
            for is_header, cells in parsed
        ]

    @staticmethod
    def _render_table_as_markdown(parsed: list[tuple[bool, list[str]]]) -> list[str]:
        width = max(len(cells) for _, cells in parsed)
        rendered: list[str] = []
        separator_written = False
        for idx, (is_header, cells) in enumerate(parsed):
            padded = [c.replace("|", "\\|") or " " for c in cells] + [" "] * (width - len(cells))
            rendered.append("| " + " | ".join(padded) + " |")
            if is_header and not separator_written:
                rendered.append("| " + " | ".join(["---"] * width) + " |")
                separator_written = True
            elif idx == 0 and not separator_written:
                rendered.insert(0, "| " + " | ".join([" "] * width) + " |")
                rendered.insert(1, "| " + " | ".join(["---"] * width) + " |")
                separator_written = True
        return rendered

    @staticmethod
    def _render_table_as_rows(header: list[str], body: list[tuple[bool, list[str]]]) -> list[str]:
        """Разворачивает таблицу построчно: «Колонка: значение».

        Альтернативная стратегия для бенчмарка: каждая строка таблицы становится
        самостоятельным предложением и переживает нарезку на чанки без потери шапки.
        """
        rendered: list[str] = []
        for is_header, cells in body:
            if is_header:
                header = cells
                continue
            pairs = [
                f"{header[j]}: {cell}" if j < len(header) and header[j] else cell
                for j, cell in enumerate(cells)
                if cell
            ]
            if pairs:
                rendered.append("- " + "; ".join(pairs))
        return rendered

    # -- инлайновый разбор ---------------------------------------------------

    def _inline(self, text: str, in_table: bool = False) -> str:
        text = _RE_FIGURE.sub(self._on_figure, text)
        text = _RE_LINK.sub(self._on_link, text)
        text = _RE_BSLINK.sub(self._on_bslink, text)
        text = _RE_MEDIA.sub(self._on_media, text)
        text = _RE_FOOTNOTE.sub(lambda m: f" ({m.group(1).strip()})", text)
        text = _RE_ITALIC.sub(r"*\1*", text)
        text = _RE_MONO.sub(r"`\1`", text)
        text = _RE_UNDERLINE.sub(r"\1", text)
        text = _RE_LINEBREAK.sub(" " if in_table else "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        # После удаления иконок остаются висящие пробелы перед пунктуацией.
        return re.sub(r" +([,.;:!?)])", r"\1", text)

    def _on_figure(self, m: re.Match[str]) -> str:
        """``[{{ path?nolink |Рисунок 1. Подпись }}]`` — картинка с подписью."""
        body = m.group(1)
        path, _, caption = body.partition("|")
        stored = self._remember_image(path)
        caption = " ".join(caption.split()).strip()
        if caption and stored:
            self._figures.append((caption, stored))
        return f"*{caption}*" if caption else ""

    def _on_link(self, m: re.Match[str]) -> str:
        """``[[target|label]]`` — метка важнее цели: цель это технический id."""
        target, sep, label = m.group(1).partition("|")
        target = target.strip()
        label = self._inline(label).strip() if sep else ""

        if target.startswith(("http://", "https://", "ftp://", "mailto:")):
            return label or target

        page_id = normalize_page_id(target)
        if page_id:
            self._links.append(page_id)
        if label:
            return label
        # Ссылка без подписи: подставляем заголовок целевой страницы, иначе
        # страницы-оглавления превращаются в набор бессмысленных путей.
        resolved = self.title_map.get(page_id)
        if resolved:
            return resolved
        anchor = _anchor_of(target)
        if anchor:
            return anchor.replace("_", " ")
        return page_id.rsplit("/", 1)[-1].replace("_", " ") if page_id else ""

    def _on_bslink(self, m: re.Match[str]) -> str:
        """``{{bslink>Главное меню → Отчеты → ...|ShowRibbonPageOrItem?GUID}}``.

        Плагин Business Studio: видимая часть — путь по меню программы, после
        первого ``|`` идут GUID объектов. Наивное удаление всей конструкции
        уничтожает ровно тот текст, по которому ищут «где это настраивается».
        """
        label = m.group(1).split("|", 1)[0]
        return self._inline(label).strip()

    def _on_media(self, m: re.Match[str]) -> str:
        """``{{common/icons/31.png?nolink}}`` — иконки внутри предложений."""
        body = m.group(1)
        path, _, caption = body.partition("|")
        self._remember_image(path)
        caption = caption.strip()
        if caption:
            return f"*{caption}*"
        return "" if _RE_IMAGE_EXT.search(path) else body.strip()

    def _remember_image(self, path: str) -> str:
        clean = path.split("?", 1)[0].strip().strip("|").strip()
        if not clean or not _RE_IMAGE_EXT.search(clean):
            return ""
        clean = clean.replace(":", "/").lstrip("/")
        self._images.append(clean)
        return clean

    # -- прочее --------------------------------------------------------------

    @staticmethod
    def _is_link_only(raw_line: str) -> bool:
        without_links = _RE_LINK.sub("", raw_line)
        without_links = _RE_MEDIA.sub("", without_links)
        return not re.sub(r"[\s*_\-|^]", "", without_links)

    @staticmethod
    def _extract_title(markdown: str) -> str:
        for line in markdown.split("\n"):
            if line.startswith("# "):
                return line[2:].strip()
        for line in markdown.split("\n"):
            if line.startswith("#"):
                return line.lstrip("# ").strip()
        return ""


def extract_title(raw: str) -> str:
    """Быстро достаёт заголовок страницы без полной конвертации (для title_map)."""
    for line in raw.lstrip("\ufeff").replace("\r\n", "\n").split("\n"):
        m = _RE_HEADING.match(line)
        if m and m.group(2):
            title = m.group(2)
            title = _RE_BSLINK.sub(lambda x: x.group(1).split("|", 1)[0], title)
            title = _RE_LINK.sub(lambda x: x.group(1).partition("|")[2] or x.group(1), title)
            return re.sub(r"[*_'`]", "", title).strip()
    return ""


def make_converter(
    title_map: dict[str, str] | None = None, **kwargs: object
) -> Callable[[str], ParsedPage]:
    converter = DokuWikiConverter(title_map=title_map, **kwargs)  # type: ignore[arg-type]
    return converter.convert
