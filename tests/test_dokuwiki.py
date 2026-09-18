"""Тесты конвертера DokuWiki на конструкциях, реально встречающихся в справке."""

from __future__ import annotations

import pytest

from bsrag.parsing.dokuwiki import DokuWikiConverter, extract_title, normalize_page_id


@pytest.fixture
def conv() -> DokuWikiConverter:
    return DokuWikiConverter(title_map={"ru/manual/terms": "Термины и определения"})


def test_headings_levels(conv: DokuWikiConverter) -> None:
    md = conv.convert("====== Справочники ======\n===== Работа в окне =====\n==== Фильтр ====").markdown
    assert md.splitlines()[0] == "# Справочники"
    assert "## Работа в окне" in md
    assert "### Фильтр" in md


def test_empty_heading_is_dropped(conv: DokuWikiConverter) -> None:
    # `== ==` используется в справке как визуальный разделитель, а не заголовок.
    md = conv.convert("== ==\n**Внимание!** Текст").markdown
    assert "#" not in md
    assert md == "**Внимание!** Текст"


def test_bslink_keeps_menu_path(conv: DokuWikiConverter) -> None:
    raw = (
        "Настройка задаётся опцией "
        "{{bslink>Главное меню → Настройки → Дополнительно|ShowOnForm?cbdeb0a9-aa22;c=017b,o=Tab}}."
    )
    md = conv.convert(raw).markdown
    assert "Главное меню → Настройки → Дополнительно" in md
    assert "ShowOnForm" not in md
    assert "cbdeb0a9" not in md


def test_link_with_label_keeps_label(conv: DokuWikiConverter) -> None:
    md = conv.convert("см. [[ru/manual/filter/filter_element|Окно фильтра]]").markdown
    assert md == "см. Окно фильтра"


def test_bare_link_resolves_to_page_title(conv: DokuWikiConverter) -> None:
    md = conv.convert("  * [[ru/manual/terms]]").markdown
    assert md == "- Термины и определения"


def test_figure_caption_survives_image_removal(conv: DokuWikiConverter) -> None:
    page = conv.convert("[{{ ru/manual/list271.png?nolink |Рисунок 1. Окно справочника }}]")
    assert page.markdown == "*Рисунок 1. Окно справочника*"
    assert page.images == ["ru/manual/list271.png"]


def test_inline_icon_is_removed_without_leaving_gaps(conv: DokuWikiConverter) -> None:
    md = conv.convert('значком "открытая папка" {{common/icons/31.png?nolink}}.').markdown
    assert md == 'значком "открытая папка".'


def test_table_box_caption_moves_before_table(conv: DokuWikiConverter) -> None:
    raw = (
        "<startTableBox>\n"
        "^  Кнопка  ^  Название  ^  Описание  ^\n"
        "|  {{common/icons/06.png?nolink}}  | Новый (Ins) | Создается новый объект. |\n"
        "<endTableBox|Таблица 1. Панель инструментов>"
    )
    md = conv.convert(raw).markdown
    lines = [line for line in md.splitlines() if line.strip()]
    assert lines[0] == "**Таблица 1. Панель инструментов**"
    # Колонка с иконками после очистки пуста и удаляется целиком.
    assert lines[1] == "| Название | Описание |"
    assert lines[2] == "| --- | --- |"
    assert lines[3] == "| Новый (Ins) | Создается новый объект. |"


def test_table_cell_with_bslink_is_not_split_on_inner_pipe(conv: DokuWikiConverter) -> None:
    raw = (
        "^  Пункт  ^  Описание  ^\n"
        "| Отчеты | Открывает {{bslink>Главное меню → Отчеты|ShowRibbon?abc;def:Item}} модели. |"
    )
    rows = [line for line in conv.convert(raw).markdown.splitlines() if line.startswith("|")]
    assert rows[-1] == "| Отчеты | Открывает Главное меню → Отчеты модели. |"


def test_rowspan_marker_inherits_value_from_row_above(conv: DokuWikiConverter) -> None:
    # Так устроена таблица горячих клавиш: клавиша указана один раз, а дальше `:::`.
    raw = (
        "^  Клавиши  ^  Действие  ^  Место применения  ^\n"
        "| F1 | Открыть справку | **Окно свойств** объекта |\n"
        "| ::: | ::: | **Окно справочника** |"
    )
    rows = [line for line in conv.convert(raw).markdown.splitlines() if line.startswith("|")]
    assert rows[-1] == "| F1 | Открыть справку | **Окно справочника** |"


def test_table_rows_mode_verbalizes_each_row() -> None:
    conv = DokuWikiConverter(table_mode="rows")
    raw = "^  Кнопка  ^  Описание  ^\n| Новый (Ins) | Создается новый объект. |"
    md = conv.convert(raw).markdown
    assert "- Кнопка: Новый (Ins); Описание: Создается новый объект." in md


def test_code_block_is_preserved_verbatim(conv: DokuWikiConverter) -> None:
    raw = "<code>\nSub Пример()\n  Set app = CreateObject(\"ByteEnterprise.OleApplication\")\nEnd Sub\n</code>"
    md = conv.convert(raw).markdown
    assert "```" in md
    assert 'Set app = CreateObject("ByteEnterprise.OleApplication")' in md


def test_contextnavigator_macro_is_dropped(conv: DokuWikiConverter) -> None:
    assert conv.convert("Текст\n\n[<contextnavigator>]\n\n\n\n").markdown == "Текст"


def test_multiline_link_is_joined(conv: DokuWikiConverter) -> None:
    md = conv.convert("(См. [[/ru/manual/branches#проверка_ветки\n| Проверка ветки]]).").markdown
    assert md == "(См. Проверка ветки)."


def test_index_page_is_detected(conv: DokuWikiConverter) -> None:
    raw = "====== Руководство ======\n  * [[ru/manual/terms]]\n  * [[ru/manual/filter]]\n"
    assert conv.convert(raw).is_index_page is True


def test_regular_page_is_not_index_page(conv: DokuWikiConverter) -> None:
    raw = "====== Фильтр ======\nФильтр отбирает объекты по условиям.\nСм. [[ru/manual/terms]].\n"
    assert conv.convert(raw).is_index_page is False


def test_footnote_becomes_inline_parenthesis(conv: DokuWikiConverter) -> None:
    assert conv.convert("Текст((Сноска про деталь)) далее").markdown == "Текст (Сноска про деталь) далее"


def test_italic_does_not_break_urls(conv: DokuWikiConverter) -> None:
    md = conv.convert("Ссылка [[https://docs.business-studio.ru/doku.php|Справка]]").markdown
    assert md == "Ссылка Справка"


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("ru/manual/manage_model//manage_model", "ru/manual/manage_model/manage_model"),
        ("ru:manual:terms", "ru/manual/terms"),
        ("ru/manual/grids#сортировка_строк", "ru/manual/grids"),
        ("/ru/manual/filter/", "ru/manual/filter"),
    ],
)
def test_normalize_page_id(target: str, expected: str) -> None:
    assert normalize_page_id(target) == expected


def test_extract_title_ignores_markup() -> None:
    assert extract_title("====== **Справочники** ======\nтекст") == "Справочники"


def test_byte_order_mark_does_not_hide_first_heading(conv: DokuWikiConverter) -> None:
    raw = "\ufeff====== Права пользователя ======\nТекст статьи."
    page = conv.convert(raw)
    assert page.title == "Права пользователя"
    assert page.markdown.startswith("# Права пользователя")
    assert extract_title(raw) == "Права пользователя"
