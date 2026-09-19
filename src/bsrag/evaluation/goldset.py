"""Золотой набор вопросов для оценки поиска.

Набор состоит из трёх частей:

* ``auto``   — вопросы, сгенерированные локальной LLM по случайным чанкам. Эталон
  известен по построению, поэтому их можно наделать много и мерить на них
  стабильно; зато они «удобные» и переоценивают качество.
* ``manual`` — каверзные вопросы, размеченные руками: формулировки пользователя,
  а не справки, синонимы, вопросы про конкретные кнопки и таблицы.
* ``trap``   — вопросы, ответа на которые в справке нет. Нужны, чтобы проверить,
  что система отказывается отвечать, а не выдумывает.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from bsrag.chunking import Chunk, chunk_pages
from bsrag.config import Settings
from bsrag.corpus import load_or_prepare_corpus

QuestionKind = Literal["auto", "manual", "trap"]

GENERATION_PROMPT = """Ниже фрагмент официальной справки программы Business Studio.

Придумай ОДИН вопрос, который реальный пользователь Business Studio задал бы \
службе поддержки, и ответ на который содержится в этом фрагменте.

Требования к вопросу:
- вопрос должен быть самодостаточным: он не должен ссылаться на «фрагмент», \
«текст выше», «данный раздел»;
- вопрос должен быть конкретным (про кнопку, настройку, окно, правило, порядок \
действий), а не общим («что такое моделирование?»);
- формулируй так, как пишет пользователь, а не как написано в справке;
- только один вопрос, одной строкой, без пояснений и нумерации.

Фрагмент (раздел «{location}»):
\"\"\"
{text}
\"\"\"

Вопрос:"""

_RE_BAD_QUESTION = re.compile(
    r"фрагмент|в тексте|в данном разделе|выше|приведённ|приведенн|согласно справке",
    re.IGNORECASE,
)


@dataclass
class QAItem:
    """Вопрос золотого набора."""

    id: str
    question: str
    gold_pages: list[str]
    kind: QuestionKind = "auto"
    gold_chunk_id: str | None = None
    reference_answer: str = ""
    topic: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GoldSet:
    items: list[QAItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def of_kind(self, *kinds: QuestionKind) -> list[QAItem]:
        return [item for item in self.items if item.kind in kinds]

    @property
    def answerable(self) -> list[QAItem]:
        """Вопросы, у которых в справке есть эталонная страница."""
        return [item for item in self.items if item.kind != "trap" and item.gold_pages]


def goldset_path(settings: Settings) -> Path:
    return settings.resolve(settings.paths.goldset) / "goldset.jsonl"


def manual_path(settings: Settings) -> Path:
    return settings.resolve(settings.paths.goldset) / "manual.yaml"


def save_goldset(goldset: GoldSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in goldset.items:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_goldset(path: Path) -> GoldSet:
    with path.open(encoding="utf-8") as fh:
        return GoldSet([QAItem(**json.loads(line)) for line in fh if line.strip()])


def load_manual(path: Path) -> list[QAItem]:
    """Читает размеченные руками вопросы из YAML."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    items: list[QAItem] = []
    for i, entry in enumerate(raw):
        items.append(
            QAItem(
                id=entry.get("id") or f"manual-{i:03d}",
                question=entry["question"],
                gold_pages=[p.lower().strip("/") for p in entry.get("gold_pages", [])],
                kind=entry.get("kind", "manual"),
                reference_answer=entry.get("reference_answer", ""),
                topic=entry.get("topic", ""),
                note=entry.get("note", ""),
            )
        )
    return items


def sample_chunks(chunks: list[Chunk], n: int, seed: int = 13) -> list[Chunk]:
    """Отбирает содержательные чанки, равномерно размазывая выборку по разделам."""
    rng = random.Random(seed)
    usable = [c for c in chunks if c.n_tokens >= 80]
    by_section: dict[str, list[Chunk]] = {}
    for chunk in usable:
        section = chunk.page_id.split("/")[1] if chunk.page_id.count("/") >= 1 else chunk.page_id
        by_section.setdefault(section, []).append(chunk)

    picked: list[Chunk] = []
    sections = sorted(by_section, key=lambda s: -len(by_section[s]))
    while len(picked) < n and any(by_section.values()):
        for section in sections:
            pool = by_section.get(section)
            if not pool:
                continue
            picked.append(pool.pop(rng.randrange(len(pool))))
            if len(picked) >= n:
                break
    return picked


def generate_auto_questions(
    settings: Settings, n: int, console=None, seed: int = 13
) -> list[QAItem]:
    from bsrag.generation import build_llm, strip_reasoning

    pages = load_or_prepare_corpus(settings)
    chunks = chunk_pages(pages, settings.chunking, settings.embedding.model_name)
    selected = sample_chunks(chunks, n, seed=seed)

    llm = build_llm(settings.generation)
    items: list[QAItem] = []
    for i, chunk in enumerate(selected):
        location = " > ".join(x for x in (chunk.title, chunk.headings) if x)
        prompt = GENERATION_PROMPT.format(location=location, text=chunk.text[:2500])
        question = strip_reasoning(str(llm.invoke(prompt).content)).strip().strip('"').split("\n")[0]
        if console:
            console.print(f"[dim]{i + 1}/{len(selected)}[/] {question[:110]}")
        if len(question) < 15 or _RE_BAD_QUESTION.search(question):
            continue
        items.append(
            QAItem(
                id=f"auto-{i:03d}",
                question=question,
                gold_pages=[chunk.page_id],
                kind="auto",
                gold_chunk_id=chunk.chunk_id,
                topic=location,
            )
        )
    return items


def build_goldset(
    settings: Settings, n_auto: int = 60, keep_existing_auto: bool = True, console=None
) -> Path:
    """Собирает goldset.jsonl из manual.yaml и, по желанию, авто-вопросов.

    ``n_auto=0`` не вызывает LLM. Если ``keep_existing_auto``, уже сгенерированные
    auto-вопросы из предыдущего файла сохраняются — иначе расширение ручной
    разметки каждый раз стоило бы десятки минут генерации.
    """
    manual = load_manual(manual_path(settings))
    auto: list[QAItem] = []
    path = goldset_path(settings)
    if n_auto:
        auto = generate_auto_questions(settings, n_auto, console=console)
    elif keep_existing_auto and path.exists():
        auto = load_goldset(path).of_kind("auto")
    goldset = GoldSet(manual + auto)
    save_goldset(goldset, path)
    if console:
        console.print(
            f"Вопросов: всего {len(goldset)}, "
            f"ручных {len(goldset.of_kind('manual'))}, "
            f"ловушек {len(goldset.of_kind('trap'))}, "
            f"автоматических {len(goldset.of_kind('auto'))}"
        )
    return path
