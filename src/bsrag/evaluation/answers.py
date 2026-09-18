"""Оценка сгенерированных ответов.

Считаются три вещи:

* детерминированные метрики — доля ответов со ссылками, корректность номеров
  ссылок, отказ отвечать на вопросы-ловушки, задержка;
* оценка судьи — локальная LLM выставляет обоснованность и полноту по рубрике;
* попадание источников — есть ли эталонная страница среди процитированных.

Судьёй выступает отдельная локальная модель: RAGAS на 12B-модели тратит по
5-8 вызовов LLM на вопрос, что при сравнении трёх генераторов превращается в часы.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from bsrag.config import Settings
from bsrag.evaluation.goldset import GoldSet, QAItem

JUDGE_PROMPT = """Ты — строгий эксперт по программе Business Studio. Оцени ответ \
консультанта, опираясь ТОЛЬКО на приведённые фрагменты справки.

Фрагменты справки:
\"\"\"
{context}
\"\"\"

Вопрос пользователя: {question}

Ответ консультанта:
\"\"\"
{answer}
\"\"\"

Оцени по двум шкалам:
- "faithfulness" (обоснованность): 2 — всё сказанное подтверждается фрагментами; \
1 — есть детали, которых во фрагментах нет; 0 — ответ противоречит фрагментам или выдуман.
- "completeness" (полнота): 2 — вопрос закрыт полностью; 1 — ответ частичный; \
0 — ответ не по существу.

Верни ТОЛЬКО JSON без пояснений: {{"faithfulness": <0-2>, "completeness": <0-2>}}"""

REFUSAL_MARKERS = ("не нашёл", "не нашел", "нет ответа", "не содержит", "отсутствует в справке")
_RE_JSON = re.compile(r"\{[^{}]*\}")


@dataclass
class AnswerRecord:
    """Ответ системы на один вопрос вместе с оценками."""

    item: QAItem
    answer: str
    latency_ms: float
    sources: list[str] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    n_context: int = 0
    faithfulness: float | None = None
    completeness: float | None = None

    @property
    def refused(self) -> bool:
        lowered = self.answer.lower()
        return any(marker in lowered for marker in REFUSAL_MARKERS)

    @property
    def has_valid_citations(self) -> bool:
        return bool(self.citations) and all(1 <= c <= self.n_context for c in self.citations)

    @property
    def cited_gold_source(self) -> bool:
        """Эталонная страница попала в процитированные фрагменты."""
        if not self.item.gold_pages:
            return False
        cited = {self.sources[c - 1] for c in self.citations if 1 <= c <= len(self.sources)}
        return bool(cited & set(self.item.gold_pages))


def judge_answer(record: AnswerRecord, context: str, settings: Settings, judge_model: str) -> None:
    from bsrag.generation import build_llm, strip_reasoning

    config = settings.generation.model_copy(update={"model": judge_model, "temperature": 0.0})
    llm = build_llm(config)
    prompt = JUDGE_PROMPT.format(
        context=context[:9000], question=record.item.question, answer=record.answer[:3000]
    )
    raw = strip_reasoning(str(llm.invoke(prompt).content))
    match = _RE_JSON.search(raw)
    if not match:
        return
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return
    record.faithfulness = float(verdict.get("faithfulness", 0)) / 2
    record.completeness = float(verdict.get("completeness", 0)) / 2


def collect_answers(
    settings: Settings, goldset: GoldSet, console=None
) -> tuple[list[AnswerRecord], dict[str, str]]:
    from bsrag.generation import format_context
    from bsrag.pipeline import RagPipeline

    pipeline = RagPipeline(settings)
    records: list[AnswerRecord] = []
    contexts: dict[str, str] = {}

    items = goldset.of_kind("manual", "trap")
    for i, item in enumerate(items):
        response = pipeline.ask(item.question)
        record = AnswerRecord(
            item=item,
            answer=response.answer.text,
            latency_ms=response.answer.latency_ms,
            sources=[hit.chunk.page_id for hit in response.hits],
            citations=response.answer.used_sources,
            n_context=len(response.hits),
        )
        records.append(record)
        contexts[item.id] = format_context(response.hits)
        if console:
            console.print(
                f"[dim]{i + 1}/{len(items)}[/] {item.question[:70]} "
                f"[dim]({response.answer.latency_ms / 1000:.1f} с)[/]"
            )
    return records, contexts


def evaluate_llm(
    settings: Settings, goldset: GoldSet, judge_model: str | None = None, console=None
) -> dict[str, Any]:
    records, contexts = collect_answers(settings, goldset, console=console)
    if not records:
        return {}

    judge = judge_model or _pick_judge(settings)
    if judge:
        for record in records:
            judge_answer(record, contexts[record.item.id], settings, judge)

    answerable = [r for r in records if r.item.kind != "trap"]
    traps = [r for r in records if r.item.kind == "trap"]

    def mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    metrics: dict[str, Any] = {
        "n_questions": len(records),
        "latency_p50_s": statistics.median(r.latency_ms for r in records) / 1000,
        "latency_p95_s": sorted(r.latency_ms for r in records)[int(0.95 * (len(records) - 1))] / 1000,
        "citation_rate": mean([float(r.has_valid_citations) for r in answerable]),
        "cited_gold_source": mean([float(r.cited_gold_source) for r in answerable]),
        "answer_rate": mean([float(not r.refused) for r in answerable]),
        "judge_model": judge or "",
    }
    if traps:
        # Главная проверка на галлюцинации: на вопрос без ответа в справке
        # система обязана отказаться отвечать.
        metrics["trap_refusal_rate"] = mean([float(r.refused) for r in traps])

    judged = [r for r in answerable if r.faithfulness is not None]
    if judged:
        metrics["faithfulness"] = mean([r.faithfulness for r in judged])
        metrics["completeness"] = mean([r.completeness for r in judged])
    return metrics


def _pick_judge(settings: Settings) -> str:
    """Судьёй берём другую модель, чтобы не оценивать саму себя."""
    from bsrag.generation import available_models

    installed = available_models(settings.generation.base_url)
    preferred = [m for m in installed if m.split(":")[0] != settings.generation.model.split(":")[0]]
    for candidate in ("gemma3:12b-it-qat", "qwen3:8b"):
        if candidate in preferred:
            return candidate
    return preferred[0] if preferred else ""
