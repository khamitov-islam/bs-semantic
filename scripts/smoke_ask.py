"""Дымовой прогон RAG: каверзные вопросы + ловушка."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsrag.config import load_settings  # noqa: E402
from bsrag.pipeline import RagPipeline  # noqa: E402

QUESTIONS = [
    "Какой горячей клавишей вызвать окно Права доступа?",
    "Можно ли сохранить базу, пока с ней работают другие сотрудники?",
    "Как поменять номер порта, на котором Business Studio отдаёт OData?",
    "Как настроить интеграцию Business Studio с Telegram-ботом?",
]


def main() -> None:
    settings = load_settings()
    pipeline = RagPipeline(settings)
    print(f"Модель: {settings.generation.model}\n")
    for question in QUESTIONS:
        print("=" * 80)
        print(f"Q: {question}")
        response = pipeline.ask(question)
        print(f"поиск {response.search.retrieval_ms:.0f} мс + реранк {response.search.rerank_ms:.0f} мс + LLM {response.answer.latency_ms / 1000:.1f} с")
        print(response.answer.text)
        print("Источники:")
        for i, hit in enumerate(response.hits, start=1):
            print(f"  [{i}] {hit.chunk.page_id}  ({hit.score:.3f})")


if __name__ == "__main__":
    main()
