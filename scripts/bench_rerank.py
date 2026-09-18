"""Замер задержки реранкинга на прогретой модели."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsrag.config import load_settings  # noqa: E402
from bsrag.retrieval import Retriever  # noqa: E402

QUESTIONS = [
    "Какой горячей клавишей вызвать окно Права доступа?",
    "Можно ли сохранить базу, пока с ней работают другие сотрудники?",
    "Как поменять номер порта, на котором отдаётся OData?",
]


def main() -> None:
    settings = load_settings()
    retriever = Retriever(settings)
    retriever.warmup()

    for question in QUESTIONS:
        result = retriever.search(question, top_k=5)
        print(
            f"поиск {result.retrieval_ms:6.0f} мс | реранк {result.rerank_ms:7.0f} мс | "
            f"{result.hits[0].chunk.page_id}"
        )

    for candidates in (10, 20, 40):
        result = retriever.search(QUESTIONS[0], top_k=5, candidates=candidates)
        print(f"кандидатов {candidates:3d}: реранк {result.rerank_ms:7.0f} мс")


if __name__ == "__main__":
    main()
