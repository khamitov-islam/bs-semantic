"""Генерация ответа локальной LLM через Ollama."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import httpx
from langchain_ollama import ChatOllama

from bsrag.config import GenerationConfig
from bsrag.retrieval import Hit

SYSTEM_PROMPT = """Ты — консультант по программе Business Studio. Ты отвечаешь на вопросы \
пользователей строго по фрагментам официальной справки, приведённым ниже.

Правила:
1. Используй только информацию из фрагментов. Не добавляй ничего от себя и не \
опирайся на общие знания о других программах.
2. После каждого утверждения ставь номер фрагмента в квадратных скобках: [1], [2]. \
Если утверждение опирается на несколько фрагментов — [1][3].
3. Если во фрагментах нет ответа, ответь ровно: «В справке Business Studio я не нашёл \
ответа на этот вопрос.» и, если уместно, укажи, какие близкие темы есть во фрагментах.
4. Сохраняй терминологию справки дословно: названия окон, кнопок, пунктов меню и \
путей вида «Главное меню → Отчеты → Шаблоны отчетов объекта».
5. Отвечай по-русски, по существу. Пошаговые инструкции оформляй нумерованным списком.
"""

USER_TEMPLATE = """Фрагменты справки:

{context}

Вопрос пользователя: {question}

Ответ (со ссылками на номера фрагментов):"""

_RE_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class Answer:
    text: str
    used_sources: list[int]
    latency_ms: float = 0.0
    model: str = ""


def format_context(hits: list[Hit], max_chars_per_chunk: int = 2500) -> str:
    """Собирает пронумерованный контекст с указанием источника каждого фрагмента."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        location = " > ".join(x for x in (chunk.breadcrumb, chunk.title, chunk.headings) if x)
        body = chunk.context_text[:max_chars_per_chunk]
        blocks.append(f"[{i}] Источник: {location}\nФайл справки: {chunk.page_id}\n\n{body}")
    return "\n\n---\n\n".join(blocks)


def extract_citations(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", text)})


def strip_reasoning(text: str) -> str:
    """Убирает блок рассуждений у think-моделей (Qwen3 и подобные)."""
    return _RE_THINK.sub("", text).strip()


def build_llm(config: GenerationConfig, streaming: bool = False) -> ChatOllama:
    kwargs: dict[str, object] = {
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "num_ctx": config.num_ctx,
        "num_predict": config.max_tokens,
        "streaming": streaming,
        "keep_alive": config.keep_alive,
    }
    if not config.think:
        # Отключает «размышления» у гибридных моделей: в RAG они дают задержку,
        # но не улучшают ответ, собранный из готовых фрагментов справки.
        kwargs["reasoning"] = False
    return ChatOllama(**kwargs)


def warmup_llm(config: GenerationConfig) -> None:
    """Держит веса модели в памяти Ollama, чтобы первый ответ не ждал загрузки."""
    try:
        httpx.post(
            f"{config.base_url.rstrip('/')}/api/generate",
            json={
                "model": config.model,
                "prompt": "ок",
                "stream": False,
                "keep_alive": config.keep_alive,
                "think": False,
                "options": {"num_predict": 1, "temperature": 0},
            },
            timeout=180.0,
        )
    except Exception:
        pass


def _messages(question: str, hits: list[Hit]) -> list[tuple[str, str]]:
    return [
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE.format(context=format_context(hits), question=question)),
    ]


def generate(question: str, hits: list[Hit], config: GenerationConfig) -> Answer:
    import time

    llm = build_llm(config)
    started = time.perf_counter()
    response = llm.invoke(_messages(question, hits))
    latency = (time.perf_counter() - started) * 1000
    text = strip_reasoning(str(response.content))
    return Answer(
        text=text, used_sources=extract_citations(text), latency_ms=latency, model=config.model
    )


def stream(question: str, hits: list[Hit], config: GenerationConfig) -> Iterator[str]:
    llm = build_llm(config, streaming=True)
    in_reasoning = False
    for piece in llm.stream(_messages(question, hits)):
        token = str(piece.content)
        if "<think>" in token:
            in_reasoning = True
        if "</think>" in token:
            in_reasoning = False
            token = token.split("</think>", 1)[1]
        if in_reasoning:
            continue
        if token:
            yield token


def available_models(base_url: str) -> list[str]:
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
        return sorted(m["name"] for m in response.json().get("models", []))
    except Exception:
        return []
