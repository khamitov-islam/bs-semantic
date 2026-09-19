"""Общие элементы интерфейса Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bsrag.config import Settings, load_settings  # noqa: E402

PROFILES = {
    "improved": ("Улучшенный", "configs/default.yaml"),
    "baseline": ("Бейслайн", "configs/baseline.yaml"),
}


def selected_profile() -> str:
    """Переключатель зафиксированного бейслайна и текущих улучшений."""
    return st.sidebar.radio(
        "Профиль",
        list(PROFILES),
        format_func=lambda key: PROFILES[key][0],
        help="Бейслайн — зафиксированное решение. Улучшенный — те же модели плюс "
        "раскрытие запроса и прогрев.",
    )


@st.cache_resource(show_spinner=False)
def get_settings(profile: str = "improved") -> Settings:
    relative = PROFILES.get(profile, PROFILES["improved"])[1]
    return load_settings(PROJECT_ROOT / relative)


@st.cache_resource(show_spinner="Загружаю модели и индекс...")
def get_retriever(profile: str, mode: str, embedding_model: str):
    """Ретривер кэшируется по профилю, режиму и модели."""
    from bsrag.retrieval import Retriever

    settings = get_settings(profile).model_copy(deep=True)
    settings.retrieval.mode = mode
    settings.embedding.model_name = embedding_model
    return Retriever(settings)


@st.cache_resource(show_spinner="Прогреваю эмбеддер, реранкер и Ollama...")
def warmup_runtime(profile: str, mode: str, embedding_model: str, llm: str) -> dict[str, float]:
    """Один раз за жизнь процесса: после этого первый вопрос не ждёт загрузки весов."""
    import time

    from bsrag.generation import warmup_llm

    settings = get_settings(profile)
    retriever = get_retriever(profile, mode, embedding_model)
    started = time.perf_counter()
    retriever.warmup()
    retrieval_s = time.perf_counter() - started

    gen = settings.generation.model_copy(update={"model": llm})
    started = time.perf_counter()
    warmup_llm(gen)
    return {"retrieval_warmup_s": retrieval_s, "llm_warmup_s": time.perf_counter() - started}


@st.cache_data(show_spinner=False)
def media_file(image_path: str) -> str | None:
    settings = get_settings("improved")
    candidate = settings.resolve(settings.corpus.media_dir) / image_path
    return str(candidate) if candidate.exists() else None


def render_source(index: int, hit, show_images: bool = True) -> None:
    """Карточка источника: откуда именно взят фрагмент справки."""
    chunk = hit.chunk
    location = " > ".join(x for x in (chunk.breadcrumb, chunk.title, chunk.headings) if x)
    with st.expander(f"**[{index}]** {location}  ·  релевантность {hit.score:.3f}"):
        st.caption(f"Файл справки: `{chunk.source_path}`")
        st.caption(f"Страница DokuWiki: `{chunk.page_id}`  ·  [открыть в справке]({chunk.url})")
        st.markdown(chunk.text)
        if show_images and chunk.images:
            for image in chunk.images[:3]:
                path = media_file(image)
                if path:
                    st.image(path, caption=image, use_container_width=True)


def sidebar_status(settings: Settings, warmup: dict[str, float] | None = None) -> None:
    from bsrag.embeddings import resolve_device
    from bsrag.index import collection_name

    with st.sidebar.expander("Состояние системы", expanded=False):
        st.caption(f"Эмбеддинги: `{settings.embedding.model_name}`")
        st.caption(f"Коллекция: `{collection_name(settings)}`")
        st.caption(f"Устройство: `{resolve_device(settings.embedding.device)}`")
        st.caption(
            "Раскрытие запроса: "
            + ("вкл" if settings.retrieval.query_expand else "выкл")
        )
        st.caption(
            "Индекс таблиц: "
            + ("вкл" if settings.retrieval.table_index else "выкл")
        )
        if warmup:
            st.caption(
                f"Прогрев поиска: {warmup['retrieval_warmup_s']:.1f} с · "
                f"LLM: {warmup['llm_warmup_s']:.1f} с"
            )
        st.caption("Всё работает локально: внешние API не используются.")
