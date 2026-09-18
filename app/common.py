"""Общие элементы интерфейса Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bsrag.config import Settings, load_settings  # noqa: E402


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return load_settings()


@st.cache_resource(show_spinner="Загружаю модели и индекс...")
def get_retriever(mode: str, embedding_model: str):
    """Ретривер кэшируется по режиму поиска и модели: их смена меняет индекс.

    Остальные параметры (top-k, реранкер) передаются прямо в запрос.
    """
    from bsrag.retrieval import Retriever

    settings = get_settings().model_copy(deep=True)
    settings.retrieval.mode = mode
    settings.embedding.model_name = embedding_model
    return Retriever(settings)


@st.cache_data(show_spinner=False)
def media_file(image_path: str) -> str | None:
    settings = get_settings()
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


def sidebar_status(settings: Settings) -> None:
    from bsrag.embeddings import resolve_device
    from bsrag.index import collection_name

    with st.sidebar.expander("Состояние системы", expanded=False):
        st.caption(f"Эмбеддинги: `{settings.embedding.model_name}`")
        st.caption(f"Коллекция: `{collection_name(settings)}`")
        st.caption(f"Устройство: `{resolve_device(settings.embedding.device)}`")
        st.caption("Всё работает локально: внешние API не используются.")
