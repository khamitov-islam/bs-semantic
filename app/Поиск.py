"""Главная страница: вопрос-ответ по справке Business Studio."""

from __future__ import annotations

import time

import streamlit as st

from common import (
    get_retriever,
    get_settings,
    render_source,
    selected_profile,
    sidebar_status,
    warmup_runtime,
)

st.set_page_config(page_title="Поиск по справке Business Studio", page_icon="📘", layout="wide")

EXAMPLES = [
    "Как настроить права доступа пользователей к модели?",
    "Какими клавишами выделить несколько объектов в справочнике?",
    "Где включается использование OLE и OData?",
    "Как перенести изменения из ветки в основную модель?",
]


def main() -> None:
    profile = selected_profile()
    settings = get_settings(profile)
    st.title("📘 Поиск по справке Business Studio")
    if profile == "baseline":
        st.caption("Зафиксированный бейслайн: hybrid + реранкер, без раскрытия запроса.")
    else:
        st.caption(
            "Семантический поиск и ответы локальной LLM. Все модели работают на этом "
            "компьютере, данные наружу не уходят."
        )

    controls = sidebar(settings)
    warmup = warmup_runtime(
        profile, controls["mode"], settings.embedding.model_name, controls["model"]
    )
    sidebar_status(settings, warmup)

    if "history" not in st.session_state:
        st.session_state.history = []

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            for i, hit in enumerate(entry["hits"], start=1):
                render_source(i, hit, controls["show_images"])

    if not st.session_state.history:
        st.markdown("**Примеры вопросов:**")
        columns = st.columns(len(EXAMPLES))
        for column, example in zip(columns, EXAMPLES):
            if column.button(example, use_container_width=True):
                st.session_state.pending = example
                st.rerun()

    question = st.chat_input("Задайте вопрос по функционалу Business Studio")
    if not question:
        question = st.session_state.pop("pending", None)
    if question:
        answer(question, settings, controls, profile)
        st.rerun()


def sidebar(settings) -> dict:
    from bsrag.generation import available_models

    st.sidebar.header("Настройки")
    models = available_models(settings.generation.base_url)
    default = models.index(settings.generation.model) if settings.generation.model in models else 0
    model = st.sidebar.selectbox("Модель ответа", models or [settings.generation.model], index=default)

    mode = st.sidebar.radio(
        "Режим поиска",
        ["hybrid", "dense", "sparse"],
        index=["hybrid", "dense", "sparse"].index(settings.retrieval.mode),
        help="hybrid — слияние векторного поиска и BM25; sparse — только BM25.",
        horizontal=True,
    )
    top_k = st.sidebar.slider("Фрагментов в контексте", 1, 10, settings.retrieval.top_k)
    use_reranker = st.sidebar.checkbox("Реранкер (кросс-энкодер)", settings.retrieval.use_reranker)
    show_images = st.sidebar.checkbox("Показывать иллюстрации справки", True)
    search_only = st.sidebar.checkbox("Только поиск, без ответа LLM", False)
    return {
        "model": model,
        "mode": mode,
        "top_k": top_k,
        "use_reranker": use_reranker,
        "show_images": show_images,
        "search_only": search_only,
    }


def answer(question: str, settings, controls: dict, profile: str) -> None:
    from bsrag.generation import stream
    from bsrag.retrieval import deduplicate_by_page

    with st.chat_message("user"):
        st.write(question)

    retriever = get_retriever(profile, controls["mode"], settings.embedding.model_name)
    with st.spinner("Ищу в справке..."):
        result = retriever.search(
            question, top_k=controls["top_k"] * 3, use_reranker=controls["use_reranker"]
        )
    hits = deduplicate_by_page(result.hits, controls["top_k"])

    with st.chat_message("assistant"):
        st.caption(
            f"Найдено за {result.retrieval_ms:.0f} мс"
            + (f" + реранкинг {result.rerank_ms:.0f} мс" if result.rerank_ms else "")
        )
        text = ""
        if not controls["search_only"]:
            config = settings.generation.model_copy(update={"model": controls["model"]})
            placeholder = st.empty()
            started = time.perf_counter()
            for token in stream(question, hits, config):
                text += token
                placeholder.markdown(text + "▌")
            placeholder.markdown(text)
            st.caption(f"Ответ сгенерирован за {time.perf_counter() - started:.1f} с · {config.model}")

        st.markdown("**Источники в справке:**")
        for i, hit in enumerate(hits, start=1):
            render_source(i, hit, controls["show_images"])

    st.session_state.history.append({"question": question, "answer": text, "hits": hits})


main()
