"""Инспектор корпуса: что стало со страницей после очистки и как она нарезана."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import get_settings, selected_profile

st.set_page_config(page_title="Инспектор корпуса", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Читаю корпус...")
def load_corpus(profile: str):
    from bsrag.corpus import load_or_prepare_corpus

    return load_or_prepare_corpus(get_settings(profile))


@st.cache_resource(show_spinner="Нарезаю на чанки...")
def load_chunks(profile: str, strategy: str, size: int, overlap: float, table_mode: str):
    from bsrag.chunking import chunk_pages

    settings = get_settings(profile)
    config = settings.chunking.model_copy(
        update={
            "strategy": strategy,
            "chunk_size": size,
            "chunk_overlap_ratio": overlap,
            "table_mode": table_mode,
        }
    )
    return chunk_pages(load_corpus(profile), config, settings.embedding.model_name)


def raw_text(page, profile: str) -> str:
    settings = get_settings(profile)
    path = settings.resolve(settings.corpus.pages_dir) / f"{page.page_id}.txt"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def main() -> None:
    st.title("🔍 Инспектор корпуса")
    st.caption(
        "Слева — исходная разметка DokuWiki, справа — то, что уходит в индекс. "
        "Ниже — как страница разрезана на чанки при текущих настройках."
    )

    profile = selected_profile()
    pages = load_corpus(profile)
    settings = get_settings(profile)

    with st.sidebar:
        st.header("Нарезка")
        strategy = st.selectbox(
            "Стратегия", ["header_recursive", "fixed", "parent"],
            index=["header_recursive", "fixed", "parent"].index(settings.chunking.strategy),
        )
        size = st.slider("Размер чанка, токенов", 128, 768, settings.chunking.chunk_size, step=64)
        overlap = st.slider("Оверлап", 0.0, 0.4, settings.chunking.chunk_overlap_ratio, step=0.05)
        table_mode = st.radio("Таблицы", ["markdown", "rows"], horizontal=True)

    chunks = load_chunks(profile, strategy, size, overlap, table_mode)

    columns = st.columns(4)
    columns[0].metric("Страниц", len(pages))
    columns[1].metric("Чанков", len(chunks))
    columns[2].metric("Токенов в чанке (среднее)", f"{sum(c.n_tokens for c in chunks) / len(chunks):.0f}")
    columns[3].metric("Самый длинный чанк", max(c.n_tokens for c in chunks))

    st.plotly_chart(
        px.histogram(
            pd.DataFrame({"Токенов в чанке": [c.n_tokens for c in chunks]}),
            x="Токенов в чанке",
            nbins=50,
            title="Распределение длины чанков",
        ),
        use_container_width=True,
    )

    st.divider()
    titles = {f"{p.title} — {p.page_id}": p for p in pages}
    selected = st.selectbox("Страница справки", sorted(titles), index=0)
    page = titles[selected]

    left, right = st.columns(2)
    with left:
        st.markdown("**Исходная разметка DokuWiki**")
        st.code(raw_text(page, profile)[:6000], language="text")
    with right:
        st.markdown("**После очистки (уходит в индекс)**")
        st.markdown(page.markdown[:6000])

    page_chunks = [c for c in chunks if c.page_id == page.page_id]
    st.markdown(f"**Чанки этой страницы: {len(page_chunks)}**")
    for chunk in page_chunks:
        with st.expander(f"#{chunk.chunk_index} · {chunk.n_tokens} токенов · {chunk.headings or '—'}"):
            st.caption("Текст, который уходит в эмбеддер:")
            st.code(chunk.embed_text, language="markdown")


main()
