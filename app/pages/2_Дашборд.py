"""Дашборд экспериментов: чем обоснован выбор каждой части решения."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from common import get_settings, selected_profile

st.set_page_config(page_title="Дашборд экспериментов", page_icon="📊", layout="wide")

METRIC_LABELS = {
    "ndcg@10": "nDCG@10",
    "mrr@10": "MRR@10",
    "hit_rate@1": "Hit@1",
    "hit_rate@3": "Hit@3",
    "hit_rate@5": "Hit@5",
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
}


@st.cache_data(show_spinner=False)
def load_results(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def results_frame(filename: str = "benchmarks.csv") -> pd.DataFrame | None:
    settings = get_settings("improved")
    path = settings.resolve(settings.paths.reports) / filename
    if not path.exists():
        return None
    return load_results(str(path), path.stat().st_mtime)


def section_chunking(frame: pd.DataFrame, metric: str) -> None:
    data = frame[frame.stage == "chunking"]
    if data.empty:
        return
    st.subheader("Этап A. Нарезка на чанки")
    st.caption(
        "Сетка «размер × оверлап» на фиксированной модели эмбеддингов, затем — "
        "вклад отдельных решений по нарезке."
    )

    grid = data[data["ablation"].isna()] if "ablation" in data.columns else data
    if not grid.empty:
        pivot = grid.pivot_table(index="chunk_size", columns="overlap", values=metric)
        heat = px.imshow(
            pivot,
            text_auto=".3f",
            color_continuous_scale="Blues",
            labels={"x": "Оверлап", "y": "Размер чанка, токенов", "color": METRIC_LABELS.get(metric, metric)},
            aspect="auto",
        )
        left, right = st.columns([3, 2])
        left.plotly_chart(heat, use_container_width=True)

        sizes = grid.groupby("chunk_size")[["n_chunks", "latency_p50_ms"]].mean().reset_index()
        right.plotly_chart(
            px.bar(
                sizes,
                x="chunk_size",
                y="n_chunks",
                labels={"chunk_size": "Размер чанка", "n_chunks": "Число чанков в индексе"},
                title="Стоимость индекса",
            ),
            use_container_width=True,
        )

    if "ablation" in data.columns and data["ablation"].notna().any():
        cols = [c for c in ("name", metric, "n_chunks", "collection") if c in data.columns]
        ablations = data[data["ablation"].notna()][cols].copy()
        baseline = grid[metric].max() if not grid.empty else None
        chart = px.bar(
            ablations.sort_values(metric),
            x=metric,
            y="name",
            orientation="h",
            text="n_chunks" if "n_chunks" in ablations.columns else None,
            labels={"name": "", metric: METRIC_LABELS.get(metric, metric)},
            title="Что будет, если сделать иначе (подпись — число чанков)",
        )
        if baseline is not None:
            chart.add_vline(
                x=baseline,
                line_dash="dash",
                annotation_text="лучшая конфигурация сетки",
                line_color="green",
            )
        st.plotly_chart(chart, use_container_width=True)
        st.caption(
            "parent-document не меняет текст эмбеддера — только контекст для LLM, "
            "поэтому retrieval-метрики совпадают с header-aware. "
            "no breadcrumbs и tables-as-rows считаются на отдельных коллекциях."
        )
        if "collection" in ablations.columns:
            st.dataframe(
                ablations.rename(columns={metric: METRIC_LABELS.get(metric, metric)}),
                use_container_width=True,
                hide_index=True,
            )


def section_embeddings(frame: pd.DataFrame, metric: str) -> None:
    data = frame[frame.stage == "embeddings"]
    if data.empty:
        return
    st.subheader("Этап B. Модели эмбеддингов")
    st.caption("Качество против стоимости: индексация корпуса и задержка поиска.")

    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(
            data.sort_values(metric),
            x=metric,
            y="name",
            orientation="h",
            labels={"name": "", metric: METRIC_LABELS.get(metric, metric)},
            title="Качество поиска",
        ),
        use_container_width=True,
    )
    scatter = px.scatter(
        data,
        x="index_build_s",
        y=metric,
        text="name",
        size="latency_p50_ms",
        labels={
            "index_build_s": "Индексация корпуса, с",
            metric: METRIC_LABELS.get(metric, metric),
        },
        title="Качество и цена (размер точки — задержка запроса)",
    )
    scatter.update_traces(textposition="top center")
    right.plotly_chart(scatter, use_container_width=True)


def section_retrieval(frame: pd.DataFrame, metric: str) -> None:
    data = frame[frame.stage == "retrieval"]
    if data.empty:
        return
    st.subheader("Этап C. Схема поиска")
    st.caption("Вклад BM25, векторного поиска и кросс-энкодера по отдельности.")

    data = data.copy()
    data["Реранкер"] = data["reranker"].map({True: "с реранкером", False: "без реранкера"})
    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(
            data,
            x="mode",
            y=metric,
            color="Реранкер",
            barmode="group",
            labels={"mode": "Режим поиска", metric: METRIC_LABELS.get(metric, metric)},
            title="Качество",
        ),
        use_container_width=True,
    )
    right.plotly_chart(
        px.bar(
            data,
            x="mode",
            y="latency_p50_ms",
            color="Реранкер",
            barmode="group",
            labels={"mode": "Режим поиска", "latency_p50_ms": "Задержка (медиана), мс"},
            title="Задержка",
        ),
        use_container_width=True,
    )


def section_llm(frame: pd.DataFrame) -> None:
    data = frame[frame.stage == "llm"]
    if data.empty:
        return
    st.subheader("Этап D. Локальные LLM")
    st.caption(
        "Обоснованность и полнота выставлены другой локальной моделью-судьёй. "
        "«Отказы на ловушках» — доля вопросов без ответа в справке, на которых "
        "система честно призналась, что ответа нет."
    )

    columns = [c for c in ("faithfulness", "completeness", "citation_rate", "trap_refusal_rate") if c in data]
    melted = data.melt(id_vars="name", value_vars=columns, var_name="Метрика", value_name="Значение")
    melted["Метрика"] = melted["Метрика"].map(
        {
            "faithfulness": "Обоснованность",
            "completeness": "Полнота",
            "citation_rate": "Корректные ссылки",
            "trap_refusal_rate": "Отказы на ловушках",
        }
    )
    left, right = st.columns([3, 2])
    left.plotly_chart(
        px.bar(melted, x="Метрика", y="Значение", color="name", barmode="group", title="Качество ответов"),
        use_container_width=True,
    )
    if "latency_p50_s" in data:
        right.plotly_chart(
            px.bar(
                data,
                x="name",
                y=["latency_p50_s", "latency_p95_s"],
                barmode="group",
                labels={"name": "", "value": "Секунды", "variable": ""},
                title="Время ответа",
            ),
            use_container_width=True,
        )


def section_improve(frame: pd.DataFrame, metric: str) -> None:
    data = frame[frame.stage == "improve"]
    if data.empty:
        return
    st.subheader("Этап E. Улучшения поверх бейслайна")
    st.caption(
        "Один фактор поверх опоры C (USER hybrid+реранкер). "
        "Бейслайн в сайдбаре — это опора C; улучшенный собирает nobc + таблицы + раскрытие."
    )
    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(
            data.sort_values(metric),
            x=metric,
            y="name",
            orientation="h",
            labels={"name": "", metric: METRIC_LABELS.get(metric, metric)},
            title="Качество поиска",
        ),
        use_container_width=True,
    )
    if "latency_p50_ms" in data.columns:
        right.plotly_chart(
            px.bar(
                data.sort_values("latency_p50_ms"),
                x="latency_p50_ms",
                y="name",
                orientation="h",
                labels={"name": "", "latency_p50_ms": "Задержка (медиана), мс"},
                title="Задержка",
            ),
            use_container_width=True,
        )


def section_goldset_expand(frame: pd.DataFrame, metric: str) -> None:
    data = frame[frame.stage == "goldset_expand"]
    if data.empty:
        return
    n_manual = int(data["n_manual"].dropna().max()) if "n_manual" in data.columns else 0
    st.subheader("Расширенный ручной goldset")
    st.caption(
        f"Только каверзные вопросы ({n_manual or 'все ручные'}), без синтетики из чанков. "
        "Один и тот же набор для бейслайна и улучшенного профиля."
    )
    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(
            data.sort_values(metric),
            x=metric,
            y="name",
            orientation="h",
            labels={"name": "", metric: METRIC_LABELS.get(metric, metric)},
            title="Качество на ручных вопросах",
        ),
        use_container_width=True,
    )
    if "hit_rate@1" in data.columns:
        right.plotly_chart(
            px.bar(
                data.sort_values("hit_rate@1"),
                x=["hit_rate@1", "hit_rate@3"],
                y="name",
                barmode="group",
                orientation="h",
                labels={"name": "", "value": "Доля", "variable": ""},
                title="Hit@1 и Hit@3",
            ),
            use_container_width=True,
        )


def section_warmup(frame: pd.DataFrame) -> None:
    data = frame[frame.stage == "warmup"]
    if data.empty:
        return
    st.subheader("Прогрев первого ответа")
    st.caption("Холодный запуск грузит веса реранкера. После прогрева в UI этот счёт не повторяется.")
    st.plotly_chart(
        px.bar(
            data,
            x="name",
            y="latency_p50_ms",
            labels={"name": "", "latency_p50_ms": "мс"},
            title="Поиск + реранкер",
        ),
        use_container_width=True,
    )


def section_baseline_compare(current: pd.DataFrame, metric: str) -> None:
    old = results_frame("benchmarks_baseline.csv")
    if old is None or old.empty:
        return
    st.subheader("Бейслайн vs пересчёт абляций")
    st.caption(
        "В бейслайне parent / rows / no breadcrumbs имели одинаковые метрики — "
        "они читали одну коллекцию. Справа — отдельные коллекции."
    )
    old_ab = old[old.get("ablation").notna()] if "ablation" in old.columns else pd.DataFrame()
    new_ab = current[current["ablation"].notna()] if "ablation" in current.columns else pd.DataFrame()
    if old_ab.empty or new_ab.empty or metric not in old_ab.columns:
        return
    merged = old_ab[["name", metric]].merge(
        new_ab[["name", metric, "n_chunks"]],
        on="name",
        suffixes=("_бейслайн", "_сейчас"),
    )
    long = merged.melt(
        id_vars="name",
        value_vars=[f"{metric}_бейслайн", f"{metric}_сейчас"],
        var_name="набор",
        value_name=metric,
    )
    long["набор"] = long["набор"].str.replace(f"{metric}_", "", regex=False)
    st.plotly_chart(
        px.bar(
            long,
            x=metric,
            y="name",
            color="набор",
            barmode="group",
            orientation="h",
            labels={"name": "", metric: METRIC_LABELS.get(metric, metric)},
        ),
        use_container_width=True,
    )


def main() -> None:
    selected_profile()
    st.title("📊 Дашборд экспериментов")
    frame = results_frame()
    if frame is None or frame.empty:
        st.info("Результатов пока нет. Запустите `bsrag bench all`.")
        return

    metric = st.selectbox(
        "Метрика качества поиска",
        [m for m in METRIC_LABELS if m in frame.columns],
        format_func=lambda m: METRIC_LABELS[m],
    )

    best = frame[frame.stage != "llm"].sort_values(metric, ascending=False).head(1)
    if not best.empty:
        row = best.iloc[0]
        columns = st.columns(4)
        columns[0].metric(METRIC_LABELS[metric], f"{row[metric]:.3f}")
        columns[1].metric("Лучшая конфигурация", str(row["name"]))
        columns[2].metric("Задержка (медиана)", f"{row.get('latency_p50_ms', 0):.0f} мс")
        columns[3].metric("Экспериментов", str(len(frame)))

    section_chunking(frame, metric)
    section_baseline_compare(frame, metric)
    section_embeddings(frame, metric)
    section_retrieval(frame, metric)
    section_improve(frame, metric)
    section_goldset_expand(frame, metric)
    section_warmup(frame)
    section_llm(frame)

    with st.expander("Все результаты таблицей"):
        st.dataframe(frame, use_container_width=True)
        st.download_button("Скачать CSV", frame.to_csv(index=False), "benchmarks.csv")


main()
