"""Раскрытие запроса и уникальность имён коллекций для абляций."""

from __future__ import annotations

from bsrag.config import ChunkingConfig, EmbeddingConfig, RetrievalConfig, Settings
from bsrag.corpus import pages_file
from bsrag.index import collection_name
from bsrag.retrieval import expand_queries, rrf_merge
from bsrag.chunking import Chunk
from bsrag.retrieval import Hit


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        page_id="ru/demo",
        title="Демо",
        breadcrumb="Справка",
        headings="Раздел",
        text="текст",
        url="https://example",
        source_path="demo.txt",
    )


def test_expand_hotkey_query_adds_help_wording() -> None:
    variants = expand_queries("Какими клавишами выделить несколько объектов?")
    assert variants[0].startswith("Какими клавишами")
    assert any("горячая клавиша" in item for item in variants)


def test_expand_unknown_query_stays_single() -> None:
    question = "Что такое функция?"
    assert expand_queries(question) == [question]


def test_rrf_merge_prefers_shared_high_ranks() -> None:
    a = Hit(_chunk("a"), score=1.0, rank=0, retriever_score=1.0)
    b = Hit(_chunk("b"), score=0.5, rank=1, retriever_score=0.5)
    c = Hit(_chunk("c"), score=0.1, rank=0, retriever_score=0.1)
    merged = rrf_merge([[a, b], [b, c]], k=60)
    assert merged[0].chunk.chunk_id == "b"


def test_collection_names_differ_for_chunking_ablations() -> None:
    base = Settings(
        embedding=EmbeddingConfig(model_name="intfloat/multilingual-e5-base"),
        chunking=ChunkingConfig(strategy="header_recursive", chunk_size=256, chunk_overlap_ratio=0.15),
        retrieval=RetrievalConfig(use_reranker=False),
    )
    names = {
        "base": collection_name(base),
        "fixed": collection_name(
            base.model_copy(update={"chunking": base.chunking.model_copy(update={"strategy": "fixed"})})
        ),
        "parent": collection_name(
            base.model_copy(update={"chunking": base.chunking.model_copy(update={"strategy": "parent"})})
        ),
        "rows": collection_name(
            base.model_copy(update={"chunking": base.chunking.model_copy(update={"table_mode": "rows"})})
        ),
        "nobc": collection_name(
            base.model_copy(
                update={"chunking": base.chunking.model_copy(update={"prepend_breadcrumb": False})}
            )
        ),
    }
    assert len(set(names.values())) == 5
    assert names["nobc"].endswith("_nobc")
    assert names["base"] != names["nobc"]


def test_table_rows_keep_header_on_each_line() -> None:
    from bsrag.chunking import iter_markdown_tables

    md = "# Клавиши\n\n| Клавиша | Действие |\n| --- | --- |\n| Ctrl+R | Права доступа |\n| F1 | Справка |\n"
    tables = iter_markdown_tables(md)
    assert len(tables) == 1
    path, header, rows = tables[0]
    assert path == "Клавиши"
    assert header[0] == "Клавиша"
    assert rows[0][0] == "Ctrl+R"


def test_lexical_query_gets_sparse_weight() -> None:
    from bsrag.retrieval import fusion_weights, is_lexical_query

    assert is_lexical_query("Какой горячей клавишей вызвать окно Права доступа?")
    hot = fusion_weights("Ctrl+R")
    long = fusion_weights("Две проектные группы правили модель в своих ветках")
    assert hot["sparse"] > hot["dense"]
    assert long["dense"] > long["sparse"]


def test_pages_cache_splits_by_table_mode() -> None:
    markdown = Settings()
    rows = Settings(chunking=ChunkingConfig(table_mode="rows"))
    assert pages_file(markdown).name in {"pages.jsonl", "pages_markdown.jsonl"}
    assert pages_file(rows).name == "pages_rows.jsonl"
    assert pages_file(markdown) != pages_file(rows)
