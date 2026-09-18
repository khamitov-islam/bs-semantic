"""Локальные эмбеддинг-модели.

Обёртка над ``langchain_huggingface`` нужна ровно для одного: модели семейства
E5 / RoSBERTa / BERTA требуют разных префиксов для запроса и документа, и если
их перепутать или не поставить, качество поиска заметно падает.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

# Префиксы, которых требуют карточки моделей.
PREFIX_PRESETS: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-": ("query: ", "passage: "),
    "deepvk/USER-base": ("query: ", "passage: "),
    "ai-forever/ru-en-RoSBERTa": ("search_query: ", "search_document: "),
    "sergeyzh/BERTA": ("search_query: ", "search_document: "),
    "sergeyzh/rubert-": ("search_query: ", "search_document: "),
    "BAAI/bge-m3": ("", ""),
    "deepvk/USER-bge-m3": ("", ""),
}


def prefixes_for(model_name: str) -> tuple[str, str]:
    for prefix, pair in PREFIX_PRESETS.items():
        if model_name.startswith(prefix):
            return pair
    return "", ""


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class PrefixedEmbeddings(HuggingFaceEmbeddings):
    """HuggingFace-эмбеддинги с раздельными префиксами запроса и документа."""

    query_prefix: str = ""
    passage_prefix: str = ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(self.query_prefix + text)


@lru_cache(maxsize=4)
def get_embeddings(
    model_name: str,
    device: str = "auto",
    batch_size: int = 32,
    normalize: bool = True,
    query_prefix: str | None = None,
    passage_prefix: str | None = None,
) -> PrefixedEmbeddings:
    preset_query, preset_passage = prefixes_for(model_name)
    model_kwargs: dict[str, Any] = {"device": resolve_device(device)}
    if model_name.startswith(("jinaai/", "nomic-ai/")):
        model_kwargs["trust_remote_code"] = True

    return PrefixedEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": normalize, "batch_size": batch_size},
        query_prefix=preset_query if query_prefix is None else query_prefix,
        passage_prefix=preset_passage if passage_prefix is None else passage_prefix,
    )


@lru_cache(maxsize=2)
def get_sparse_embeddings(model_name: str = "Qdrant/bm25"):
    """Разрежённые векторы BM25 для гибридного поиска.

    Русская морфология здесь критична: без стеммера «справочника» и «справочник»
    считаются разными термами.
    """
    from langchain_qdrant import FastEmbedSparse

    try:
        return FastEmbedSparse(model_name=model_name, language="russian")
    except TypeError:
        return FastEmbedSparse(model_name=model_name)


def embedding_dimension(model_name: str, device: str = "auto") -> int:
    return len(get_embeddings(model_name, device=device).embed_query("тест"))
