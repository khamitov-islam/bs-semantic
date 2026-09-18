"""Векторное хранилище: Qdrant в локальном (embedded) режиме.

Qdrant выбран из-за встроенного гибридного поиска: плотные и разрежённые (BM25)
векторы лежат в одной коллекции, слияние результатов делает сам движок через RRF,
поэтому ансамблирование ретриверов не нужно писать руками.
"""

from __future__ import annotations

import json
import re
import shutil
import atexit
from dataclasses import asdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from bsrag.chunking import Chunk
from bsrag.config import Settings
from bsrag.embeddings import get_embeddings, get_sparse_embeddings

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


def collection_name(settings: Settings) -> str:
    """Имя коллекции кодирует конфигурацию, чтобы эксперименты не перетирали друг друга."""
    model = re.sub(r"[^a-zA-Z0-9]+", "_", settings.embedding.model_name).strip("_").lower()
    chunk = settings.chunking
    overlap = int(chunk.chunk_overlap_ratio * 100)
    breadcrumb = "" if chunk.prepend_breadcrumb else "_nobc"
    return f"bs_{model}_{chunk.strategy}_{chunk.chunk_size}_{overlap}_{chunk.table_mode}{breadcrumb}"


def chunk_to_document(chunk: Chunk) -> Document:
    """В индекс уходит ``embed_text`` (с контекстом), в ответ — исходный ``text``."""
    metadata = asdict(chunk)
    metadata["text"] = chunk.text
    metadata["context_text"] = chunk.context_text
    return Document(page_content=chunk.embed_text, metadata=metadata)


def document_to_chunk(doc: Document) -> Chunk:
    fields = {k: v for k, v in doc.metadata.items() if k in Chunk.__dataclass_fields__}
    return Chunk(**fields)


def _retrieval_mode(mode: str) -> RetrievalMode:
    return {
        "dense": RetrievalMode.DENSE,
        "sparse": RetrievalMode.SPARSE,
        "bm25": RetrievalMode.SPARSE,
        "hybrid": RetrievalMode.HYBRID,
    }[mode]


def index_path(settings: Settings) -> Path:
    return settings.resolve(settings.paths.index) / "qdrant"


_CLIENTS: dict[str, QdrantClient] = {}


def get_client(path: str | Path) -> QdrantClient:
    """Один клиент на каталог в пределах процесса.

    Qdrant в embedded-режиме держит эксклюзивную блокировку каталога, поэтому при
    переборе конфигураций нельзя открывать новое соединение на каждый прогон.
    """
    key = str(path)
    if key not in _CLIENTS:
        Path(key).mkdir(parents=True, exist_ok=True)
        _CLIENTS[key] = QdrantClient(path=key)
    return _CLIENTS[key]


def close_clients() -> None:
    for client in _CLIENTS.values():
        try:
            client.close()
        except Exception:
            pass
    _CLIENTS.clear()


atexit.register(close_clients)


def build_index(
    chunks: list[Chunk],
    settings: Settings,
    recreate: bool = True,
    batch_size: int = 128,
) -> QdrantVectorStore:
    name = collection_name(settings)
    # Коллекция всегда собирается гибридной: плотные и BM25-векторы лежат рядом,
    # а режим поиска выбирается уже при запросе. Это позволяет сравнивать
    # dense / sparse / hybrid на одном и том же индексе.
    mode = RetrievalMode.HYBRID
    client = get_client(index_path(settings))

    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    dense, sparse = _encoders(settings, mode)
    vectors_config: dict[str, qmodels.VectorParams] = {}
    sparse_config: dict[str, qmodels.SparseVectorParams] = {}
    if dense is not None:
        size = len(dense.embed_query("тест"))
        vectors_config[DENSE_VECTOR] = qmodels.VectorParams(
            size=size, distance=qmodels.Distance.COSINE
        )
    if sparse is not None:
        sparse_config[SPARSE_VECTOR] = qmodels.SparseVectorParams(index=qmodels.SparseIndexParams())

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )

    store = QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=mode,
        vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
    )

    documents = [chunk_to_document(chunk) for chunk in chunks]
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])

    _write_manifest(settings, chunks)
    return store


def _encoders(settings: Settings, mode: RetrievalMode):
    dense = sparse = None
    if mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID):
        dense = get_embeddings(
            settings.embedding.model_name,
            device=settings.embedding.device,
            batch_size=settings.embedding.batch_size,
            normalize=settings.embedding.normalize,
        )
    if mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID):
        sparse = get_sparse_embeddings()
    return dense, sparse


def index_exists(settings: Settings) -> bool:
    return manifest_path(settings).exists()


def open_index(settings: Settings) -> QdrantVectorStore:
    name = collection_name(settings)
    mode = _retrieval_mode(settings.retrieval.mode)
    dense, sparse = _encoders(settings, mode)

    client = get_client(index_path(settings))
    if not client.collection_exists(name):
        raise FileNotFoundError(
            f"Коллекция {name} не найдена. Сначала постройте индекс: `bsrag index`."
        )
    return QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=mode,
        vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
    )


def manifest_path(settings: Settings) -> Path:
    return index_path(settings) / f"{collection_name(settings)}.manifest.json"


def read_manifest(settings: Settings) -> dict:
    path = manifest_path(settings)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_manifest(settings: Settings, chunks: list[Chunk]) -> None:
    tokens = [c.n_tokens for c in chunks] or [0]
    manifest = {
        "collection": collection_name(settings),
        "n_chunks": len(chunks),
        "n_pages": len({c.page_id for c in chunks}),
        "embedding_model": settings.embedding.model_name,
        "chunking": settings.chunking.model_dump(),
        "retrieval_mode": settings.retrieval.mode,
        "tokens_mean": sum(tokens) / len(tokens),
        "tokens_max": max(tokens),
    }
    manifest_path(settings).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def drop_index(settings: Settings) -> None:
    close_clients()
    shutil.rmtree(index_path(settings), ignore_errors=True)
