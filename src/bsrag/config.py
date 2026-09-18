"""Конфигурация проекта: YAML + переменные окружения."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class CorpusConfig(BaseModel):
    pages_dir: Path = Path("bs_6/docs/data/pages")
    media_dir: Path = Path("bs_6/docs/data/media")
    namespaces: list[str] = Field(default_factory=lambda: ["ru"])
    # `playground` — песочница DokuWiki, `dirout` — выгрузка списка файлов с диска.
    exclude: list[str] = Field(default_factory=lambda: ["ru/playground", "ru/dirout"])
    doc_url_template: str = "https://docs.business-studio.ru/doku.php/{page_id}"
    min_chars: int = 120
    drop_index_pages: bool = True


class ChunkingConfig(BaseModel):
    strategy: str = "header_recursive"
    chunk_size: int = 512
    chunk_overlap_ratio: float = 0.15
    table_mode: str = "markdown"
    prepend_breadcrumb: bool = True


class EmbeddingConfig(BaseModel):
    model_name: str = "deepvk/USER-bge-m3"
    query_prefix: str = ""
    passage_prefix: str = ""
    batch_size: int = 32
    device: str = "auto"
    normalize: bool = True


class RetrievalConfig(BaseModel):
    mode: str = "hybrid"
    top_k: int = 5
    candidates: int = 16
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    score_threshold: float | None = None


class GenerationConfig(BaseModel):
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    num_ctx: int = 8192
    max_tokens: int = 1024
    think: bool = False


class PathsConfig(BaseModel):
    processed: Path = Path("data/processed")
    index: Path = Path("data/index")
    cache: Path = Path("data/cache")
    goldset: Path = Path("data/goldset")
    reports: Path = Path("reports")


class Settings(BaseModel):
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(config_path: str | Path | None = None, **overrides: Any) -> Settings:
    path = Path(config_path or os.getenv("BSRAG_CONFIG") or DEFAULT_CONFIG)
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
