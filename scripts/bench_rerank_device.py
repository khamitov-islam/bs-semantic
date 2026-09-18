"""Где быстрее считать кросс-энкодер на Apple Silicon: MPS или CPU."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentence_transformers import CrossEncoder  # noqa: E402

QUERY = "Какой горячей клавишей вызвать окно Права доступа?"
PASSAGE = "Горячие клавиши. Ctrl+R — вызвать окно Права доступа в Окне свойств объекта. " * 12

CONFIGS = [
    ("BAAI/bge-reranker-v2-m3", "mps", "float16"),
    ("BAAI/bge-reranker-v2-m3", "cpu", None),
    ("DiTy/cross-encoder-russian-msmarco", "mps", None),
    ("DiTy/cross-encoder-russian-msmarco", "cpu", None),
]


def main() -> None:
    for model_name, device, dtype in CONFIGS:
        kwargs = {"device": device, "max_length": 512}
        if dtype:
            kwargs["model_kwargs"] = {"torch_dtype": dtype}
        try:
            encoder = CrossEncoder(model_name, **kwargs)
        except Exception as error:
            print(f"{model_name} / {device}: {str(error)[:80]}")
            continue

        pairs = [(QUERY, PASSAGE)] * 40
        encoder.predict(pairs[:4], show_progress_bar=False)
        for batch in (16, 40):
            started = time.perf_counter()
            encoder.predict(pairs, batch_size=batch, show_progress_bar=False)
            elapsed = (time.perf_counter() - started) * 1000
            print(f"{model_name:38s} {device:4s} {dtype or 'fp32':8s} batch={batch:2d}: {elapsed:6.0f} мс / 40 пар")


if __name__ == "__main__":
    main()
