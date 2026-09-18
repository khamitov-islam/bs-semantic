"""Сравнение кросс-энкодеров: качество переупорядочивания против задержки."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bsrag.config import load_settings  # noqa: E402
from bsrag.embeddings import resolve_device  # noqa: E402
from bsrag.retrieval import Retriever  # noqa: E402

CANDIDATES = [
    ("BAAI/bge-reranker-v2-m3", 1024, None),
    ("BAAI/bge-reranker-v2-m3", 512, None),
    ("BAAI/bge-reranker-v2-m3", 512, "float16"),
    ("DiTy/cross-encoder-russian-msmarco", 512, None),
    ("qilowoq/bge-reranker-v2-m3-en-ru", 512, None),
]

PROBES = [
    ("Какой горячей клавишей вызвать окно Права доступа?", "ru/manual/shortcuts"),
    ("Можно ли сохранить базу, пока с ней работают другие?", "ru/manual/administration/backup"),
    ("Как поменять номер порта OData?", "ru/technical_manual/work_via_odata"),
    ("Какой порт нужен для PostgreSQL?", "ru/manual/install/install/used_ports"),
    ("Чем ограничить доступ группы к справочникам?", "ru/manual/administration/user_rights"),
    ("Как затянуть данные из экселя в модель?", "ru/manual/export_import/customizable_data_exchange"),
]


def main() -> None:
    settings = load_settings()
    retriever = Retriever(settings)

    pools = []
    for question, gold in PROBES:
        result = retriever.search(question, top_k=40, use_reranker=False)
        pools.append((question, gold, result.hits))

    baseline = sum(
        any(h.chunk.page_id == gold for h in hits[:5]) for _, gold, hits in pools
    )
    print(f"без реранкера: попаданий в топ-5 {baseline}/{len(PROBES)}\n")

    for model_name, max_length, dtype in CANDIDATES:
        try:
            from sentence_transformers import CrossEncoder

            kwargs = {"device": resolve_device("auto"), "max_length": max_length}
            if dtype:
                kwargs["model_kwargs"] = {"torch_dtype": dtype}
            encoder = CrossEncoder(model_name, **kwargs)
            encoder.predict([("прогрев", "прогрев модели")], show_progress_bar=False)
        except Exception as error:
            print(f"{model_name} ({max_length}, {dtype}): недоступен — {str(error)[:90]}")
            continue

        hits_at_5 = 0
        elapsed = 0.0
        for question, gold, hits in pools:
            pairs = [(question, h.chunk.embed_text) for h in hits]
            started = time.perf_counter()
            scores = encoder.predict(pairs, show_progress_bar=False)
            elapsed += time.perf_counter() - started
            ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:5]
            hits_at_5 += any(h.chunk.page_id == gold for h, _ in ranked)

        label = f"{model_name} (max_len={max_length}, {dtype or 'float32'})"
        print(f"{label:62s} топ-5 {hits_at_5}/{len(PROBES)} · {elapsed / len(PROBES) * 1000:6.0f} мс/запрос")


if __name__ == "__main__":
    main()
