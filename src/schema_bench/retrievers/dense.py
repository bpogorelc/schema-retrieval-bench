from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from schema_bench import DENSE_MODEL_ID
from schema_bench.catalog import SchemaDoc
from schema_bench.metrics import deterministic_rank
from schema_bench.paths import CACHE_DIR, ensure_output_dirs
from schema_bench.threading_utils import cap_cpu_threads


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        docs: Sequence[SchemaDoc],
        model_id: str = DENSE_MODEL_ID,
        cache_dir: Path | None = None,
        device: str = "cpu",
    ):
        cap_cpu_threads(1)
        ensure_output_dirs()
        self.docs = list(docs)
        self.doc_ids = [d.id for d in self.docs]
        self.model_id = model_id
        self.cache_dir = cache_dir or CACHE_DIR
        self.device = device
        self._model = None
        self.doc_embeddings = self._load_or_encode_docs()

    def _cache_path(self) -> Path:
        payload = "\n".join(f"{d.id}\t{d.text}" for d in self.docs)
        digest = hashlib.sha256(f"{self.model_id}\n{payload}".encode("utf-8")).hexdigest()[:16]
        safe_model = self.model_id.replace("/", "__")
        return self.cache_dir / f"doc_emb__{safe_model}__{digest}.npy"

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def _load_or_encode_docs(self) -> np.ndarray:
        path = self._cache_path()
        if path.exists():
            return np.load(path)
        model = self._get_model()
        texts = [d.text for d in self.docs]
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(emb, dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        return arr

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        model = self._get_model()
        emb = model.encode(
            list(queries),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def score(self, query: str) -> list[tuple[str, float]]:
        q = self.encode_queries([query])[0]
        sims = self.doc_embeddings @ q
        return [(doc_id, float(score)) for doc_id, score in zip(self.doc_ids, sims)]

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        return deterministic_rank(self.score(query), top_k=top_k)
