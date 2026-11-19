from __future__ import annotations

import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from schema_bench.catalog import SchemaDoc
from schema_bench.metrics import deterministic_rank

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self, docs: Sequence[SchemaDoc]):
        self.docs = list(docs)
        self.doc_ids = [d.id for d in self.docs]
        corpus = [tokenize(d.text) for d in self.docs]
        self._bm25 = BM25Okapi(corpus)

    def score(self, query: str) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        paired = [(doc_id, float(score)) for doc_id, score in zip(self.doc_ids, scores)]
        return paired

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        return deterministic_rank(self.score(query), top_k=top_k)
