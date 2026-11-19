from __future__ import annotations

from typing import Sequence

from schema_bench.metrics import deterministic_rank
from schema_bench.retrievers.bm25 import BM25Retriever
from schema_bench.retrievers.dense import DenseRetriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match rankings")
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (k + rank))
    return list(scores.items())


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        candidate_pool: int = 50,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.candidate_pool = candidate_pool

    def score(self, query: str) -> list[tuple[str, float]]:
        bm25_rank = self.bm25.retrieve(query, top_k=self.candidate_pool)
        dense_rank = self.dense.retrieve(query, top_k=self.candidate_pool)
        fused = reciprocal_rank_fusion(
            [bm25_rank, dense_rank],
            k=self.rrf_k,
            weights=[self.bm25_weight, self.dense_weight],
        )
        return fused

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        return deterministic_rank(self.score(query), top_k=top_k)


def tune_hybrid_on_dev(
    bm25: BM25Retriever,
    dense: DenseRetriever,
    questions: Sequence,
    top_k: int = 5,
) -> HybridRetriever:
    """Grid-search simple RRF weights on the development split only."""
    from schema_bench.metrics import mean, ndcg_at_k

    grid = [
        (60, 1.0, 1.0),
        (60, 1.5, 1.0),
        (60, 1.0, 1.5),
        (40, 1.0, 1.0),
        (80, 1.0, 1.0),
    ]
    best = None
    best_score = -1.0
    for rrf_k, bw, dw in grid:
        retriever = HybridRetriever(
            bm25, dense, rrf_k=rrf_k, bm25_weight=bw, dense_weight=dw
        )
        scores = []
        for q in questions:
            ranked = [doc_id for doc_id, _ in retriever.retrieve(q.text, top_k=top_k)]
            scores.append(ndcg_at_k(ranked, q.relevant, k=top_k))
        avg = mean(scores)
        if avg > best_score:
            best_score = avg
            best = retriever
    assert best is not None
    return best
