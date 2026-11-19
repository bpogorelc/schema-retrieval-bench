from schema_bench.catalog import load_catalog
from schema_bench.retrievers.bm25 import BM25Retriever
from schema_bench.retrievers.hybrid import reciprocal_rank_fusion


def test_bm25_stable_ranking():
    docs = load_catalog()
    retriever = BM25Retriever(docs)
    a = retriever.retrieve("net revenue", top_k=5)
    b = retriever.retrieve("net revenue", top_k=5)
    assert a == b


def test_rrf_deterministic():
    left = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
    right = [("b", 1.0), ("a", 0.9), ("d", 0.8)]
    fused1 = reciprocal_rank_fusion([left, right], k=60)
    fused2 = reciprocal_rank_fusion([left, right], k=60)
    assert fused1 == fused2
