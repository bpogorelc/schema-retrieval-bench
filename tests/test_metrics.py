from schema_bench.metrics import deterministic_rank, ndcg_at_k, percentile, recall_at_k


def test_deterministic_tie_break():
    scored = [("b", 1.0), ("a", 1.0), ("c", 0.5)]
    ranked = deterministic_rank(scored, top_k=3)
    assert [doc for doc, _ in ranked] == ["a", "b", "c"]


def test_ndcg_perfect():
    relevance = {"a": 3, "b": 2}
    assert ndcg_at_k(["a", "b"], relevance, k=2) == 1.0


def test_recall_at_k():
    relevance = {"a": 2, "b": 1, "c": 0}
    assert recall_at_k(["c", "a"], relevance, k=2) == 0.5


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
