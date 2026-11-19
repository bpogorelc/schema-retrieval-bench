from __future__ import annotations

import math
from typing import Iterable, Sequence


def deterministic_rank(
    scored: Sequence[tuple[str, float]],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Sort by descending score, then ascending doc_id for stable ties."""
    ranked = sorted(scored, key=lambda x: (-x[1], x[0]))
    if top_k is not None:
        return ranked[:top_k]
    return ranked


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    total = 0.0
    for i, gain in enumerate(gains[:k]):
        total += (2**gain - 1) / math.log2(i + 2)
    return total


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: dict[str, int],
    k: int = 5,
) -> float:
    gains = [float(relevance.get(doc_id, 0)) for doc_id in ranked_ids[:k]]
    ideal = sorted((float(v) for v in relevance.values() if v > 0), reverse=True)
    if not ideal:
        return 0.0
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / ideal_dcg


def recall_at_k(
    ranked_ids: Sequence[str],
    relevance: dict[str, int],
    k: int = 5,
    min_grade: int = 1,
) -> float:
    relevant_ids = {doc_id for doc_id, grade in relevance.items() if grade >= min_grade}
    if not relevant_ids:
        return 0.0
    hit = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_ids)
    return hit / len(relevant_ids)


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)
