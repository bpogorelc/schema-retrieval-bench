-- Top 5 questions where hybrid lost the most nDCG@5 against BM25.
-- Sequential quality rows only.
-- Placeholders: {per_query} replaced by analyze.py

WITH base AS (
    SELECT
        question_id,
        query_text,
        split,
        method,
        ndcg_at_5
    FROM file('{per_query}', Parquet)
    WHERE workload != 'throughput'
      AND topology = 'local_sequential'
),
pivoted AS (
    SELECT
        b.question_id,
        b.query_text,
        b.split,
        maxIf(b.ndcg_at_5, b.method = 'bm25') AS bm25_ndcg,
        maxIf(b.ndcg_at_5, b.method = 'hybrid') AS hybrid_ndcg
    FROM base AS b
    GROUP BY b.question_id, b.query_text, b.split
)
SELECT
    question_id,
    split,
    query_text,
    bm25_ndcg,
    hybrid_ndcg,
    (bm25_ndcg - hybrid_ndcg) AS hybrid_loss_vs_bm25
FROM pivoted
ORDER BY hybrid_loss_vs_bm25 DESC
LIMIT 5
