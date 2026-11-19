-- Quality and latency comparison by method and split.
-- Use sequential quality rows only so each question is counted once.
-- Placeholders: {per_query} replaced by analyze.py

SELECT
    method,
    split,
    countDistinct(question_id) AS n_questions,
    avg(ndcg_at_5) AS ndcg_at_5,
    avg(recall_at_5) AS recall_at_5,
    quantile(0.5)(latency_ms) AS p50_latency_ms,
    quantile(0.95)(latency_ms) AS p95_latency_ms
FROM file('{per_query}', Parquet)
WHERE workload != 'throughput'
  AND topology = 'local_sequential'
GROUP BY method, split
ORDER BY split, method
