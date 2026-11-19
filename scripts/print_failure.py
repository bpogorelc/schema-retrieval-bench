import os
from pathlib import Path

import pandas as pd

pq = pd.read_parquet(
    "outputs/runs/benchmark_20261119T222606Z_a1361f0f/per_query.parquet"
)
q = pq[
    (pq.question_id == "q16")
    & (pq.topology == "local_sequential")
    & (pq.workload != "throughput")
]
for _, r in q.iterrows():
    print(r.method, round(float(r.ndcg_at_5), 3), list(r.ranked_ids))

cache = Path.home() / ".cache" / "huggingface"
print("hf_cache", cache)
if cache.exists():
    for p in cache.rglob("*"):
        if "MiniLM" in str(p):
            print("model_path_sample", p)
            break
