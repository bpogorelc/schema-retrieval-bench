#!/usr/bin/env python3
"""Print markdown-friendly summaries for README tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main(run_dir: str) -> None:
    run_path = Path(run_dir)
    pq = pd.read_parquet(run_path / "per_query.parquet")
    meta = pd.read_parquet(run_path / "run_meta.parquet")

    quality = pq[pq["workload"] != "throughput"].copy()
    # Unique question/method once
    quality = quality.sort_values("repeat_index").drop_duplicates(
        ["question_id", "method"], keep="first"
    )

    print("## Quality (unique questions)")
    g = (
        quality.groupby(["method", "split"], as_index=False)
        .agg(
            n=("question_id", "nunique"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            p50_latency_ms=("latency_ms", lambda s: float(s.quantile(0.5))),
            p95_latency_ms=("latency_ms", lambda s: float(s.quantile(0.95))),
        )
        .sort_values(["split", "method"])
    )
    print(g.to_string(index=False))

    held = quality[quality["split"] == "held_out"]
    print("\n## Held-out means")
    print(
        held.groupby("method")[["ndcg_at_5", "recall_at_5", "latency_ms"]]
        .agg({"ndcg_at_5": "mean", "recall_at_5": "mean", "latency_ms": ["median", lambda s: s.quantile(0.95)]})
        .to_string()
    )

    print("\n## Timing summaries")
    for _, row in meta.iterrows():
        phase = row.get("phase")
        if isinstance(phase, str) and phase.endswith("_summary"):
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "workers": row.get("workers"),
                        "startup_s_mean": row.get("startup_s_mean"),
                        "warm_eval_s_mean": row.get("warm_eval_s_mean"),
                        "startup_s_values": row.get("startup_s_values"),
                        "warm_eval_s_values": row.get("warm_eval_s_values"),
                    },
                    default=str,
                )
            )

    # Failure example: lowest hybrid nDCG on held_out among exact-ish misses
    hybrid = quality[quality["method"] == "hybrid"].sort_values("ndcg_at_5")
    print("\n## Lowest hybrid nDCG examples")
    cols = ["question_id", "split", "query_text", "ndcg_at_5", "recall_at_5", "ranked_ids"]
    print(hybrid[cols].head(5).to_string(index=False))

    # Hybrid vs BM25 losses
    pivot = quality.pivot_table(
        index=["question_id", "query_text", "split"],
        columns="method",
        values="ndcg_at_5",
        aggfunc="first",
    ).reset_index()
    if "bm25" in pivot.columns and "hybrid" in pivot.columns:
        pivot["loss"] = pivot["bm25"] - pivot["hybrid"]
        print("\n## Hybrid losses vs BM25")
        print(
            pivot.sort_values("loss", ascending=False)
            .head(5)[["question_id", "split", "query_text", "bm25", "hybrid", "loss"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main(sys.argv[1])
