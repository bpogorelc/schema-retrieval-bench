from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from schema_bench.eval_ray import assert_same_rankings, run_ray_eval
from schema_bench.eval_sequential import run_sequential_eval
from schema_bench.hardware import hardware_fingerprint
from schema_bench.io_parquet import write_run_outputs
from schema_bench.labels import load_questions
from schema_bench.paths import ensure_output_dirs
from schema_bench.timing import cap_workers, parse_worker_list, summarize_timing_runs


def _run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def _dedupe_quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count each unique question once for quality metrics."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("workload") == "throughput":
            continue
        key = (row["question_id"], row["method"])
        prev = best.get(key)
        if prev is None or row["repeat_index"] < prev["repeat_index"]:
            best[key] = row
    return list(best.values())


def run_full_benchmark(
    ray_address: str | None = None,
    workers: str = "1,2,4",
    throughput_repeats: int = 10,
    timing_repeats: int = 3,
    batch_size: int = 5,
    top_k: int = 5,
) -> dict[str, Any]:
    ensure_output_dirs()
    run_id = _run_id("benchmark")
    bank = load_questions()
    all_ids = [q.id for q in bank.questions]
    held_out_ids = [q.id for q in bank.by_split("held_out")]
    worker_counts = cap_workers(parse_worker_list(workers))

    all_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    hw = hardware_fingerprint()

    # Initialize Ray once with enough CPUs for the largest worker config.
    import os

    import ray

    os.environ.setdefault("RAY_AUTH_MODE", "disabled")
    if ray.is_initialized():
        ray.shutdown()
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        ray.init(
            num_cpus=max(worker_counts),
            ignore_reinit_error=True,
            include_dashboard=False,
        )

    # --- Sequential quality on all questions (unique) ---
    seq_rows, seq_meta = run_sequential_eval(
        question_ids=all_ids,
        tune_hybrid=True,
        run_id=run_id,
        workload="quality",
    )
    seq_meta = {**seq_meta, **hw, "phase": "quality_sequential"}
    all_rows.extend(seq_rows)
    meta_rows.append(seq_meta)

    # --- Ray quality on held-out (ranking equality check vs sequential) ---
    seq_held = [r for r in seq_rows if r["question_id"] in set(held_out_ids)]
    ray_rows, ray_meta = run_ray_eval(
        held_out_ids,
        ray_address=ray_address,
        num_workers=worker_counts[0],
        batch_size=batch_size,
        top_k=top_k,
        tune_hybrid=True,
        run_id=run_id,
        workload="quality",
    )
    assert_same_rankings(seq_held, ray_rows)
    ray_meta = {
        **ray_meta,
        **hw,
        "phase": "quality_ray_rank_check",
        "rankings_match_sequential": True,
    }
    all_rows.extend(ray_rows)
    meta_rows.append(ray_meta)

    # --- Timing: sequential x repeats (throughput replay for larger workload) ---
    throughput_ids = all_ids * max(1, throughput_repeats)
    seq_timing_runs = []
    for i in range(timing_repeats):
        rows, meta = run_sequential_eval(
            question_ids=throughput_ids,
            tune_hybrid=False,
            run_id=run_id,
            workload="throughput",
        )
        for row in rows:
            row["repeat_index"] = i
        meta = {
            **meta,
            **hw,
            "phase": "timing_sequential",
            "timing_repeat": i,
            "workload": "throughput",
            "throughput_repeats": throughput_repeats,
            "unique_questions": len(all_ids),
        }
        all_rows.extend(rows)
        meta_rows.append(meta)
        seq_timing_runs.append(meta)
    meta_rows.append(
        {
            "run_id": run_id,
            "phase": "timing_sequential_summary",
            **summarize_timing_runs(seq_timing_runs),
            **hw,
        }
    )

    # --- Timing: Ray workers x repeats ---
    for n_workers in worker_counts:
        ray_timing_runs = []
        for i in range(timing_repeats):
            rows, meta = run_ray_eval(
                throughput_ids,
                ray_address=ray_address,
                num_workers=n_workers,
                batch_size=batch_size,
                top_k=top_k,
                tune_hybrid=False,
                run_id=run_id,
                workload="throughput",
            )
            for row in rows:
                row["repeat_index"] = i
            meta = {
                **meta,
                **hw,
                "phase": f"timing_ray_workers_{n_workers}",
                "timing_repeat": i,
                "workload": "throughput",
                "throughput_repeats": throughput_repeats,
                "unique_questions": len(all_ids),
            }
            all_rows.extend(rows)
            meta_rows.append(meta)
            ray_timing_runs.append(meta)
        meta_rows.append(
            {
                "run_id": run_id,
                "phase": f"timing_ray_workers_{n_workers}_summary",
                "workers": n_workers,
                **summarize_timing_runs(ray_timing_runs),
                **hw,
            }
        )

    quality_rows = _dedupe_quality_rows(all_rows)
    # Keep throughput rows for latency analysis, but mark quality subset clearly.
    for row in all_rows:
        row["counts_for_quality"] = row.get("workload") != "throughput"

    per_query_path, run_meta_path = write_run_outputs(run_id, all_rows, meta_rows)

    if ray.is_initialized() and not ray_address:
        ray.shutdown()

    return {
        "run_id": run_id,
        "per_query": str(per_query_path),
        "run_meta": str(run_meta_path),
        "n_rows": len(all_rows),
        "n_quality_unique_rows": len(quality_rows),
        "worker_counts": worker_counts,
        "rankings_match_sequential": True,
        "label_status": bank.label_status,
        "topology_note": (
            "remote_ray" if ray_address else "local_ray / local_sequential"
        ),
    }
