from __future__ import annotations

import os
import time
from typing import Any

from schema_bench.threading_utils import cap_cpu_threads

# Per-worker process cache (reused across tasks in the same worker).
_WORKER_STATE: dict[str, Any] = {}


def _get_worker_state(tune_hybrid: bool = True) -> dict[str, Any]:
    global _WORKER_STATE
    if _WORKER_STATE.get("ready"):
        return _WORKER_STATE
    cap_cpu_threads(1)
    from schema_bench.catalog import load_catalog
    from schema_bench.eval_sequential import build_retrievers
    from schema_bench.labels import load_questions

    docs = load_catalog()
    bank = load_questions()
    retrievers = build_retrievers(
        docs,
        dev_questions=bank.by_split("dev"),
        tune_hybrid=tune_hybrid,
    )
    node_id = "local"
    try:
        import ray

        node_id = ray.get_runtime_context().get_node_id()
    except Exception:
        pass
    _WORKER_STATE = {
        "ready": True,
        "retrievers": retrievers,
        "bank": bank,
        "node_id": node_id,
    }
    return _WORKER_STATE


def evaluate_batch_local(
    question_ids: list[str],
    top_k: int = 5,
    workload: str = "quality",
    topology: str = "local_ray",
    run_id: str = "",
    repeat_index: int = 0,
    tune_hybrid: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    from schema_bench.eval_sequential import evaluate_questions

    state = _get_worker_state(tune_hybrid=tune_hybrid)
    questions = state["bank"].by_ids(question_ids)
    rows = evaluate_questions(
        questions,
        state["retrievers"],
        top_k=top_k,
        workload=workload,
        topology=topology,
        run_id=run_id,
        repeat_index=repeat_index,
        worker_id=str(id(state)),
        node_id=state["node_id"],
    )
    return rows, state["node_id"]


def _chunked(items: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def run_ray_eval(
    question_ids: list[str],
    *,
    ray_address: str | None = None,
    num_workers: int = 2,
    batch_size: int = 5,
    top_k: int = 5,
    tune_hybrid: bool = True,
    run_id: str = "ray",
    workload: str = "quality",
    timing_warmup: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import ray

    topology = "remote_ray" if ray_address else "local_ray"
    startup_t0 = time.perf_counter()
    if not ray.is_initialized():
        os.environ.setdefault("RAY_AUTH_MODE", "disabled")
        if ray_address:
            ray.init(address=ray_address, ignore_reinit_error=True)
        else:
            ray.init(
                num_cpus=num_workers,
                ignore_reinit_error=True,
                include_dashboard=False,
            )

    @ray.remote
    def remote_batch(
        qids: list[str],
        top_k: int,
        workload: str,
        topology: str,
        run_id: str,
        repeat_index: int,
        tune_hybrid: bool,
    ) -> tuple[list[dict[str, Any]], str]:
        return evaluate_batch_local(
            qids,
            top_k=top_k,
            workload=workload,
            topology=topology,
            run_id=run_id,
            repeat_index=repeat_index,
            tune_hybrid=tune_hybrid,
        )

    batches = _chunked(question_ids, batch_size)
    if timing_warmup and batches:
        # Force workers to load model/index before warm timing.
        warm_refs = [
            remote_batch.remote(
                batches[i % len(batches)],
                top_k,
                workload,
                topology,
                run_id,
                0,
                tune_hybrid,
            )
            for i in range(min(num_workers, len(batches)))
        ]
        ray.get(warm_refs)

    startup_s = time.perf_counter() - startup_t0

    warm_t0 = time.perf_counter()
    refs = [
        remote_batch.remote(
            batch,
            top_k,
            workload,
            topology,
            run_id,
            0,
            tune_hybrid,
        )
        for batch in batches
    ]
    results = ray.get(refs)
    warm_s = time.perf_counter() - warm_t0

    rows: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    for batch_rows, node_id in results:
        rows.extend(batch_rows)
        node_ids.add(node_id)

    meta = {
        "run_id": run_id,
        "engine": "ray",
        "topology": topology,
        "n_questions": len(question_ids),
        "startup_s": startup_s,
        "warm_eval_s": warm_s,
        "workers": num_workers,
        "batch_size": batch_size,
        "node_ids": sorted(node_ids),
        "ray_address": ray_address or "local",
    }
    return rows, meta


def rankings_by_question_method(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row["question_id"], row["method"])
        out[key] = list(row["ranked_ids"])
    return out


def assert_same_rankings(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> None:
    left_map = rankings_by_question_method(left)
    right_map = rankings_by_question_method(right)
    if left_map.keys() != right_map.keys():
        raise AssertionError(
            f"Ranking key mismatch: {left_map.keys() ^ right_map.keys()}"
        )
    for key in left_map:
        if left_map[key] != right_map[key]:
            raise AssertionError(
                f"Ranking mismatch for {key}: {left_map[key]} != {right_map[key]}"
            )
