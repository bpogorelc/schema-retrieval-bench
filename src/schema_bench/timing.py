from __future__ import annotations

import os
from typing import Any


def parse_worker_list(spec: str) -> list[int]:
    workers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        workers.append(int(part))
    return workers


def cap_workers(requested: list[int], cpu_count: int | None = None) -> list[int]:
    cpus = cpu_count or os.cpu_count() or 1
    capped = sorted({max(1, min(w, cpus)) for w in requested})
    return capped


def summarize_timing_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    startups = [r["startup_s"] for r in runs]
    warms = [r["warm_eval_s"] for r in runs]
    return {
        "n_repeats": len(runs),
        "startup_s_mean": sum(startups) / len(startups),
        "warm_eval_s_mean": sum(warms) / len(warms),
        "startup_s_values": startups,
        "warm_eval_s_values": warms,
    }
