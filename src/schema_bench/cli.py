from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from typing import Sequence

from schema_bench.eval_sequential import run_sequential_eval, smoke_question_ids
from schema_bench.hardware import hardware_fingerprint
from schema_bench.io_parquet import write_run_outputs
from schema_bench.paths import ensure_output_dirs


def _run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def cmd_smoke(_args: argparse.Namespace) -> int:
    ensure_output_dirs()
    run_id = _run_id("smoke")
    qids = smoke_question_ids()
    rows, meta = run_sequential_eval(
        question_ids=qids,
        tune_hybrid=False,
        run_id=run_id,
        workload="smoke",
    )
    meta.update(hardware_fingerprint())
    meta["batch_size"] = len(qids)
    per_query_path, run_meta_path = write_run_outputs(run_id, rows, [meta])
    summary = {
        "run_id": run_id,
        "n_rows": len(rows),
        "question_ids": qids,
        "startup_s": meta["startup_s"],
        "warm_eval_s": meta["warm_eval_s"],
        "per_query": str(per_query_path),
        "run_meta": str(run_meta_path),
        "methods": sorted({r["method"] for r in rows}),
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from schema_bench.benchmark import run_full_benchmark

    result = run_full_benchmark(
        ray_address=args.ray_address,
        workers=args.workers,
        throughput_repeats=args.throughput_repeats,
        timing_repeats=args.timing_repeats,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from schema_bench.analyze import run_analysis

    run_analysis(args.run_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="schema-bench",
        description="Schema Retrieval Benchmark for an Analytics Copilot",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Tiny sequential smoke test")
    smoke.set_defaults(func=cmd_smoke)

    bench = sub.add_parser("benchmark", help="Full quality + timing benchmark")
    bench.add_argument("--ray-address", default=None, help="Optional Ray cluster address")
    bench.add_argument(
        "--workers",
        default="1,2,4",
        help="Comma-separated Ray worker counts (capped by CPU)",
    )
    bench.add_argument("--throughput-repeats", type=int, default=10)
    bench.add_argument("--timing-repeats", type=int, default=3)
    bench.add_argument("--batch-size", type=int, default=5)
    bench.set_defaults(func=cmd_benchmark)

    analyze = sub.add_parser("analyze", help="Run chDB SQL reports on a run directory")
    analyze.add_argument(
        "--run-dir",
        required=True,
        help="Path to outputs/runs/<run_id>",
    )
    analyze.set_defaults(func=cmd_analyze)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
