from __future__ import annotations

from pathlib import Path

from schema_bench.paths import SQL_DIR


def _load_sql(name: str, per_query: Path) -> str:
    raw = (SQL_DIR / name).read_text(encoding="utf-8")
    # chDB on Windows/WSL needs forward slashes in file() paths.
    return raw.replace("{per_query}", per_query.resolve().as_posix())


def run_analysis(run_dir: str | Path) -> None:
    import chdb

    run_path = Path(run_dir)
    per_query = run_path / "per_query.parquet"
    if not per_query.exists():
        raise FileNotFoundError(f"Missing {per_query}")

    print("=== Quality / latency comparison ===")
    q1 = _load_sql("quality_latency_comparison.sql", per_query)
    print(chdb.query(q1, "Pretty"))

    print("\n=== Top 5 hybrid losses vs BM25 ===")
    q2 = _load_sql("hybrid_vs_bm25_losses.sql", per_query)
    print(chdb.query(q2, "Pretty"))
