from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from schema_bench.paths import RUNS_DIR, ensure_output_dirs


def write_run_outputs(
    run_id: str,
    per_query_rows: list[dict[str, Any]],
    run_meta_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    ensure_output_dirs()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = run_dir / "per_query.parquet"
    run_meta_path = run_dir / "run_meta.parquet"
    pd.DataFrame(per_query_rows).to_parquet(per_query_path, index=False)
    pd.DataFrame(run_meta_rows).to_parquet(run_meta_path, index=False)
    return per_query_path, run_meta_path
