from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SQL_DIR = REPO_ROOT / "sql"
OUTPUTS_DIR = REPO_ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "cache"
RUNS_DIR = OUTPUTS_DIR / "runs"

CATALOG_PATH = DATA_DIR / "schema_catalog.json"
QUESTIONS_PATH = DATA_DIR / "questions.yaml"


def ensure_output_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
