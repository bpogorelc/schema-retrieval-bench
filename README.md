# Schema Retrieval Benchmark for an Analytics Copilot

CPU-only Python benchmark that answers two product questions:

1. **Which retriever to ship** — BM25 vs dense (`sentence-transformers/all-MiniLM-L6-v2`) vs hybrid RRF.
2. **When Ray is worth it** — sequential vs Ray Core at 1/2/4 workers for offline evaluation.

No cloud account, API keys, frontend, or LLM calls. Designed to run on a Windows laptop through WSL2.

## Setup (WSL2)

```bash
# From the repo root inside WSL
bash scripts/setup_wsl.sh
source ~/.venvs/schema-bench/bin/activate
```

Notes:

- The setup script installs **CPU-only** PyTorch first (`https://download.pytorch.org/whl/cpu`). Installing plain `torch` from PyPI can pull multi-GB CUDA wheels and fill the disk.
- The venv lives on the Linux filesystem at `~/.venvs/schema-bench` (not a repo `.venv` symlink — Windows Git errors on WSL symlinks).
- `chDB` requires Linux wheels — use WSL, not native Windows Python.

### First model download

The first dense retrieval run downloads `sentence-transformers/all-MiniLM-L6-v2` (~88MB total Hugging Face cache observed here) to:

`~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`

Document embeddings are then cached under `outputs/cache/`.

## Reproduction commands

```bash
source ~/.venvs/schema-bench/bin/activate

# Smoke: 3 questions, sequential BM25/dense/hybrid
python -m schema_bench.cli smoke

# Full benchmark: quality + Ray timing (1/2/4 workers × 3 repeats) + throughput replay
python -m schema_bench.cli benchmark --workers 1,2,4 --timing-repeats 3 --throughput-repeats 10 --batch-size 5

# Optional remote Ray cluster
python -m schema_bench.cli benchmark --ray-address HOST:PORT

# chDB SQL reports on a run directory
python -m schema_bench.cli analyze --run-dir outputs/runs/<run_id>

# Tests
pytest -q
```

Draft relevance labels live in [`data/questions.yaml`](data/questions.yaml) (`label_status: draft`). Schema catalog: [`data/schema_catalog.json`](data/schema_catalog.json) (52 entries).

## Measured results

Hardware (WSL2): `DESKTOP-AD82P60`, Linux 4.4.0-19041-Microsoft, **8 logical CPUs**, Python 3.10.12.

Run id: `benchmark_20261119T222606Z_a1361f0f`  
Topology: `local_sequential` / `local_ray`  
Batch size: 5  
Throughput workload: same 30 questions replayed ×10 (`workload: throughput`); unique questions counted once for quality.

### Retrieval quality (held-out, sequential)

| Method | nDCG@5 | Recall@5 | p50 latency (ms) | p95 latency (ms) |
|--------|--------|----------|------------------|------------------|
| BM25   | 0.682  | 0.705    | 0.14             | 0.21             |
| Dense  | 0.815  | 0.871    | 16.1             | 28.0             |
| Hybrid | 0.786  | 0.811    | 16.1             | 20.1             |

Dev-split hybrid beat dense slightly after RRF tuning on **dev only** (hybrid nDCG@5 0.866 vs dense 0.857). Held-out favors dense — treat labels as draft.

### Sequential vs Ray (warm evaluation, throughput replay)

Means over 3 repeats. Startup recorded separately from warm eval.

| Config | Workers | Batch size | Startup mean (s) | Warm eval mean (s) | Topology |
|--------|---------|------------|------------------|--------------------|----------|
| Sequential | 1 | 300 queries | 0.80 | 10.50 | local_sequential |
| Ray | 1 | 5 | 0.17 | 8.35 | local_ray |
| Ray | 2 | 5 | 0.18 | 4.47 | local_ray |
| Ray | 4 | 5 | 0.29 | 4.38 | local_ray |

Ray worker node id observed: `044075ddf2d7fc2d3fc5fbea45202a69a5cc0054ae05379af5dba583` (single local node). Sequential and Ray rankings matched under deterministic tie-breaking (`rankings_match_sequential: true`).

## One retrieval failure

**q16** (dev): *“What's the typical basket size in dollars?”*  
Expected: `metric_aov` (grade 3), `metric_median_order_value` (grade 2).

| Method | nDCG@5 | Top-5 |
|--------|--------|-------|
| BM25 | 0.000 | converted / CAC / cancellation / category / repeat_purchase |
| Dense | 0.373 | **median_order_value**, quantity, units_sold, order_items, conversion |
| Hybrid | 0.160 | quantity, converted, **median_order_value**, category, conversion |

None ranked `average_order_value` in the top-5. “Basket size” is a paraphrase the lexical index misses and the MiniLM model only partially maps (median over AOV).

## Recommendation

- **Ship dense** for the analytics-copilot schema retriever on this catalog: best held-out nDCG@5 / Recall@5 (0.815 / 0.871). Hybrid helps a bit on exact-name/dev cases but lost on held-out after RRF tuning on draft labels. BM25 remains useful as a cheap baseline and for exact identifiers (`orders.order_date`), not as the primary ship candidate.
- **Use Ray for offline evaluation starting at 2 workers** on this 8-vCPU laptop: warm throughput dropped from ~10.5s sequential to ~4.5s (~2.3×). Four workers did not meaningfully beat two on this small replay workload, while each worker still loads the embedding model — so Ray pays off for larger question sets / repeated offline sweeps, not for single interactive queries (dense p50 ≈ 16ms already).

## Project layout

```text
data/                  schema catalog + draft graded questions
sql/                   chDB quality/latency and hybrid-vs-BM25 reports
src/schema_bench/      retrievers, sequential + Ray eval, CLI
tests/                 metrics, splits, determinism, seq-vs-Ray rankings
outputs/               gitignored Parquet runs + embedding cache
scripts/setup_wsl.sh   WSL bootstrap (CPU torch + venv on Linux FS)
```

## Implementation notes

- Retrievers: BM25 (`rank_bm25`), dense cosine, hybrid RRF (`k`/weights tuned on `dev` only).
- Ties broken by `(-score, doc_id)`.
- Intent families stay entirely in `dev` or `held_out` (no paraphrase leakage).
- Thread caps (`OMP`/`MKL`/`TORCH`=1) avoid oversubscription under Ray.
