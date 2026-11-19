import os

import pytest

from schema_bench.eval_ray import assert_same_rankings, run_ray_eval
from schema_bench.eval_sequential import run_sequential_eval, smoke_question_ids


@pytest.mark.slow
def test_sequential_and_ray_rankings_match():
    # Skip if Ray/torch stack unavailable in the environment.
    pytest.importorskip("ray")
    pytest.importorskip("sentence_transformers")

    qids = smoke_question_ids()
    seq_rows, _ = run_sequential_eval(
        question_ids=qids,
        tune_hybrid=False,
        run_id="test_seq",
        workload="quality",
    )
    # Keep Ray tiny for CI/laptop.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray_rows, _ = run_ray_eval(
        qids,
        ray_address=None,
        num_workers=1,
        batch_size=2,
        tune_hybrid=False,
        run_id="test_ray",
        workload="quality",
    )
    assert_same_rankings(seq_rows, ray_rows)

    import ray

    if ray.is_initialized():
        ray.shutdown()
