from __future__ import annotations

import os


def cap_cpu_threads(n: int = 1) -> None:
    """Avoid BLAS / torch oversubscription under Ray workers."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass
