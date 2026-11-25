import time
import numpy as np

try:
    import cupy as _cp
except ImportError:
    _cp = None


def _get_array_module(arr):
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp
    return np


def _checkpoint(enabled, tag, t_step_start, t_stage_start):
    if not enabled:
        return time.perf_counter()
    now = time.perf_counter()
    print(
        f"  [{tag}] +{now - t_stage_start:.3f}s (elapsed: {now - t_step_start:.3f}s)",
        flush=True,
    )
    return time.perf_counter()
