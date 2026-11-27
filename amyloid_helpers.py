import numpy as np
import cupy as _cp
import time

def harmonic_mean_8_corners(c000, c001, c010, c011, c100, c101, c110, c111):
    corners = (c000, c001, c010, c011, c100, c101, c110, c111)
    out_shape = _cp.broadcast_shapes(*(_cp.shape(c) for c in corners))
    has_zero = _cp.zeros(out_shape, dtype=bool)
    recip_sum = _cp.zeros(out_shape, dtype=_cp.float32)

    for c in corners:
        c_arr = _cp.asarray(c, dtype=_cp.float32)
        zero_mask = c_arr <= 0.0
        has_zero |= zero_mask
        recip_sum += _cp.where(zero_mask, 0.0, 1.0 / _cp.where(zero_mask, 1.0, c_arr))

    return _cp.where(has_zero, 0.0, 8.0 / recip_sum)


def get_array_module(arr):
    if _cp is not None and isinstance(arr, _cp.ndarray):
        return _cp
    return np


def checkpoint(enabled, tag, t_step_start, t_stage_start):
    if not enabled:
        return time.perf_counter()
    now = time.perf_counter()
    print(
        f"  [{tag}] +{now - t_stage_start:.3f}s (elapsed: {now - t_step_start:.3f}s)",
        flush=True,
    )
    return time.perf_counter()
