import numpy as np
import cupy as cp

def harmonic_mean_8_corners(c000, c001, c010, c011, c100, c101, c110, c111):
    corners = (c000, c001, c010, c011, c100, c101, c110, c111)
    out_shape = cp.broadcast_shapes(*(cp.shape(c) for c in corners))
    has_zero = cp.zeros(out_shape, dtype=bool)
    recip_sum = cp.zeros(out_shape, dtype=cp.float32)

    for c in corners:
        c_arr = cp.asarray(c, dtype=cp.float32)
        zero_mask = c_arr <= 0.0
        has_zero |= zero_mask
        recip_sum += cp.where(zero_mask, 0.0, 1.0 / cp.where(zero_mask, 1.0, c_arr))

    return cp.where(has_zero, 0.0, 8.0 / recip_sum)
