import numpy as np

def harmonic_mean_8_corners(c000, c001, c010, c011, c100, c101, c110, c111):
    corners = (c000, c001, c010, c011, c100, c101, c110, c111)
    out_shape = np.broadcast_shapes(*(np.shape(c) for c in corners))
    has_zero = np.zeros(out_shape, dtype=bool)
    recip_sum = np.zeros(out_shape, dtype=np.float64)

    for c in corners:
        c_arr = np.asarray(c, dtype=np.float64)
        zero_mask = c_arr <= 0.0
        has_zero |= zero_mask
        recip_sum += np.where(zero_mask, 0.0, 1.0 / np.where(zero_mask, 1.0, c_arr))

    return np.where(has_zero, 0.0, 8.0 / recip_sum)
