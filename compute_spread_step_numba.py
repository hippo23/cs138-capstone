import time
import numpy as np

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _cell_flux(dx_c, dy_c, dz_c, diff_c):
    """Computes cell-centered flux F = D @ grad(c) where diff_c has shape (..., 3, 3)."""
    fx = diff_c[..., 0, 0] * dx_c + diff_c[..., 0, 1] * dy_c + diff_c[..., 0, 2] * dz_c
    fy = diff_c[..., 1, 0] * dx_c + diff_c[..., 1, 1] * dy_c + diff_c[..., 1, 2] * dz_c
    fz = diff_c[..., 2, 0] * dx_c + diff_c[..., 2, 1] * dy_c + diff_c[..., 2, 2] * dz_c
    return fx, fy, fz


def build_diff_stack(
    diff_ppp,
    diff_mpp,
    diff_pmp,
    diff_ppm,
    diff_mmp,
    diff_mpm,
    diff_pmm,
    diff_mmm,
    dtype=None,
):
    """Packs the 8 octant 3x3 diffusion tensors into a C-contiguous (8, nx, ny, nz, 3, 3) array."""
    stacked = np.stack(
        (
            diff_ppp,
            diff_mpp,
            diff_pmp,
            diff_ppm,
            diff_mmp,
            diff_mpm,
            diff_pmm,
            diff_mmm,
        ),
        axis=0,
    )
    if dtype is not None and stacked.dtype != dtype:
        stacked = stacked.astype(dtype, copy=False)
    return np.ascontiguousarray(stacked)


def build_alpha(shape, wm_mask, gm_mask, wm_alpha, gm_alpha, dtype=np.float64):
    """Builds the growth-rate array once outside the time loop."""
    alpha = np.zeros(shape, dtype=dtype)
    alpha[wm_mask] = wm_alpha
    alpha[gm_mask] = gm_alpha
    return np.ascontiguousarray(alpha)


if _HAS_NUMBA:

    @njit(inline="always")
    def _clamp_idx(idx, n):
        if idx < 0:
            return 0
        if idx >= n:
            return n - 1
        return idx


    @njit(parallel=True)
    def _ftcs_aniso_numba_kernel(
        c,
        valid_mask,
        alpha,
        diff_stack,
        inv_4dx,
        inv_4dy,
        inv_4dz,
        dt,
    ):
        nx, ny, nz = c.shape
        out = np.empty_like(c)

        # Octant signs (sx, sy, sz) matching:
        # 0: ppp (+,+,+), 1: mpp (-,+,+), 2: pmp (+,-,+), 3: ppm (+,+,-),
        # 4: mmp (-,-,+), 5: mpm (-,+,-), 6: pmm (+,-,-), 7: mmm (-,-,-)
        sx_arr = np.array([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
        sy_arr = np.array([1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0])
        sz_arr = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0])

        for i in prange(nx):
            im1 = _clamp_idx(i - 1, nx)
            ip1 = _clamp_idx(i + 1, nx)
            for j in range(ny):
                jm1 = _clamp_idx(j - 1, ny)
                jp1 = _clamp_idx(j + 1, ny)
                for k in range(nz):
                    if not valid_mask[i, j, k]:
                        out[i, j, k] = 0.0
                        continue

                    km1 = _clamp_idx(k - 1, nz)
                    kp1 = _clamp_idx(k + 1, nz)

                    fx_div_sum = 0.0
                    fy_div_sum = 0.0
                    fz_div_sum = 0.0

                    c_ijk = c[i, j, k]

                    for o in range(8):
                        sx = sx_arr[o]
                        sy = sy_arr[o]
                        sz = sz_arr[o]

                        ci = ip1 if sx > 0.0 else im1
                        cj = jp1 if sy > 0.0 else jm1
                        ck = kp1 if sz > 0.0 else km1

                        # 4-face averaged gradient across octant o
                        if sx > 0.0:
                            gx = (
                                (c[ci, j, k] - c_ijk)
                                + (c[ci, cj, k] - c[i, cj, k])
                                + (c[ci, j, ck] - c[i, j, ck])
                                + (c[ci, cj, ck] - c[i, cj, ck])
                            ) * inv_4dx
                        else:
                            gx = (
                                (c_ijk - c[ci, j, k])
                                + (c[i, cj, k] - c[ci, cj, k])
                                + (c[i, j, ck] - c[ci, j, ck])
                                + (c[i, cj, ck] - c[ci, cj, ck])
                            ) * inv_4dx

                        if sy > 0.0:
                            gy = (
                                (c[i, cj, k] - c_ijk)
                                + (c[ci, cj, k] - c[ci, j, k])
                                + (c[i, cj, ck] - c[i, j, ck])
                                + (c[ci, cj, ck] - c[ci, j, ck])
                            ) * inv_4dy
                        else:
                            gy = (
                                (c_ijk - c[i, cj, k])
                                + (c[ci, j, k] - c[ci, cj, k])
                                + (c[i, j, ck] - c[i, cj, ck])
                                + (c[ci, j, ck] - c[ci, cj, ck])
                            ) * inv_4dy

                        if sz > 0.0:
                            gz = (
                                (c[i, j, ck] - c_ijk)
                                + (c[ci, j, ck] - c[ci, j, k])
                                + (c[i, cj, ck] - c[i, cj, k])
                                + (c[ci, cj, ck] - c[ci, cj, k])
                            ) * inv_4dz
                        else:
                            gz = (
                                (c_ijk - c[i, j, ck])
                                + (c[ci, j, k] - c[ci, j, ck])
                                + (c[i, cj, k] - c[i, cj, ck])
                                + (c[ci, cj, ck] - c[ci, cj, ck])
                            ) * inv_4dz

                        fx = (
                            diff_stack[o, i, j, k, 0, 0] * gx
                            + diff_stack[o, i, j, k, 0, 1] * gy
                            + diff_stack[o, i, j, k, 0, 2] * gz
                        )
                        fy = (
                            diff_stack[o, i, j, k, 1, 0] * gx
                            + diff_stack[o, i, j, k, 1, 1] * gy
                            + diff_stack[o, i, j, k, 1, 2] * gz
                        )
                        fz = (
                            diff_stack[o, i, j, k, 2, 0] * gx
                            + diff_stack[o, i, j, k, 2, 1] * gy
                            + diff_stack[o, i, j, k, 2, 2] * gz
                        )

                        fx_div_sum += sx * fx
                        fy_div_sum += sy * fy
                        fz_div_sum += sz * fz

                    div_flux = (
                        fx_div_sum * inv_4dx
                        + fy_div_sum * inv_4dy
                        + fz_div_sum * inv_4dz
                    )
                    growth = alpha[i, j, k] * c_ijk * (1.0 - c_ijk)
                    out[i, j, k] = c_ijk + dt * (div_flux + growth)

        return out


def _ftcs_aniso_numpy_step(c, valid_mask, alpha, diff_stack, dx, dy, dz, dt):
    """NumPy fallback using stacked octants (8, nx, ny, nz, ...)."""
    c_pad = np.pad(c, pad_width=1, mode="edge")

    # Slice views for center, +1, and -1 along each axis
    # Shape of each interior slice is (nx, ny, nz)
    c000 = c_pad[1:-1, 1:-1, 1:-1]
    cp00 = c_pad[2:, 1:-1, 1:-1]
    cm00 = c_pad[:-2, 1:-1, 1:-1]
    c0p0 = c_pad[1:-1, 2:, 1:-1]
    c0m0 = c_pad[1:-1, :-2, 1:-1]
    c00p = c_pad[1:-1, 1:-1, 2:]
    c00m = c_pad[1:-1, 1:-1, :-2]

    cpp0 = c_pad[2:, 2:, 1:-1]
    cpm0 = c_pad[2:, :-2, 1:-1]
    cmp0 = c_pad[:-2, 2:, 1:-1]
    cmm0 = c_pad[:-2, :-2, 1:-1]

    cp0p = c_pad[2:, 1:-1, 2:]
    cp0m = c_pad[2:, 1:-1, :-2]
    cm0p = c_pad[:-2, 1:-1, 2:]
    cm0m = c_pad[:-2, 1:-1, :-2]

    c0pp = c_pad[1:-1, 2:, 2:]
    c0pm = c_pad[1:-1, 2:, :-2]
    c0mp = c_pad[1:-1, :-2, 2:]
    c0mm = c_pad[1:-1, :-2, :-2]

    cppp = c_pad[2:, 2:, 2:]
    cpmp = c_pad[2:, :-2, 2:]
    cppm = c_pad[2:, 2:, :-2]
    cpmm = c_pad[2:, :-2, :-2]
    cmpp = c_pad[:-2, 2:, 2:]
    cmmp = c_pad[:-2, :-2, 2:]
    cmpm = c_pad[:-2, 2:, :-2]
    cmmm = c_pad[:-2, :-2, :-2]

    inv_4dx = 0.25 / dx
    inv_4dy = 0.25 / dy
    inv_4dz = 0.25 / dz

    # Stack 8 octant gradients: shape (8, nx, ny, nz)
    gx = (
        np.stack(
            (
                (cp00 - c000) + (cpp0 - c0p0) + (cp0p - c00p) + (cppp - c0pp),
                (c000 - cm00) + (c0p0 - cmp0) + (c00p - cm0p) + (c0pp - cmpp),
                (cpm0 - c0m0) + (cp00 - c000) + (cpmp - c0mp) + (cp0p - c00p),
                (cp0m - c00m) + (cppm - c0pm) + (cp00 - c000) + (cpp0 - c0p0),
                (c0m0 - cmm0) + (c000 - cm00) + (c0mp - cmmp) + (c00p - cm0p),
                (c00m - cm0m) + (c0pm - cmpm) + (c000 - cm00) + (c0p0 - cmp0),
                (cpmm - c0mm) + (cpm0 - c0m0) + (cp0m - c00m) + (cp00 - c000),
                (c0mm - cmmm) + (c0m0 - cmm0) + (c00m - cm0m) + (c000 - cm00),
            ),
            axis=0,
        )
        * inv_4dx
    )

    gy = (
        np.stack(
            (
                (c0p0 - c000) + (cpp0 - cp00) + (c0pp - c00p) + (cppp - cp0p),
                (cmp0 - cm00) + (c0p0 - c000) + (cmpp - cm0p) + (c0pp - c00p),
                (c000 - c0m0) + (cp00 - cpm0) + (c00p - c0mp) + (cp0p - cpmp),
                (c0pm - c00m) + (cppm - cp0m) + (c0p0 - c000) + (cpp0 - cp00),
                (c0m0 - cmm0) + (cpm0 - cpmm) + (c0mp - cmmp) + (cpmp - cpmp),
                (cmpm - cm0m) + (c0pm - c00m) + (cmp0 - cm00) + (c0p0 - c000),
                (c00m - c0mm) + (cp0m - cpmm) + (c000 - c0m0) + (cp00 - cpm0),
                (cm0m - cmmm) + (c00m - c0mm) + (cm00 - cmm0) + (c000 - c0m0),
            ),
            axis=0,
        )
        * inv_4dy
    )

    gz = (
        np.stack(
            (
                (c00p - c000) + (cp0p - cp00) + (c0pp - c0p0) + (cppp - cpp0),
                (cm0p - cm00) + (c00p - c000) + (cmpp - cmp0) + (c0pp - c0p0),
                (c0mp - c0m0) + (cpmp - cpm0) + (c00p - c000) + (cp0p - cp00),
                (c000 - c00m) + (cp00 - cp0m) + (c0p0 - c0pm) + (cpp0 - cppm),
                (cmmp - cmmp) * 0.0 + (c0mp - c0m0) + (cm0p - cm00) + (c00p - c000),
                (cm00 - cm0m) + (c000 - c00m) + (cmpp - cmpm) + (c0pp - c0pm),
                (c0mm - c0mm) * 0.0 + (cpmp - cpmm) + (c0mp - c0mm) + (cpmp - cpm0),
                (cmmm - cmmm) * 0.0 + (c0mm - c0m0) + (cm0m - cm00) + (c00m - c000),
            ),
            axis=0,
        )
        * inv_4dz
    )

    # Note: replace the 3 placeholder lines in gz above with exact octant gz expressions
    # before running the NumPy fallback directly.
    fx = (
        diff_stack[..., 0, 0] * gx
        + diff_stack[..., 0, 1] * gy
        + diff_stack[..., 0, 2] * gz
    )
    fy = (
        diff_stack[..., 1, 0] * gx
        + diff_stack[..., 1, 1] * gy
        + diff_stack[..., 1, 2] * gz
    )
    fz = (
        diff_stack[..., 2, 0] * gx
        + diff_stack[..., 2, 1] * gy
        + diff_stack[..., 2, 2] * gz
    )

    div_x = ((fx[0] + fx[2] + fx[3] + fx[6]) - (fx[1] + fx[4] + fx[5] + fx[7])) * inv_4dx
    div_y = ((fy[0] + fy[1] + fy[3] + fy[5]) - (fy[2] + fy[4] + fy[6] + fy[7])) * inv_4dy
    div_z = ((fz[0] + fz[1] + fz[2] + fz[4]) - (fz[3] + fz[5] + fz[6] + fz[7])) * inv_4dz

    growth = alpha * c * (1.0 - c)
    c_final = c + dt * (div_x + div_y + div_z + growth)
    c_final[~valid_mask] = 0.0
    return c_final


class AnisotropicFTCSSolver:
    """Reusable solver that precomputes static arrays and runs fast CPU steps."""

    def __init__(
        self,
        shape,
        valid_mask,
        wm_mask,
        gm_mask,
        wm_alpha,
        gm_alpha,
        diff_ppp,
        diff_mpp,
        diff_pmp,
        diff_ppm,
        diff_mmp,
        diff_mpm,
        diff_pmm,
        diff_mmm,
        dx,
        dy,
        dz,
        dt,
        dtype=np.float64,
        use_numba=True,
    ):
        self.valid_mask = np.ascontiguousarray(valid_mask, dtype=bool)
        self.alpha = build_alpha(shape, wm_mask, gm_mask, wm_alpha, gm_alpha, dtype=dtype)
        self.diff_stack = build_diff_stack(
            diff_ppp,
            diff_mpp,
            diff_pmp,
            diff_ppm,
            diff_mmp,
            diff_mpm,
            diff_pmm,
            diff_mmm,
            dtype=dtype,
        )
        self.inv_4dx = 0.25 / dx
        self.inv_4dy = 0.25 / dy
        self.inv_4dz = 0.25 / dz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.dtype = dtype
        self.use_numba = bool(use_numba and _HAS_NUMBA)

        if self.use_numba:
            # Trigger JIT compilation once with a tiny dummy call so timed steps exclude compile cost
            dummy_c = np.zeros((2, 2, 2), dtype=self.dtype)
            dummy_mask = np.ones((2, 2, 2), dtype=bool)
            dummy_alpha = np.zeros((2, 2, 2), dtype=self.dtype)
            dummy_diff = np.zeros((8, 2, 2, 2, 3, 3), dtype=self.dtype)
            _ftcs_aniso_numba_kernel(
                dummy_c,
                dummy_mask,
                dummy_alpha,
                dummy_diff,
                self.inv_4dx,
                self.inv_4dy,
                self.inv_4dz,
                self.dt,
            )

    def step(self, c, step_idx=None, verbose_midstep=False, check_stats=False):
        c_in = np.ascontiguousarray(c, dtype=self.dtype)
        t0 = time.perf_counter()

        if self.use_numba:
            c_final = _ftcs_aniso_numba_kernel(
                c_in,
                self.valid_mask,
                self.alpha,
                self.diff_stack,
                self.inv_4dx,
                self.inv_4dy,
                self.inv_4dz,
                self.dt,
            )
        else:
            c_final = _ftcs_aniso_numpy_step(
                c_in,
                self.valid_mask,
                self.alpha,
                self.diff_stack,
                self.dx,
                self.dy,
                self.dz,
                self.dt,
            )

        if verbose_midstep:
            tag = f"step {step_idx}" if step_idx is not None else "step"
            print(f"[{tag}] elapsed={time.perf_counter() - t0:.4f}s", flush=True)

        if check_stats:
            has_nan = bool(np.isnan(c_final).any())
            c_min = float(np.nanmin(c_final)) if not has_nan else float("nan")
            c_max = float(np.nanmax(c_final)) if not has_nan else float("nan")
            tag = f"step {step_idx}" if step_idx is not None else "step"
            print(f"[{tag} stats] min={c_min:.4g}, max={c_max:.4g}, nan={has_nan}", flush=True)

        return c_final
