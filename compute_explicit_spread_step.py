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


def _tensor_vec_mul(diff_c, gx, gy, gz, xp):
    fx = diff_c[..., 0, 0] * gx + diff_c[..., 0, 1] * gy + diff_c[..., 0, 2] * gz
    fy = diff_c[..., 1, 0] * gx + diff_c[..., 1, 1] * gy + diff_c[..., 1, 2] * gz
    fz = diff_c[..., 2, 0] * gx + diff_c[..., 2, 1] * gy + diff_c[..., 2, 2] * gz
    return fx, fy, fz


def compute_explicit_ftcs_step_gpu(
    c,
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
    verbose_midstep=True,
    check_stats=False,
    step_idx=None,
):
    xp = _get_array_module(c)
    step_tag = f"step {step_idx}" if step_idx is not None else "step"
    if verbose_midstep:
        print(f"--- {step_tag} started (shape={c.shape}, dtype={c.dtype}, backend={xp.__name__}) ---", flush=True)

    t_step_start = time.perf_counter()
    t0 = t_step_start

    c_g = xp.asarray(c)
    valid_g = xp.asarray(valid_mask, dtype=bool)
    wm_g = xp.asarray(wm_mask, dtype=bool)
    gm_g = xp.asarray(gm_mask, dtype=bool)

    diff_ppp_g = xp.asarray(diff_ppp)
    diff_mpp_g = xp.asarray(diff_mpp)
    diff_pmp_g = xp.asarray(diff_pmp)
    diff_ppm_g = xp.asarray(diff_ppm)
    diff_mmp_g = xp.asarray(diff_mmp)
    diff_mpm_g = xp.asarray(diff_mpm)
    diff_pmm_g = xp.asarray(diff_pmm)
    diff_mmm_g = xp.asarray(diff_mmm)

    alpha = xp.zeros(c_g.shape, dtype=xp.float32)
    alpha[wm_g] = wm_alpha
    alpha[gm_g] = gm_alpha
    c_pad = xp.pad(c_g, pad_width=1, mode="edge")
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 1/5 setup+pad", t_step_start, t0)

    inv_4dx = 0.25 / dx
    inv_4dy = 0.25 / dy
    inv_4dz = 0.25 / dz

    c_mmm = c_pad[:-2, :-2, :-2]
    c_mm0 = c_pad[:-2, :-2, 1:-1]
    c_mmp = c_pad[:-2, :-2, 2:]
    c_m0m = c_pad[:-2, 1:-1, :-2]
    c_m00 = c_pad[:-2, 1:-1, 1:-1]
    c_m0p = c_pad[:-2, 1:-1, 2:]
    c_mpm = c_pad[:-2, 2:, :-2]
    c_mp0 = c_pad[:-2, 2:, 1:-1]
    c_mpp = c_pad[:-2, 2:, 2:]

    c_0mm = c_pad[1:-1, :-2, :-2]
    c_0m0 = c_pad[1:-1, :-2, 1:-1]
    c_0mp = c_pad[1:-1, :-2, 2:]
    c_00m = c_pad[1:-1, 1:-1, :-2]
    c_000 = c_pad[1:-1, 1:-1, 1:-1]
    c_00p = c_pad[1:-1, 1:-1, 2:]
    c_0pm = c_pad[1:-1, 2:, :-2]
    c_0p0 = c_pad[1:-1, 2:, 1:-1]
    c_0pp = c_pad[1:-1, 2:, 2:]

    c_pmm = c_pad[2:, :-2, :-2]
    c_pm0 = c_pad[2:, :-2, 1:-1]
    c_pmp = c_pad[2:, :-2, 2:]
    c_p0m = c_pad[2:, 1:-1, :-2]
    c_p00 = c_pad[2:, 1:-1, 1:-1]
    c_p0p = c_pad[2:, 1:-1, 2:]
    c_ppm = c_pad[2:, 2:, :-2]
    c_pp0 = c_pad[2:, 2:, 1:-1]
    c_ppp = c_pad[2:, 2:, 2:]

    # NOTE: compute the gradients (this is for the inner gradient of c, if you look at the equation (9) in the documentation of
    # the methods).
    dx_ppp = ((c_p00 - c_000) + (c_pp0 - c_0p0) + (c_p0p - c_00p) + (c_ppp - c_0pp)) * inv_4dx
    dy_ppp = ((c_0p0 - c_000) + (c_pp0 - c_p00) + (c_0pp - c_00p) + (c_ppp - c_p0p)) * inv_4dy
    dz_ppp = ((c_00p - c_000) + (c_p0p - c_p00) + (c_0pp - c_0p0) + (c_ppp - c_pp0)) * inv_4dz

    dx_mpp = ((c_000 - c_m00) + (c_0p0 - c_mp0) + (c_00p - c_m0p) + (c_0pp - c_mpp)) * inv_4dx
    dy_mpp = ((c_mp0 - c_m00) + (c_0p0 - c_000) + (c_mpp - c_m0p) + (c_0pp - c_00p)) * inv_4dy
    dz_mpp = ((c_m0p - c_m00) + (c_00p - c_000) + (c_mpp - c_mp0) + (c_0pp - c_0p0)) * inv_4dz

    dx_pmp = ((c_pm0 - c_0m0) + (c_p00 - c_000) + (c_pmp - c_0mp) + (c_p0p - c_00p)) * inv_4dx
    dy_pmp = ((c_000 - c_0m0) + (c_p00 - c_pm0) + (c_00p - c_0mp) + (c_p0p - c_pmp)) * inv_4dy
    dz_pmp = ((c_0mp - c_0m0) + (c_pmp - c_pm0) + (c_00p - c_000) + (c_p0p - c_p00)) * inv_4dz

    dx_ppm = ((c_p0m - c_00m) + (c_ppm - c_0pm) + (c_p00 - c_000) + (c_pp0 - c_0p0)) * inv_4dx
    dy_ppm = ((c_0pm - c_00m) + (c_ppm - c_p0m) + (c_0p0 - c_000) + (c_pp0 - c_p00)) * inv_4dy
    dz_ppm = ((c_000 - c_00m) + (c_p00 - c_p0m) + (c_0p0 - c_0pm) + (c_pp0 - c_ppm)) * inv_4dz

    dx_mmp = ((c_0m0 - c_mm0) + (c_000 - c_m00) + (c_0mp - c_mmp) + (c_00p - c_m0p)) * inv_4dx
    dy_mmp = ((c_m00 - c_mm0) + (c_000 - c_0m0) + (c_m0p - c_mmp) + (c_00p - c_0mp)) * inv_4dy
    dz_mmp = ((c_mmp - c_mm0) + (c_0mp - c_0m0) + (c_m0p - c_m00) + (c_00p - c_000)) * inv_4dz

    dx_mpm = ((c_00m - c_m0m) + (c_0pm - c_mpm) + (c_000 - c_m00) + (c_0p0 - c_mp0)) * inv_4dx
    dy_mpm = ((c_mpm - c_m0m) + (c_0pm - c_00m) + (c_mp0 - c_m00) + (c_0p0 - c_000)) * inv_4dy
    dz_mpm = ((c_m00 - c_m0m) + (c_000 - c_00m) + (c_mp0 - c_mpm) + (c_0p0 - c_0pm)) * inv_4dz

    dx_pmm = ((c_pmm - c_0mm) + (c_p0m - c_00m) + (c_pm0 - c_0m0) + (c_p00 - c_000)) * inv_4dx
    dy_pmm = ((c_00m - c_0mm) + (c_p0m - c_pmm) + (c_000 - c_0m0) + (c_p00 - c_pm0)) * inv_4dy
    dz_pmm = ((c_0m0 - c_0mm) + (c_pm0 - c_pmm) + (c_000 - c_00m) + (c_p00 - c_p0m)) * inv_4dz

    dx_mmm = ((c_0mm - c_mmm) + (c_00m - c_m0m) + (c_0m0 - c_mm0) + (c_000 - c_m00)) * inv_4dx
    dy_mmm = ((c_m0m - c_mmm) + (c_00m - c_0mm) + (c_m00 - c_mm0) + (c_000 - c_0m0)) * inv_4dy
    dz_mmm = ((c_mm0 - c_mmm) + (c_0m0 - c_0mm) + (c_m00 - c_m0m) + (c_000 - c_00m)) * inv_4dz
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 2/5 8-octant gradients", t_step_start, t0)

    # NOTE: This is where we combine the gradients with the diffusion matrix, referring to the numerator of
    # eq. (8) in the documentation.
    fx_ppp, fy_ppp, fz_ppp = _tensor_vec_mul(diff_ppp_g, dx_ppp, dy_ppp, dz_ppp, xp)
    fx_mpp, fy_mpp, fz_mpp = _tensor_vec_mul(diff_mpp_g, dx_mpp, dy_mpp, dz_mpp, xp)
    fx_pmp, fy_pmp, fz_pmp = _tensor_vec_mul(diff_pmp_g, dx_pmp, dy_pmp, dz_pmp, xp)
    fx_ppm, fy_ppm, fz_ppm = _tensor_vec_mul(diff_ppm_g, dx_ppm, dy_ppm, dz_ppm, xp)
    fx_mmp, fy_mmp, fz_mmp = _tensor_vec_mul(diff_mmp_g, dx_mmp, dy_mmp, dz_mmp, xp)
    fx_mpm, fy_mpm, fz_mpm = _tensor_vec_mul(diff_mpm_g, dx_mpm, dy_mpm, dz_mpm, xp)
    fx_pmm, fy_pmm, fz_pmm = _tensor_vec_mul(diff_pmm_g, dx_pmm, dy_pmm, dz_pmm, xp)
    fx_mmm, fy_mmm, fz_mmm = _tensor_vec_mul(diff_mmm_g, dx_mmm, dy_mmm, dz_mmm, xp)
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 3/5 fluxes", t_step_start, t0)

    # NOTE: This is the full calculation of the divergence, referring to the full equation in eq. (8) of
    # the documentation.
    div_x = (
        (fx_ppp + fx_pmp + fx_ppm + fx_pmm)
        - (fx_mpp + fx_mmp + fx_mpm + fx_mmm)
    ) / (4.0 * dx)

    div_y = (
        (fy_ppp + fy_mpp + fy_ppm + fy_mpm)
        - (fy_pmp + fy_mmp + fy_pmm + fy_mmm)
    ) / (4.0 * dy)

    div_z = (
        (fz_ppp + fz_mpp + fz_pmp + fz_mmp)
        - (fz_ppm + fz_mpm + fz_pmm + fz_mmm)
    ) / (4.0 * dz)

    div_flux = div_x + div_y + div_z
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 4/5 divergence", t_step_start, t0)

    growth = alpha * c_g * (1.0 - c_g)
    c_final = c_g + dt * (div_flux + growth)
    c_final *= valid_g
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 5/5 growth + update", t_step_start, t0)

    if check_stats:
        has_nan = bool(xp.isnan(c_final).any())
        c_min = float(xp.nanmin(c_final)) if not has_nan else float("nan")
        c_max = float(xp.nanmax(c_final)) if not has_nan else float("nan")
        print(f"  [{step_tag} stats] min={c_min:.4g}, max={c_max:.4g}, nan={has_nan}", flush=True)

    return c_final
