import time
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


def _cell_flux(dx_c, dy_c, dz_c, diff_c):
    """Computes cell-centered flux F = D @ grad(c) where diff_c has shape (..., 3, 3)."""
    fx = diff_c[..., 0, 0] * dx_c + diff_c[..., 0, 1] * dy_c + diff_c[..., 0, 2] * dz_c
    fy = diff_c[..., 1, 0] * dx_c + diff_c[..., 1, 1] * dy_c + diff_c[..., 1, 2] * dz_c
    fz = diff_c[..., 2, 0] * dx_c + diff_c[..., 2, 1] * dy_c + diff_c[..., 2, 2] * dz_c
    return fx, fy, fz


def _checkpoint(enabled, tag, t_step_start, t_stage_start):
    if not enabled:
        return time.perf_counter()
    now = time.perf_counter()
    print(
        f"  [{tag}] +{now - t_stage_start:.3f}s (elapsed: {now - t_step_start:.3f}s)",
        flush=True,
    )
    return time.perf_counter()


def compute_explicit_ftcs_step(
    c,
    valid_mask,
    fiber,
    wm_mask,
    gm_mask,
    wm_ext,
    wm_axn,
    wm_alpha,
    gm_ext,
    gm_axn,
    gm_alpha,
    dx,
    dy,
    dz,
    dt,
    verbose_midstep=True,
    check_stats=False,
    step_idx=None,
):
    step_tag = f"step {step_idx}" if step_idx is not None else "step"
    if verbose_midstep:
        print(f"--- {step_tag} started (shape={c.shape}, dtype={c.dtype}) ---", flush=True)

    t_step_start = time.perf_counter()
    t0 = t_step_start

    alpha = np.zeros(c.shape, dtype=np.float64)
    alpha[wm_mask] = wm_alpha
    alpha[gm_mask] = gm_alpha
    c_pad = np.pad(c, pad_width=1, mode="edge")
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 1/6 setup+pad", t_step_start, t0)

    # 1. (i + 1/2, j + 1/2, k + 1/2)
    dx_ppp = (
        (c_pad[2:, 1:-1, 1:-1] - c_pad[1:-1, 1:-1, 1:-1])
        + (c_pad[2:, 2:, 1:-1] - c_pad[1:-1, 2:, 1:-1])
        + (c_pad[2:, 1:-1, 2:] - c_pad[1:-1, 1:-1, 2:])
        + (c_pad[2:, 2:, 2:] - c_pad[1:-1, 2:, 2:])
    ) / (4 * dx)
    dy_ppp = (
        (c_pad[1:-1, 2:, 1:-1] - c_pad[1:-1, 1:-1, 1:-1])
        + (c_pad[2:, 2:, 1:-1] - c_pad[2:, 1:-1, 1:-1])
        + (c_pad[1:-1, 2:, 2:] - c_pad[1:-1, 1:-1, 2:])
        + (c_pad[2:, 2:, 2:] - c_pad[2:, 1:-1, 2:])
    ) / (4 * dy)
    dz_ppp = (
        (c_pad[1:-1, 1:-1, 2:] - c_pad[1:-1, 1:-1, 1:-1])
        + (c_pad[2:, 1:-1, 2:] - c_pad[2:, 1:-1, 1:-1])
        + (c_pad[1:-1, 2:, 2:] - c_pad[1:-1, 2:, 1:-1])
        + (c_pad[2:, 2:, 2:] - c_pad[2:, 2:, 1:-1])
    ) / (4 * dz)

    # 2. (i - 1/2, j + 1/2, k + 1/2)
    dx_mpp = (
        (c_pad[1:-1, 1:-1, 1:-1] - c_pad[:-2, 1:-1, 1:-1])
        + (c_pad[1:-1, 2:, 1:-1] - c_pad[:-2, 2:, 1:-1])
        + (c_pad[1:-1, 1:-1, 2:] - c_pad[:-2, 1:-1, 2:])
        + (c_pad[1:-1, 2:, 2:] - c_pad[:-2, 2:, 2:])
    ) / (4 * dx)
    dy_mpp = (
        (c_pad[:-2, 2:, 1:-1] - c_pad[:-2, 1:-1, 1:-1])
        + (c_pad[1:-1, 2:, 1:-1] - c_pad[1:-1, 1:-1, 1:-1])
        + (c_pad[:-2, 2:, 2:] - c_pad[:-2, 1:-1, 2:])
        + (c_pad[1:-1, 2:, 2:] - c_pad[1:-1, 1:-1, 2:])
    ) / (4 * dy)
    dz_mpp = (
        (c_pad[:-2, 1:-1, 2:] - c_pad[:-2, 1:-1, 1:-1])
        + (c_pad[1:-1, 1:-1, 2:] - c_pad[1:-1, 1:-1, 1:-1])
        + (c_pad[:-2, 2:, 2:] - c_pad[:-2, 2:, 1:-1])
        + (c_pad[1:-1, 2:, 2:] - c_pad[1:-1, 2:, 1:-1])
    ) / (4 * dz)

    # 3. (i + 1/2, j - 1/2, k + 1/2)
    dx_pmp = (
        (c_pad[2:, :-2, 1:-1] - c_pad[1:-1, :-2, 1:-1])
        + (c_pad[2:, 1:-1, 1:-1] - c_pad[1:-1, 1:-1, 1:-1])
        + (c_pad[2:, :-2, 2:] - c_pad[1:-1, :-2, 2:])
        + (c_pad[2:, 1:-1, 2:] - c_pad[1:-1, 1:-1, 2:])
    ) / (4 * dx)
    dy_pmp = (
        (c_pad[1:-1, 1:-1, 1:-1] - c_pad[1:-1, :-2, 1:-1])
        + (c_pad[2:, 1:-1, 1:-1] - c_pad[2:, :-2, 1:-1])
        + (c_pad[1:-1, 1:-1, 2:] - c_pad[1:-1, :-2, 2:])
        + (c_pad[2:, 1:-1, 2:] - c_pad[2:, :-2, 2:])
    ) / (4 * dy)
    dz_pmp = (
        (c_pad[1:-1, :-2, 2:] - c_pad[1:-1, :-2, 1:-1])
        + (c_pad[2:, :-2, 2:] - c_pad[2:, :-2, 1:-1])
        + (c_pad[1:-1, 1:-1, 2:] - c_pad[1:-1, 1:-1, 1:-1])
        + (c_pad[2:, 1:-1, 2:] - c_pad[2:, 1:-1, 1:-1])
    ) / (4 * dz)

    # 4. (i + 1/2, j + 1/2, k - 1/2)
    dx_ppm = (
        (c_pad[2:, 1:-1, :-2] - c_pad[1:-1, 1:-1, :-2])
        + (c_pad[2:, 2:, :-2] - c_pad[1:-1, 2:, :-2])
        + (c_pad[2:, 1:-1, 1:-1] - c_pad[1:-1, 1:-1, 1:-1])
        + (c_pad[2:, 2:, 1:-1] - c_pad[1:-1, 2:, 1:-1])
    ) / (4 * dx)
    dy_ppm = (
        (c_pad[1:-1, 2:, :-2] - c_pad[1:-1, 1:-1, :-2])
        + (c_pad[2:, 2:, :-2] - c_pad[2:, 1:-1, :-2])
        + (c_pad[1:-1, 2:, 1:-1] - c_pad[1:-1, 1:-1, 1:-1])
        + (c_pad[2:, 2:, 1:-1] - c_pad[2:, 1:-1, 1:-1])
    ) / (4 * dy)
    dz_ppm = (
        (c_pad[1:-1, 1:-1, 1:-1] - c_pad[1:-1, 1:-1, :-2])
        + (c_pad[2:, 1:-1, 1:-1] - c_pad[2:, 1:-1, :-2])
        + (c_pad[1:-1, 2:, 1:-1] - c_pad[1:-1, 2:, :-2])
        + (c_pad[2:, 2:, 1:-1] - c_pad[2:, 2:, :-2])
    ) / (4 * dz)

    # 5. (i - 1/2, j - 1/2, k + 1/2)
    dx_mmp = (
        (c_pad[1:-1, :-2, 1:-1] - c_pad[:-2, :-2, 1:-1])
        + (c_pad[1:-1, 1:-1, 1:-1] - c_pad[:-2, 1:-1, 1:-1])
        + (c_pad[1:-1, :-2, 2:] - c_pad[:-2, :-2, 2:])
        + (c_pad[1:-1, 1:-1, 2:] - c_pad[:-2, 1:-1, 2:])
    ) / (4 * dx)
    dy_mmp = (
        (c_pad[:-2, 1:-1, 1:-1] - c_pad[:-2, :-2, 1:-1])
        + (c_pad[1:-1, 1:-1, 1:-1] - c_pad[1:-1, :-2, 1:-1])
        + (c_pad[:-2, 1:-1, 2:] - c_pad[:-2, :-2, 2:])
        + (c_pad[1:-1, 1:-1, 2:] - c_pad[1:-1, :-2, 2:])
    ) / (4 * dy)
    dz_mmp = (
        (c_pad[:-2, :-2, 2:] - c_pad[:-2, :-2, 1:-1])
        + (c_pad[1:-1, :-2, 2:] - c_pad[1:-1, :-2, 1:-1])
        + (c_pad[:-2, 1:-1, 2:] - c_pad[:-2, 1:-1, 1:-1])
        + (c_pad[1:-1, 1:-1, 2:] - c_pad[1:-1, 1:-1, 1:-1])
    ) / (4 * dz)

    # 6. (i - 1/2, j + 1/2, k - 1/2)
    dx_mpm = (
        (c_pad[1:-1, 1:-1, :-2] - c_pad[:-2, 1:-1, :-2])
        + (c_pad[1:-1, 2:, :-2] - c_pad[:-2, 2:, :-2])
        + (c_pad[1:-1, 1:-1, 1:-1] - c_pad[:-2, 1:-1, 1:-1])
        + (c_pad[1:-1, 2:, 1:-1] - c_pad[:-2, 2:, 1:-1])
    ) / (4 * dx)
    dy_mpm = (
        (c_pad[:-2, 2:, :-2] - c_pad[:-2, 1:-1, :-2])
        + (c_pad[1:-1, 2:, :-2] - c_pad[1:-1, 1:-1, :-2])
        + (c_pad[:-2, 2:, 1:-1] - c_pad[:-2, 1:-1, 1:-1])
        + (c_pad[1:-1, 2:, 1:-1] - c_pad[1:-1, 1:-1, 1:-1])
    ) / (4 * dy)
    dz_mpm = (
        (c_pad[:-2, 1:-1, 1:-1] - c_pad[:-2, 1:-1, :-2])
        + (c_pad[1:-1, 1:-1, 1:-1] - c_pad[1:-1, 1:-1, :-2])
        + (c_pad[:-2, 2:, 1:-1] - c_pad[:-2, 2:, :-2])
        + (c_pad[1:-1, 2:, 1:-1] - c_pad[1:-1, 2:, :-2])
    ) / (4 * dz)

    # 7. (i + 1/2, j - 1/2, k - 1/2)
    dx_pmm = (
        (c_pad[2:, :-2, :-2] - c_pad[1:-1, :-2, :-2])
        + (c_pad[2:, 1:-1, :-2] - c_pad[1:-1, 1:-1, :-2])
        + (c_pad[2:, :-2, 1:-1] - c_pad[1:-1, :-2, 1:-1])
        + (c_pad[2:, 1:-1, 1:-1] - c_pad[1:-1, 1:-1, 1:-1])
    ) / (4 * dx)
    dy_pmm = (
        (c_pad[1:-1, 1:-1, :-2] - c_pad[1:-1, :-2, :-2])
        + (c_pad[2:, 1:-1, :-2] - c_pad[2:, :-2, :-2])
        + (c_pad[1:-1, 1:-1, 1:-1] - c_pad[1:-1, :-2, 1:-1])
        + (c_pad[2:, 1:-1, 1:-1] - c_pad[2:, :-2, 1:-1])
    ) / (4 * dy)
    dz_pmm = (
        (c_pad[1:-1, :-2, 1:-1] - c_pad[1:-1, :-2, :-2])
        + (c_pad[2:, :-2, 1:-1] - c_pad[2:, :-2, :-2])
        + (c_pad[1:-1, 1:-1, 1:-1] - c_pad[1:-1, :-2, 1:-1])
        + (c_pad[2:, 1:-1, 1:-1] - c_pad[2:, :-2, 1:-1])
    ) / (4 * dz)

    # 8. (i - 1/2, j - 1/2, k - 1/2)
    dx_mmm = (
        (c_pad[1:-1, :-2, :-2] - c_pad[:-2, :-2, :-2])
        + (c_pad[1:-1, 1:-1, :-2] - c_pad[:-2, 1:-1, :-2])
        + (c_pad[1:-1, :-2, 1:-1] - c_pad[:-2, :-2, 1:-1])
        + (c_pad[1:-1, 1:-1, 1:-1] - c_pad[:-2, 1:-1, 1:-1])
    ) / (4 * dx)
    dy_mmm = (
        (c_pad[:-2, 1:-1, :-2] - c_pad[:-2, :-2, :-2])
        + (c_pad[1:-1, 1:-1, :-2] - c_pad[1:-1, :-2, :-2])
        + (c_pad[:-2, 1:-1, 1:-1] - c_pad[:-2, :-2, 1:-1])
        + (c_pad[1:-1, 1:-1, 1:-1] - c_pad[1:-1, :-2, 1:-1])
    ) / (4 * dy)
    dz_mmm = (
        (c_pad[:-2, :-2, 1:-1] - c_pad[:-2, :-2, :-2])
        + (c_pad[1:-1, :-2, 1:-1] - c_pad[1:-1, :-2, :-2])
        + (c_pad[:-2, 1:-1, 1:-1] - c_pad[:-2, 1:-1, :-2])
        + (c_pad[1:-1, 1:-1, 1:-1] - c_pad[1:-1, :-2, 1:-1])
    ) / (4 * dz)
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 2/6 8-octant gradients", t_step_start, t0)

    # Per-cell diffusion tensor D = D_ext * I + D_axn * (v v^T)
    ext_diff = np.zeros(c.shape, dtype=np.float64)
    ext_diff[wm_mask] = wm_ext
    ext_diff[gm_mask] = gm_ext
    ext_diff[~valid_mask] = 0.0

    dir_outer_prod = fiber[..., :, None] * fiber[..., None, :]
    axonal_diff = np.zeros(c.shape + (3, 3), dtype=np.float64)

    if np.ndim(wm_axn) == 0:
        axonal_diff[wm_mask] = wm_axn * dir_outer_prod[wm_mask]
    else:
        axonal_diff[wm_mask] = wm_axn[wm_mask, None, None] * dir_outer_prod[wm_mask]

    if np.ndim(gm_axn) == 0:
        axonal_diff[gm_mask] = gm_axn * dir_outer_prod[gm_mask]
    else:
        axonal_diff[gm_mask] = gm_axn[gm_mask, None, None] * dir_outer_prod[gm_mask]

    axonal_diff[~valid_mask] = 0.0

    diffusion = ext_diff[..., None, None] * np.eye(3) + axonal_diff
    diffusion_pad = np.pad(diffusion, pad_width=((1, 1), (1, 1), (1, 1), (0, 0), (0, 0)), mode="edge")
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 3/6 tensor build+pad", t_step_start, t0)

    # 8 cell-centered harmonic-mean tensors
    diff_ppp = harmonic_mean_8_corners(
        diffusion_pad[1:-1, 1:-1, 1:-1], diffusion_pad[1:-1, 1:-1, 2:],
        diffusion_pad[1:-1, 2:, 1:-1], diffusion_pad[1:-1, 2:, 2:],
        diffusion_pad[2:, 1:-1, 1:-1], diffusion_pad[2:, 1:-1, 2:],
        diffusion_pad[2:, 2:, 1:-1], diffusion_pad[2:, 2:, 2:],
    )
    diff_mpp = harmonic_mean_8_corners(
        diffusion_pad[:-2, 1:-1, 1:-1], diffusion_pad[:-2, 1:-1, 2:],
        diffusion_pad[:-2, 2:, 1:-1], diffusion_pad[:-2, 2:, 2:],
        diffusion_pad[1:-1, 1:-1, 1:-1], diffusion_pad[1:-1, 1:-1, 2:],
        diffusion_pad[1:-1, 2:, 1:-1], diffusion_pad[1:-1, 2:, 2:],
    )
    diff_pmp = harmonic_mean_8_corners(
        diffusion_pad[1:-1, :-2, 1:-1], diffusion_pad[1:-1, :-2, 2:],
        diffusion_pad[1:-1, 1:-1, 1:-1], diffusion_pad[1:-1, 1:-1, 2:],
        diffusion_pad[2:, :-2, 1:-1], diffusion_pad[2:, :-2, 2:],
        diffusion_pad[2:, 1:-1, 1:-1], diffusion_pad[2:, 1:-1, 2:],
    )
    diff_ppm = harmonic_mean_8_corners(
        diffusion_pad[1:-1, 1:-1, :-2], diffusion_pad[1:-1, 1:-1, 1:-1],
        diffusion_pad[1:-1, 2:, :-2], diffusion_pad[1:-1, 2:, 1:-1],
        diffusion_pad[2:, 1:-1, :-2], diffusion_pad[2:, 1:-1, 1:-1],
        diffusion_pad[2:, 2:, :-2], diffusion_pad[2:, 2:, 1:-1],
    )
    diff_mmp = harmonic_mean_8_corners(
        diffusion_pad[:-2, :-2, 1:-1], diffusion_pad[:-2, :-2, 2:],
        diffusion_pad[:-2, 1:-1, 1:-1], diffusion_pad[:-2, 1:-1, 2:],
        diffusion_pad[1:-1, :-2, 1:-1], diffusion_pad[1:-1, :-2, 2:],
        diffusion_pad[1:-1, 1:-1, 1:-1], diffusion_pad[1:-1, 1:-1, 2:],
    )
    diff_mpm = harmonic_mean_8_corners(
        diffusion_pad[:-2, 1:-1, :-2], diffusion_pad[:-2, 1:-1, 1:-1],
        diffusion_pad[:-2, 2:, :-2], diffusion_pad[:-2, 2:, 1:-1],
        diffusion_pad[1:-1, 1:-1, :-2], diffusion_pad[1:-1, 1:-1, 1:-1],
        diffusion_pad[1:-1, 2:, :-2], diffusion_pad[1:-1, 2:, 1:-1],
    )
    diff_pmm = harmonic_mean_8_corners(
        diffusion_pad[1:-1, :-2, :-2], diffusion_pad[1:-1, :-2, 1:-1],
        diffusion_pad[1:-1, 1:-1, :-2], diffusion_pad[1:-1, 1:-1, 1:-1],
        diffusion_pad[2:, :-2, :-2], diffusion_pad[2:, :-2, 1:-1],
        diffusion_pad[2:, 1:-1, :-2], diffusion_pad[2:, 1:-1, 1:-1],
    )
    diff_mmm = harmonic_mean_8_corners(
        diffusion_pad[:-2, :-2, :-2], diffusion_pad[:-2, :-2, 1:-1],
        diffusion_pad[:-2, 1:-1, :-2], diffusion_pad[:-2, 1:-1, 1:-1],
        diffusion_pad[1:-1, :-2, :-2], diffusion_pad[1:-1, :-2, 1:-1],
        diffusion_pad[1:-1, 1:-1, :-2], diffusion_pad[1:-1, 1:-1, 1:-1],
    )
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 4/6 8-corner harmonic means", t_step_start, t0)

    # Cell-centered fluxes F = D @ grad(c)
    fx_ppp, fy_ppp, fz_ppp = _cell_flux(dx_ppp, dy_ppp, dz_ppp, diff_ppp)
    fx_mpp, fy_mpp, fz_mpp = _cell_flux(dx_mpp, dy_mpp, dz_mpp, diff_mpp)
    fx_pmp, fy_pmp, fz_pmp = _cell_flux(dx_pmp, dy_pmp, dz_pmp, diff_pmp)
    fx_ppm, fy_ppm, fz_ppm = _cell_flux(dx_ppm, dy_ppm, dz_ppm, diff_ppm)
    fx_mmp, fy_mmp, fz_mmp = _cell_flux(dx_mmp, dy_mmp, dz_mmp, diff_mmp)
    fx_mpm, fy_mpm, fz_mpm = _cell_flux(dx_mpm, dy_mpm, dz_mpm, diff_mpm)
    fx_pmm, fy_pmm, fz_pmm = _cell_flux(dx_pmm, dy_pmm, dz_pmm, diff_pmm)
    fx_mmm, fy_mmm, fz_mmm = _cell_flux(dx_mmm, dy_mmm, dz_mmm, diff_mmm)

    # Outer divergence div(D grad c) from the 8 octant cells around each node
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
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 5/6 fluxes + divergence", t_step_start, t0)

    growth = alpha * c * (1.0 - c)
    c_final = c + dt * (div_flux + growth)
    c_final[~valid_mask] = 0.0
    t0 = _checkpoint(verbose_midstep, f"{step_tag} 6/6 growth + update", t_step_start, t0)

    if check_stats:
        has_nan = bool(np.isnan(c_final).any())
        c_min = float(np.nanmin(c_final)) if not has_nan else float("nan")
        c_max = float(np.nanmax(c_final)) if not has_nan else float("nan")
        print(f"  [{step_tag} stats] min={c_min:.4g}, max={c_max:.4g}, nan={has_nan}", flush=True)

    return c_final
