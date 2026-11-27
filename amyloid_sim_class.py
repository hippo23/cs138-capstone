import time
import numpy as np
import cupy as cp
import nibabel as nib
from compute_explicit_spread_step import compute_explicit_ftcs_step_gpu
from cupyx.scipy.ndimage import gaussian_filter
from amyloid_helpers import harmonic_mean_8_corners

cp.random.seed(24)

class AmyloidSimulation:
    def __init__(self, valid_mask, fiber, seed_mask, wm_mask, gm_mask, max_seed_value=0.05, min_seed_value=0.001, wm_ext = 1.5, wm_axn = 24.0, wm_alpha = 0.6, gm_ext = 1.5, gm_axn = 0.0, gm_alpha = 0.3, stability_factor=0.6, target_time=1.0):

        """
        Parameters:
        -----------
        valid_mask : numpy.ndarray : Mask indicating valid areas for simulation
        seed_mask : numpy.ndarray : Mask indicating seeding region.
        wm_mask : numpy.ndarray : Mask indicating all valid white matter.
        gm_mask : numpy.ndarray : Mask indicating all valid gray matter.
        stability_factor : float : Factor to reduce timestep (0-1) for stability control
        target_time : float : Target simulation time (seconds)
        min_seed_value : float : minimum initial concentration within seeding region
        max_seed_value : float : maximum initiial concentration within seeding region
        wm_ext : float : Extracellular diffusion for white matter
        wm_axn : float : Axonal transportation for white matter
        wm_alpha : float : Growth rate for white matter
        gm_ext : float : Extracellular diffusion for gray matter
        gm_axn : float : Axonal transportation for gray matter
        gm_alpha : float : Growth rate for gray matter
        """

        self.valid_mask = valid_mask.astype(bool)
        self.fiber = fiber
        self.wm_mask = wm_mask
        self.gm_mask = gm_mask

        self.c = cp.zeros(self.valid_mask.shape, dtype=cp.float32)
        noise = cp.random.uniform(min_seed_value, max_seed_value, size=self.valid_mask.shape)
        noise = gaussian_filter(noise, sigma=2.0)
        self.c[seed_mask] = noise[seed_mask]

        self.wm_ext = float(wm_ext) # cm^2 / year
        self.wm_axn = float(wm_axn) # cm^2 / year
        self.wm_alpha = float(wm_alpha) # year^(-1)
        self.gm_ext = float(gm_ext) # cm^2 / year
        self.gm_axn = float(gm_axn) # cm^2 / year
        self.gm_alpha = float(gm_alpha) # year^(-1)

        print(f'WM: Extracellular Diffusion {self.wm_ext} cm^2 / year; Axonal Transport {self.wm_axn} cm^2 / year; Growth Rate: {self.wm_alpha} / year')
        print(f'GM: Extracellular Diffusion {self.gm_ext} cm^2 / year; Axonal Transport {self.gm_axn} cm^2 / year; Growth Rate: {self.gm_alpha} / year')

        self.stability_factor = float(stability_factor)
        self.target_time = float(target_time)

        self.dx = .1
        self.dy = .1
        self.dz = .1

        # we pre-compute the diffusion matrix used in calculation, as this is a heavy computation step.
        print("PRE-PROCESSING DIFFUSION MATRIX...")
        ext_diff = cp.zeros(self.c.shape, dtype=cp.float32)
        ext_diff[self.wm_mask] = self.wm_ext
        ext_diff[self.gm_mask] = self.gm_ext
        ext_diff[~self.valid_mask] = 0.0

        dir_outer_prod = self.fiber[..., :, None] * self.fiber[..., None, :]
        axonal_diff = cp.zeros(self.c.shape + (3, 3), dtype=cp.float32)
        axonal_diff[self.wm_mask] = self.wm_axn * dir_outer_prod[self.wm_mask]
        axonal_diff[self.gm_mask] = self.gm_axn * dir_outer_prod[self.gm_mask]

        # NOTE: (ANISOTROPIC) just the axonal diffusion.
        axonal_diff[~self.valid_mask] = 0.0
        diffusion = ext_diff[..., None, None] * cp.eye(3) + axonal_diff
        diffusion_pad = cp.pad(diffusion, pad_width=((1, 1), (1, 1), (1, 1), (0, 0), (0, 0)), mode="edge")


        # NOTE: (ANISOTROPIC) this is the calculation of the axonal diffusion matrix to avoid re-calculating every computation step.
        self.diff_ppp = harmonic_mean_8_corners(
            diffusion_pad[1:-1, 1:-1, 1:-1], diffusion_pad[1:-1, 1:-1, 2:],
            diffusion_pad[1:-1, 2:, 1:-1], diffusion_pad[1:-1, 2:, 2:],
            diffusion_pad[2:, 1:-1, 1:-1], diffusion_pad[2:, 1:-1, 2:],
            diffusion_pad[2:, 2:, 1:-1], diffusion_pad[2:, 2:, 2:],
        )
        self.diff_mpp = harmonic_mean_8_corners(
            diffusion_pad[:-2, 1:-1, 1:-1], diffusion_pad[:-2, 1:-1, 2:],
            diffusion_pad[:-2, 2:, 1:-1], diffusion_pad[:-2, 2:, 2:],
            diffusion_pad[1:-1, 1:-1, 1:-1], diffusion_pad[1:-1, 1:-1, 2:],
            diffusion_pad[1:-1, 2:, 1:-1], diffusion_pad[1:-1, 2:, 2:],
        )
        self.diff_pmp = harmonic_mean_8_corners(
            diffusion_pad[1:-1, :-2, 1:-1], diffusion_pad[1:-1, :-2, 2:],
            diffusion_pad[1:-1, 1:-1, 1:-1], diffusion_pad[1:-1, 1:-1, 2:],
            diffusion_pad[2:, :-2, 1:-1], diffusion_pad[2:, :-2, 2:],
            diffusion_pad[2:, 1:-1, 1:-1], diffusion_pad[2:, 1:-1, 2:],
        )
        self.diff_ppm = harmonic_mean_8_corners(
            diffusion_pad[1:-1, 1:-1, :-2], diffusion_pad[1:-1, 1:-1, 1:-1],
            diffusion_pad[1:-1, 2:, :-2], diffusion_pad[1:-1, 2:, 1:-1],
            diffusion_pad[2:, 1:-1, :-2], diffusion_pad[2:, 1:-1, 1:-1],
            diffusion_pad[2:, 2:, :-2], diffusion_pad[2:, 2:, 1:-1],
        )
        self.diff_mmp = harmonic_mean_8_corners(
            diffusion_pad[:-2, :-2, 1:-1], diffusion_pad[:-2, :-2, 2:],
            diffusion_pad[:-2, 1:-1, 1:-1], diffusion_pad[:-2, 1:-1, 2:],
            diffusion_pad[1:-1, :-2, 1:-1], diffusion_pad[1:-1, :-2, 2:],
            diffusion_pad[1:-1, 1:-1, 1:-1], diffusion_pad[1:-1, 1:-1, 2:],
        )
        self.diff_mpm = harmonic_mean_8_corners(
            diffusion_pad[:-2, 1:-1, :-2], diffusion_pad[:-2, 1:-1, 1:-1],
            diffusion_pad[:-2, 2:, :-2], diffusion_pad[:-2, 2:, 1:-1],
            diffusion_pad[1:-1, 1:-1, :-2], diffusion_pad[1:-1, 1:-1, 1:-1],
            diffusion_pad[1:-1, 2:, :-2], diffusion_pad[1:-1, 2:, 1:-1],
        )
        self.diff_pmm = harmonic_mean_8_corners(
            diffusion_pad[1:-1, :-2, :-2], diffusion_pad[1:-1, :-2, 1:-1],
            diffusion_pad[1:-1, 1:-1, :-2], diffusion_pad[1:-1, 1:-1, 1:-1],
            diffusion_pad[2:, :-2, :-2], diffusion_pad[2:, :-2, 1:-1],
            diffusion_pad[2:, 1:-1, :-2], diffusion_pad[2:, 1:-1, 1:-1],
        )
        self.diff_mmm = harmonic_mean_8_corners(
            diffusion_pad[:-2, :-2, :-2], diffusion_pad[:-2, :-2, 1:-1],
            diffusion_pad[:-2, 1:-1, :-2], diffusion_pad[:-2, 1:-1, 1:-1],
            diffusion_pad[1:-1, :-2, :-2], diffusion_pad[1:-1, :-2, 1:-1],
            diffusion_pad[1:-1, 1:-1, :-2], diffusion_pad[1:-1, 1:-1, 1:-1],
        )

        print("DONE PRE-PROCESSING DIFFUSION MATRIX...")

        max_diff = max(self.wm_ext + self.wm_axn, self.gm_ext + self.gm_axn)
        self.dt = self.stability_factor / (
            2.0
            * max_diff
            * (1.0 / self.dx**2 + 1.0 / self.dy**2 + 1.0 / self.dz**2)
            + max(self.wm_alpha, self.gm_alpha)
        )

        print(f'\u0394t = {self.dt}')
        self.current_step = 0

    def explicit_ftcs(self):
        self.c = compute_explicit_ftcs_step_gpu(
                    self.c, self.valid_mask, self.wm_mask, self.gm_mask,
                    self.wm_alpha, self.gm_alpha, self.diff_ppp, self.diff_mpp,
                    self.diff_pmp, self.diff_ppm, self.diff_mmp, self.diff_mpm,
                    self.diff_pmm, self.diff_mmm, self.dx, self.dy,
                    self.dz, self.dt
        )

    def runge_kutta(self):
        # NOTE: I'm not sure if this is the correct way, but I'm pretty sure Runge-Kutta is just the
        # same step but weighed four times with slightly different c.
        k1 = compute_explicit_ftcs_step_gpu(
                    self.c, self.valid_mask, self.wm_mask, self.gm_mask,
                    self.wm_alpha, self.gm_alpha, self.diff_ppp, self.diff_mpp,
                    self.diff_pmp, self.diff_ppm, self.diff_mmp, self.diff_mpm,
                    self.diff_pmm, self.diff_mmm, self.dx, self.dy,
                    self.dz, self.dt
        )

        rk2_c = self.c + 0.5 * self.dt * k1

        k2 = compute_explicit_ftcs_step_gpu(
                    rk2_c, self.valid_mask, self.wm_mask, self.gm_mask,
                    self.wm_alpha, self.gm_alpha, self.diff_ppp, self.diff_mpp,
                    self.diff_pmp, self.diff_ppm, self.diff_mmp, self.diff_mpm,
                    self.diff_pmm, self.diff_mmm, self.dx, self.dy,
                    self.dz, self.dt
        )

        rk3_c = self.c + 0.5 * self.dt * k2

        k3 = compute_explicit_ftcs_step_gpu(
                    rk3_c, self.valid_mask, self.wm_mask, self.gm_mask,
                    self.wm_alpha, self.gm_alpha, self.diff_ppp, self.diff_mpp,
                    self.diff_pmp, self.diff_ppm, self.diff_mmp, self.diff_mpm,
                    self.diff_pmm, self.diff_mmm, self.dx, self.dy,
                    self.dz, self.dt
        )

        rk4_c = self.c + self.dt * k3

        k4 = compute_explicit_ftcs_step_gpu(
                    rk4_c, self.valid_mask, self.wm_mask, self.gm_mask,
                    self.wm_alpha, self.gm_alpha, self.diff_ppp, self.diff_mpp,
                    self.diff_pmp, self.diff_ppm, self.diff_mmp, self.diff_mpm,
                    self.diff_pmm, self.diff_mmm, self.dx, self.dy,
                    self.dz, self.dt
        )

        self.c = self.c + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def implicit_euler():
        ...

    # TODO
    def run_simulation(self, output_freq=100, method=0):
        results = []
        time_steps_list = []
        total_time = 0.0
        start_run_time = time.time()

        initial_sim_time_offset = 0.0

        step = 0
        while True:
            self.current_step = step

            if method == 0:
                self.explicit_ftcs()
            elif method == 1:
                self.runge_kutta()
            else:
                self.implicit_euler()

            current_dt = self.dt
            total_time += current_dt

            if step % output_freq == 0:
                # compute maximum concentration
                max_c = cp.nanmax(self.c)

                print(
                    f"Step {step} ",
                    f"Sim Time: {total_time + initial_sim_time_offset:.7f} years / {self.target_time} years (+{current_dt:.7e} years), "
                    f"Max Concentration: {max_c:.10f}"
                )

            if total_time >= self.target_time:
                print(f" Simulation time reached target time {self.target_time} years. Stopping.")
                # Ensure the very last state is saved if stopping due to time
                results.append(self.c.copy())
                time_steps_list.append(total_time + initial_sim_time_offset)
                break

            step += 1

        end_run_time = time.time()
        print(f"\nSimulation finished in {end_run_time - start_run_time:.2f} seconds (wall clock).")

        return results, time_steps_list

    def visualize_results(self, results, affine, save_final_frame=False, output_path=None):
        final_frame = cp.asarray(results[-1], dtype=cp.float32)

        if save_final_frame and output_path:
            img = nib.Nifti1Image(final_frame, affine)
            nib.save(img, output_path)

            print(f"Saved final frame to {output_path}")
            return output_path
