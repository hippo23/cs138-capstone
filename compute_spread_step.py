import numba
import numpy as np

np.random.seed(24) 

@numba.njit(cache=True)
def compute_explicit_ftcs_step(c, valid_mask, wm_mask, gm_mask, wm_ext, wm_axn, wm_alpha, gm_ext, gm_axn, gm_alpha, dx, dy, dz, dt):
    """
        Performs a single step of amyloid spreading using explicit FTCS.

        Parameters:
        ----------
        c : 3D array : Current concentration at cell in brain relative to max value (valid cells only).
        dx : float : Grid cell size in x-direction (m)
        dy : float : Grid cell size in y-direction (m)
        dt : float : Time step size (s)
    """
    diffusion = np.zeros(c.shape, c.dtype) 
    diffusion[wm_mask] = wm_ext
    diffusion[gm_mask] = gm_ext

    alpha = np.zeros(c.shape, c.dtype) 
    alpha[wm_mask] = wm_alpha
    alpha[gm_mask] = gm_alpha

    # create a padded array to avoid any out-of-bounds error. However, we will cut out
    # the pad at the end, so that it does not affect future calculations.

    nx_int, ny_int, nz_int = c.shape
    nx_pad, ny_pad, nz_pad = nx_int + 2, ny_int + 2, nz_int + 2
    c_pad = np.zeros((nx_pad, ny_pad, nz_pad), dtype=c.dtype)
    c_pad[1:-1, 1:-1, 1:-1] = c

    ddcdx = np.zeros_like(c)
    ddcdy = np.zeros_like(c)
    ddcdz = np.zeros_like(c)

    # now that we have the diffusion variables for each cell, we can compute the next timestep
    # all invalid cells should not have any contribution towards the spread. 

    ddcdx = (
        c_pad[2:, 1:-1, 1:-1] - 2.0 * c_pad[1:-1, 1:-1, 1:-1] + c_pad[:-2, 1:-1, 1:-1]
    ) / (dx ** 2)

    ddcdy = (
        c_pad[1:-1, 2:, 1:-1] - 2.0 * c_pad[1:-1, 1:-1, 1:-1] + c_pad[1:-1, :-2, 1:-1]
    ) / (dy ** 2)

    ddcdz = (
        c_pad[1:-1, 1:-1, 2:] - 2.0 * c_pad[1:-1, 1:-1, 1:-1] + c_pad[1:-1, 1:-1, :-2]
    ) / (dz ** 2)

    c_final = (diffusion * (ddcdx + ddcdy + ddcdz) + alpha * c * (1 - c)) * dt + c

    # zero out interior invalid cells again
    c_final[~valid_mask] = 0

    return c_final
