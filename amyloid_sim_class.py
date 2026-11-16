import numpy as np

class AmyloidSimulation:
    def __init__(self, tissue_model, valid_mask, wm_ext = 1.5, wm_axn = 3.0, wm_alpha = 0.6, gm_ext = 1.5, gm_axn = 0.0, gm_alpha = 0.3, stability_factor=0.4, target_time=3600.0):

    """
    Parameters:
    -----------
    tissue_model : numpy.ndarray : model of brain, with each cell categorized into gray matter, white matter, or CSF
    valid_mask : numpy.ndarray : Mask indicating valid areas for simulation 
    stability_factor : float : Factor to reduce timestep (0-1) for stability control 
    target_time : float : Target simulation time (seconds)
    wm_ext : float : Extracellular diffusion for white matter
    wm_axn : float : Axonal transportation for white matter
    wm_alpha : float : Growth rate for white matter
    gm_ext : float : Extracellular diffusion for gray matter
    gm_axn : float : Axonal transportation for gray matter
    gm_alpha : float : Growth rate for gray matter
    """

        self.tissue_model = tissue_model.astype(np.int8)
        self.valid_mask = valid_mask.astype(bool)
        self.amyloid_model = np.zeros_like(valid_mask)

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

    # TODO
    def explicit_euler():
        ...

    # TODO
    def runge_kutta():
        ...

    # TODO
    def crank_nicolson():
        ...

    # TODO
    def run_simulation(self, num_steps, output_freq=10):
        results = []
        time_steps_list = []
        total_time = 0.0
        start_run_time = time.time()

        initial_sim_time_offset = 0.0 

        step = 0
        while True:
            self.current_step = step

            # NOTE: make sure to set initial timestep depending on method being used, referring to Von Neumann Stability Analysis
            # NOTE: also take note of boundary conditions (some parts inside the brain are no flux as well (CSF portions))

            # explicit euler: simply subsitute forward time difference and central difference into fisher-kolmogrov to get
            # the system, and then solve.

            # runge-kutta: again, advance in time using the system. This becomes fourth-order accurate in time, however,
            # you might opt to use a fourth-order accurate approximation to the spatial derivatives as well for it to make sense.

            # implicit / semi-implicit (crank-nicolson): 
            # IMEX / semi-implicit to reduce computational cost to not have to solve
            # nonlinear algebraic systems
            # still, cannot be solved using LU due to brain mesh size. Need iterative methods
            # i.e. Gauss-Siedel, Jacobi

            if step % output_freq == 0:
                ...
        
            if total_time >= self.target_time:
                print(f" Simulation time reached target time {self.target_time}. Stopping.")
                # Ensure the very last state is saved if stopping due to time
                if step % output_freq != 0:
                    results.append(self.h.copy())
                    time_steps_list.append(total_time + initial_sim_time_offset)
                break

            step += 1

        end_run_time = time.time()
        print(f"\nSimulation finished {step+1} steps in {end_run_time - start_run_time:.2f} seconds (wall clock).")

        
        # TODO: return the results (could return results at each freq to show animation)
        return results, time_steps_list

    # TODO
    def visualize_results(self, results, time_steps, output_path=None, show_animation=True, save_last_frame=True):
        ...
