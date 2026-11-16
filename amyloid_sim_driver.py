import argparse
import os
import sys
import time
import numpy as np
from amyloid_sim_class import AmyloidSimulation

np.random.seed(24) 

def amyloid_simulation_driver(args):
    """
    Loads data, sets up, runs, and visualizes the amyloid simulation
    based on command-line arguments.

    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    """

    print("--- Starting Amyloid Simulation Driver ---")

    if not args.anatomy_path:
        print("ERROR: anatomy_path argument is required for a standard simulation.")
        sys.exit(1)

    print(f"Loading anatomy from: {args.dem_path}")

    if not args.wm_path:
        print("ERROR: wm_path argument is required for a standard simulation.")
        sys.exit(1)

    print(f"Loading white matter from: {args.wm_path}")

    if not args.gm_path:
        print("ERROR: wm_path argument is required for a standard simulation.")
        sys.exit(1)

    print(f"Loading gray matter from: {args.gm_path}")

    if not args.csf_path:
        print("ERROR: wm_path argument is required for a standard simulation.")
        sys.exit(1)

    print(f"Loading cerebrospinal fluid from: {args.csf_path}")

    if not args.mask_path:
        print("ERROR: mask argument is required for a standard simulation.")
        sys.exit(1)

    print(f"Loading mask from: {args.mask_path}")

    try:
        mask_img = nib.load(args.mask_path)
        gm_img   = nib.load(args.gm_path)
        wm_img   = nib.load(args.wm_path)
        csf_img  = nib.load(args.csf_path) 

    except FileNotFoundError:
        sys.exit(1)
    except ImportError as e:
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # using the probability files and the mask (which essentially mirrors the anatomy)
    # we decide what is the most probably tissue category at each "cell" in the brain
    mask = mask_img.get_fdata() > 0
    gm = gm_img.get_fdata()
    wm = wm_img.get_fdata()
    csf = csf_img.get_fdata()

    tissue_probs = np.stack([gm, wm, csf], axis=-1)
    tissue_max = np.argmax(tissue_probs, axis=-1) + 1

    mask = mask or (tissue_max == 3)

    tissue = np.zeros(mask.shape, dtype=np.uint8)    
    tissue = tissue + tissue_max
    tissue[~mask] = 0

    # Initialize Simulation
    print("Initializing simulation...")
    try:
        # TODO
        sim = AmyloidSimulation(
            ...
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize AmyloidSimulation. {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        )

    # Run Simulation 
    print(f"\n--- Running Simulation for {args.steps} steps ---")
    start_time = time.time()
    try:
        # TODO
        results, time_steps = sim.run_simulation(
            num_steps=args.steps,
            output_freq=args.output_freq
        )
    except Exception as e:
        print(f"\n--- ERROR occurred during simulation run ---")
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1) # Exit if simulation crashes
        
    end_time = time.time()
    print(f"Simulation compute time: {end_time - start_time:.2f} seconds")

    # Visualize and Save Results
    if not results:
        print("Simulation finished, but no results were generated (possibly 0 steps or immediate instability).")
        sys.exit(0) 

    print("\n--- Visualizing and Saving Results ---")
    try:
        # Check if output path is specified for saving
        save_anim = args.output_path is not None
        save_frame = save_anim and args.save_last_frame # Only save frame if anim is saved

        # TODO
        _ = sim.visualize_results(
            ...
        )

    except ImportError as e:
        print(f"\nERROR during visualization/saving: Missing dependency. {e}")
    except Exception as e:
        print(f"\nERROR during visualization or saving: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Amyloid Simulation Driver Finished ---")

# Main function to run the simulation from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Amyloid Simulation.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input/Output Arguments
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument("--wm_path", "--wm-path", dest="wm_path", type=str, default=None,
                      help="Path to file determining white matter probability.")
    io_group.add_argument("--gm_path", "--gm-path", dest="gm_path", type=str, default=None,
                      help="Path to file determining gray matter probability.")
    io_group.add_argument("--csf_path", "--csf-path", dest="csf_path", type=str, default=None,
                      help="Path to file determining cerebrospinal fluid probability.")
    io_group.add_argument("--mask_path", "--mask-path", dest="mask_path", type=str, default=None,
                      help="Path to file for the anatomy of the brain (all cells inside this mask are considered a valid simulation cell).")
    io_group.add_argument("-o", "--output-path", type=str, default=None, # Default None -> don't save animation
                        help="Where to save the final model (mask with amyloid concentration).")

    # Simulation Run Arguments
    sim_run_group = parser.add_argument_group('Simulation Run Control')
    sim_run_group.add_argument("--output-freq", type=int, default=50, help="Frequency (steps) for saving results for animation.")
    sim_run_group.add_argument("--target-time", type=float, default=3600.0, help="Target simulation time (seconds) for adaptive timestep stop.")

    # Parse Arguments 
    args = parser.parse_args()

    amyloid_simulation_driver(args)
