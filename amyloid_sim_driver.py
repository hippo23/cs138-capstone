import argparse
import os
import sys
import time
import nibabel as nib
import numpy as np
import cupy as cp
from typing import cast
from amyloid_sim_class import AmyloidSimulation

cp.random.seed(24)

def amyloid_simulation_driver(args):
    """
    Loads data, sets up, runs, and visualizes the amyloid simulation
    based on command-line arguments.

    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    """

    print("--- Starting Amyloid Simulation Driver ---")

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
        print("ERROR: mask_path argument is required for a standard simulation.")
        sys.exit(1)

    print(f"Loading mask from: {args.mask_path}")

    if not args.fiber_path:
        print("ERROR: fiber_path argument is required for a standard simulation.")
        sys.exit(1)

    print(f"Loading mask from: {args.fiber_path}")

    if not args.seed_path:
        print("ERROR: seed_path argument is required for a standard simulation.")
        sys.exit(1)

    print(f"Loading mask from: {args.seed_path}")

    try:
        mask_img = cast(nib.Nifti1Image, nib.load(args.mask_path))
        gm_img   = cast(nib.Nifti1Image, nib.load(args.gm_path))
        wm_img   = cast(nib.Nifti1Image, nib.load(args.wm_path))
        csf_img  = cast(nib.Nifti1Image, nib.load(args.csf_path))
        fiber_img  = cast(nib.Nifti1Image, nib.load(args.fiber_path))
        seed_img  = cast(nib.Nifti1Image, nib.load(args.seed_path))

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
    mask = cp.asarray(mask_img.get_fdata() > 0)
    gm = cp.asarray(gm_img.get_fdata())
    wm = cp.asarray(wm_img.get_fdata())
    csf = cp.asarray(csf_img.get_fdata())
    fiber = cp.asarray(fiber_img.get_fdata())
    fiber_mask = cp.linalg.norm(fiber, axis=-1) > 1e-6
    seed = cp.asarray(seed_img.get_fdata() > 0)

    tissue_probs = cp.stack([wm, gm, csf], axis=-1)
    tissue_max = cp.argmax(tissue_probs, axis=-1) + 1

    valid_mask = (tissue_max != 3) & mask & fiber_mask
    seed_mask = valid_mask & seed
    wm_mask = valid_mask & (tissue_max == 1)
    gm_mask = valid_mask & (tissue_max == 2)

    # Initialize Simulation
    print("Initializing simulation...")
    try:
        sim = AmyloidSimulation(
            valid_mask=valid_mask,
            fiber=fiber,
            seed_mask=seed_mask,
            wm_mask=wm_mask,
            gm_mask=gm_mask,
            max_seed_value=args.max_seed_value,
            min_seed_value=args.min_seed_value,
            target_time=args.target_time
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize AmyloidSimulation. {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # Run Simulation
    print(f"\n--- Running Simulation for {args.target_time} years ---")
    start_time = time.time()
    try:
        results, time_steps = sim.run_simulation(
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
        _ = sim.visualize_results(
            results=results, affine=mask_img.affine, save_final_frame=True, output_path=args.output_path
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
    io_group.add_argument("--fiber_path", "--fiber-path", dest="fiber_path", type=str, default=None)
    io_group.add_argument("--seed_path", "--seed-path", dest="seed_path", type=str, default=None,
                      help="Initial seeding region for misfolded protein.")
    io_group.add_argument("-o", "--output-path", type=str, default=None, # Default None -> don't save animation
                        help="Where to save the final model (mask with amyloid concentration).")

    # Simulation Run Arguments
    sim_run_group = parser.add_argument_group('Simulation Run Control')
    sim_run_group.add_argument("--output-freq", type=int, default=100, help="Frequency (steps) for saving results for animation.")
    sim_run_group.add_argument("--target-time", type=float, default=1, help="Target simulation time (years) for adaptive timestep stop.")
    sim_run_group.add_argument("--min-seed-value", type=float, default=0.05, help="Maximum possible initial concentration of amyloid inside the seeding region.")
    sim_run_group.add_argument("--max-seed-value", type=float, default=0.001, help="Minimum possible initial concentration of amyloid inside seeding region.")

    # Parse Arguments
    args = parser.parse_args()

    amyloid_simulation_driver(args)
