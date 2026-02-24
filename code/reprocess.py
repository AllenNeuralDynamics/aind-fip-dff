import argparse
import glob
import json
import logging
import os
import shutil
import sys
from datetime import datetime as dt
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path

import zarr
from aind_data_schema.core.processing import ProcessName
from aind_data_schema.core.quality_control import QualityControl
from hdmf_zarr import NWBZarrIO

from run_capsule import (
    generate_qc_plots,
    process_nwb_file,
    write_output_metadata,
    setup_logging_from_metadata,
)

"""
This script reprocesses fiber photometry data from multiple datasets in parallel.
The subfolder for each dataset includes the NWB file as well as metadata JSONs.
For each dataset, the script processes each channel (typically 4) of each ROI
(typically 4) by generating baseline-corrected (ΔF/F) and motion-corrected traces,
which are then overwritten in the NWB file. It also updates the processing.json
and quality_control.json files for each dataset.
"""


def process1dataset(source_path, args, start_time):
    """Process a single dataset/NWB file.

    Parameters
    ----------
    source_path : str
        Path to the NWB file to process.
    args : argparse.Namespace
        Command-line arguments containing processing parameters.
    start_time : dt
        Start time of the processing run.
    """
    fiber_path = Path(source_path).parent.parent

    # Setup logging
    setup_logging_from_metadata(fiber_path)

    # Copy files to the destination directory
    destination_path = Path(args.output_dir) / fiber_path.name
    shutil.copytree(
        fiber_path,
        destination_path,
        ignore=shutil.ignore_patterns("output", "dff-qc", "processing.json"),
    )
    # Update path to the NWB file within the copied directory
    nwb_file_path = destination_path / "nwb" / os.path.basename(source_path)
    logging.info(f"Processing NWB file: {nwb_file_path}")

    with NWBZarrIO(nwb_file_path, mode="r+", load_namespaces=True) as io:
        nwb_file = io.read()
        has_fiber = nwb_file.acquisition.get("G_0") is not None

    if has_fiber:
        # 1) Remove from NWB object
        with NWBZarrIO(nwb_file_path, mode="r+", load_namespaces=True) as io:
            nwb_file = io.read()
            if "fiber_photometry" in nwb_file.processing:
                del nwb_file.processing["fiber_photometry"]
                io.write(nwb_file)

        # 2) Physically delete leftover group
        store = zarr.open(nwb_file_path, mode="r+")
        if "processing" in store:
            del store["processing"]

        # Use the shared processing function
        df_fip_pp, df_pp_params, coeffs, intercepts, weights, methods = (
            process_nwb_file(nwb_file_path, args)
        )

        # Generate QC plots if requested
        if not args.no_qc:
            new_qc = generate_qc_plots(
                df_fip_pp,
                df_pp_params,
                coeffs,
                intercepts,
                weights,
                methods,
                args,
                destination_path,
            )

            # Update quality_control.json
            with open(destination_path / "quality_control.json") as f:
                old_qc = QualityControl.model_validate(json.load(f))

            new_qc.evaluations = [
                e for e in old_qc.evaluations if not e.name.startswith("Preprocessing")
            ] + new_qc.evaluations
            new_qc.write_standard_file(destination_path)

            # Remove the temporary QC file from dff-qc subdirectory
            new_qc_path = destination_path / "dff-qc" / "quality_control.json"
            if new_qc_path.exists():
                new_qc_path.unlink()

        # Append DataProcess to processing.json
        process_name = ProcessName.DF_F_ESTIMATION

    else:
        logging.info(
            "No fiber photometry data found, only behavior data. Preprocessing not needed."
        )
        process_name = None  # Update processing.json without appending DataProcess

    write_output_metadata(
        metadata=vars(args),
        json_dir=fiber_path,
        process_name=process_name,
        input_fp=source_path,
        output_fp=destination_path / "nwb",
        start_date_time=start_time,
    )


if __name__ == "__main__":
    start_time = dt.now()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_pattern",
        type=str,
        default=r"/data/*/nwb/*.nwb",
        help="Source pattern to find nwb input files",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="/results/", help="Output directory"
    )
    parser.add_argument(
        "--dff_methods",
        nargs="+",
        default=["poly", "exp", "bright"],
        help=(
            "List of dff methods to run. Available options are:\n"
            "  'poly': Fit with 4th order polynomial using ordinary least squares (OLS)\n"
            "  'exp': Fit with biphasic exponential using OLS\n"
            "  'tri-exp': Fit with triphasic exponential using OLS\n"
            "  'bright': Robust fit with [Bi- or Tri-phasic exponential decay (bleaching)] x "
            "[Increasing saturating exponential (brightening)] using iteratively "
            "reweighted least squares (IRLS)"
        ),
    )
    parser.add_argument(
        "--cutoff_freq_motion",
        type=float,
        default=0.05,
        help=(
            "Cutoff frequency of the lowpass Butterworth filter that's only "
            "applied for estimating the regression coefficient, in Hz."
        ),
    )
    parser.add_argument(
        "--cutoff_freq_noise",
        type=float,
        default=3,
        help=(
            "Cutoff frequency of the lowpass Butterworth filter "
            "that's applied to filter out noise, in Hz."
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use multiple processes and threads to parallelize fibers and channels.",
    )
    parser.add_argument("--no_qc", action="store_true", help="Skip QC plots.")
    args = parser.parse_args()
    args.serial = not args.parallel

    # Create the destination directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all files matching the source pattern
    source_paths = glob.glob(args.source_pattern)

    if len(source_paths) == 0:
        logging.error(f"No files found matching pattern: {args.source_pattern}")
        sys.exit(1)

    if len(source_paths) > 1:
        with Pool(
            min(len(source_paths), int(os.getenv("CO_CPUS", cpu_count())))
        ) as pool:
            pool.starmap(
                process1dataset, [(path, args, start_time) for path in source_paths]
            )
    else:
        process1dataset(source_paths[0], args, start_time)
