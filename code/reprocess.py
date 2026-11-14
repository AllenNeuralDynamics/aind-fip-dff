import utils.blas_threads  # noqa: F401 -- sets BLAS env vars, must precede numpy
import argparse
import glob
import json
import logging
import os
import shutil
import sys
from collections import Counter
from datetime import datetime as dt
from pathlib import Path

import zarr
from aind_data_schema.core.processing import ProcessName
from aind_data_schema.core.quality_control import QualityControl
from hdmf_zarr import NWBZarrIO
from joblib import Parallel, delayed

from run_capsule import (
    generate_qc_plots,
    process_nwb_file,
    write_fit_timing,
    write_output_metadata,
    setup_logging_from_metadata,
    _b_percentile_type,
    _cutoff_freq_motion_type,
    _cutoff_freq_noise_type,
)

"""
This script reprocesses fiber photometry data from multiple datasets in parallel.
The folder for each dataset includes the NWB file as well as metadata JSONs.
For each dataset, the script processes each channel (typically 3) of each ROI
(typically 4) by generating baseline-corrected (ΔF/F) and motion-corrected traces,
which are then overwritten in the NWB file. It also updates the processing.json
and quality_control.json files for each dataset.
"""

#: Name of the NWB file in each output dataset, matching run_capsule.py.
NWB_NAME = "fiber.nwb.zarr"
#: Where to look for inputs without --source_pattern: this branch's outputs,
#: and older ones with the NWB in an `nwb/` subfolder.
DEFAULT_SOURCE_PATTERNS = ("/data/*/" + NWB_NAME, "/data/*/nwb/*.nwb")
#: `type` tag of the metrics run_capsule.py writes (see `_metric_schema_fields`).
DFF_METRIC_TYPE = "FIP DFF Processing"


def _dataset_dir(source_path: Path) -> Path:
    """Dataset folder of an NWB file: its parent, or the parent of the older
    `nwb/` subfolder layout."""
    return source_path.parent.parent if source_path.parent.name == "nwb" else source_path.parent


def _schema_major(metadata: dict) -> int:
    """Major version of a metadata file's `schema_version`; 1 if it is
    missing or unparsable, or the file has schema 1's `evaluations`."""
    if "evaluations" in metadata:
        return 1
    try:
        return int(str(metadata.get("schema_version", "1")).split(".")[0])
    except ValueError:
        return 1


def _is_dff_metric(metric) -> bool:
    """Whether a QC metric was written by this capsule, so a rerun replaces it.

    Matches the current `type` tag, and the "Fiber ROI" tag key older runs of
    this branch used.
    """
    tags = metric.tags or {}
    if isinstance(tags, dict):
        return tags.get("type") == DFF_METRIC_TYPE or "Fiber ROI" in tags
    return DFF_METRIC_TYPE in tags


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
    source_path = Path(source_path)
    fiber_path = _dataset_dir(source_path)

    # Setup logging
    setup_logging_from_metadata(fiber_path)

    # Copy files to the destination directory
    destination_path = args.output_dir / fiber_path.name
    shutil.copytree(
        fiber_path,
        destination_path,
        ignore=shutil.ignore_patterns("output", "dff-qc", "processing.json"),
    )
    # Move the NWB file to the top level of the copied directory under its
    # standard name (older outputs kept it in an `nwb/` subfolder).
    old_nwb_file_path = destination_path / source_path.relative_to(fiber_path)
    nwb_file_path = destination_path / NWB_NAME
    if old_nwb_file_path != nwb_file_path and old_nwb_file_path.exists():
        shutil.move(old_nwb_file_path, nwb_file_path)
        nwb_dir = old_nwb_file_path.parent
        if nwb_dir != destination_path and not any(nwb_dir.iterdir()):
            nwb_dir.rmdir()
    logging.info(f"Processing NWB file: {nwb_file_path}")

    with NWBZarrIO(nwb_file_path, mode="r+", load_namespaces=True) as io:
        nwb_file = io.read()
        fiber_prefixes = ("G_", "R_", "Iso_", "Signal_")
        has_fiber = any(
            isinstance(name, str) and any(name.startswith(p) for p in fiber_prefixes)
            for name in nwb_file.acquisition.keys()
        )

    if has_fiber:
        # 1) Remove from NWB object
        with NWBZarrIO(nwb_file_path, mode="r+", load_namespaces=True) as io:
            nwb_file = io.read()
            if "fiber_photometry" in nwb_file.processing:
                del nwb_file.processing["fiber_photometry"]
                io.write(nwb_file)

        # 2) Physically delete leftover fiber_photometry group
        store = zarr.open(nwb_file_path, mode="r+")
        if "processing" in store:
            processing_group = store["processing"]
            if "fiber_photometry" in processing_group:
                del processing_group["fiber_photometry"]
                # Optionally remove the processing group if it is now empty
                if len(processing_group.keys()) == 0:
                    del store["processing"]

        # Use the shared processing function
        (
            df_fip_pp,
            df_pp_params,
            coeffs,
            intercepts,
            weights,
            methods,
        ) = process_nwb_file(nwb_file_path, args)

        write_fit_timing(df_pp_params, destination_path)

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

            # Merge into the dataset's quality_control.json, replacing this
            # capsule's metrics from the earlier run and keeping the rest.
            # Without a schema 2 one, the new QC stays in dff-qc as
            # run_capsule.py leaves it.
            qc_path = destination_path / "quality_control.json"
            old_qc_data = None
            if qc_path.exists():
                with open(qc_path) as f:
                    old_qc_data = json.load(f)
            if old_qc_data is None:
                logging.info(
                    f"No {qc_path.name} in {fiber_path}; QC left in the dff-qc folder"
                )
            elif _schema_major(old_qc_data) < 2:
                # Schema 1 QC (evaluations, not a flat metric list) cannot be
                # merged with schema 2 metrics; keep it untouched rather than
                # dropping other capsules' evaluations.
                logging.warning(
                    f"{qc_path} is schema "
                    f"{old_qc_data.get('schema_version', '1.x')}; left unchanged, "
                    "new QC is in the dff-qc folder"
                )
            else:
                old_qc = QualityControl.model_validate(old_qc_data)

                new_qc.metrics = [
                    m for m in old_qc.metrics if not _is_dff_metric(m)
                ] + new_qc.metrics
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
        process_name=process_name,
        input_fp=source_path,
        output_fp=destination_path,
        start_date_time=start_time,
    )


def _process1dataset_safe(source_path, args, start_time):
    """Wrap process1dataset so one dataset's unhandled exception logs and
    gets skipped (partial output removed) instead of aborting the whole
    outer Parallel job. Returns the failed path for the caller to log."""
    destination_path = args.output_dir / _dataset_dir(Path(source_path)).name
    destination_existed = destination_path.exists()
    try:
        process1dataset(source_path, args, start_time)
        return None
    except Exception:
        logging.exception(f"Failed processing {source_path}")
        if not destination_existed and destination_path.exists():
            shutil.rmtree(destination_path, ignore_errors=True)
        return source_path


if __name__ == "__main__":
    start_time = dt.now()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_pattern",
        type=str,
        default=None,
        help=(
            "Source pattern to find nwb input files. Default: both "
            + " and ".join(DEFAULT_SOURCE_PATTERNS)
            + " (current and older output layouts)"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("/results/"),
        help="Output directory",
    )
    parser.add_argument(
        "--dff_methods",
        nargs="+",
        default=["poly", "exp", "bright-v2"],
        help=(
            "List of dff methods to run. Available options are:\n"
            "  'poly': Fit with 4th order polynomial using ordinary least squares (OLS)\n"
            "  'exp': Fit with biphasic exponential using OLS\n"
            "  'tri-exp': Fit with triphasic exponential using OLS\n"
            "  'bright': Robust fit with [Bi- or Tri-phasic exponential decay "
            "(bleaching)] x [Increasing saturating exponential (brightening)] using "
            "iteratively reweighted least squares (IRLS)\n"
            "  'bright-v2': Robust fit with a sum-of-exponentials baseline (bleaching, "
            "optionally with a brightening term) selected and fit via "
            "aind_ophys_utils.nonlinear_fit (see utils.preprocess.tc_brightfit_v2); "
            "much faster than 'bright'"
        ),
    )
    parser.add_argument(
        "--b_percentile",
        type=_b_percentile_type,
        default=0.7,
        help=(
            "Percentile (or 'mode') for baseline calculation in tc_dFF -- "
            "'poly'/'exp'/'tri-exp' only, no effect on 'bright'/'bright-v2'. "
            "A fraction in (0, 1]: the plain median of the lowest "
            "`b_percentile` fraction of the whole ratio distribution is "
            "subtracted (e.g. 0.7 == median of the lowest 70%%, 1.0 == plain "
            "median of everything). 'mode': the half-sample mode of the "
            "whole ratio distribution instead. Default is 0.7 (median of "
            "the lowest 70%%), matching production."
        ),
    )
    parser.add_argument(
        "--cutoff_freq_motion",
        type=_cutoff_freq_motion_type,
        default=0.05,
        help=(
            "Cutoff frequency of the lowpass Butterworth filter that's only "
            "applied for estimating the regression coefficient, in Hz."
        ),
    )
    parser.add_argument(
        "--cutoff_freq_noise",
        type=_cutoff_freq_noise_type,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all files matching the source pattern
    patterns = [args.source_pattern] if args.source_pattern else DEFAULT_SOURCE_PATTERNS
    source_paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})

    if len(source_paths) == 0:
        logging.error(f"No files found matching pattern(s): {', '.join(patterns)}")
        sys.exit(1)

    # Each input writes to <output_dir>/<dataset dir name>; two inputs in one dataset
    # would write (and, on failure, clean up) the same directory concurrently.
    destinations = Counter(_dataset_dir(Path(p)).name for p in source_paths)
    shared = sorted(name for name, n in destinations.items() if n > 1)
    if shared:
        logging.error(f"Several inputs map to the same output directory: {', '.join(shared)}")
        sys.exit(1)

    if len(source_paths) > 1:
        # Force matplotlib to finish building its font cache here, once, in
        # the single-threaded parent -- on a fresh image, N workers racing
        # to build it on first use corrupts figure rendering. Render one
        # throwaway figure so workers see an already-finished cache.
        import matplotlib.pyplot as _plt

        _fig = _plt.figure()
        _fig.text(0.5, 0.5, "warm font cache")
        _fig.canvas.draw()
        _plt.close(_fig)

        n_jobs = min(len(source_paths), int(os.getenv("CO_CPUS", -1)))
        failures = Parallel(n_jobs=n_jobs)(
            delayed(_process1dataset_safe)(path, args, start_time) for path in source_paths
        )
        failures = [path for path in failures if path is not None]
        if failures:
            # Every worker has finished (completed outputs are kept); exit
            # nonzero so orchestration can tell an incomplete batch from success.
            logging.error("Failed datasets: %s", ", ".join(map(str, failures)))
            sys.exit(1)
    else:
        process1dataset(source_paths[0], args, start_time)
