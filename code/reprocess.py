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
from aind_data_schema.core.data_description import DataDescription, DataLevel
from aind_data_schema.core.processing import ProcessName, Processing
from aind_data_schema.core.quality_control import QualityControl
from pydantic import ValidationError
from hdmf_zarr import NWBZarrIO
from aind_data_schema.utils.inheritance import (
    derive_data_description_from_derived,
    derive_data_description_from_raw,
)
from joblib import Parallel, delayed

from run_capsule import (
    generate_qc_plots,
    process_nwb_file,
    write_fit_timing,
    write_output_metadata,
    setup_logging_from_metadata,
    _b_percentile_type,
    _crop_start_s_type,
    _cutoff_freq_motion_type,
    _cutoff_freq_noise_type,
)

"""
This script reprocesses fiber photometry data from multiple datasets in parallel.
The subfolder for each dataset includes the NWB file as well as metadata JSONs.
For each dataset, the script processes each channel (typically 3) of each ROI
(typically 4) by generating baseline-corrected (ΔF/F) and motion-corrected traces,
which are then overwritten in the NWB file. It also updates the processing.json
and quality_control.json files for each dataset.
"""


#: `tags["type"]` of the dF/F metrics written by `run_capsule` (see `_metric_schema_fields`).
DFF_QC_TYPE = "FIP DFF Processing"


def _dff_metric_names(source_path):
    """Names of the dF/F metrics in `source_path/dff-qc/quality_control.json`, if present."""
    path = source_path / "dff-qc" / "quality_control.json"
    try:
        return {m["name"] for m in json.loads(path.read_text())["metrics"]}
    except (OSError, KeyError, ValueError):
        return set()


def merge_qc(destination_path, new_qc, source_path=None):
    """Write `new_qc` to `quality_control.json`, keeping the other metrics already in it.

    The old dF/F metrics are dropped: those tagged `DFF_QC_TYPE`, and those listed in the
    source's `dff-qc/quality_control.json` (outputs of earlier versions are not tagged).

    Parameters
    ----------
    destination_path : Path
        Dataset directory, with an existing `quality_control.json` (optional).
    new_qc : QualityControl
        The dF/F metrics of this run.
    source_path : Path, optional
        Dataset directory the run started from.
    """
    qc_path = destination_path / "quality_control.json"
    if qc_path.exists():
        try:
            old_qc = QualityControl.model_validate(json.loads(qc_path.read_text()))
        except ValidationError:
            logging.warning(
                f"{qc_path} is not valid for this schema. Writing only the new metrics."
            )
        else:
            old_dff = _dff_metric_names(source_path) if source_path else set()
            kept = [
                m
                for m in old_qc.metrics
                if (m.tags or {}).get("type") != DFF_QC_TYPE and m.name not in old_dff
            ]
            # Rebuild instead of assigning `metrics`, so that `status` is recomputed
            new_qc = QualityControl(
                metrics=kept + new_qc.metrics,
                default_grouping=list(
                    dict.fromkeys(old_qc.default_grouping + new_qc.default_grouping)
                ),
                allow_tag_failures=old_qc.allow_tag_failures,
                key_experimenters=old_qc.key_experimenters,
                notes=old_qc.notes,
            )
    new_qc.write_standard_file(destination_path)
    # Remove the temporary QC file from dff-qc subdirectory
    (destination_path / "dff-qc" / "quality_control.json").unlink(missing_ok=True)


def append_processing(destination_path, old_path):
    """Append the dF/F process in `destination_path/processing.json` to the history in `old_path`.

    Parameters
    ----------
    destination_path : Path
        Dataset directory whose `processing.json` holds only this run's process.
    old_path : Path
        `processing.json` of the dataset this run started from (optional).
    """
    if not old_path.exists():
        return
    try:
        old = Processing.model_validate(json.loads(old_path.read_text()))
    except ValidationError:
        logging.warning(f"{old_path} is not valid for this schema. Writing only the new process.")
        return
    new = Processing.model_validate(json.loads((destination_path / "processing.json").read_text()))

    graph = dict(old.dependency_graph) if old.dependency_graph is not None else None
    previous = old.data_processes[-1].name if old.data_processes else None
    names = set(old.process_names)
    for process in new.data_processes:
        base, i = process.name, 0
        while process.name in names:  # a dF/F process from an earlier run is kept
            i += 1
            process.name = f"{base}_{i}"
        names.add(process.name)
        if graph is not None:
            graph[process.name] = [previous] if previous else []
        previous = process.name

    pipelines = list(old.pipelines or [])
    pipelines += [p for p in new.pipelines or [] if p.name and p not in pipelines]
    Processing(
        data_processes=old.data_processes + new.data_processes,
        pipelines=pipelines or None,
        notes=old.notes,
        dependency_graph=graph,
    ).write_standard_file(destination_path)


def derive_description(fiber_path):
    """Data description of the reprocessed asset: a new derived asset of the input one.

    Parameters
    ----------
    fiber_path : Path
        Dataset directory with the input `data_description.json`.

    Returns
    -------
    DataDescription
        Named `<original input>_processed_<now>`, with the input asset in `source_data`.
    """
    old = DataDescription.model_validate(
        json.loads((fiber_path / "data_description.json").read_text())
    )
    derive = (
        derive_data_description_from_derived
        if old.data_level == DataLevel.DERIVED
        else derive_data_description_from_raw
    )
    return derive(old, "processed", source_data=[old.name])


def process1dataset(source_path, args, start_time, data_description=None):
    """Process a single dataset/NWB file.

    Parameters
    ----------
    source_path : str
        Path to the NWB file to process.
    args : argparse.Namespace
        Command-line arguments containing processing parameters.
    start_time : dt
        Start time of the processing run.
    data_description : DataDescription, optional
        Description of the output asset, whose name is the output directory.
        Derived from the input's if not given.
    """
    source_path = Path(source_path)
    fiber_path = source_path.parent
    if data_description is None:
        data_description = derive_description(fiber_path)

    # Copy files to the destination directory
    destination_path = args.output_dir / data_description.name
    shutil.copytree(
        fiber_path,
        destination_path,
        ignore=shutil.ignore_patterns("output", "dff-qc"),
    )
    data_description.write_standard_file(destination_path)

    # Setup logging, labelled with the name of the new asset
    setup_logging_from_metadata(destination_path)

    # The output asset has a fixed NWB file name, which `processing.json` records
    nwb_file_path = destination_path / "fiber.nwb.zarr"
    if source_path.name != nwb_file_path.name:
        if nwb_file_path.exists():
            raise FileExistsError(f"{fiber_path} already has a {nwb_file_path.name}")
        shutil.move(destination_path / source_path.name, nwb_file_path)
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
            merge_qc(destination_path, new_qc, fiber_path)

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
    append_processing(destination_path, fiber_path / "processing.json")


def _process1dataset_safe(source_path, args, start_time):
    """Wrap process1dataset so one dataset's unhandled exception logs and
    gets skipped (partial output removed) instead of aborting the whole
    outer Parallel job. Returns the failed path for the caller to log."""
    destination_path = None
    destination_existed = True
    try:
        data_description = derive_description(Path(source_path).parent)
        destination_path = args.output_dir / data_description.name
        destination_existed = destination_path.exists()
        process1dataset(source_path, args, start_time, data_description)
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
        default=r"/data/*/fiber.nwb.zarr",
        help="Source pattern to find nwb input files",
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
        "--crop_start_s",
        type=_crop_start_s_type,
        default=5.0,
        help=(
            "Seconds cropped from the start of each fiber before fitting, to leave out the "
            "start-up transient or an off-level start. Their dF/F is NaN."
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
    source_paths = glob.glob(args.source_pattern)

    if len(source_paths) == 0:
        logging.error(f"No files found matching pattern: {args.source_pattern}")
        sys.exit(1)

    # Each input writes to <output_dir>/<dataset dir name>; two inputs in one dataset
    # would write (and, on failure, clean up) the same directory concurrently.
    destinations = Counter(Path(p).parent.name for p in source_paths)
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
    else:
        # Same wrapper, so a failure removes the partial output here too
        failures = [_process1dataset_safe(source_paths[0], args, start_time)]
    failures = [path for path in failures if path is not None]
    if failures:
        # Every dataset has finished (completed outputs are kept); exit nonzero so
        # orchestration can tell an incomplete run from success.
        logging.error("Failed datasets: %s", ", ".join(map(str, failures)))
        sys.exit(1)
