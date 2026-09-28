import argparse
import glob
import json
import logging
import os
import shutil
import sys
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
    write_output_metadata,
    setup_logging_from_metadata,
    _correction_type,
    _b_percentile_type,
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
    source_path = Path(source_path)
    fiber_path = source_path.parent.parent

    # Setup logging
    setup_logging_from_metadata(fiber_path)

    # Copy files to the destination directory
    destination_path = args.output_dir / fiber_path.name
    shutil.copytree(
        fiber_path,
        destination_path,
        ignore=shutil.ignore_patterns("output", "dff-qc", "processing.json"),
    )
    # Update path to the NWB file within the copied directory
    old_nwb_file_path = destination_path / "nwb" / source_path.name
    nwb_file_path = destination_path / "fib.nwb.zarr"
    if old_nwb_file_path.exists():
        shutil.move(old_nwb_file_path, nwb_file_path)
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
            pregocue_starts,
            pregocue_ends,
            event_label,
        ) = process_nwb_file(nwb_file_path, args)

        # Per-(method, fiber, channel) fit timing -- NOT just cumulative
        # processing time. `fit_time_s` is wall-clock time inside
        # `_process1channel`'s `chunk_processing` call; when channels run
        # concurrently (`--parallel`, threading backend), that wall-clock
        # time can include contention with sibling threads rather than
        # pure per-trace fit cost -- still useful for relative comparisons
        # across methods, but not a substitute for a serial (`--serial`,
        # the default) run if precise absolute timings are needed.
        df_pp_params[["preprocess", "channel", "fiber_number", "fit_time_s"]].to_csv(
            destination_path / "dff_timing.csv", index=False
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
                pregocue_starts,
                pregocue_ends,
                event_label,
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


def _process1dataset_safe(source_path, args, start_time):
    """Wrap process1dataset so one dataset's unhandled exception (e.g. the
    matplotlib/Agg RendererAgg crash seen on rare degenerate-data QC plots)
    logs and gets skipped instead of aborting the whole outer Parallel job --
    joblib's default fail-fast behavior means a single bad dataset otherwise
    sacrifices every other still-queued dataset's output too, which is far
    worse than losing QC output for just the one dataset that triggered it."""
    try:
        process1dataset(source_path, args, start_time)
    except Exception:
        logging.exception(f"Skipping {source_path}: unhandled exception")


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
        "-o",
        "--output-dir",
        type=Path,
        default=Path("/results/"),
        help="Output directory",
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
            "  'bright': Robust fit with a sum-of-exponentials baseline (bleaching, "
            "optionally with a brightening term) selected and fit via "
            "aind_ophys_utils.nonlinear_fit (see utils.preprocess.tc_brightfit_v2)\n"
            "  'bright_legacy': The previous 'bright' implementation -- robust fit "
            "with [Bi- or Tri-phasic exponential decay (bleaching)] x [Increasing "
            "saturating exponential (brightening)] using iteratively reweighted "
            "least squares (IRLS)"
        ),
    )
    parser.add_argument(
        "--correction",
        type=_correction_type,
        default=None,
        help=(
            "Optional per-trace, post-hoc centering correction applied to the "
            "'bright' method's fitted baseline (see tc_brightfit_v2). Either "
            "a percentile in [0, 100] or the literal string 'mode'. "
            "50 (median): shift by the plain median of the residuals -- "
            "exactly zero-centered on clean data, 50%% breakdown point. "
            "35: approximates the old 'pct70' recipe (median of the lowest "
            "70%% of residuals, the same convention 'poly'/'exp'/'tri-exp' "
            "use in production via tc_dFF's b_percentile) -- not "
            "bit-identical (uses np.percentile's interpolation instead), "
            "but the same idea: a small guaranteed offset on clean data "
            "for robustness up to 30%% one-sided contamination. Any other "
            "percentile can be swept directly. 'mode': shift by the "
            "residuals' half-sample mode instead -- theoretically less "
            "biased by transient-driven skew, though real-data testing so "
            "far shows median still wins in practice (its own estimator "
            "has lower variance at typical per-trace sample sizes). The "
            "legacy literal strings 'median' and 'pct70' are also accepted, "
            "as aliases for 50 and 35 respectively (backward-compatible "
            "with scripts written before this flag took a percentile). Has "
            "no effect on methods other than 'bright' -- see --b_percentile "
            "for the analogous (but mechanistically different: mandatory, "
            "0.0-1.0 scale) knob for 'poly'/'exp'/'tri-exp', which also "
            "accepts 'mode'. Default is no correction."
        ),
    )
    parser.add_argument(
        "--correction_space",
        choices=["raw", "ratio"],
        default="raw",
        help=(
            "Where --correction is applied (only 'bright' method; no effect "
            "if --correction is not given). 'raw' (default): shift the "
            "fitted baseline additively in raw-fluorescence units, from the "
            "pre-division residual, before computing dF/F -- the original "
            "behavior. 'ratio': compute dF/F first, then shift the dF/F "
            "trace itself additively (same percentile/mode statistic, "
            "computed from dF/F's own distribution instead) -- mirrors how "
            "--b_percentile's correction works for 'poly'/'exp'/'tri-exp', "
            "but for 'bright'. NOT equivalent to 'raw': division is "
            "nonlinear, so the two orderings give different results (by a "
            "term proportional to the correction size over F0, scaled by "
            "the instantaneous dF/F -- small in practice, but not exactly "
            "zero). Added to test this ordering question directly."
        ),
    )
    parser.add_argument(
        "--b_percentile",
        type=_b_percentile_type,
        default=0.7,
        help=(
            "Percentile (or 'mode') for baseline calculation in tc_dFF -- "
            "'poly'/'exp'/'tri-exp' only, no effect on 'bright'/'bright_legacy' "
            "(which use --correction instead). Looks similar to --correction "
            "(both pick a percentile) but is mechanistically different, not "
            "just a differently-scoped copy of it: this is a MANDATORY, "
            "built-in part of tc_dFF's own ratio-based dF/F formula (there "
            "is no 'off' state -- every poly/exp/tri-exp trace uses some "
            "percentile/mode), whereas --correction is an OPTIONAL additive "
            "shift bolted on after bright's fit is already complete "
            "(default is no shift at all). Also note the different scale "
            "for the numeric case: a 0.0-1.0 fraction here (of the lowest "
            "values), vs. --correction's direct 0-100 percentile -- not "
            "interchangeable. 1.0 gives the plain median of the whole "
            "ratio distribution (no truncation) -- the same idea as "
            "--correction 50, just applied to these methods' own ratio-based "
            "residual instead of bright's additive one. 'mode': the "
            "half-sample mode of the whole ratio distribution instead -- "
            "mirrors --correction's own percentile-vs-mode option, added "
            "for a direct real-data comparison. Default is 0.7 (median of "
            "the lowest 70%%), matching production."
        ),
    )
    parser.add_argument(
        "--c_pos",
        type=float,
        default=None,
        help=(
            "Optional override for the 'bright' method's dF/F IRLS M-estimator "
            "positive-residual threshold (AsymmetricTukeyBiweight(c_pos, c_neg), "
            "see tc_brightfit_v2's M_DFF). Has no effect on other methods. Must "
            "be given together with --c_neg. Default is None, which leaves "
            "tc_brightfit_v2 on its own default (c_pos=3.5, c_neg=4.0)."
        ),
    )
    parser.add_argument(
        "--c_neg",
        type=float,
        default=None,
        help=(
            "Optional override for the 'bright' method's dF/F IRLS M-estimator "
            "negative-residual threshold -- see --c_pos. Must be given together "
            "with --c_pos."
        ),
    )
    parser.add_argument(
        "--motion_correction_mode",
        choices=["demean", "intercept"],
        default="demean",
        help=(
            "How motion_correct applies the fitted Iso regression: 'demean' "
            "discards the fitted intercept (safe default -- a channel's own "
            "F0-fitting bias passes through unchanged); 'intercept' also "
            "subtracts the fitted intercept, which additionally removes each "
            "channel's own constant F0-fitting bias but assumes that channel "
            "has no genuine tonic (constant, non-transient) signal of interest. "
            "Default is 'demean'."
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

    if (args.c_pos is None) != (args.c_neg is None):
        parser.error("--c_pos and --c_neg must be given together.")

    # Create the destination directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all files matching the source pattern
    source_paths = glob.glob(args.source_pattern)

    if len(source_paths) == 0:
        logging.error(f"No files found matching pattern: {args.source_pattern}")
        sys.exit(1)

    if len(source_paths) > 1:
        # Force matplotlib to finish building its font cache exactly once,
        # here in the single-threaded parent process, before any worker is
        # forked below. On a freshly-built Docker image (no pre-existing
        # ~/.cache/matplotlib font cache), N outer worker processes each
        # hitting matplotlib for the first time *simultaneously* race to
        # build/write that cache -- observed both as pyparsing mathtext
        # ParseExceptions on otherwise-valid strings, and (this run) as
        # RendererAgg.__init__ receiving a corrupted garbage height
        # (~4.7e14). Rendering one throwaway figure here completes the
        # font-cache build before Parallel(...) starts, so workers see an
        # already-finished cache instead of racing to build it themselves.
        import matplotlib.pyplot as _plt

        _fig = _plt.figure()
        _fig.text(0.5, 0.5, "warm font cache")
        _fig.canvas.draw()
        _plt.close(_fig)

        n_jobs = min(len(source_paths), int(os.getenv("CO_CPUS", -1)))
        Parallel(n_jobs=n_jobs)(
            delayed(_process1dataset_safe)(path, args, start_time) for path in source_paths
        )
    else:
        process1dataset(source_paths[0], args, start_time)
