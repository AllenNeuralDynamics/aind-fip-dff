import argparse
import glob
import itertools
import json
import logging
import os
import shutil
from datetime import datetime as dt
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
from aind_data_schema.core.data_description import DerivedDataDescription
from aind_data_schema.core.processing import (
    DataProcess,
    PipelineProcess,
    Processing,
    ProcessName,
)
from aind_data_schema.core.quality_control import (
    QCEvaluation,
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_log_utils import log
from aind_metadata_upgrader.data_description_upgrade import DataDescriptionUpgrade
from aind_metadata_upgrader.processing_upgrade import ProcessingUpgrade
from aind_qcportal_schema.metric_value import DropdownMetric
from hdmf_zarr import NWBZarrIO
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, sosfiltfilt, welch

import utils.nwb_dict_utils as nwb_utils
from utils.preprocess import chunk_processing, motion_correct

"""
This capsule takes in an NWB file containing raw fiber photometry data 
then process each channel (usually 4) of each ROI (usually 4) by
generating baseline-corrected (ΔF/F) and motion-corrected traces,
which are then appended back to the NWB file.
"""


def write_output_metadata(
    metadata: dict,
    json_dir: str,
    process_name: Union[str, None],
    input_fp: Union[str, Path],
    output_fp: Union[str, Path],
    start_date_time: dt,
) -> None:
    """Writes output metadata to processing.json

    Parameters
    ----------
    metadata : dict
        Parameters passed to the capsule.
    json_dir : str
        Directory where the processing.json and data_description.json file is located.
    process_name : str
        Name of the process being recorded.
    input_fp : Union[str, Path]
        Path to the data input.
    output_fp : Union[str, Path]
        Path to the data output.
    start_date_time : dt
        Start date and time of the process.
    """
    proc_path = Path(json_dir) / "processing.json"

    dp = (
        [
            DataProcess(
                name=process_name,
                software_version=os.getenv("VERSION", ""),
                start_date_time=start_date_time,
                end_date_time=dt.now(),
                input_location=str(input_fp),
                output_location=str(output_fp),
                code_url=(os.getenv("DFF_EXTRACTION_URL")),
                parameters=metadata,
            )
        ]
        if process_name is not None
        else []
    )

    if os.path.exists(proc_path):
        with open(proc_path, "r") as f:
            proc_data = json.load(f)

        proc_upgrader = ProcessingUpgrade(old_processing_model=proc_data)
        processing = proc_upgrader.upgrade(
            processor_full_name="Fiberphotometry Processing Pipeline"
        )
        p = processing.processing_pipeline
        p.data_processes += dp
    else:
        p = PipelineProcess(
            processor_full_name="Fiberphotometry Processing Pipeline",
            data_processes=dp,
        )
        processing = Processing(processing_pipeline=p)
    if u := os.getenv("PIPELINE_URL", ""):
        p.pipeline_url = u
    if v := os.getenv("PIPELINE_VERSION", ""):
        p.pipeline_version = v
    processing.write_standard_file(output_directory=Path(output_fp).parent)

    dd_file = Path(json_dir) / "data_description.json"
    if os.path.exists(dd_file):
        with open(dd_file, "r") as f:
            dd_data = json.load(f)
        dd_upgrader = DataDescriptionUpgrade(old_data_description_dict=dd_data)
        new_dd = dd_upgrader.upgrade()
        derived_dd = DerivedDataDescription.from_data_description(
            data_description=new_dd, process_name="processed"
        )
        derived_dd.write_standard_file(output_directory=Path(output_fp).parent)
    else:
        logging.error("no input data description")


def plot_raw_dff_mc(
    nwb_file: pynwb.NWBFile,
    fiber: str,
    channels: list[str],
    method: str,
    fig_path: str = "/results/dff-qc/",
):
    """Plot raw, dF/F, and preprocessed (dF/F with motion correction) photometry traces
    for multiple channels from an NWB file.

    Parameters
    ----------
    nwb_file : NWBFile
        The Neurodata Without Borders (NWB) file containing photometry signal traces
        and their associated metadata.
    fiber : str
        The name of the fiber for which the signals should be plotted.
    channels : list of str
        A list of channel names to be plotted (e.g., ['G', 'R', 'Iso']).
    method : str
        The name of the preprocessing method used ("poly", "exp", or "bright").
    fig_path : str, optional
        The path where the generated plot will be saved. Defaults to "/results/dff-qc/".
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 4), sharex=True)
    for i, suffix in enumerate(("", f"_dff-{method}", f"_dff-{method}_mc-iso-IRLS")):
        for ch in sorted(channels):
            if i == 0:
                trace = nwb_file.acquisition[ch + f"_{fiber}"]
            else:
                trace = nwb_file.processing["fiber_photometry"].data_interfaces[
                    ch + f"_{fiber}{suffix}"
                ]
            t, d = trace.timestamps[:], trace.data[:]
            t -= t[0]
            if ~np.isnan(t).all():
                ax[i].plot(
                    t,
                    d * 100 if i else d,
                    label=ch,
                    alpha=0.8,
                    # more color-blind-friendly g, b, and r
                    c={"G": "#009E73", "Iso": "#0072B2", "R": "#D55E00"}.get(
                        ch, f"C{i}"
                    ),
                )
        if i == 0:
            ax[i].legend()
        ax[i].set_title(
            (
                "Raw",
                r"$\Delta$F/F ('dff')",
                r"$\Delta$F/F + motion-correction ('dff_mc')",
            )[i]
        )
        ax[i].set_ylabel(("F [a.u.]", r"$\Delta$F/F [%]", r"$\Delta$F/F [%]")[i])
    tmin, tmax = np.nanmin(t), np.nanmax(t)
    ax[i].set_xlim(tmin - (tmax - tmin) / 100, tmax + (tmax - tmin) / 100)
    plt.suptitle(f"Method: {method},  ROI: {fiber}", y=1)
    plt.xlabel("Time [" + trace.unit + "]")
    plt.tight_layout(pad=0.2)
    os.makedirs(fig_path, exist_ok=True)
    fig_file = os.path.join(fig_path, f"ROI{fiber}_{method}.png")
    plt.savefig(fig_file, dpi=300)
    return fig_file


def plot_dff(
    df_fip_pp: pd.DataFrame,
    fiber: str,
    channels: list[str],
    method: str,
    fig_path: str,
):
    """Plot raw and dF/F photometry traces for multiple channels.

    Parameters
    ----------
    df_fip_pp : pd.DataFrame
        The dataframe with the preprocessed FIP data containing F, dF/F and F0 traces.
    fiber : str
        The name of the fiber for which the signals should be plotted.
    channels : list of str
        A list of channel names to be plotted (e.g., ['G', 'R', 'Iso']).
    method : str
        The name of the preprocessing method used ("poly", "exp", or "bright").
    fig_path : str
        The path where the generated plot will be saved.
    """
    fig, ax = plt.subplots(
        2 * len(channels), 1, figsize=(12, 2 * len(channels)), sharex=True
    )
    for c, ch in enumerate(sorted(channels)):
        df = df_fip_pp[
            (df_fip_pp.channel == ch)
            & (df_fip_pp.fiber_number == fiber)
            & (df_fip_pp.preprocess == method)
        ]
        t = df.time_fip.values
        t -= t[0]
        if ~np.isnan(t).all():
            for i in (0, 1):
                a = ax[2 * c + i]
                a.plot(
                    t,
                    (df.signal, df.dFF * 100)[i],
                    label=(("raw ", r"$\Delta$F/F ")[i] + ch),
                    # more color-blind-friendly g, b, and r
                    c={"G": "#009E73", "Iso": "#0072B2", "R": "#D55E00"}.get(
                        ch, f"C{c}"
                    ),
                )
                if i == 0:
                    a.plot(t, df.F0, label=r"fitted F$_0$", c="#F0E442")
                else:
                    a.axhline(0, c="k", ls="--")
                a.legend(
                    loc=(0.01, 0.77), ncol=2 - i, borderpad=0.05
                ).get_frame().set_linewidth(0.0)
                a.set_ylabel(("F [a.u.]", r"$\Delta$F/F [%]")[i])

    tmin, tmax = np.nanmin(t), np.nanmax(t)
    ax[i].set_xlim(tmin - (tmax - tmin) / 100, tmax + (tmax - tmin) / 100)
    plt.suptitle(f"$\\bf{{\Delta F/F_0}}$  Method: {method},  ROI: {fiber}", y=1)
    plt.xlabel("Time [s]")
    plt.tight_layout(pad=0.2, h_pad=0)

    os.makedirs(fig_path, exist_ok=True)
    fig_file = os.path.join(fig_path, f"ROI{fiber}_dff-{method}.png")
    plt.savefig(fig_file, dpi=300)


def plot_motion_correction(
    df_fip_pp: pd.DataFrame,
    fiber: str,
    channels: list[str],
    method: str,
    fig_path: str,
    coeffs: list[dict],
    intercepts: list[dict],
    cutoff_freq_motion: float,
    cutoff_freq_noise: float,
    fs: float = 20,
):
    """Plot dF/F and motion-corrected dF/F photometry traces for multiple channels.

    Parameters
    ----------
    df_fip_pp : pd.DataFrame
        The dataframe with the preprocessed FIP data containing F, dF/F and F0 traces.
    fiber : str
        The name of the fiber for which the signals should be plotted.
    channels : list of str
        A list of channel names to be plotted (e.g., ['G', 'R', 'Iso']).
    method : str
        The name of the preprocessing method used ("poly", "exp", or "bright").
    fig_path : str
        The path where the generated plot will be saved.
    coeffs : dict of list of dict
        The regression coefficients for each method/fiber/channel combination.
    intercepts : dict of list of dict
        The regression intercepts for each method/fiber/channel combination.
    cutoff_freq_motion : float
        Cutoff frequency of the lowpass Butterworth filter that's only
        applied for estimating the regression coefficient, in Hz.
    cutoff_freq_noise : float
        Cutoff frequency of the lowpass Butterworth filter
        that's applied to filter out noise, in Hz.
    fs : float, optional
        Sampling rate of the signal, in Hz. Defaults to 20.

    Returns
    -------
    None
        The function saves the plot to the specified fig_path.
    """
    cut = cutoff_freq_noise is not None and cutoff_freq_noise < fs / 2
    colors = {"G": "C2", "Iso": "C0", "R": "C3"}
    rows = 3 * len(channels) - 3
    fig = plt.figure(figsize=(15, rows))
    gs = GridSpec(rows, 3, width_ratios=[11, 1, 3])

    def plot_psd(ax, data, color, cut=False):
        """Helper function to create Power Spectral Density plots.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to plot the PSD.
        data : array-like
            The data to compute the PSD for.
        color : str
            The color to use for the plot.
        cut : bool, optional
            Whether to cut the frequency range. Defaults to False.

        Returns
        -------
        matplotlib.axes.Axes
            The axis with the PSD plot.
        """
        psd = np.array(welch(data * 100, nperseg=1024))[:, 1:-1]
        if cut:
            psd = psd[:, psd[0] < min(0.5, 1.25 * cutoff_freq_noise / fs)]
        ax.loglog(psd[0] * fs, psd[1], c=color)
        return ax

    left_axes = []
    center_axes = []
    right_axes = []
    df_iso = df_fip_pp[
        (df_fip_pp.channel == "Iso")
        & (df_fip_pp.fiber_number == fiber)
        & (df_fip_pp.preprocess == method)
    ]
    t = df_iso.time_fip.values
    t -= t[0]
    for c, ch in enumerate(sorted([ch for ch in channels if ch != "Iso"])):
        df = df_fip_pp[
            (df_fip_pp.channel == ch)
            & (df_fip_pp.fiber_number == fiber)
            & (df_fip_pp.preprocess == method)
        ]
        color = colors.get(ch, f"C{c}")
        # Create subplots in the left and center column (sharing x-axis)
        for i in range(3):
            ax = fig.add_subplot(
                gs[3 * c + i, 0], sharex=(None if c + i == 0 else left_axes[0])
            )
            ax2 = fig.add_subplot(
                gs[3 * c + i, 1], sharex=(None if c + i == 0 else center_axes[0])
            )
            if i < 2:
                l = ("", "low-passed")[i]
                if cut:
                    sos = butter(N=2, Wn=cutoff_freq_noise, fs=fs, output="sos")
                    noise_filt = lambda x: sosfiltfilt(sos, x)
                else:
                    noise_filt = lambda x: x
                ax.plot(
                    t,
                    (noise_filt(df["dFF"]) if i == 0 else df["filtered"]) * 100,
                    c=color,
                    label=(("", "low-passed ")[i] + ch),
                )
                plot_psd(
                    ax2, df["dFF"] if i == 0 else df["filtered"], color, i == 1 and cut
                )
                coef = coeffs[method][int(fiber)][ch]
                intercept = intercepts[method][int(fiber)][ch]
                if i == 0:
                    ax.plot(
                        t,
                        (intercept + noise_filt(df_iso["dFF"]) * coef) * 100,
                        c=colors["Iso"],
                        label="regressed Iso",
                        alpha=0.5,
                    )
                    plot_psd(ax2.twinx(), df_iso["dFF"], colors["Iso"]).tick_params(
                        axis="y", which="both", colors=colors["Iso"]
                    )
                else:
                    ax2.axvline(cutoff_freq_motion, c="k", ls="--")
                    plot_psd(ax2.twinx(), df_iso["filtered"], "C1", cut).tick_params(
                        axis="y", which="both", colors="C1"
                    )
                ax.plot(
                    t,
                    (intercept + df_iso["filtered"] * coef) * 100,
                    c="C1",
                    label="low-passed Iso",
                )
            else:
                ax.plot(
                    t, df["motion_corrected"] * 100, c=color, label=f"corrected {ch}"
                )
                plot_psd(ax2, df["motion_corrected"], color, cut)
                if cut:
                    ax2.axvline(cutoff_freq_noise, c="k", ls="--")
            ax.legend(
                ncol=3, loc=(0.01, 0.77), borderpad=0.05
            ).get_frame().set_linewidth(0.0)
            ax2.tick_params(axis="y", which="both", colors=color)
            left_axes.append(ax)
            center_axes.append(ax2)

        # Create subplots in the right column, each spanning 3 rows
        ax = fig.add_subplot(
            gs[3 * c : 3 * c + 3, 2], sharex=(None if c == 0 else right_axes[0])
        )
        ax.scatter(
            df_iso["dFF"] * 100, df["dFF"] * 100, s=0.1, c="C0", label="original"
        )
        ax.scatter(
            df_iso["filtered"] * 100,
            df["filtered"] * 100,
            s=0.1,
            c="C1",
            label="low-passed",
        )
        x, y = np.array(ax.get_xlim()), ax.get_ylim()
        ax.plot(x, intercept * 100 + coef * x, c="r", label="regression")
        ax.set_ylim(y)
        ax.legend(
            loc="lower right", markerscale=12, borderpad=0.05
        ).get_frame().set_linewidth(0.0)
        ax.set_ylabel(ch, color=color, labelpad=-3)
        ax.set_title(f"Regression coeff.: {coef:.4f}", fontsize=12, y=0.9)
        right_axes.append(ax)

    # Hide x-tick labels for all but the bottom subplots
    for ax in left_axes[:-1] + center_axes[:-1] + right_axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    tmin, tmax = np.nanmin(t), np.nanmax(t)
    left_axes[-1].set_xlim(tmin - (tmax - tmin) / 100, tmax + (tmax - tmin) / 100)
    left_axes[-1].set_xlabel("Time [s]")
    left_axes[rows // 2].set_ylabel(
        "$\Delta$F/F [%]", y=(1.1, 0.5)[rows % 2], labelpad=10
    )
    center_axes[-1].set_xlabel("Frequency [Hz]")
    center_axes[rows // 2].set_ylabel("PSD", y=(1.1, 0.5)[rows % 2])
    right_axes[-1].set_xlabel("Iso", color=colors["Iso"])
    plt.suptitle(
        f"$\\bf{{Motion\;correction}}$   Methods: {method} & iso-IRLS,  ROI: {fiber}",
        y=1,
    )
    plt.tight_layout(pad=0.2, h_pad=0, w_pad=0)
    for ax in center_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 - 0.025, pos.y0, pos.width + 0.015, pos.height])

    os.makedirs(fig_path, exist_ok=True)
    fig_file = os.path.join(fig_path, f"ROI{fiber}_dff-{method}_mc-iso-IRLS.png")
    plt.savefig(fig_file, dpi=300)


def create_metric(fiber, method, reference, motion=False):
    """Create a QC metric for baseline or motion correction.

    Parameters
    ----------
    fiber : str
        The fiber/ROI identifier.
    method : str
        The preprocessing method used.
    reference : str
        Path to the reference image for this metric.
    motion : bool, optional
        Whether this is a motion correction metric. Defaults to False.

    Returns
    -------
    QCMetric
        The created quality control metric.
    """
    return QCMetric(
        name=f"{'Motion' if motion else 'Baseline'} correction of ROI {fiber} using method '{method}'",
        reference=reference,
        status_history=[
            QCStatus(
                evaluator="Pending review",
                timestamp=dt.now(),
                status=Status.PENDING,
            )
        ],
        value=DropdownMetric(
            options=[
                "Preprocessing successful",
                (
                    "Motion correction failed"
                    if motion
                    else "Baseline correction (dF/F) failed"
                ),
            ],
            status=[
                Status.PASS,
                Status.FAIL,
            ],
        ),
    )


def create_evaluation(method, metrics):
    """Create a QC evaluation for a specific preprocessing method.

    Parameters
    ----------
    method : str
        The preprocessing method being evaluated.
    metrics : list of QCMetric
        The metrics included in this evaluation.

    Returns
    -------
    QCEvaluation
        The created quality control evaluation.
    """
    name = f"Preprocessing using method '{method}'"
    return QCEvaluation(
        name=name,
        modality=Modality.FIB,
        stage=Stage.PROCESSING,
        metrics=metrics,
        allow_failed_metrics=False,
        description=(
            "Review the preprocessing plots to ensure accurate "
            "baseline (dF/F) and motion correction."
        ),
    )


if __name__ == "__main__":
    start_time = dt.now()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_pattern",
        type=str,
        default=r"/data/nwb/*.nwb",
        help="Source pattern to find nwb input files",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="/results/", help="Output directory"
    )
    parser.add_argument(
        "--fiber_path",
        type=str,
        default="/data/fiber_raw_data",
        help="Directory of fiber raw data",
    )
    parser.add_argument(
        "--dff_methods",
        nargs="+",
        default=["poly", "exp", "bright"],
        help=(
            "List of dff methods to run. Available options are:\n"
            "  'poly': Fit with 4th order polynomial using ordinary least squares (OLS)\n"
            "  'exp': Fit with biphasic exponential using OLS\n"
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
    parser.add_argument("--no_qc", action="store_true", help="Skip QC plots.")
    args = parser.parse_args()
    fiber_path = Path(args.fiber_path)

    # Load subject data
    subject_json_path = fiber_path / "subject.json"
    with open(subject_json_path, "r") as f:
        subject_data = json.load(f)

    # Grab the subject_id and times for logging
    subject_id = subject_data.get("subject_id", None)

    # Raise an error if subject_id is None
    if subject_id is None:
        logging.info("No subject_id in subject file")
        raise ValueError("subject_id is missing from the subject_data.")

    # Load data description
    data_description_path = fiber_path / "data_description.json"
    with open(data_description_path, "r") as f:
        data_description = json.load(f)

    asset_name = data_description.get("name", None)

    log.setup_logging(
        "aind-fip-dff",
        subject_id=subject_id,
        asset_name=asset_name,
    )

    # Create the destination directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all files matching the source pattern
    source_paths = glob.glob(args.source_pattern)

    # Copy each matching file to the destination directory
    for source_path in source_paths:
        destination_path = os.path.join(
            args.output_dir, "nwb", os.path.basename(source_path)
        )
        shutil.copytree(source_path, destination_path)
        # Update path to the NWB file within the copied directory
        nwb_file_path = destination_path
        if os.path.isdir(os.path.join(args.fiber_path, "FIP")) or os.path.isdir(
            os.path.join(args.fiber_path, "fib")
        ):
            # Print the path to ensure correctness
            logging.info(f"Processing NWB file: {nwb_file_path}")

            with NWBZarrIO(path=str(nwb_file_path), mode="r+") as io:
                nwb_file = io.read()
                # convert nwb to dataframe
                df_fip = nwb_utils.nwb_to_dataframe(nwb_file)
                # add the session column
                filename = os.path.basename(nwb_file_path)
                if "behavior" in filename:
                    session_name = filename.split(".")[0]
                    session_name = session_name.split("behavior_")[1]
                else:
                    session_name = filename.split(".")[0]
                    session_name = session_name.split("FIP_")[1]

                df_fip.insert(0, "session", session_name)

                # now pass the dataframe through the preprocessing functions
                df_fip_pp = pd.DataFrame()
                df_pp_params = pd.DataFrame()
                coeffs, intercepts = {}, {}
                fiber_numbers = df_fip["fiber_number"].unique()
                channels = df_fip["channel"].unique()
                channels = channels[~pd.isna(channels)]
                for pp_name in args.dff_methods:
                    if pp_name in ["poly", "exp", "bright"]:

                        def process1fiber(
                            fiber_number: str,
                        ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
                            """Preprocess the fiber photometry signal of one ROI (dF/F + motion correction).

                            Parameters
                            ----------
                            fiber_number : str
                                Fiber/ROI number.

                            Returns
                            -------
                            tuple
                                Contains four elements:
                                - df_1fiber : pd.DataFrame
                                    Dataframe with preprocessed fiber photometry signals.
                                - df_pp_params : pd.DataFrame
                                    Dataframe with the parameters of the preprocessing.
                                - coeff : dict
                                    Dictionary mapping channels to regression coefficients for motion correction.
                                - intercept : dict
                                    Dictionary mapping channels to regression intercepts for motion correction.
                            """

                            # dF/F
                            def process1channel(channel):
                                df_fip_iter = df_fip[
                                    (df_fip["fiber_number"] == fiber_number)
                                    & (df_fip["channel"] == channel)
                                ].copy()

                                NM_values = df_fip_iter["signal"].values
                                NM_preprocessed, NM_fitting_params, NM_fit = (
                                    chunk_processing(NM_values, method=pp_name)
                                )
                                df_fip_iter.loc[:, "dFF"] = NM_preprocessed
                                df_fip_iter.loc[:, "preprocess"] = pp_name
                                df_fip_iter.loc[:, "F0"] = NM_fit

                                NM_fitting_params.update(
                                    {
                                        "preprocess": pp_name,
                                        "channel": channel,
                                        "fiber_number": fiber_number,
                                    }
                                )
                                df_pp_params_ses = pd.DataFrame(
                                    NM_fitting_params, index=[0]
                                )
                                return df_fip_iter, df_pp_params_ses

                            with ThreadPool(len(channels)) as tp:
                                res = tp.map(process1channel, channels)
                            df_1fiber = pd.concat(
                                [r[0] for r in res], ignore_index=True
                            )
                            df_pp_params = pd.concat([r[1] for r in res])

                            # motion correction
                            df_dff_iter = (
                                pd.DataFrame(  # convert to #frames x #channels
                                    np.column_stack(
                                        [
                                            df_1fiber[df_1fiber["channel"] == c][
                                                "dFF"
                                            ].values
                                            for c in channels
                                        ]
                                    ),
                                    columns=channels,
                                )
                            )
                            # run motion correction
                            df_mc_iter, df_filt_iter, coeff, intercept = motion_correct(
                                df_dff_iter,
                                cutoff_freq_motion=args.cutoff_freq_motion,
                                cutoff_freq_noise=args.cutoff_freq_noise,
                            )
                            # convert back to a table with columns channel and signal
                            df_1fiber["motion_corrected"] = df_mc_iter.melt(
                                var_name="channel", value_name="motion_corrected"
                            ).motion_corrected
                            df_1fiber["filtered"] = df_filt_iter.melt(
                                var_name="channel", value_name="filtered"
                            ).filtered
                            return df_1fiber, df_pp_params, coeff, intercept

                        with Pool(len(fiber_numbers)) as pool:
                            res = pool.map(process1fiber, fiber_numbers)
                        df_fip_pp = pd.concat([df_fip_pp] + [r[0] for r in res])
                        df_pp_params = pd.concat([df_pp_params] + [r[1] for r in res])
                        coeffs[pp_name] = [r[2] for r in res]
                        intercepts[pp_name] = [r[3] for r in res]

                methods = df_fip_pp.preprocess.unique()
                for method in methods:
                    for signal, suffix in (
                        ("dFF", f"_dff-{method}"),
                        ("motion_corrected", f"_dff-{method}_mc-iso-IRLS"),
                    ):
                        # format the processed traces as dict for conversion to nwb
                        dict_from_df = nwb_utils.split_fip_traces(
                            df_fip_pp[df_fip_pp.preprocess == method], signal=signal
                        )
                        # and add them to the original nwb
                        nwb_file = nwb_utils.attach_dict_fip(
                            nwb_file, dict_from_df, suffix
                        )

                io.write(nwb_file)
                logging.info(
                    "Successfully updated the nwb with preprocessed data"
                    f" using methods {methods}"
                )
                if not args.no_qc:
                    channels = df_fip_pp["channel"].unique()
                    fibers = df_fip_pp["fiber_number"].unique()

                    def foo(a):
                        fiber, method = a
                        return plot_dff(
                            df_fip_pp,
                            fiber,
                            channels,
                            method,
                            os.path.join(args.output_dir, "dff-qc"),
                        )

                    def bar(a):
                        fiber, method = a
                        return plot_motion_correction(
                            df_fip_pp,
                            fiber,
                            channels,
                            method,
                            os.path.join(args.output_dir, "dff-qc"),
                            coeffs,
                            intercepts,
                            args.cutoff_freq_motion,
                            args.cutoff_freq_noise,
                        )

                    with Pool(len(fibers)) as pool:
                        pool.map(foo, itertools.product(fibers, methods))
                        pool.map(bar, itertools.product(fibers, methods))
                    evaluations = []
                    for method in methods:
                        metrics = []
                        for fiber in fibers:
                            metrics.append(
                                create_metric(
                                    fiber, method, f"dff-qc/ROI{fiber}_dff-{method}.png"
                                )
                            )
                            metrics.append(
                                create_metric(
                                    fiber,
                                    method,
                                    f"dff-qc/ROI{fiber}_dff-{method}_mc-iso-IRLS.png",
                                    True,
                                )
                            )
                        evaluations.append(create_evaluation(method, metrics))
                    # Create QC object and save
                    qc = QualityControl(evaluations=evaluations)
                    qc.write_standard_file(
                        output_directory=os.path.join(args.output_dir, "dff-qc")
                    )
            process_name = (
                ProcessName.DF_F_ESTIMATION
            )  # append DataProcess to processing.json

        else:
            logging.info("NO Fiber but only Behavior data, preprocessing not needed")
            os.mkdir(os.path.join(args.output_dir, "dff-qc"))
            qc_file_path = Path(args.output_dir) / "dff-qc" / "no_fip_to_qc.txt"
            # Create an empty file
            with open(qc_file_path, "w") as file:
                file.write(
                    "FIP data files are missing. This may be a behavior session."
                )
            process_name = None  # update processing.json w/o appending DataProcess

        write_output_metadata(
            metadata=vars(args),
            json_dir=args.fiber_path,
            process_name=process_name,
            input_fp=source_path,
            output_fp=os.path.join(args.output_dir, "nwb"),
            start_date_time=start_time,
        )

    src_directory = args.fiber_path
    # Iterate over all .json files in the source directory
    if os.path.exists(src_directory):
        for filename in ["subject.json", "procedures.json", "session.json", "rig.json"]:
            src_file = os.path.join(src_directory, filename)
            if os.path.exists(src_file):
                dest_file = os.path.join(args.output_dir, filename)
                # Move the file
                shutil.copy2(src_file, dest_file)
                logging.info(f"Moved: {src_file} to {dest_file}")
