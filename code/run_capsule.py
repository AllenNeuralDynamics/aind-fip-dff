import argparse
import glob
import json
import os
import shutil
from datetime import datetime as dt
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from aind_data_schema.core.processing import (
    DataProcess,
    PipelineProcess,
    Processing,
    ProcessName,
)
from aind_data_schema.core.data_description import DerivedDataDescription
from hdmf_zarr import NWBZarrIO

from aind_metadata_upgrader.data_description_upgrade import DataDescriptionUpgrade

import utils.nwb_dict_utils as nwb_utils
from utils.preprocess import batch_processing

"""
This capsule should take in an NWB file,
check the number of subjects (confirm this),
check the number of channels,
check the number of fibers,
then preprocess the arrays with the dF_F signal
"""


def write_output_metadata(
    metadata: dict,
    json_dir: str,
    process_name: str,
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
    with open(Path(json_dir) / "processing.json", "r") as f:
        proc_data = json.load(f)
    processing = Processing(**proc_data)
    p = processing.processing_pipeline
    p.data_processes.append(
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
    )
    p.processor_full_name = "Fiberphotometry Processing Pipeline"
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
            data_description=new_dd,
            process_name="processed"

        derived_dd.write_standard_file(output_directory=Path(output_fp).parent)



def plot_raw_dff_mc(nwb_file, fiber, channels, method, fig_path="/results/plots/"):
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
        The path where the generated plot will be saved. Defaults to "/results/plots/".
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 4), sharex=True)
    for i, suffix in enumerate(("", f"_dff-{method}", f"_preprocessed-{method}")):
        for ch in sorted(channels):
            trace = nwb_file.acquisition[ch + f"_{fiber}{suffix}"]
            t, d = trace.timestamps[:], trace.data[:]
            ax[i].plot(
                t,
                d * 100 if i else d,
                label=ch,
                alpha=0.8,
                # more color-blind-friendly g, b, and r
                c={"G": "#009E73", "Iso": "#0072B2", "R": "#D55E00"}.get(ch, f"C{i}"),
            )
        if i == 0:
            ax[i].legend()
        ax[i].set_title(
            (
                "Raw",
                r"$\Delta$F/F ('dff')",
                r"$\Delta$F/F + motion-correction ('preprocessed')",
            )[i]
        )
        ax[i].set_ylabel(("F [a.u.]", r"$\Delta$F/F [%]", r"$\Delta$F/F [%]")[i])
    ax[i].set_xlim(t[0] - (t[-1] - t[0]) / 100, t[-1] + (t[-1] - t[0]) / 100)
    plt.suptitle(f"Method: {method},  Fiber: {fiber}", y=1)
    plt.xlabel("Time [" + trace.unit + "]")
    plt.tight_layout(pad=0.2)
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, f"Fiber{fiber}_{method}.png"))


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
            "  'bright': Robust fit with [Biphasic exponential decay (bleaching)] x "
            "[Increasing saturating exponential (brightening)] using iteratively "
            "reweighted least squares (IRLS)"
        ),
    )
    parser.add_argument("--no_qc", action="store_true", help="Skip QC plots.")
    args = parser.parse_args()

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
            print(f"Processing NWB file: {nwb_file_path}")

            with NWBZarrIO(path=str(nwb_file_path), mode="r+") as io:
                nwb_file = io.read()
                # convert nwb to dataframe
                df_from_nwb = nwb_utils.nwb_to_dataframe(nwb_file)
                # add the session column
                filename = os.path.basename(nwb_file_path)
                if "behavior" in filename:
                    session_name = filename.split(".")[0]
                    session_name = session_name.split("behavior_")[1]
                else:
                    session_name = filename.split(".")[0]
                    session_name = session_name.split("FIP_")[1]

                df_from_nwb.insert(0, "session", session_name)

                # now pass the dataframe through the preprocessing function:
                df_fip_pp_nwb, df_PP_params, df_fip_mc = batch_processing(
                    df_from_nwb, args.dff_methods
                )

                methods = df_fip_pp_nwb.preprocess.unique()
                for method in methods:
                    for df, suffix in (
                        (df_fip_pp_nwb, "dff"),
                        (df_fip_mc, "preprocessed"),
                    ):
                        # format the processed traces as dict for conversion to nwb
                        dict_from_df = nwb_utils.split_fip_traces(
                            df[df.preprocess == method]
                        )
                        # and add them to the original nwb
                        nwb_file = nwb_utils.attach_dict_fip(
                            nwb_file, dict_from_df, f"_{suffix}-{method}"
                        )

                io.write(nwb_file)
                print(
                    "Successfully updated the nwb with preprocessed data"
                    f" using methods {methods}"
                )
                if not args.no_qc:
                    for method in methods:
                        keys_split = [
                            k.split("_")
                            for k in nwb_file.acquisition.keys()
                            if k.endswith(method)
                        ]
                        channels = sorted(set([k[0] for k in keys_split]))
                        fibers = sorted(set([k[1] for k in keys_split]))
                        for fiber in fibers:
                            plot_raw_dff_mc(
                                nwb_file,
                                fiber,
                                channels,
                                method,
                                os.path.join(args.output_dir, "plots"),
                            )
        else:
            print("NO Fiber but only Behavior data, preprocessing not needed")

    src_directory = args.fiber_path

    # Iterate over all .json files in the source directory
    if os.path.exists(src_directory):
        for filename in [ 'subject.json', 'procedures.json', 'session.json' ]
            src_file = os.path.join(src_directory, filename)
            if os.path.exists(src_file):
                dest_file = os.path.join(args.output_dir, filename)

                # Move the file
                shutil.copy2(src_file, dest_file)
                print(f"Moved: {src_file} to {dest_file}")

    # Append to processing.json
    write_output_metadata(
        vars(args),
        src_directory,
        ProcessName.DF_F_ESTIMATION,
        source_path,
        os.path.join(args.output_dir, "nwb"),
        start_time,
    )
