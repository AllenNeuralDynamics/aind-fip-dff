import numpy as np
import pandas as pd
import pynwb


def is_numeric(obj) -> bool:
    """Check if an array or object is numeric.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool
        True if the object is numeric, False otherwise.
    """
    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)


def attach_dict_fip(
    nwb: pynwb.NWBFile, dict_fip: dict[str, list[np.ndarray]], suffix: str
) -> pynwb.NWBFile:
    """Attach a dictionary of fiber photometry data to an NWB file.

    Parameters
    ----------
    nwb : pynwb.NWBFile
        The NWB file to attach the data to.
    dict_fip : dict[str, list[np.ndarray]]
        Dictionary containing fiber photometry data.
    suffix : str
        Suffix to add to the name of each TimeSeries.

    Returns
    -------
    pynwb.NWBFile
        The NWB file with the attached data.
    """
    # Create or retrieve a processing module
    module_name = "fiber_photometry"
    if module_name not in nwb.processing:
        processing_module = pynwb.ProcessingModule(
            name=module_name, description="Fiber photometry data"
        )
        nwb.add_processing_module(processing_module)
    else:
        processing_module = nwb.processing[module_name]

    # Add TimeSeries to the processing module
    for neural_stream in dict_fip:
        ts = pynwb.TimeSeries(
            name=neural_stream + suffix,
            data=dict_fip[neural_stream][1],
            unit="s",
            timestamps=dict_fip[neural_stream][0],
        )
        processing_module.add(ts)

    return nwb


def split_fip_traces(
    df_fip: pd.DataFrame,
    split_by: list[str] = ["channel", "fiber_number"],
    signal: str = "signal",
) -> dict:
    """Split a dataframe with fiber photometry data into individual traces.

    Parameters
    ----------
    df_fip : pd.DataFrame
        Time series DataFrame with columns signal, time, channel, and channel number.
        Contains signals for different channels and channel numbers mixed together.
    split_by : list[str], optional
        Column names to group by. Default is ["channel", "fiber_number"].
    signal : str, optional
        Column name containing the signal values. Default is "signal".

    Returns
    -------
    dict
        Dictionary with keys formed by joining the group values with '_'.
        Values are 2D arrays where the first row contains timestamps and
        the second row contains signal values.
    """
    dict_fip = {}
    groups = df_fip.groupby(split_by)
    for group_name, df_group in list(groups):
        df_group = df_group.sort_values("time_fip")
        # Transforms integers in the name into int type strings.
        # This is needed because nan in the dataframe entries
        # automatically transform entire columns into float type
        group_name_string = [
            str(int(x)) if (is_numeric(x) and x == int(x)) else str(x)
            for x in group_name
        ]
        group_string = "_".join(group_name_string)
        dict_fip[group_string] = np.vstack(
            [df_group.time_fip.values, df_group[signal].values]
        )
    return dict_fip


def nwb_to_dataframe(nwbfile: pynwb.NWBFile) -> pd.DataFrame:
    """Convert NWB file time series data to a pandas DataFrame.

    Reads time series data from an NWB file, extracts data for channels
    containing 'R_', 'G_', or 'Iso_', and organizes it into a structured DataFrame.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        NWB file containing fiber photometry data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time_fip: timestamps
        - channel: channel name (R, G, Iso)
        - fiber_number: fiber/ROI identifier
        - signal: raw signal values
    """
    # Define the list of required substrings
    required_substrings = ["R_", "G_", "Iso_"]

    data_dict = {}
    timestamps = {}

    # Iterate over all TimeSeries in the NWB file
    for key, time_series in nwbfile.acquisition.items():
        # Check if the key contains any of the required substrings
        if any(substring in key for substring in required_substrings):
            # Store only the 'data' part of the TimeSeries
            data_dict[time_series.name] = time_series.data[:]
            timestamps[key] = time_series.timestamps[:]
        transformed_data = []

        # Transform the data to have a single column for channel names
        for channel, data in data_dict.items():
            channel, fiber_number = channel.split("_")
            for i in range(len(timestamps[channel + "_" + fiber_number])):
                transformed_data.append(
                    {
                        "time_fip": timestamps[channel + "_" + fiber_number][i],
                        "channel": channel,
                        "fiber_number": fiber_number,
                        "signal": data[i],
                    }
                )

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(transformed_data)

    return df
