import numpy as np
import pandas as pd
import pynwb


def is_numeric(obj):
    """
    Check if an array or object is numeric
    """
    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)


def attach_dict_fip(nwb, dict_fip, suffix):
    for neural_stream in dict_fip:
        ts = pynwb.TimeSeries(
            name=neural_stream + suffix,
            data=dict_fip[neural_stream][1],
            unit="s",
            timestamps=dict_fip[neural_stream][0],
        )
        nwb.add_acquisition(ts)
    return nwb


def split_fip_traces(df_fip, split_by=["channel", "fiber_number"]):
    """
    split_neural_traces takes in a dataframe with fiber photometry data series and splits it into
    individual traces for each channel and each channel number.

    Parameters
    ----------
    df_fip: DataFrame
        Time series Dataframe with columns signal, time, channel, and channel number.
        Has the signals for variations of channel and channel numbers are mixed together

    Returns
    ----------
    dict_fip: dictionary
        Dictionary that takes in channel name and channel number as key, and time series and signal
        bundled together as a 2x<TIMESERIES_LEN> as the value

    """
    dict_fip = {}
    groups = df_fip.groupby(split_by)
    for group_name, df_group in list(groups):
        df_group = df_group.sort_values("time_fip")
        # Transforms integers in the name into int type strings. This is needed because nan in the dataframe entries automatically transform entire columns into float type
        group_name_string = [
            str(int(x)) if (is_numeric(x) and x == int(x)) else str(x)
            for x in group_name
        ]
        group_string = "_".join(group_name_string)
        dict_fip[group_string] = np.vstack(
            [df_group.time_fip.values, df_group.signal.values]
        )
    return dict_fip


def nwb_to_dataframe(nwbfile):
    """
    Reads time series data from an NWB file, converts it into a dictionary,
    including only keys that contain 'R_', 'G_', or 'Iso_', and stores only the 'data' part.
    Also adds a single 'timestamps' field from the first matching key and converts the dictionary to a pandas DataFrame.

    Parameters:
    nwbfile: NWB zarr file including aligned times

    Returns:
    pd.DataFrame: A pandas DataFrame with the time series data and timestamps.
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
