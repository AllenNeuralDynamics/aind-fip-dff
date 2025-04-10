import numpy as np
import pandas as pd
import pynwb
import zarr


def np_to_zarr(array, path="array", store=None, chunks=None, dtype=None):
    """
    Convert a NumPy array to a Zarr array.

    Args:
        array (np.ndarray): The NumPy array to convert.
        path (str, optional): The path within the Zarr store to save the array. Defaults to "array".
        store (zarr.Store, optional): The Zarr store to use. Defaults to an in-memory store.
                                      Use `zarr.DirectoryStore` for persistent storage.
        chunks (tuple, optional): Chunk size for the Zarr array. Defaults to the shape of the array.
        dtype (str or np.dtype, optional): Data type for the Zarr array. Defaults to the dtype of the input array.

    Returns:
        zarr.core.Array: The created Zarr array.
    """
    # Use an in-memory store if no store is provided
    if store is None:
        store = zarr.MemoryStore()

    # Create a Zarr group in the store
    root = zarr.group(store=store)

    # Create the Zarr array
    zarr_array = root.create_dataset(
        path,
        data=array,
        chunks=(
            chunks if chunks else array.shape
        ),  # Default to the full array as one chunk
        dtype=(
            dtype if dtype else array.dtype
        ),  # Default to the dtype of the input array
    )

    return zarr_array


def is_numeric(obj) -> bool:
    """
    Check if an array or object is numeric
    """
    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)


def attach_dict_fip(
    nwb: pynwb.NWBFile, dict_fip: dict[str, list[np.ndarray]], suffix: str
) -> pynwb.NWBFile:
    """
    Attach a dictionary of fiber photometry data to an NWB file.
    Args:
        nwb: pynwb.NWBFile
            The NWB file to attach the data to.
        dict_fip: dict[str, list[np.ndarray]]
            Dictionary containing fiber photometry data.
        suffix: str
            Suffix to add to the name of each TimeSeries.
    Returns:
        pynwb.NWBFile: The NWB file with the attached data.
    """
    nwb.add_processing_module(
        [
            pynwb.TimeSeries(
                name=neural_stream + suffix,
                data=np_to_zarr(
                    dict_fip[neural_stream][1],
                    f"/processing/{neural_stream + suffix}/data",
                ),
                unit="s",
                timestamps=np_to_zarr(
                    dict_fip[neural_stream][0],
                    f"/processing/{neural_stream + suffix}/timestamps",
                ),
            )
            for neural_stream in dict_fip
        ]
    )
    return nwb



# def attach_dict_fip(
#     nwb: pynwb.NWBFile, dict_fip: dict[str, list[np.ndarray]], suffix: str
# ) -> pynwb.NWBFile:
#     """
#     Attach a dictionary of fiber photometry data to an NWB file.

#     Args:
#         nwb: pynwb.NWBFile
#             The NWB file to attach the data to.
#         dict_fip: dict[str, list[np.ndarray]]
#             Dictionary containing fiber photometry data.
#         suffix: str
#             Suffix to add to the name of each TimeSeries.

#     Returns:
#         pynwb.NWBFile: The NWB file with the attached data.
#     """
#     # Create or retrieve a processing module
#     module_name = "fiber_photometry"
#     if module_name not in nwb.processing:
#         processing_module = pynwb.ProcessingModule(
#             name=module_name, description="Fiber photometry data"
#         )
#         nwb.add_processing_module(processing_module)
#     else:
#         processing_module = nwb.processing[module_name]

#     # Add TimeSeries to the processing module
#     for neural_stream in dict_fip:
#         ts = pynwb.TimeSeries(
#             name=neural_stream + suffix,
#             data=dict_fip[neural_stream][1],
#             unit="s",
#             timestamps=dict_fip[neural_stream][0],
#         )
#         processing_module.add(ts)

#     return nwb


def split_fip_traces(
    df_fip: pd.DataFrame, split_by: list[str] = ["channel", "fiber_number"]
) -> dict:
    """
    split_neural_traces takes in a dataframe with fiber photometry data series and splits it into
    individual traces for each channel and each channel number.
    Args:
        df_fip: pd.DataFrame
            Time series Dataframe with columns signal, time, channel, and channel number.
            Has the signals for variations of channel and channel numbers are mixed together
    Returns:
        dict_fip: dict
            Dictionary that takes in channel name and channel number as key, and time series and signal
            bundled together as a 2x<TIMESERIES_LEN> as the value
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
            [df_group.time_fip.values, df_group.signal.values]
        )
    return dict_fip


def nwb_to_dataframe(nwbfile: pynwb.NWBFile) -> pd.DataFrame:
    """
    Reads time series data from an NWB file, converts it into a dictionary,
    including only keys that contain 'R_', 'G_', or 'Iso_', and stores only the 'data' part.
    Also adds a single 'timestamps' field from the first matching key
    and converts the dictionary to a pandas DataFrame.
    Args:
        nwbfile:
            NWB zarr file including aligned times
    Returns: pynwb.NWBFile
        df: pd.DataFrame
            A pandas DataFrame with the time series data and timestamps.
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
