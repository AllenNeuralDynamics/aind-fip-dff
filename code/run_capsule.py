import glob
import os
import shutil

from hdmf_zarr import NWBZarrIO

import utils.nwb_dict_utils as nwb_utils
from utils.preprocess import batch_processing

"""
This capsule should take in an NWB file, 
check the number of subjects (confirm this),
check the number of channels,
check the number of fibers,
then preprocess the arrays with the dF_F signal
"""


source_pattern = r"/data/nwb/*.nwb"
destination_dir = "/results/nwb/"
fiber_path = '/data/fiber_raw_data' 


# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Find all files matching the source pattern
source_paths = glob.glob(source_pattern)

# Copy each matching file to the destination directory
for source_path in source_paths:
    destination_path = os.path.join(destination_dir, os.path.basename(source_path))
    shutil.copytree(source_path, destination_path)
    # Update path to the NWB file within the copied directory
    nwb_file_path = destination_path
    if os.path.isdir(os.path.join(fiber_path, "FIP")) or os.path.isdir(os.path.join(fiber_path, "fib")):
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
            df_fip_pp_nwb, df_PP_params = batch_processing(df_fip=df_from_nwb)

            methods = df_fip_pp_nwb.preprocess.unique()
            for method in methods:
                # format the processed traces as dict to allow for proper conversion to nwb
                dict_from_df = nwb_utils.split_fip_traces(
                    df_fip_pp_nwb[df_fip_pp_nwb.preprocess == method]
                )
                # and add them to the original nwb
                nwb_file = nwb_utils.attach_dict_fip(nwb_file, dict_from_df, method)

            io.write(nwb_file)
            print(
                f"Successfully updated the nwb with preprocessed data using methods {methods}"
            )
    else:
        print("No Behavior data, preproccesing unneeded") 

src_directory = "/data/fiber_raw_data/"
dest_directory = "/results/"

# Iterate over all files in the source directory
if os.path.exists(src_directory):
    for filename in os.listdir(src_directory):
        if filename.endswith(".json"):
            # Construct full file path
            src_file = os.path.join(src_directory, filename)
            dest_file = os.path.join(dest_directory, filename)

            # Move the file
            shutil.copy2(src_file, dest_file)
            print(f"Moved: {src_file} to {dest_file}")

