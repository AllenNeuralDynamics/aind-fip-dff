
#%%
import os
import pandas as pd
import numpy as np
import itertools
#from pynwb import NWBHDF5IO
from pathlib import Path
import glob
import re
import argparse
import os
import numpy as  np
from scipy.signal import medfilt, butter, filtfilt
from scipy.optimize import curve_fit
import glob
import itertools
import pandas as pd
import shutil

from datetime import datetime
from pynwb import NWBHDF5IO, NWBFile
from hdmf_zarr import NWBZarrIO
import utils.new_preprocess as nwp
import utils.nwb_dict_utils as nwb_utils


"""
This capsule should take in an NWB file, 
check the number of subjects (confirm this),
check the number of channels,
check the number of fibers,
then preprocess the arrays with the dF_F signal
"""


source_pattern = r'/data/nwb/*.nwb'  
destination_dir = '/results/nwb/'
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

        with NWBZarrIO(path=str(nwb_file_path), mode='r+') as io:
            nwb_file = io.read()
            #%% convert nwb to dataframe
            df_from_nwb = nwb_utils.nwb_to_dataframe(nwb_file)

            #%% add the session column
            filename  = os.path.basename(nwb_file_path)
            if "behavior" in filename:
                session_name = filename.split('.')[0]
                session_name = session_name.split("behavior_")[1]
            else:
                session_name = filename.split('.')[0]
                session_name = session_name.split("FIP_")[1]

            df_from_nwb.insert(0, 'session', session_name)

            #%% now pass the dataframe through the preprocessing function:
            df_fip_pp_nwb, df_PP_params = nwp.batch_processing_new(df_fip=df_from_nwb)

            #df_fip_pp_nwb, df_PP_params = nwp.batch_processing(df_fip=df_from_nwb)

            #%% Step to allow for proper conversion to nwb 
            df_from_nwb_s = nwb_utils.split_fip_traces(df_fip_pp_nwb)

            #%% format the processed traces and add them to the original nwb
            nwb_file = nwb_utils.attach_dict_fip(nwb_file,df_from_nwb_s)

            io.write(nwb_file)


        src_directory = '/data/fiber_raw_data/'
        dest_directory = '/results/'

        # Iterate over all files in the source directory
        for filename in os.listdir(src_directory):
            if filename.endswith('.json'):
                # Construct full file path
                src_file = os.path.join(src_directory, filename)
                dest_file = os.path.join(dest_directory, filename)

                # Move the file
                shutil.copy2(src_file, dest_file)
                print(f"Moved: {src_file} to {dest_file}")
                print('Succesfully updated the nwb with preprocessed data')




