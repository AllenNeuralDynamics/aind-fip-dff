
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
import hdmf_zarr.nwb
from hdmf_zarr.nwb import NWBZarrIO
import new_preprocess as nwp


"""
This capsule should take in an NWB file, 
check the number of subjects (confirm this),
check the number of channels,
check the number of fibers,
then preprocess the arrays with the dF_F signal
"""


#%% origin and destination directories for the nwb file
nwb_original_dir = '/data/'
nwb_results_dir = '/results/nwb/'


source_pattern = r'/data/nwb/*.nwb'  
destination_dir = '/results/nwb/'

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

# Print the path to ensure correctness
print(f"Processing NWB file: {nwb_file_path}")

with NWBZarrIO(path=str(nwb_file_path), mode='r+') as io:
    nwb_file = io.read()
    #%% convert nwb to dataframe
    df_from_nwb = nwp.nwb_to_dataframe(nwb_file)
    print(df_from_nwb)

    #%% add the session column
    filename  = os.path.basename(nwb_file_path)
    session_name = filename.split('.')[0]
    df_from_nwb.insert(0, 'session', session_name)

    #%% now pass the dataframe through the preprocessing function:
    df_fip_pp_nwb, df_PP_params = nwp.batch_processing(df_fip=df_from_nwb)

    #%% Step to allow for proper conversion to nwb 
    df_from_nwb_s = nwp.split_fip_traces(df_fip_pp_nwb)

    #%% format the processed traces and add them to the original nwb
    nwb_file = nwp.attach_dict_fip(nwb_file,df_from_nwb_s)

    io.write(nwb_file)
    print('Succesfully updated the nwb with preprocessed data')




