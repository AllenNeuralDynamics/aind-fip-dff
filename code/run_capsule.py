
#%%
import os
import pandas as pd
import numpy as np
import itertools
#from pynwb import NWBHDF5IO
from pathlib import Path
import glob
#import util_download as util_dl
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
nwb_original_dir = '../data/'
nwb_results_dir = '../results/nwb/'


#%% 1. copy the original nwb directory from the origin to the destination directory
nwb_files = glob.glob(os.path.join(nwb_original_dir, '*.nwb'))

#%%
# assuming there's multiple nwb files: copy all of them to the dest folder
for nwb_file in nwb_files:
    shutil.copytree(nwb_file, nwb_results_dir)

#%%
nwb_files_results = glob.glob(os.path.join(nwb_results_dir, '*.nwb'))


#%% 2. Test the new NWBfunction: convert nwb to dataframe

try:
    nwb_path = "/Users/brian.gitahi/Desktop/AIND/FIP/Git/aind-fip-dff/data/655100_2023-03-15_11-16-51.nwb"
    df_from_nwb = nwp.nwb_to_dataframe(nwb_path)
    print("loaded nwb file from local path")
except Exception:
    nwb_path = "../data/655100_2023-03-15_11-16-51.nwb"
    df_from_nwb = nwp.nwb_to_dataframe(nwb_path)
    print("loaded nwb file from alternate path")

#%%
# Use regex to find the session name
match = re.search(r'(\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.nwb', nwb_path)
if match:
    session_name = match.group(1)
    print("Session name:", session_name)
else:
    print("Session name not found")
#%%
# add the session column
df_from_nwb.insert(0, 'session', session_name)

#%% now pass the dataframe through the preprocessing function:
df_fip_pp_nwb, df_PP_params = nwp.batch_processing(df_fip=df_from_nwb)


#%% Step to allow for proper conversion to nwb 
df_from_nwb_s = nwp.split_fip_traces(df_fip_pp_nwb)

#%%  then pass the preprocessed data back to the nwb file
try:
    processed_nwb = nwp.attach_dict_fip("/Users/brian.gitahi/Desktop/AIND/FIP/Git/aind-fip-dff/655100_2023-03-15_11-16-51.nwb",df_from_nwb_s)
except Exception:
    processed_nwb = nwp.attach_dict_fip("../data/655100_2023-03-15_11-16-51.nwb",df_from_nwb_s)



#%% Create a new NWB file and copy data from the processed NWB
new_nwb = NWBFile(session_description='Processed session',
                  identifier='NWB123',
                  session_start_time=datetime.now().astimezone())

# Copy data from processed_nwb to new_nwb
for item in processed_nwb.acquisition.values():
    new_nwb.add_acquisition(item)


#%%
# POST THE NWB TO THE RESULTS HERE -- fix this
# Define the new file path
results_path= "../results/"


# Write the updated NWB file to the new location
with NWBZarrIO(results_path, mode='w') as io:
    io.write(new_nwb)



