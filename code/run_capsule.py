
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


from pynwb import NWBHDF5IO
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


#%% 1. Create preprocessed df using old functions for comparison with new version

AnalDir = '../trial_data/700708_2024-06-14_08-38-31/'

# define the files with the traces from each of the channels
filenames = []
for name in ['FIP_DataG', 'FIP_DataR', 'FIP_DataIso']:
    if bool(glob.glob(AnalDir + os.sep +  "**" + os.sep + name +'*',recursive=True)) == True:
        filenames.extend(glob.glob(AnalDir + os.sep + "**" + os.sep + name +'*', recursive=True)) 

# create the df for input to the batch preprocessing function and then preprocess it
df_fip_ses = nwp.load_Homebrew_fip_data(filenames=filenames)
df_fip_pp, df_PP_params = nwp.batch_processing(df_fip=df_fip_ses)



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
    processed_nwb = nwp.attach_dict_fip("/Users/brian.gitahi/Desktop/AIND/FIP/Git/aind-fip-dff/data/655100_2023-03-15_11-16-51.nwb",df_from_nwb_s)
except Exception:
    processed_nwb = nwp.attach_dict_fip("../data/655100_2023-03-15_11-16-51.nwb",df_from_nwb_s)


#%%
# POST THE NWB TO THE RESULTS HERE -- fix this
# Define the new file path
results_path= "../results/655100_2023-03-15_11-16-51_processed.nwb"


# Write the updated NWB file to the new location
with NWBZarrIO(results_path, mode='w') as io:
    io.write(processed_nwb)

