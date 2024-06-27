
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
    #join directory and the name of the nwb file then copy the origin --> destination
    destination_path = os.path.join(nwb_results_dir, os.path.basename(nwb_file))
    shutil.copytree(nwb_file, destination_path)

    #%% convert nwb to dataframe
    df_from_nwb = nwp.nwb_to_dataframe(destination_path)

    #%% add the session column
    filename  = os.path.basename(nwb_file)
    session_name = filename.split('.')[0]
    df_from_nwb.insert(0, 'session', session_name)

    #%% now pass the dataframe through the preprocessing function:
    df_fip_pp_nwb, df_PP_params = nwp.batch_processing(df_fip=df_from_nwb)

    #%% Step to allow for proper conversion to nwb 
    df_from_nwb_s = nwp.split_fip_traces(df_fip_pp_nwb)

    #%% format the processed traces and add them to the original nwb
    processed_nwb = nwp.attach_dict_fip(destination_path,df_from_nwb_s)

    #%% then save the updated nwb in the results folder
    with NWBZarrIO(destination_path, mode='r+') as io:
        io.write(processed_nwb)
        print('Succesfully updated the nwb with preprocessed data')




