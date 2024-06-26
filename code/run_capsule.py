
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

import new_preprocess as nwp


# get the following from the nwb
#%%
subject_id = "652738"
session_id = "2021-06-01_11-00-00"
N_assets_per_subject = 1
fibers_per_file = 2

# Main loop 

#%%
# New chunk
AnalDir = '../trial_data/700708_2024-06-14_08-38-31/'
# (add system later)
filenames = []
for name in ['FIP_DataG', 'FIP_DataR', 'FIP_DataIso']:
    if bool(glob.glob(AnalDir + os.sep +  "**" + os.sep + name +'*',recursive=True)) == True:
        filenames.extend(glob.glob(AnalDir + os.sep + "**" + os.sep + name +'*', recursive=True)) 

#%%
# create the df for input to the new function, chunk processing
df_fip_ses = nwp.load_Homebrew_fip_data(filenames=filenames)
#%%
df_fip_pp, df_PP_params = nwp.batch_processing(df_fip=df_fip_ses)


#%%
df_fip = pd.DataFrame()
df_pp_params = pd.DataFrame()
df_logging = pd.DataFrame()


#%% test the NWBfunction
df_from_nwb = nwp.nwb_to_dataframe("/Users/brian.gitahi/Desktop/AIND/FIP/Git/aind-fip-dff/655100_2023-03-15_11-16-51.nwb")


    #%%








        

    # 