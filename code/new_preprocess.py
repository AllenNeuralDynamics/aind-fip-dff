import os
import pandas as pd
import numpy as np
import itertools
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

"""
This capsule should take in an NWB file, 
check the number of subjects (confirm this),
check the number of channels,
check the number of fibers,
then preprocess the arrays with the dF_F signal
"""

# Function to create the input to the batch processing function
def load_Homebrew_fip_data(filenames,  fibers_per_file=2):     
    """
    This function loops over the filenames for the channels 
    in the NPM system 'L415', 'L470', 'L560'
    The created dataframe has the following fields:
        - session
        - time
        - signal
        - fiber_number
        - channel
        - excitation
        - camera
        - system
    """
                   
    df_fip = pd.DataFrame()
    # df_data_acquisition = pd.DataFrame()
    save_fip_channels= np.arange(1, fibers_per_file+1)
    for filename in filenames:
        subject_id, session_date, session_time = re.search(r"\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filename).group().split('_')
        session_name = subject_id+'_'+session_date+'_'+session_time       
        header = os.path.basename(filename).split('/')[-1]
        channel = ('_'.join(header.split('_')[:2])).replace('FIP_Data','')        
        try:
            df_fip_file = pd.read_csv(filename, header=None)  #read the CSV file        
        except pd.errors.EmptyDataError:
            continue
        except FileNotFoundError:
            continue
        df_file = pd.DataFrame()
        for col in df_fip_file.columns[save_fip_channels]:
            df_fip_file_renamed = df_fip_file[[0, col]].rename(columns={0:'time_fip', col:'signal'})
            channel_number = int(col)
            df_fip_file_renamed['fiber_number'] = channel_number
            df_fip_file_renamed.loc[:, 'frame_number'] = df_fip_file.index.values
            df_file = pd.concat([df_file, df_fip_file_renamed])
            # df_data_acquisition = pd.concat([df_data_acquisition, pd.DataFrame({'session':ses_idx, 'system':'FIP', channel+str(channel_number):1.,'N_files':len(filenames)}, index=[0])])                                           
        df_file['channel'] = channel            
        camera = {'Iso':'G', 'G':'G', 'R':'R'}[channel]        
        excitation = {'Iso':415, 'G':470, 'R':560}[channel]
        df_file['excitation'] = excitation
        df_file['camera'] = camera
        df_fip = pd.concat([df_fip, df_file], axis=0)     

    if len(df_fip) > 0:       
        df_fip['system'] = 'FIP' 
        df_fip['preprocess'] = 'None'
        df_fip['session'] = subject_id+'_'+session_date+'_'+session_time        
        df_fip_ses = df_fip.loc[:,['session', 'frame_number', 'time_fip',	'signal', 'channel', 'fiber_number', 'excitation', 'camera', 'system', 'preprocess']]
    else:
        df_fip_ses = df_fip
    return df_fip_ses


