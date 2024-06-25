#%%
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
from pynwb import NWBHDF5IO
from hdmf_zarr.nwb import NWBZarrIO

#%%
"""
This capsule should take in an NWB file, 
check the number of subjects (confirm this),
check the number of channels,
check the number of fibers,
then preprocess the arrays with the dF_F signal
"""

# Preprocessing functions
#---------------------------------------------------------------------------------------------
# removing first few seconds
def tc_crop(tc, nFrame2cut):
    tc_cropped = tc[nFrame2cut:]
    return tc_cropped

# Median filtering to remove electrical artifact.
def tc_medfilt(tc, kernelSize):
    tc_filtered = medfilt(tc, kernel_size=kernelSize)
    return tc_filtered

# Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
def tc_lowcut(tc, sampling_rate):
    b,a = butter(2, 9, btype='low', fs=sampling_rate)
    tc_filtered = filtfilt(b,a, tc)
    return tc_filtered

# fit with polynomial to remove bleaching artifact 
def tc_polyfit(tc, sampling_rate, degree):
    time_seconds = np.arange(len(tc)) /sampling_rate 
    coefs = np.polyfit(time_seconds, tc, deg=degree)
    tc_poly = np.polyval(coefs, time_seconds)
    return tc_poly, coefs

# setting up sliding baseline to calculate dF/F
def tc_slidingbase(tc, sampling_rate):
    b,a = butter(2, 0.0001, btype='low', fs=sampling_rate)
    tc_base = filtfilt(b,a, tc, padtype='even')
    return tc_base

# obtain dF/F using median of values within sliding baseline 
def tc_dFF(tc, tc_base, b_percentile):
    tc_dFoF = tc/tc_base
    sort = np.sort(tc_dFoF)
    b_median = np.median(sort[0:round(len(sort) * b_percentile)])
    tc_dFoF = tc_dFoF - b_median
    return tc_dFoF

# fill in the gap left by cropping out the first few timesteps
def tc_filling(tc, nFrame2cut):
    tc_filled = np.append(np.ones([nFrame2cut,1])*tc[0], tc)
    return tc_filled
    
def tc_expfit(tc, sampling_rate=20):
    # bi-exponential fit
    def func(x, a, b, c, d):
        return a * np.exp(-b * x) + c * np.exp(-d * x)
    
    time_seconds = np.arange(len(tc))/sampling_rate
    popt, pcov = curve_fit(func,time_seconds,tc)
    tc_exp = func(time_seconds, popt[0], popt[1], popt[2], popt[3])
    return tc_exp, popt

# Preprocessing total function
def chunk_processing(tc, method = 'poly', nFrame2cut=100, kernelSize=1, sampling_rate=20, degree=4, b_percentile=0.7):
    """
    Preprocesses the fiber photometry signal.
    Args:
        tc: np.array
            Fiber photometry signal
        method: str
            Method to preprocess the data. Options: poly, exp
        nFrame2cut: int
            Number of frames to crop from the beginning of the signal
        kernelSize: int
            Size of the kernel for median filtering
        sampling_rate: int
            Sampling rate of the signal
        degree: int
            Degree of the polynomial to fit
        b_percentile: float
            Percentile to calculate the baseline
    Returns:
        tc_dFoF: np.array
            Preprocessed fiber photometry signal
        tc_params: dict
            Dictionary with the parameters of the preprocessing    
    """
    tc_cropped = tc_crop(tc, nFrame2cut)
    tc_filtered = medfilt(tc_cropped, kernel_size=kernelSize)
    tc_filtered = tc_lowcut(tc_filtered, sampling_rate)
    
    if method == 'poly':
        tc_fit, tc_coefs = tc_polyfit(tc_filtered, sampling_rate, degree)
    if method == 'exp':
        tc_fit, tc_coefs = tc_expfit(tc_filtered, sampling_rate)           
    tc_estim = tc_filtered - tc_fit # 
    tc_base = tc_slidingbase(tc_filtered, sampling_rate)
    tc_dFoF = tc_dFF(tc_estim, tc_base, b_percentile)    
    tc_dFoF = tc_filling(tc_dFoF, nFrame2cut)    
    tc_params = {i_coef:tc_coefs[i_coef] for i_coef in range(len(tc_coefs))}
    tc_qualitymetrics = {'QC_metric':np.nan}    
    tc_params.update(tc_qualitymetrics)

    return tc_dFoF, tc_params


# Processing Functions
#---------------------------------------------------------------------------------------------
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

# run the total preprocessing on multiple sessions -- future iteration: collect exceptions in a log file


# FUNCTION to convert NWB to df_fip? -- this is the method decided on
# OR FUNCTION to go from NWB to the traces that we need for batch processing function

#---------------------------------------------------------------------------------------------
"""
This function takes in an nwb file and converts it to a 
dataframe to be processed by batch processing
"""

def nwb_to_dataframe(nwb_file_path):
    """
    Reads time series data from an NWB file, converts it into a dictionary,
    including only keys that contain 'R_', 'G_', or 'Iso_', and stores only the 'data' part.
    Also adds a single 'timestamps' field from the first matching key and converts the dictionary to a pandas DataFrame.

    Parameters:
    nwb_file_path (str): The path to the NWB file.

    Returns:
    pd.DataFrame: A pandas DataFrame with the time series data and timestamps.
    """
    # Define the list of required substrings
    required_substrings = ['R_', 'G_', 'Iso_']

    # Open the NWB file
    with NWBHDF5IO(nwb_file_path, 'r') as io:
        nwbfile = io.read()

        data_dict = {}
        timestamps_added = False

        # Iterate over all TimeSeries in the NWB file
        for key, time_series in nwbfile.acquisition.items():
            # Check if the key contains any of the required substrings
            if any(substring in key for substring in required_substrings):
                # Store only the 'data' part of the TimeSeries
                data_dict[time_series.name] = time_series.data[:]
                
                # Add 'timestamps' field from the first matching key
                if not timestamps_added:
                    data_dict['timestamps'] = time_series.timestamps[:]
                    timestamps_added = True

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(data_dict)

        return df

# Example usage
nwb_file_path = 'path_to_your_nwb_file.nwb'
time_series_df = nwb_to_dataframe(nwb_file_path)
print(time_series_df)

#---------------------------------------------------------------------------------------------



def batch_processing(df_fip, methods=['poly', 'exp']):
    df_fip_pp = pd.DataFrame()    
    df_pp_params = pd.DataFrame() 
    
    # df_fip = pd.read_pickle(filename)        
    if len(df_fip) == 0:
        return df_fip, df_pp_params

    sessions = pd.unique(df_fip['session'].values)
    sessions = sessions[~pd.isna(sessions)]
    fiber_numbers = np.unique(df_fip['fiber_number'].values)    
    channels = pd.unique(df_fip['channel']) # ['G', 'R', 'Iso']    
    channels = channels[~pd.isna(channels)]
    for pp_name in methods:     
        if pp_name in ['poly', 'exp']:   
            for i_iter, (channel, fiber_number, session) in enumerate(itertools.product(channels, fiber_numbers, sessions)):            
                df_fip_iter = df_fip[(df_fip['session']==session) & (df_fip['fiber_number']==fiber_number) & (df_fip['channel']==channel)]        
                if len(df_fip_iter) == 0:
                    continue
                
                NM_values = df_fip_iter['signal'].values   
                try:      
                    NM_preprocessed, NM_fitting_params = chunk_processing(NM_values, method=pp_name)
                except:
                    continue                                       
                df_fip_iter.loc[:,'signal'] = NM_preprocessed                            
                df_fip_iter.loc[:,'preprocess'] = pp_name
                df_fip_pp = pd.concat([df_fip_pp, df_fip_iter], axis=0)                    
                
                NM_fitting_params.update({'preprocess':pp_name, 'channel':channel, 'fiber_number':fiber_number, 'session':session})
                df_pp_params_ses = pd.DataFrame(NM_fitting_params, index=[0])
                df_pp_params = pd.concat([df_pp_params, df_pp_params_ses], axis=0)     

                # Below is where Johannes's method is commented out   
                """         if pp_name in ['double_exp']:
                            for i_iter, (channel, fiber_number, session) in enumerate(itertools.product(channels, fiber_numbers, sessions)):            
                                df_fip_iter = df_fip[(df_fip['session']==session) & (df_fip['fiber_number']==fiber_number)]        
                                F = list()
                                for i_channel, channel in enumerate(['iso', 'G', 'R']):
                                    F.append(df_fip_iter[df_fip_iter['channel'] == channel].signal.values.flatten())
                                F = np.vstack(F)
                                dff_mc = preprocess(F)
                                for i_channel, channel in enumerate(['iso', 'G', 'R']):
                                    df_fip_channel = df_fip_iter[df_fip_iter['channel'] == channel]
                                    df_fip_channel.loc[:,'signal'] = dff_mc[i_channel]
                                    df_fip_channel.loc[:,'preprocess'] = pp_name
                                df_fip_pp = pd.concat([df_fip_pp, df_fip_channel], axis=0)                    
                                
                                NM_fitting_params.update({'preprocess':pp_name, 'channel':channel, 'fiber_number':fiber_number, 'session':session})
                                df_pp_params_ses = pd.DataFrame(NM_fitting_params, index=[0])
                                df_pp_params = pd.concat([df_pp_params, df_pp_params_ses], axis=0)    """     


    return df_fip_pp, df_pp_params
#---------------------------------------------------------------------------------------------


# input is nwb file now

# so we want to access traces: 

def batch_processing_new(df_fip, methods=['poly', 'exp']):
    df_fip_pp = pd.DataFrame()    
    df_pp_params = pd.DataFrame() 
    
    # df_fip = pd.read_pickle(filename)        
    if len(df_fip) == 0:
        return df_fip, df_pp_params

    sessions = pd.unique(df_fip['session'].values)
    sessions = sessions[~pd.isna(sessions)]
    fiber_numbers = np.unique(df_fip['fiber_number'].values)    
    channels = pd.unique(df_fip['channel']) # ['G', 'R', 'Iso']    
    channels = channels[~pd.isna(channels)]
    for pp_name in methods:     
        if pp_name in ['poly', 'exp']:   
            for i_iter, (channel, fiber_number, session) in enumerate(itertools.product(channels, fiber_numbers, sessions)):            
                df_fip_iter = df_fip[(df_fip['session']==session) & (df_fip['fiber_number']==fiber_number) & (df_fip['channel']==channel)]        
                if len(df_fip_iter) == 0:
                    continue
                
                NM_values = df_fip_iter['signal'].values   
                try:      
                    NM_preprocessed, NM_fitting_params = chunk_processing(NM_values, method=pp_name)
                except:
                    continue                                       
                df_fip_iter.loc[:,'signal'] = NM_preprocessed                            
                df_fip_iter.loc[:,'preprocess'] = pp_name
                df_fip_pp = pd.concat([df_fip_pp, df_fip_iter], axis=0)                    
                
                NM_fitting_params.update({'preprocess':pp_name, 'channel':channel, 'fiber_number':fiber_number, 'session':session})
                df_pp_params_ses = pd.DataFrame(NM_fitting_params, index=[0])
                df_pp_params = pd.concat([df_pp_params, df_pp_params_ses], axis=0)     

            

    return df_fip_pp, df_pp_params


#%%
def nwb_to_dataframe(nwb_file_path):
    """
    Reads time series data from an NWB file, converts it into a dictionary,
    including only keys that contain 'R_', 'G_', or 'Iso_', and stores only the 'data' part.
    Also adds a single 'timestamps' field from the first matching key and converts the dictionary to a pandas DataFrame.

    Parameters:
    nwb_file_path (str): The path to the NWB file.

    Returns:
    pd.DataFrame: A pandas DataFrame with the time series data and timestamps.
    """
    # Define the list of required substrings
    required_substrings = ['R_', 'G_', 'Iso_']

    # Open the NWB file -- HD5 version
    # with NWBHDF5IO(nwb_file_path, 'r') as io:
    #     nwbfile = io.read()

    with NWBZarrIO(nwb_file_path, 'r') as io:
        nwbfile = io.read()

        data_dict = {}
        timestamps_added = False
        timestamps  = {}

        # Iterate over all TimeSeries in the NWB file
        for key, time_series in nwbfile.acquisition.items():
            # Check if the key contains any of the required substrings
            if any(substring in key for substring in required_substrings):
                # Store only the 'data' part of the TimeSeries
                data_dict[time_series.name] = time_series.data[:]

                timestamps[key] = (time_series.timestamps[:])

        
            print(f"{key} timestamps", timestamps)


        transformed_data = []

        # Transform the data to have a single column for channel names

        print(data_dict)


        for channel, data in data_dict.items():  
            channel, fiber_number = channel.split('_')
            for i in range(len(timestamps[channel + '_'+ fiber_number])):
                transformed_data.append({
                'time_fip': timestamps[channel + '_'+ fiber_number][i],
                'channel': channel,
                'fiber_number': fiber_number,
                'signal': data[i]
            })
            print("data",len(data))

    

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(transformed_data)

        return df

#%%


#%%
df_from_nwb = nwb_to_dataframe("/Users/brian.gitahi/Desktop/AIND/FIP/Git/aind-fip-dff/655100_2023-03-15_11-16-51.nwb")

# %%
