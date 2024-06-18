import os
import pandas as pd
import numpy as np
import itertools
from pynwb import NWBHDF5IO
from pathlib2 import Path
import glob
import util_download as util_dl
import re
import argparse
import os
import numpy as  np
from scipy.signal import medfilt, butter, filtfilt
from scipy.optimize import curve_fit
import glob
import itertools
import pandas as pd

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


data_folder = "../data/"
results_folder = "../results/"
scratch_folder = "../results/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='all', help='Method to preprocess the data. Options: all, poly, exp, double_exp')
    parser.add_argument("--input_nwb", type=str, help='Path to the NWB file to preprocess')
    args = parser.parse_args()

    method_to_preprocess = args.method

    #  Search all fiber photometry datasets and define a list of assets
    data_assets = search_all_assets('FIP*')

    # The following will be extracted from NWB file. 
    subject_id = "652738"
    session_id = "2021-06-01_11-00-00"
    N_assets_per_subject = 1
    fibers_per_file = 2

    #  Main loop
    df_fip = pd.DataFrame()
    df_pp_params = pd.DataFrame()
    df_logging = pd.DataFrame()

    try:
        print(subject_id)

        # Download data assets related to subject
        folder = '../scratch/'+subject_id+'/'
        Path(folder).mkdir(parents=True, exist_ok=True)
        download_assets(query_asset='FIP_'+subject_id+'*', query_asset_files='.csv', download_folder=folder, max_assets_to_download=N_assets_per_subject)
        download_assets(query_asset='FIP_'+subject_id+'*', query_asset_files='TTL_', download_folder=folder, max_assets_to_download=N_assets_per_subject)
        download_assets(query_asset='FIP_'+subject_id+'*', query_asset_files='.json', download_folder=folder, max_assets_to_download=N_assets_per_subject)

        folders_sessions = glob.glob(folder+'*')    
        for i_folder, AnalDir in enumerate(folders_sessions[:1]):
            print(subject_id + ' ' + AnalDir)
            subject_id, session_date, session_time = re.search(r"\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", AnalDir).group().split('_')            
            session_name = subject_id+'_'+session_date+'_'+session_time
            error = 'nan'

            #% Detect Fiber photometry and behavioral systems 
            fiber_photometry_system, filenames = detect_fiber_photometry_system(AnalDir)
            behavior_system = detect_behavior_recording_system(AnalDir)

            if fiber_photometry_system is None or behavior_system is None:
                error = 'system_detection'
                df_logging = pd.concat([df_logging, pd.DataFrame({'subject_id':subject_id, 'session':session_name, 'date':session_date, 'behavior_system':behavior_system, 'fiber_photometry_system':fiber_photometry_system, 'error':error}, index=[0])])
                continue

            #% Load FIP data and create fip dataframe 
            if fiber_photometry_system == 'NPM':
                df_fip_ses = load_NPM_fip_data(filenames, fibers_per_file=fibers_per_file)
                df_fip_ses_cleaned = clean_timestamps_NPM(df_fip_ses)            

            elif fiber_photometry_system == 'Homebrew':
                df_fip_ses_cleaned = load_Homebrew_fip_data(filenames, fibers_per_file=fibers_per_file)
            
            if len(df_fip_ses_cleaned) == 0:
                print('Could not processs session because of loading the data:'+session_name)
                error = 'loading'
                df_logging = pd.concat([df_logging, pd.DataFrame({'subject_id':subject_id, 'session':session_name, 'date':session_date, 'behavior_system':behavior_system, 'fiber_photometry_system':fiber_photometry_system, 'error':error}, index=[0])])
                continue

            #% Align FIP to behavioral system clock
            if behavior_system == 'bpod':
                timestamps_bitcodes, frame_number_bitcodes, bitcodes = compute_timestamps_bitcodes(AnalDir)
                frame_number_convertor_FIP_to_bpod = alignment_fip_time_to_bpod(AnalDir, folder_nwb='/data/foraging_nwb_bpod/')
                if frame_number_convertor_FIP_to_bpod is np.nan:
                    print('Could not processs session because of bitcodes alignment:'+session_name)
                    error = 'nwb/bitcodes'
                    df_logging = pd.concat([df_logging, pd.DataFrame({'subject_id':subject_id, 'session':session_name, 'date':session_date, 'behavior_system':behavior_system, 'fiber_photometry_system':fiber_photometry_system, 'error':error}, index=[0])])
                    continue
                df_fip_ses_cleaned.loc[:, 'time'] = frame_number_convertor_FIP_to_bpod(df_fip_ses_cleaned['frame_number'].values)           
                df_fip_ses_aligned = df_fip_ses_cleaned.loc[:,['session', 'frame_number', 'time', 'time_fip','signal', 'channel', 'fiber_number', 'excitation', 'camera', 'system']]
                
            elif behavior_system == 'bonsai':
                timestamps_Harp_cleaned = clean_timestamps_Harp(AnalDir)
                df_fip_ses_aligned = alignment_fip_time_to_harp(df_fip_ses_cleaned, timestamps_Harp_cleaned)            
            
            #% Preprocessing of FIP data
            df_fip_ses_aligned.loc[:,'preprocess'] = 'None'        
            df_fip_ses_pp, df_pp_params_ses = batch_processing(df_fip_ses_aligned, methods=['poly', 'exp'])
            df_fip_ses = pd.concat([df_fip_ses_aligned, df_fip_ses_pp])     
            df_pp_params = pd.concat([df_pp_params, df_pp_params_ses])   

            #% Adding fip to nwb
            dict_fip = split_fip_traces(df_fip_ses, split_by=['channel', 'fiber_number', 'preprocess'])        
            nwb, src_io = attach_dict_fip(AnalDir, folder_nwb='/data/foraging_nwb_'+behavior_system+'/', dict_fip=dict_fip)

            #% Storing new nwb
            save_dirname = '../results/'+AnalDir.strip('../scratch/')        
            if not os.path.exists(save_dirname):
                Path(save_dirname).mkdir(parents=True, exist_ok=True)         
            save_filename_nwb = save_dirname+os.sep+ session_name +'.nwb'        
            with NWBHDF5IO(save_filename_nwb, mode='w') as export_io:
                export_io.export(src_io=src_io, nwbfile=nwb)
            src_io.close()     

            #% Storing of dataframes
            save_filename_df_files = save_dirname+os.sep+ session_name +'_df'        
            df_fip_ses.to_pickle(save_filename_df_files+'_fip.pkl')
            df_pp_params_ses.to_pickle(save_filename_df_files+'_pp.pkl')  
            
            df_logging = pd.concat([df_logging, pd.DataFrame({'subject_id':subject_id, 'session':session_name, 'date':session_date, 'behavior_system':behavior_system, 'fiber_photometry_system':fiber_photometry_system, 'error':error}, index=[0])])
    except:
        df_logging = pd.concat([df_logging, pd.DataFrame({'subject_id':subject_id, 'session':session_name, 'date':session_date, 'behavior_system':np.nan, 'fiber_photometry_system':np.nan, 'error':'not detected error'}, index=[0])])
    
    df_logging.to_pickle('../results/df_logging.pkl')

        

    # 