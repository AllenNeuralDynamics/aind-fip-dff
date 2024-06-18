
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

import new_preprocess as nwp

#%%

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
    data_assets = nwp.search_all_assets('FIP*')

    # The following will be extracted from NWB file. 
    subject_id = "652738"
    session_id = "2021-06-01_11-00-00"
    N_assets_per_subject = 1
    fibers_per_file = 2


    #  Main loop 

    #%%
    # New chunk
    AnalDir = '../trial_data/'
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