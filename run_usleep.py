# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:12:51 2021

@author: Simon
"""
import os
import ospath
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import mne
import pyedflib
from pyedflib import highlevel
import scipy
import seaborn as sns
from scipy.io import loadmat
import pandas as pd
import sleep_utils
import warnings;warnings.filterwarnings("ignore")
from usleep_api import USleepAPI
import itertools
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm
from getpass import getpass


#%%
def predict_usleep(edf_file, ch_groups, saveto=None):
    
    # Create an API token at https://sleep.ai.ku.dk.
    api_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2NjQzMTE4MTEsImlhdCI6MTY2NDI2ODYxMSwibmJmIjoxNjY0MjY4NjExLCJpZGVudGl0eSI6IjUzMTU5NGUwMTc0MSJ9.y7b_X1_pY1BupoLSuOOdqBeEgqW24u-OQbOktE72D7I"  # Insert here

    # Create an API object and (optionally) a new session.
    try:
        api = USleepAPI(api_token=api_token)
    except:
        raise ConnectionRefusedError('Probably wrong API token. Please get new API token from https://sleep.ai.ku.dk/sleep_stager#')
    session = api.new_session(session_name=os.path.basename(edf_file))

    # See a list of valid models and set which model to use
    session.set_model('U-Sleep v1.0')

    # Upload a local file (usually .edf format)
    print(f'uploading {edf_file}')
    session.upload_file(edf_file)

    # Start the prediction on two channel groups:
    #   1: EEG Fpz-Cz + EOG horizontal
    #   2: EEG Pz-Oz + EOG horizontal
    # Using 30 second windows (note: U-Slep v1.0 uses 128 Hz re-sampled signals)

    
    session.predict(data_per_prediction=128*30, channel_groups=ch_groups)
    
    # Wait for the job to finish or stream to the log output
    # session.stream_prediction_log()
    print('waiting for prediction')
    success = session.wait_for_completion()
    
    if success:
        # Fetch hypnogram
        hypno = session.get_hypnogram()['hypnogram']
        if saveto:
            sleep_utils.write_hypno(hypno, saveto, mode='csv', 
                                seconds_per_annotation=1, overwrite=True)
        # Download hypnogram file
        # session.download_hypnogram(out_path="./hypnogram", file_type="tsv")
    else:
        raise Exception(f"Prediction failed.\n\n{hypno}")

    # Delete session (i.e., uploaded file, prediction and logs)
    session.delete_session()
    return hypno


#%% plot individual confmats
plt.maximize=False
if __name__=='__main__':
    
    files = ospath.choose_files('Z:/Emo_React/Raw_data/', exts=['vhdr'])
    
    
    for file in tqdm(files, desc='Creating hypnograms and spectrograms'):
        
        hypno_file = file + '.txt'
        raw = mne.io.read_raw_brainvision(file, preload=True, 
                                          eog=['HEOG', 'VEOG'],
                                          misc=['EMG', 'ECG'])
        if not os.path.exists(hypno_file):       
            if not file.endswith('.edf'):
                edf_file = f'{file}.edf'
                if not ospath.exists(edf_file):
                    include = ['C3', 'C4', 'F7', 'F8', 'Fz', 
                               'Pz', 'O2', 'O1', 'VEOG', 'HEOG']
                    drop = [ch for ch in raw.ch_names if ch not in include]
                    raw.drop_channels(drop)
                    raw.resample(100, n_jobs=-1)
                
                    raw = raw.filter(0.1, 30, verbose=False)   
                    sleep_utils.write_mne_edf(raw, edf_file, overwrite=True)
            eeg_chs = ['C3', 'C4', 'F7', 'F8', 'Fz', 'Pz', 'O2', 'O1']
            eog_chs = ['VEOG', 'HEOG']
            ch_groups = list(itertools.product(eeg_chs, eog_chs))
            hypno = predict_usleep(edf_file, ch_groups=ch_groups, saveto=hypno_file)       
            
        hypno = sleep_utils.read_hypno(hypno_file)    
        sleep_utils.hypno_summary(hypno[3:])
        fig, axs = plt.subplots(2, 1, figsize=[8,6])
        
        axs = axs.flatten()
        sleep_utils.plot_hypnogram(hypno, ax=axs[0], title=f'{ospath.basename(file)}')
    
        sleep_utils.specgram_multitaper(raw.get_data(2).squeeze()*1e6, 
                                        raw.info['sfreq'], ax=axs[1], ufreq=30,
                                        title=f'ch: {raw.ch_names[2]}')
        plt.tight_layout()
        plt.pause(0.01)
        fig.savefig(f'{file}.png')
