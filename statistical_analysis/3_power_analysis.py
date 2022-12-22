# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:41:18 2022

@author: Simon
"""
import sleep_utils
import sys; sys.path.append('..')
import ospath
import pandas as pd
import seaborn as sns
import settings
from settings import stage_map
import data_loading
import yasa
from tqdm import tqdm
from scipy.stats import ttest_rel
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
sns.set(font_scale=1.2)

# check which data is loadable
folders_subj = ospath.list_folders(f'{settings.data_dir}/Raw_data/', pattern='PN*')


folders_nights = []  # store all night folders for which we have two in here

for folder_subj in tqdm(folders_subj, desc='loading participant responses'):
    subj = data_loading.get_subj(folder_subj)
    nights_folders = ospath.list_folders(folder_subj, pattern='*night*')
    if len(nights_folders)<2:
        print(f'{len(nights_folders)} night(s) for {subj}, skipping')
        continue
    
    # for each subject, load the individual nights
    for folder_night in nights_folders:
        folders_nights.append(folder_night)

# asd
#%% perform

tqdm_loop = tqdm(total=len(folders_nights), desc='creating spectrograms')

df = pd.DataFrame()

# loop over all experimental nights that are included
for folder in folders_nights:
    
    # retrieve which night type we have: low or high arousal
    night_type = data_loading.get_learning_type(folder)
    subj = data_loading.get_subj(folder)
    
    tqdm_loop.set_description('Loading file')
    
    # we load the file into memory using the function `utils.read_edf`
    raw = data_loading.load_sleep(folder)

    
    # now load the hypnogram that fits to this data
    hypno = data_loading.get_hypno(folder)
    hypno_upsampled = yasa.hypno_upsample_to_data(hypno, sf_hypno=1/30, data=raw)
    assert len(raw)//raw.info['sfreq']//30==len(hypno)  # sanity check
    
    
    tqdm_loop.set_description('Calculating spectrogram')
    
    # map the beginning letter of the channels to the region, e.g. FCz->F
    ch2region = lambda x: x[0] if not 'E' in x else x[-3:]
    
    # calculate the bandpower using YASA, split up by sleep stages
    df_subj = yasa.bandpower(raw, include=(2, 3, 4), hypno=hypno_upsampled).reset_index()
    df_subj['Region']  = df_subj['Chan'].apply(ch2region)
    df_subj['Subject'] = subj
    df_subj['Condition'] = night_type
    df_subj['Stage'] = df_subj['Stage'].apply(lambda x: stage_map[x])

    df = pd.concat([df, df_subj])
    tqdm_loop.update()
    
#%% Plot power values of participants  

bands = ('Delta', 'Theta', 'Alpha', 'Beta', 'Sigma')

 # plot all possible markers for low and high nights
fig, axs = plt.subplots(3, len(bands))
for i, stage in enumerate(df['Stage'].unique()):
    for j, band in enumerate(bands):
        # subselect entries with the current region
        sns.boxplot(data=df, x='Condition', y=band,ax=axs[i, j])
        sns.scatterplot(data=df, x='Condition', y=band, ax=axs[i, j])
        vals1 = df[(df.Condition=='low') & (df.Stage==stage)][band]
        vals2 = df[(df.Condition=='high') & (df.Stage==stage)][band]
        _, p = ttest_rel(vals1, vals2)
        axs[i, j].set_title(f'{stage}, {band}, {p=:.3f}')
plt.suptitle(f'Power band analysis per sleep stage n={len(df.Subject.unique())}')
plt.pause(0.1)  
plt.tight_layout()
