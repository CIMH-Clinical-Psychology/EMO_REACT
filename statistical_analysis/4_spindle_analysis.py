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
import plotting
import yasa
from tqdm import tqdm
from scipy.stats import ttest_rel
import numpy as np
from joblib import Parallel, delayed, Memory
import matplotlib.pyplot as plt
sns.set(font_scale=1.2)
memory = Memory(settings.cache_dir)

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
chs = ['Pz', 'CPz', 'P1', 'P2', 'POz']
tqdm_loop = tqdm(total=len(folders_nights), desc='creating spectrograms')

df = pd.DataFrame()
spindle_func = memory.cache(yasa.spindles_detect)

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
    
    # go through the stages and stage combinations that we want

    ch2region = lambda x: x[0] if not 'E' in x else x[-3:]
    raw.drop_channels([ch for ch in raw.ch_names if not ch in chs])
    spindle_res = spindle_func(raw, include=(2, 3), ch_names=chs,
                               hypno=hypno_upsampled, multi_only=True)
    df_subj = spindle_res.summary(grp_stage=True).reset_index()
    df_subj['Subject'] = subj
    df_subj['Condition'] = night_type
    df_subj['Stage'] = df_subj['Stage'].apply(lambda x: stage_map[x])

    df = pd.concat([df, df_subj])
    tqdm_loop.update()
    
#%% Plot power values of participants  

markers = ('Density', 'Duration', 'Amplitude', 'RMS', 'AbsPower', 'RelPower', 
           'Frequency', 'Oscillations', 'Symmetry')

fig, axs = plotting.make_fig(len(markers))
 # plot all possible markers for low and high nights

for i, marker in enumerate(markers):
    # subselect entries with the current region
    sns.boxplot(data=df, x='Stage', y=marker, hue='Condition', ax=axs[i])
    # sns.scatterplot(data=df, x='Condition', y=band, ax=axs[i, j])
    pvals = {}
    for stage in df.Stage.unique():
        vals1 = df[(df.Condition=='low') & (df.Stage==stage)][marker]
        vals2 = df[(df.Condition=='high') & (df.Stage==stage)][marker]
        _, p = ttest_rel(vals1, vals2)
        pvals[stage] = np.round(p, 3)
    axs[i].set_title(f'{marker}, p={pvals}')
plt.suptitle(f'Spindle analysis per sleep stage using YASA algorithm n={len(df.Subject.unique())}')
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()