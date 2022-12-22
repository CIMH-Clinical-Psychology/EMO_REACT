# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:41:45 2022

Correlate night type with sleep parameters

@author: Simon
"""

import sleep_utils
import sys; sys.path.append('..')
import ospath
import pandas as pd
import seaborn as sns
import settings
import data_loading
from tqdm import tqdm
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
sns.set(font_scale=1.2)


#%%

# list all the folders of the participants
folders_subj = ospath.list_folders(f'{settings.data_dir}/Raw_data/', pattern='PN*')


# data is a very nested dictionary 
# with levels data[subj][night_type][test_type]
# subj = subject id, e.g. PN01
# night_type = high or low arousal
# test_type = BS or AS, i.e., before sleep or after sleep
data = {}

for folder in tqdm(folders_subj, desc='loading participant responses'):
    subj = data_loading.get_subj(folder)
    nights_folders = ospath.list_folders(folder, pattern='*night*')
    if len(nights_folders)<2:
        print(f'{len(nights_folders)} night(s) for {subj}, skipping')
        continue
    
    data[subj] = {}
    # for each subject, load the individual nights
    for folder_night in nights_folders:
        # retrieve which night type we have: low or high arousal
        night_type = data_loading.get_learning_type(folder_night)
        
        data[subj][night_type] = {}
        data[subj][night_type]['hypno'] = data_loading.get_hypno(folder_night)
        
        #  load the tests
        for test_type in ['BS', 'AS']:
            resp = data_loading.load_test_responses(folder_night, which=test_type)
            data[subj][night_type][test_type] = resp

# stop

#%% First very basic overall sleep stage distribution analyis

df = pd.DataFrame()
for subj in data:
    for night in data[subj]:
        # for this specific night, 
        info = {'subject': subj, 'night':night}
        hypno = data[subj][night]['hypno']  # load the hypnogram
        summary = sleep_utils.hypno_summary(hypno)  # calculate summary stats
        df = pd.concat([df, pd.DataFrame(summary | info,
                                         index=[0])], ignore_index=True)

# plot all possible markers for low and high nights
fig, axs = plt.subplots(4, 5); axs=axs.flatten()
for i, name in enumerate(summary):
    sns.boxplot(data=df, x='night', y=name, ax=axs[i])
    sns.scatterplot(data=df, x='night', y=name, ax=axs[i])
    _, pval = ttest_rel(df[df.night=='low'][name], df[df.night=='high'][name])
    axs[i].set_title(f'{name} {pval=:.2f}')

plt.suptitle(f'Sleep parameters for n={len(data.keys())} [{list(data)}]')

plt.tight_layout()
plt.pause(0.1)
plt.savefig('../plots/2_sleep_parameters.png')