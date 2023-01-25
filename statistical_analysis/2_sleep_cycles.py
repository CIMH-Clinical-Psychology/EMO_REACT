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
from pandas.api.types import is_numeric_dtype
sns.set(font_scale=1.2)


def aggregate(df, by1='image arousal', test_func=ttest_rel):
    df_mean = df.groupby(by1).mean(True)
    df_std = df.groupby(by1).std(True).rename(lambda x: x+' std', axis=1)
    df_aggr = pd.concat([df_mean, df_std], axis=1)
    df_aggr = df_aggr.sort_index(axis=1)
    # df_aggr = df_aggr.reset_index()
    pvals = {}
    tstats = {}
    vals = [x for _,x in df.groupby(by1)]
    for marker in df.columns:
        if not is_numeric_dtype(df[marker]): 
            tstats[marker] = ['N/A']
            pvals[marker] = ['N/A']
            continue
        stat, pval = test_func(vals[0][marker], vals[1][marker])
        tstats[marker] = [stat]
        pvals[marker] = [pval]
    df_aggr = pd.concat([df_aggr, pd.DataFrame(tstats, index=['rel tstat'])], axis=0)
    df_aggr = pd.concat([df_aggr, pd.DataFrame(pvals, index=['pvalue'])], axis=0)

    return df_aggr

#%% load data

# list all the folders of the participants
folders_subj = ospath.list_folders(f'{settings.data_dir}/Raw_data/', pattern='PN*')


# data is a very nested dictionary 
# with levels data[subj][night_type][test_type]
# subj = subject id, e.g. PN01
# night_type = high or low arousal
# test_type = BS or AS, i.e., before sleep or after sleep
data = {}

for folder in tqdm(folders_subj, desc='loading participant sleep stages'):
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

#%% 2.1 Sleep stage analysis

df = pd.DataFrame()
for subj in data:
    for night in data[subj]:
        # for this specific night, 
        info = {'subject': subj, 'image arousal':night}
        hypno = data[subj][night]['hypno']  # load the hypnogram
        summary = sleep_utils.hypno_summary(hypno)  # calculate summary stats
        df = pd.concat([df, pd.DataFrame(summary | info,
                                         index=[0])], ignore_index=True)



#%% 2.1 General markers
# plot all possible markers for low and high nights
fig, axs = plt.subplots(2, 3); axs=axs.flatten()

markers = [x for x in summary if not x.startswith(('min', 'perc', 'lat'))]

for i, name in enumerate(markers):
    sns.boxplot(data=df, x='image arousal', y=name, ax=axs[i])
    sns.scatterplot(data=df, x='image arousal', y=name, ax=axs[i])
    _, pval = ttest_rel(df[df['image arousal']=='low'][name], df[df['image arousal']=='high'][name])
    axs[i].set_title(f'{name} {pval=:.2f}')

name = '2.1 general sleep markers'
plt.suptitle(f'Sleep parameters for n={len(data.keys())} [{list(data)}]')
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
plt.savefig(f'./results/{name}.png')
df.to_excel(f'./results/{name}_raw.xlsx')
aggregate(df).to_excel(f'./results/{name}_stats.xlsx')

#%% 2.2 Sleep Stage Markers
fig, axs = plt.subplots(3, 4); axs=axs.flatten()

stagemarkers = [x for x in summary if x.startswith(('min', 'perc', 'lat'))]
stagemarkers.remove('perc_W')

for i, name in enumerate(stagemarkers):
    sns.boxplot(data=df, x='image arousal', y=name, ax=axs[i])
    sns.scatterplot(data=df, x='image arousal', y=name, ax=axs[i])
    _, pval = ttest_rel(df[df['image arousal']=='low'][name], df[df['image arousal']=='high'][name])
    axs[i].set_title(f'{name} {pval=:.2f}')

name = '2.2 sleep cycle markers'
plt.suptitle(f'Sleep parameters for n={len(data.keys())} [{list(data)}]')
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
plt.savefig(f'./results/{name}.png')
df.to_excel(f'./results/{name}_raw.xlsx')
aggregate(df).to_excel(f'./results/{name}_stats.xlsx')

#%% 2.3 Hypnogram summary
fig, axs = plt.subplots( 2, 1); axs=axs.flatten()

for i, arousal in enumerate(['low', 'high']):
    hypnos = [data[subj][arousal]['hypno'] for subj in data]
    sleep_utils.plot_hypnogram_overview(hypnos, ax=axs[i])
name = '2.3 hypnogram overview'
plt.suptitle(f'overview over hypnograms n={len(data.keys())} [{list(data)}]')
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
plt.savefig(f'./results/{name}.png')
