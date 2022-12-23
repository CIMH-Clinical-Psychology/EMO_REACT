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
from joblib import Parallel, delayed, Memory
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
sns.set(font_scale=1.2)
memory = Memory(settings.cache_dir)

# check which data is loadable
folders_subj = ospath.list_folders(f'{settings.data_dir}/Raw_data/', pattern='PN*')
def aggregate(df, by1='Condition', test_func=ttest_rel):
    df_mean = df.groupby(by1).mean(True)
    df_std = df.groupby(by1).std(True).rename(lambda x: x+' std', axis=1)
    df_aggr = pd.concat([df_mean, df_std], axis=1)
    df_aggr = df_aggr.sort_index(axis=1)
    # df_aggr = df_aggr.reset_index()
    pvals = {}
    tstats = {}
    vals = [x for _,x in df.groupby(by1)]
    for marker in df.columns:
        if not is_numeric_dtype(df[marker]) or df[marker].dtype==bool: 
            tstats[marker] = ['N/A']
            pvals[marker] = ['N/A']
            continue
        stat, pval = test_func(vals[0][marker], vals[1][marker])
        tstats[marker] = [stat]
        pvals[marker] = [pval]
    df_aggr = pd.concat([df_aggr, pd.DataFrame(tstats, index=['rel tstat'])], axis=0)
    df_aggr = pd.concat([df_aggr, pd.DataFrame(pvals, index=['pvalue'])], axis=0)

    return df_aggr
# %% data loading
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
#%% perform power band analysis

tqdm_loop = tqdm(total=len(folders_nights), desc='creating spectrograms')

df = pd.DataFrame()
bandpower_func = memory.cache(yasa.bandpower)
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
    df_subj = bandpower_func(raw, include=(2, 3, 4), hypno=hypno_upsampled).reset_index()
    df_subj['Region']  = df_subj['Chan'].apply(ch2region)
    df_subj['Subject'] = subj
    df_subj['Condition'] = night_type
    df_subj['Stage'] = df_subj['Stage'].apply(lambda x: stage_map[x])

    df = pd.concat([df, df_subj], ignore_index=True)
    tqdm_loop.update()
    
#%% 3.1 power values per sleep stage

bands = ('Delta', 'Theta', 'Alpha', 'Beta', 'Sigma')

df_bands = pd.concat([pd.DataFrame({'Rel. Power': df[band],
                          'Band': band,
                          'Subject': df.Subject,
                          'Region': df.Region,
                          'Condition': df.Condition,
                          'Chan': df.Chan,
                          'Stage': df.Stage}) for band in bands], ignore_index=True)

 # plot all possible markers for low and high nights
fig, axs = plt.subplots(1, 3, figsize=[18, 8])
for i, stage in enumerate(df['Stage'].unique()):
    df_sel = df_bands[df_bands.Stage==stage]
    # subselect entries with the current region
    ax = axs[i]
    sns.boxplot(data=df_sel, x='Band', y='Rel. Power', hue='Condition', ax=ax)
    ax.set_title(f'Stage {stage}')
    
    vals = [x for _,x in df_sel.groupby('Condition')]
    pvals = []
    for band in df_sel.Band.unique():
        _, pval = ttest_rel(*[x[x.Band==band]['Rel. Power'] for x in vals])
        pvals.append(pval)
    ax.set_xticks(range(len(bands)), [f'{b}\n{p=:.3f}' for b,p in zip(bands, pvals)])
    
plt.suptitle(f'Power band analysis per sleep stage n={len(df.Subject.unique())}')
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
name = '3.1 Power band analysis per sleep stage'
fig.savefig(f'./results/{name}.png')
df.to_excel(f'./results/{name}_raw.xlsx')
aggregate(df, 'Condition').to_excel(f'./results/{name}_stats.xlsx')

#%% 3.2 power values for different regions
from scipy.stats import ttest_ind
regions = ['F', 'T', 'C', 'P', 'O']


df_regions = df[df.Region.isin(regions)]
# df_regions = df_regions[df_regions.TotalAbsPow<df_regions.TotalAbsPow.quantile(0.95)]
df_regions.TotalAbsPow = df_regions.TotalAbsPow .apply(lambda x: 0 if x>df_regions.TotalAbsPow.quantile(0.95) else x)
df_regions = df_regions.groupby(['Region', 'Subject', 'Stage', 'Condition']).mean().reset_index()

 # plot all possible markers for low and high nights
fig, axs = plt.subplots(3, 1, figsize=[18, 10])
for i, stage in enumerate(df['Stage'].unique()):
    df_sel = df_regions[df_regions.Stage==stage]
    # subselect entries with the current region
    ax = axs[i]
    sns.boxplot(data=df_sel, x='Region', y='TotalAbsPow', hue='Condition', ax=ax)
    ax.set_title(f'Stage {stage}')
    
    vals = [x for _,x in df_sel.groupby('Condition')]
    pvals = []
    for region in regions:
        _, pval = ttest_rel(*[x[x.Region==region]['TotalAbsPow'] for x in vals])
        pvals.append(pval)
    ax.set_xticks(range(len(regions)), [f'{b}\n{p=:.3f}' for b,p in zip(regions, pvals)])
    
plt.suptitle(f'Power band analysis per region n={len(df.Subject.unique())}')
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
name = '3.2 Power band analysis for different sleep stages'
fig.savefig(f'./results/{name}.png')
df_regions.to_excel(f'./results/{name}_raw.xlsx')
aggregate(df_regions).to_excel(f'./results/{name}_stats.xlsx')
