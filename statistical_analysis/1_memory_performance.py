# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:57:30 2022

Calculate and visualize the memory performance of the participants

@author: Simon
"""
import sys; sys.path.append('..')
import ospath
import pandas as pd
import seaborn as sns
import plotting
import matplotlib
import settings
import data_loading
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.metrics import classification_report
import numpy as np

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
    for folder_night in nights_folders:
        night_type = data_loading.get_learning_type(folder_night)
        night_number = data_loading.get_night(folder_night)
        data[subj][night_type] = {}
        data[subj][night_number] = {}

        for test_type in ['BS', 'AS']:
            resp = data_loading.load_test_responses(folder_night, which=test_type)
            data[subj][night_type][test_type] = resp
            data[subj][night_number][test_type]  = resp

stop

#%% memory performance high vs low 
fig, axs, ax_b = plotting.make_fig(len(data), 1)

df = pd.DataFrame()
for i, subj in enumerate(data):
    
    df_subj = pd.DataFrame()
    for ntype in ['high', 'low']:
        for ttype in ['BS', 'AS']:
            resp = data[subj][ntype][ttype]
            perf = classification_report(resp['seen_before'].values, 
                                         resp['seen_before_resp'].values,
                                         output_dict=True)
            quad = resp['quad'].values
            quad_resp = resp['quad_resp'].values
            idx_valid = ~np.isnan(quad) & ~np.isnan(quad_resp)
            quad = quad[idx_valid]
            quad_resp = quad_resp[idx_valid]
           
            perf_quad = classification_report(quad, quad_resp, output_dict=True)
            print(perf_quad)
            df_tmp = pd.DataFrame({'subject': subj,
                                   'arousal learning type': ntype,
                                   'timepoint': ttype,
                                   } | perf['weighted avg'], 
                                  index=[0])
            df_subj = pd.concat([df_subj, df_tmp], ignore_index=True)
            
    ax = axs[i]
    ax.set_title(subj)
    sns.barplot(data=df_subj,hue='timepoint', y='f1-score', x='arousal learning type', ax=ax)
    
    df = pd.concat([df, df_subj])
    
sns.barplot(data=df_subj, hue='timepoint', y='f1-score', x='arousal learning type', ax=ax_b)
  
ax_b.set_title(f'Avg of all {len(data)} PN')

# calculate p values for performance measures
for metric in ['precision', 'recall', 'f1-score']:
    for ntype in ['high', 'low']:
        val_before = df[(df['arousal learning type']==ntype) & (df['timepoint']=='BS')][metric]
        val_after  = df[(df['arousal learning type']==ntype) & (df['timepoint']=='AS')][metric]
        print(ttest_rel(val_before, val_after))
        
#%% plot subj <-> obj correlation results

# fig, axs = plt.subplots(2, 6); axs=axs.flatten()
fig, axs, ax_b = plotting.make_fig(6, [0,0,1])

df = pd.DataFrame()
for i, night in enumerate(data):
    *_, data_y = data[night]
    data_y = data_y.copy()
    data_y['valence_subj'] = data_y['valence_subj'] + 0.02
    df_subj = pd.DataFrame(data_y | {'night': night})
    df = pd.concat([df, df_subj], ignore_index=True)
    ax = axs[i]
    sns.regplot(data=df_subj, y='valence_subj', x='valence_mean', ax=ax)
    sns.regplot(data=df_subj, y='arousal_subj', x='arousal_mean', ax=ax, color='darkorange')
    ax.legend(['Valence', '', 'Arousal', ''])
    ax.set_xlabel('OASIS mean')
    ax.set_ylabel('subjective rating')
    ax.set_title(f'{night}')

sns.regplot(data=df, y='valence_subj', x='valence_mean', ax=ax_b)
sns.regplot(data=df, y='arousal_subj', x='arousal_mean', ax=ax_b, color='darkorange')
ax_b.legend(['Valence', '', 'Arousal', ''])
ax_b.set_xlabel('OASIS mean')
ax_b.set_ylabel('subjective rating')
ax_b.set_title(f'mean of n={len(data)}')
fig.suptitle('Correlation between OASIS and subjective ratings')
fig.tight_layout()


#%% 

# fig, axs = plt.subplots(2, 6); axs=axs.flatten()
fig, axs, ax_b = plotting.make_fig(6, [0,0,1])

df = pd.DataFrame()
for i, night in enumerate(data):
    *_, data_y = data[night]
    data_y = data_y.copy()
    data_y['valence_subj'] = data_y['valence_subj'] + 0.02
    df_subj = pd.DataFrame(data_y | {'night': night})
    df = pd.concat([df, df_subj], ignore_index=True)
    ax = axs[i]
    sns.regplot(data=df_subj, y='valence_subj', x='arousal_subj', ax=ax, 
                x_jitter=0.1, y_jitter=0.1)
    ax.set_title(f'{night}')

sns.regplot(data=df, y='valence_subj', x='arousal_subj', ax=ax_b, 
            x_jitter=0.1, y_jitter=0.11)
ax_b.set_title(f'mean of n={len(data)}')
fig.suptitle('Correlation between arousal and valence')
fig.tight_layout()


#%% category

fig, axs, ax_b = plotting.make_fig(6, [0,0,1])

df = pd.DataFrame()
for i, night in enumerate(data):
    *_, data_y = data[night]
    df_subj = pd.DataFrame(data_y)
    ax = axs[i]
    df = pd.concat([df, df_subj], ignore_index=True)
    sns.scatterplot(data=df_subj, y='arousal_mean', x='img_category', ax=ax)
    ax.set_title(f'{night}')

sns.scatterplot(data=df, y='arousal_mean', x='img_category', ax=ax_b)
