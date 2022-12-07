# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:15:16 2022

plotting behavioural outputs

@author: Simon
"""
import ospath
from tqdm import tqdm
from data_loading import load_localizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import settings
import plotting
import matplotlib

matplotlib.rc('font', size=14)

folders = ospath.list_folders(f'{settings.data_dir}/Raw_data/', pattern='*night*', recursive=True)

data = {}

for folder in tqdm(folders, desc='loading localizer data'):
    night = settings.get_night(folder)
    try:
        res = load_localizer(folder)
        times, data_x, data_y = res
        data[night] = times, data_x, data_y
    except:
        print(f'ERROR: {folder}')
        
for folder in tqdm(folders, desc='loading localizer data'):
    night = settings.get_night(folder)
    try:
        res = load_localizer(folder)
        times, data_x, data_y = res
        data[night] = times, data_x, data_y
    except:
        print(f'ERROR: {folder}')
stop




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
