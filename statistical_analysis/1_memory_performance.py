# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:57:30 2022

Calculate and visualize the memory performance of the participants

@author: Simon
"""
import os, sys; sys.path.append('..')
import ospath
import pandas as pd
import seaborn as sns
import plotting
import matplotlib
import settings
import data_loading
from tqdm import tqdm
from scipy.stats import ttest_rel, pearsonr, f_oneway, ttest_ind
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

sns.set(font_scale=1.4)
os.makedirs('./results/', exist_ok=True)

def aggregate(df, by1='image arousal', by2='timepoint', metric='f1-score',
              test_func=ttest_rel):
    df_mean = df.groupby([by1, by2]).mean(True)
    df_std = df.groupby([by1, by2]).std(True).rename(lambda x: x+' std', axis=1)
    df_aggr = pd.concat([df_mean, df_std], axis=1)
    df_aggr = df_aggr.sort_index(axis=1)
    df_aggr = df_aggr.reset_index()
    pvals = []
    tstats = []
    for ntype in df_aggr[by1].unique():
        var = df_aggr[by2].unique()
        assert len(var)==2
        vals1 = df[(df[by1]==ntype) & (df[by2]==var[0])][metric]
        vals2 = df[(df[by1]==ntype) & (df[by2]==var[1])][metric]
        vals1 = vals1[~vals1.isna()]
        vals2 = vals2[~vals2.isna()]
        res = test_func(vals1, vals2)
        pvals += [res.pvalue]*2
        tstats += [res.statistic]*2

    df_aggr[f'{metric} p'] = pvals
    df_aggr[f'{metric} relative tstat'] = tstats
    return df_aggr

#%% data loading
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
            resp['subj_valence']+=1
            resp['subj_arousal']+=1
            data[subj][night_type][test_type] = resp
            data[subj][night_number][test_type]  = resp
# store all variables loaded so far in here , this way we can remove all
# computed variables later without losing them or leaking into other analyis
new_vars = set()


#%% 1.1 memory performance high vs low
settings.clear(locals(), new_vars)  # clear all variables for good measures
curr_locals = set(locals())

fig, axs, ax_b = plotting.make_fig(len(data), [0,1])

df = pd.DataFrame()
for i, subj in enumerate(data):

    df_subj = pd.DataFrame()
    for ntype in ['high', 'low']:
        for ttype in ['BS', 'AS']:
            resp = data[subj][ntype][ttype]
            perf = classification_report(resp['seen_before'].values,
                                         resp['seen_before_resp'].values,
                                         output_dict=True)
            df_tmp = pd.DataFrame({'subject': subj,
                                   'image arousal': ntype,
                                   'timepoint': ttype,
                                   } | perf['weighted avg'],
                                  index=[0])
            df_subj = pd.concat([df_subj, df_tmp], ignore_index=True)

    ax = axs[i]
    ax.set_title(f'{subj} memory performance old/new')
    sns.barplot(data=df_subj,hue='timepoint', y='f1-score', x='image arousal', ax=ax)
    ax.set_ylim(0.8, 1.05)
    ax.legend(loc='upper center')
    df = pd.concat([df, df_subj])


sns.boxplot(data=df, hue='timepoint', y='f1-score', x='image arousal', ax=ax_b,)
# sns.violinplot(data=df, hue='timepoint', y='f1-score', x=' learning'
#                , split=True, ax=ax_b)

ax_b.set_ylim(0.8, 1.05)
# calculate p values for performance measures
pvals = {}
for ntype in ['high', 'low']:
    val_before = df[(df['image arousal']==ntype) & (df['timepoint']=='BS')]['f1-score']
    val_after  = df[(df['image arousal']==ntype) & (df['timepoint']=='AS')]['f1-score']
    pvals[ntype] = ttest_rel(val_before, val_after)
    # print(ttest_rel(val_before, val_after))

name = './results/1.1 memory performance old-new'
p = ', '.join([f'{key} before/after p={val.pvalue:.3f}'for key, val in pvals.items()])
ax_b.set_title(f'memory performance old/new - n={len(data)}\n{p}')
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
fig.savefig(f'{name}.png')
# save data to disk as well
df = df.sort_values(['image arousal', 'subject'])
df.to_excel(f'{name}_raw.xlsx')
df_aggr = aggregate(df)
df_aggr.to_excel(f'{name}_mean.xlsx')

# record all new variables that have been created inside of this function
# so that they can be removed at the next section to prevent
# variables from accidentially bleeding into other analaysis
new_vars = set(locals()).difference(curr_locals)

#%% 1.2 memory performance quadrant
settings.clear(locals(), new_vars)  # clear all variables for good measures
curr_locals = set(locals())

fig, axs, ax_b = plotting.make_fig(len(data), [0,1])

df = pd.DataFrame()
for i, subj in enumerate(data):

    df_subj = pd.DataFrame()
    for ntype in ['high', 'low']:
        for ttype in ['BS', 'AS']:
            resp = data[subj][ntype][ttype]
            quad = resp['quad'].values
            quad_resp = resp['quad_resp'].values
            idx_valid = ~np.isnan(quad) & ~np.isnan(quad_resp)
            quad = quad[idx_valid]
            quad_resp = quad_resp[idx_valid]
            acc = (quad==quad_resp).mean()
            perf_quad = classification_report(quad, quad_resp, output_dict=True)
            df_tmp = pd.DataFrame({'subject': subj,
                                   'image arousal': ntype,
                                   'timepoint': ttype,
                                   'accuracy': acc
                                   }, index=[0])
            df_subj = pd.concat([df_subj, df_tmp], ignore_index=True)

    ax = axs[i]
    ax.set_title(f'{subj} memory performance quadrant')
    sns.barplot(data=df_subj,hue='timepoint', y='accuracy', x='image arousal', ax=ax)
    ax.set_ylim(0.6, 1.05)
    df = pd.concat([df, df_subj])

sns.boxplot(data=df, hue='timepoint', y='accuracy', x='image arousal', ax=ax_b)
# sns.violinplot(data=df, hue='timepoint', y='f1-score', x='image arousal'
#                , split=True, ax=ax_b)

ax_b.set_ylim(0.6, 1.05)
# calculate p values for performance measures
pvals = {}
for ntype in ['high', 'low']:
    val_before = df[(df['image arousal']==ntype) & (df['timepoint']=='BS')]['accuracy']
    val_after  = df[(df['image arousal']==ntype) & (df['timepoint']=='AS')]['accuracy']
    pvals[ntype] = ttest_rel(val_before, val_after)

name = './results/1.2 memory performance quadrant'
p = ', '.join([f'{key} before/after p={val.pvalue:.3f}'for key, val in pvals.items()])
ax_b.set_title(f'memory performance quadrant for both nights - n={len(data)}\n{p=}')
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
fig.savefig(f'{name}.png')

# save data to disc
df = df.sort_values(['image arousal', 'subject'])
df.to_excel(f'{name}_raw.xlsx')
df_aggr = aggregate(df, metric='accuracy')
df_aggr.to_excel(f'{name}_mean.xlsx')

# record all new variables that have been created inside of this function
# so that they can be removed at the next section to prevent
# variables from accidentially bleeding into other analaysis
new_vars = set(locals()).difference(curr_locals)

#%% 1.3 plot subj <-> obj correlation results
settings.clear(locals(), new_vars)  # clear all variables for good measures
curr_locals = set(locals())

# fig, axs = plt.subplots(2, 6); axs=axs.flatten()
fig, axs, ax_b = plotting.make_fig(7, [0,0,1])

df = pd.DataFrame()
for i, subj in enumerate(data):
    ratings = []
    for night in data[subj]:
        for timepoint in data[subj][night]:
            ratings.append(data[subj][night][timepoint])
    df_subj = pd.concat(ratings).drop_duplicates()
    df_subj['subject'] = subj
    df = pd.concat([df, df_subj], ignore_index=True)
    ax = axs[i]
    sns.regplot(data=df_subj, y='subj_valence', x='valence_mean', ax=ax)
    sns.regplot(data=df_subj, y='subj_arousal', x='arousal_mean', ax=ax, color='darkorange')
    ax.legend(['Valence', '_','_', 'Arousal'])
    ax.set_xlabel('OASIS mean')
    ax.set_ylabel('subjective rating')
    ax.set_title(f'{subj}')

sns.regplot(data=df, y='subj_valence', x='valence_mean', ax=ax_b)
sns.regplot(data=df, y='subj_arousal', x='arousal_mean', ax=ax_b, color='darkorange')
ax_b.legend(['Valence', '_','_', 'Arousal'])
ax_b.set_xlabel('OASIS mean')
ax_b.set_ylabel('subjective rating')
ax_b.set_title(f'mean of n={len(data)}')

# need to remove NANs first
idx1 = ~df['subj_valence'].isna()
idx2 = ~df['subj_arousal'].isna()
r_valence, p_valence = pearsonr(df['subj_valence'][idx1], df['valence_mean'][idx1])
r_arousal, p_arousal = pearsonr(df['subj_arousal'][idx2], df['arousal_mean'][idx2])

ax_b.set_title(f'Correlation between OASIS and subjective ratings, n={len(data)}\n'
             f'Valence: r={r_valence:.3f} p={p_valence:.5f}\n'
             f'Arousal: r={r_arousal:.3f} p={p_arousal:.5f}')
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
name = '1.3 Correlation subjective and OASIS ratings'
fig.savefig(f'./results/{name}.png')
df.to_excel(f'./results/{name}_raw.xlsx')
df = pd.DataFrame({'p value': [p_valence, p_arousal],
              'Pearsons r': [r_valence, r_arousal],
              'subj mean': [df['subj_valence'][idx1].mean(), df['subj_arousal'][idx2].mean()],
              'subj std': [df['subj_valence'][idx1].std(), df['subj_arousal'][idx2].std()],
              'OASIS mean': [df['valence_mean'][idx1].mean(), df['arousal_mean'][idx2].mean()],
              'OASIS std': [df['valence_mean'][idx1].std(), df['arousal_mean'][idx2].std()],

              'rating': ['valence', 'arousal']}).to_excel(f'./results/{name}_mean.xlsx')
# plt.pause(0.1)
new_vars = set(locals()).difference(curr_locals)


#%% 1.4 Arousal ratings for different nights
settings.clear(locals(), new_vars)  # clear all variables for good measures
curr_locals = set(locals())

fig, axs, ax_a = plotting.make_fig(7, [0,0,1])

df = pd.DataFrame()
for i, subj in enumerate(data):
    ratings = []
    for night in ['high', 'low']:
        for timepoint in data[subj][night]:
            df_tmp = data[subj][night][timepoint]
            df_tmp['night'] = night
            ratings.append(df_tmp)
    df_subj = pd.concat(ratings, ignore_index=True).drop_duplicates()
    df = pd.concat([df, df_subj], ignore_index=True)

    ax = axs[i]
    sns.swarmplot(data=df_subj, y='subj_arousal', x='night', hue='night', dodge=True,
                  ax=ax, legend=False)
    ax.set_title(f'{subj}')
filter_nan = lambda df: df[~df.isna()].values
values_arousal = [filter_nan(g['subj_arousal']) for _,g in df.groupby('night')]
tstat, pval = ttest_ind(*values_arousal)

ax_a.set_ylabel('subjective arousal rating')
ax_a.set_title(f'Arousal rating for diff. nights\nttest: {tstat=:.4f} {pval=:.4f}')
sns.swarmplot(data=df_subj, y='subj_arousal', x='night', hue='night', dodge=True,
              ax=ax_a, legend=True)
plt.tight_layout()
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
name = '1.4 arousal rating by night'
df.to_excel(f'./results/{name}_raw.xlsx')
df['dummy']=1
df_aggr = aggregate(df, 'dummy', 'night', metric='subj_arousal', test_func=ttest_ind)
df_aggr.to_excel(f'./results/{name}_mean.xlsx')
fig.savefig(f'./results/{name}.png')
new_vars = set(locals()).difference(curr_locals)
#%% 1.5 Valence ratings for different nights
settings.clear(locals(), new_vars)  # clear all variables for good measures
curr_locals = set(locals())

fig, axs, ax_a = plotting.make_fig(7, [0,0,1])

df = pd.DataFrame()
for i, subj in enumerate(data):
    ratings = []
    for night in ['high', 'low']:
        for timepoint in data[subj][night]:
            df_tmp = data[subj][night][timepoint]
            df_tmp['night'] = night
            ratings.append(df_tmp)
    df_subj = pd.concat(ratings, ignore_index=True).drop_duplicates()
    df = pd.concat([df, df_subj], ignore_index=True)

    ax = axs[i]
    sns.swarmplot(data=df_subj, y='subj_valence', x='night', hue='night', dodge=True,
                  ax=ax, legend=False)
    ax.set_title(f'{subj}')
filter_nan = lambda df: df[~df.isna()].values
values_arousal = [filter_nan(g['subj_valence']) for _,g in df.groupby('night')]
tstat, pval = ttest_ind(*values_arousal)

ax_a.set_ylabel('subjective valence rating')
ax_a.set_title(f'valence rating for diff. nights\nttest: {tstat=:.4f} {pval=:.4f}')
sns.swarmplot(data=df_subj, y='subj_valence', x='night', hue='night', dodge=True,
              ax=ax_a, legend=True)
plt.tight_layout()
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
name = '1.5 valence rating by night'
df.to_excel(f'./results/{name}_raw.xlsx')
df['dummy']=1
df_aggr = aggregate(df, 'dummy', 'night', metric='subj_valence', test_func=ttest_ind)
df_aggr.to_excel(f'./results/{name}_mean.xlsx')
fig.savefig(f'./results/{name}.png')
new_vars = set(locals()).difference(curr_locals)

#%% 1.5b Valence ratings high arousal night only
settings.clear(locals(), new_vars)  # clear all variables for good measures
curr_locals = set(locals())

fig, axs, ax_a = plotting.make_fig(7, [0,0,1])

def bin_valence(val):
    return 'low' if val < settings.split_valence else 'high'

df = pd.DataFrame()
for i, subj in enumerate(data):
    ratings = []
    for timepoint in data[subj]['high']:
        df_tmp = data[subj][night][timepoint].copy()
        df_tmp['valence_mean'] = df_tmp['valence_mean'].apply(bin_valence)
        ratings.append(df_tmp)
    df_subj = pd.concat(ratings, ignore_index=True).drop_duplicates()
    df = pd.concat([df, df_subj], ignore_index=True)

    ax = axs[i]
    sns.swarmplot(data=df_subj, y='subj_valence', x='valence_mean', dodge=True,
                  ax=ax, legend=False)
    ax.set_title(f'{subj}')
filter_nan = lambda df: df[~df.isna()].values
values_arousal = [filter_nan(g['subj_valence']) for _,g in df.groupby('valence_mean')]
tstat, pval = ttest_ind(*values_arousal)

ax_a.set_ylabel('subjective valence rating')
ax_a.set_title(f'valence rating for high arousal nights\nttest: {tstat=:.4f} {pval=:.4f}')
ax_a.set_xlabel(f'valence category\n(by median split of OASIS)')
sns.swarmplot(data=df_subj, y='subj_valence', x='valence_mean', dodge=True,
              ax=ax_a, legend=True)
plt.tight_layout()
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
name = '1.5 valence rating for high arousal nights'
df.to_excel(f'./results/{name}_raw.xlsx')
df['dummy']=1
df_aggr = aggregate(df, 'dummy', 'night', metric='subj_valence', test_func=ttest_ind)
df_aggr.to_excel(f'./results/{name}_mean.xlsx')
fig.savefig(f'./results/{name}.png')
new_vars = set(locals()).difference(curr_locals)


#%% 1.6 Category effect on ratings
settings.clear(locals(), new_vars)  # clear all variables for good measures
curr_locals = set(locals())

fig, axs, ax_a, ax_b = plotting.make_fig(7, [1,1], xlabel='category')

df = pd.DataFrame()
for i, subj in enumerate(data):
    ratings = []
    for night in ['high', 'low']:
        for timepoint in data[subj][night]:
            df_tmp = data[subj][night][timepoint]
            df_tmp['night'] = night
            ratings.append(df_tmp)
    df_subj = pd.concat(ratings, ignore_index=True).drop_duplicates()
    df = pd.concat([df, df_subj], ignore_index=True)

    ax = axs[i]
    sns.swarmplot(data=df_subj, y='subj_arousal', x='img_category', dodge=True,
                  hue='img_category', ax=ax, legend=False, size=4)
    ax.set_title(f'{subj}')
    ax.set_ylabel('rating')

filter_nan = lambda df: df[~df.isna()].values
values_valence = [filter_nan(g['subj_valence']) for _,g in df.groupby('img_category')]
values_arousal = [filter_nan(g['subj_arousal']) for _,g in df.groupby('img_category')]
f_valence, p_valence = f_oneway(*values_valence)
f_arousal, p_arousal = f_oneway(*values_arousal)

jitter = lambda x:x*((np.random.rand()-0.5)*0.2+1)
df['subj_arousal'] = df['subj_arousal'].apply(jitter)
df['subj_valence'] = df['subj_valence'].apply(jitter)

ax_a.set_ylabel('subjective valence rating')
ax_a.set_title(f'ANOVA: {f_valence=:.4f} {p_valence=:.4f}')
sns.swarmplot(data=df, y='subj_valence', x='img_category', dodge=True,
              hue='img_category', ax=ax_a, legend=True, size=4)
ax_b.set_ylabel('subjective arousal rating')
ax_b.set_title(f'ANOVA: {f_arousal=:.4f} {p_arousal=:.4f}')
sns.swarmplot(data=df, y='subj_arousal', x='img_category', dodge=True,
              hue='img_category', ax=ax_b, legend=True, size=4)

name = '1.6 category effect'
pd.DataFrame({'p value': [p_valence, p_arousal],
              '1-way ANOVA f value': [f_valence, f_arousal],
              'rating': ['valence', 'arousal']}).to_excel(f'./results/{name}_stats.xlsx')
plt.pause(0.1)
fig.tight_layout()
plt.pause(0.1)
fig.tight_layout()
fig.savefig(f'./results/{name}.png')

new_vars = set(locals()).difference(curr_locals)