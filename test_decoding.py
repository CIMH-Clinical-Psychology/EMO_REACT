# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:07:18 2022

Dummy testing of decoders

@author: Simon Kern
"""
import ospath
import mne
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import make_pipeline
import numpy as np
from data_loading import load_localizer
import matplotlib.pyplot as plt
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from scipy.stats import zscore

from settings import trigger
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from mne import io, pick_types, read_events, Epochs, EvokedArray, create_info
from mne.datasets import sample
from mne.preprocessing import Xdawn
from mne.decoding import Vectorizer
from sklearn.metrics import f1_score
import pandas as pd
from scipy.ndimage import gaussian_filter

import settings
import plotting

print(__doc__)

#%% settings
mse = make_scorer(mean_squared_error, greater_is_better=False)
# scoring = make_scorer(f1_score, average='macro')
kwargs = dict(n_jobs=-1, scoring='accuracy')

clf_rfr = SlidingEstimator(RandomForestRegressor(1000), **kwargs)
clf_rfc = SlidingEstimator(RandomForestClassifier(1000), **kwargs)
clf_svr = SlidingEstimator(SVR(), **kwargs)
clf_svc = SlidingEstimator(SVC( class_weight='balanced'), **kwargs)
clf_log = SlidingEstimator(LogisticRegression(C=10, penalty='l1', class_weight='balanced',
                                               solver='liblinear', max_iter=1000), 
                          **kwargs)
csp = CSP(n_components=3, norm_trace=False)
clf_csp = make_pipeline(csp,
                        LinearModel(LogisticRegression(solver='liblinear'))
                        )


clfs = {
        # 'CSP': clf_csp,
        # 'SVR': clf_svr,
        f'SVC-{C}':clf_svc for C in [0.1, 1, 10, 100]
        # 'RFR': clf_rfr, 
        # 'RFC':clf_rfc,
        # 'LogReg': clf_log
        }
colors = sns.color_palette()

#%% data loading

folders = ospath.list_folders(f'{settings.data_dir}/Raw_data/', pattern='*night*', recursive=True)

sfreq = 100
tmin= -1
tmax = 2

data = {}

for folder in tqdm(folders, desc='loading localizer data'):
    night = settings.get_night(folder)
    try:
        res = load_localizer(folder, sfreq=sfreq, tmin=tmin, tmax=tmax, event_id=trigger.STIMULUS)
        times, data_x, data_y = res
        data[night] = times, data_x, data_y
    except:
        print(f'ERROR: {folder}')

stop


#%% run calculations
# plt.maximize=False

fig, axs, ax_b = plotting.make_fig(len(data), n_bottom=[0,0,1])

df = pd.DataFrame()

for i, night in enumerate(tqdm(data, desc='subj')):
    times, data_x, data_y = data[night]
    
    subj = settings.get_subj(night)
    train_x = zscore(data_x, axis=2)
    
    
    thresh = {'valence_subj':0, 'arousal_subj':2,
              'valence_mean':3, 'arousal_mean':3,
              'img_category':-1}

    masks = {t:(v<thresh[t]) | (v>thresh[t]) for t, v in data_y.items()}
    
    df_subj = pd.DataFrame()
    for target, train_y in data_y.items():
        train_x_sel = train_x.copy()[masks[target]]
        train_y_sel = train_y[masks[target]]        
        if target!='img_category':
            train_y_sel = train_y_sel>thresh[target]
        if any(np.bincount(train_y_sel)<5):
            print(f'Too few values in classes: {night} {target=} {np.bincount(train_y_sel)}')
            continue
        for clf_name, clf in clfs.items():
            # try:
                scores = cross_val_multiscore(clf, train_x_sel, train_y_sel, n_jobs=1, cv=5, 
                                              verbose=False)
                scores = gaussian_filter(scores, (0, 2))
                
                df_tmp = pd.DataFrame({f'{scoring._score_func.__name__}': scores.ravel(),
                                       'timepoint': [x for x in times]*5,
                                       'night': night,
                                       'clf': clf_name,
                                       'target': target})
                df_subj = pd.concat([df_subj, df_tmp], ignore_index=True)
            # except:
                # pass
    # for target in train_ys:
    #     df_target = df_subj.groupby('target').get_group(target)
    #     sns.lineplot(data=df_target, x='timepoint', y=scoring, ax=axs[i], errorbar='sd')
    if len(df_subj)>0:
        sns.lineplot(data=df_subj, x='timepoint', y=scoring._score_func.__name__, 
                     hue='target', ax=axs[i], errorbar='sd')
        df = pd.concat([df, df_subj], ignore_index=True)
    
sns.lineplot(data=df[df['target']!='img_category'], x='timepoint', y=f'{scoring._score_func.__name__}', hue='target')
sns.lineplot(data=df, x='timepoint', y=f'{scoring._score_func.__name__}', hue='target')

#%% try frequency bands
import features
fig, axs, ax_b = plotting.make_fig(len(data), n_bottom=[0,0,1])

df = pd.DataFrame()

for i, night in enumerate(tqdm(data, desc='subj')):
    times, data_x, data_y = data[night]
    data_y = data_y.copy()
    
    subj = settings.get_subj(night)
    
    times, bands = features.get_bands(data_x, tstep=0.25, n_pca=False, wlen=1, relative=False)
    
    df_subj = pd.DataFrame()
    
    for target, train_y in data_y.items():
        y = train_y.copy()

        if 'subj' in target: 
            continue
                
        if target!='img_category':
            l, u = np.quantile(np.hstack([d[-1][target] for d in data.values()]), [0.2, 0.8])
            y[y<=l], y[(l<y) & (u>y)], y[y>=u] = 0, 1, 2

        for clf_name, clf in clfs.items():

            scores = cross_val_multiscore(clf, bands, y, n_jobs=5, cv=5, 
                                          verbose=False)
            # scores = gaussian_filter(scores, (0, 2))
            
            df_tmp = pd.DataFrame({'accuracy': scores.ravel(),
                                   'timepoint': [x for x in times-0.5]*5,
                                   'night': night,
                                   'clf': clf_name,
                                   'target': target})
            df_subj = pd.concat([df_subj, df_tmp], ignore_index=True)

    if len(df_subj)>0:
        sns.lineplot(data=df_subj, x='timepoint', y='accuracy' ,
                     hue='target', ax=axs[i], errorbar='sd')
        df = pd.concat([df, df_subj], ignore_index=True)
    plt.pause(0.1)
    
# sns.lineplot(data=df[df['target']!='img_category'], x='timepoint', y='accuracy', hue='target')
sns.lineplot(data=df, x='timepoint', y='accuracy', hue='target', style='clf')

#%% try PyRiemann
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

fig, axs, ax_b = plotting.make_fig(len(data), n_bottom=[0,0,1])

# load your data

wlen = 20

accs = []
df = pd.DataFrame()
for i, night in enumerate(tqdm(data, desc='subj')):
    times, data_x, data_y = data[night]
    
    # subj = settings.get_subj(night)
 
    data_x = data_x[:,:,:]
    # build your pipeline
    covest = Covariances('oas')
    ts = TangentSpace()
    svc = SVC(kernel='linear')
    clf = make_pipeline(covest, ts, svc)

    for target, train_y in data_y.items():
        if target=='img_category': continue
        # cross validation
        for t in np.arange(0, data_x.shape[-1]-wlen, 10):
            train_x = data_x[:,:,t:t+wlen]
            accuracy = cross_val_score(clf, train_x, train_y.round().astype(int), cv=5, n_jobs=-1)
            df_tmp = pd.DataFrame({'accuracy': accuracy.mean(),
                                   'target': target,
                                   'night': night,
                                   'wlen': wlen,
                                   'timepoint': t/100-0.5}, index=[0])
            df = pd.concat([df, df_tmp], ignore_index=True)            
    ax = axs[i]
    
    sns.lineplot(data=df[df['night']==night], x='timepoint', y='accuracy', hue='target', ax=ax)
    plt.pause(0.01)
    
sns.lineplot(data=df, x='timepoint', y='accuracy', hue='target', ax=ax_b)
    
#%% try TPOTClassifier
#  see also https://epistasislab.github.io/tpot/api/
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
#%% try XDAWN
tmin = -0.1
tmax = 0.5
sfreq = 150
for folder in folders[:1]:
    fig, axs = plt.subplots(2, 2, figsize=[5,8]); axs=axs.flatten()
    
    fig.suptitle(f'Classifier Target: VALENCE & AROUSAL \n {folder}\n{sfreq=}')
    
    tqdm_loop = tqdm(total=2*len(clfs))
    epochs, (train_y_valence, train_y_arousal) = load_localizer(folder, sfreq=sfreq,
                                                                     tmin=tmin, tmax=tmax, 
                                                                     event_id=trigger.STIMULUS,
                                                                     return_epochs=True)
    epochs.load_data()

    epochs.filter(1, 30, fir_design='firwin')
    eeg_channels = mne.pick_types(epochs.info, eeg=True)
    epochs.pick(eeg_channels)
    
    mask = (np.abs(train_y_valence)!=2) & (train_y_arousal>1)
    epochs = epochs[mask]
   
    train_y_valence = train_y_valence[mask]
    train_y_arousal = train_y_arousal[mask]
    
    # epochs = zscore(epochs, axis=2)
    
    clf = make_pipeline(Xdawn(n_components=10),
                        Vectorizer(),
                        MinMaxScaler(),
                        LogisticRegression(penalty='l1', solver='liblinear',
                                           multi_class='auto'))

    
    # Cross validator
    
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for name, target in {'valence':train_y_valence, 'arousal':train_y_arousal}.items():
        # Do cross-validation
        preds = np.empty(len(target))
        for train, test in cv.split(epochs, target):
            clf.fit(epochs[train], target[train])
            preds[test] = clf.predict(epochs[test])
    
        
        # Classification report
        # target_names = ['aud_l', 'aud_r', 'vis_l', 'vis_r']
        target_names = np.unique(target)
        # report = classification_report(labels, preds, target_names=target_names)
        # print(report)
        
        # Normalized confusion matrix
        cm = confusion_matrix(target, preds)
        cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        fig, ax = plt.subplots(1)
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set(title='Normalized Confusion matrix')
        fig.colorbar(im)
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        fig.tight_layout()
        ax.set(ylabel='True label', xlabel='Predicted label')
        fig.suptitle(f'{name}')