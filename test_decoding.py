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

import settings
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


print(__doc__)

data_path = sample.data_path()


#%% settings
folders = [f'{settings.data_dir}/Raw_data/PN4/Experimental night 1/',
           f'{settings.data_dir}/Raw_data/PN4/Experimental night 2/']

mse = make_scorer(mean_squared_error, greater_is_better=False)
tmin= -0.5
tmax = 1.5

clf_rfr = SlidingEstimator(RandomForestRegressor(500), n_jobs=-1)
clf_rfc = SlidingEstimator(RandomForestClassifier(500), n_jobs=-1)
clf_svr = SlidingEstimator(SVR(), n_jobs=-1)
clf_svc = SlidingEstimator(SVC(), n_jobs=-1)
clf_log = SlidingEstimator(LogisticRegression(C=100, penalty='l1', 
                                               solver='liblinear', max_iter=1000), 
                            n_jobs=-1)
csp = CSP(n_components=3, norm_trace=False)
clf_csp = make_pipeline(csp,
                        LinearModel(LogisticRegression(solver='liblinear'))
                        )


clfs = {
        # 'CSP': clf_csp,
        'SVR': clf_svr,
        'SVC':clf_svc,
        'RFR': clf_rfr, 
        'RFC':clf_rfc,
        'LogReg': clf_log
        }
colors = sns.color_palette()

#%% run calculations
# plt.maximize=False
for sfreq in [100]:
    for folder in folders[1:]:
        fig, axs = plt.subplots(len(clfs), 2, figsize=[5,8]); axs=axs.flatten()
        
        fig.suptitle(f'Classifier Target: VALENCE & AROUSAL \n {folder}\n{sfreq=}')
        
        tqdm_loop = tqdm(total=2*len(clfs))
        times, train_x, (train_y_valence, train_y_arousal) = load_localizer(folder, sfreq=sfreq,
                                                                         tmin=tmin, tmax=tmax, 
                                                                         event_id=trigger.STIMULUS)
        mask = (np.abs(train_y_valence)!=2) & (train_y_arousal>1)
        train_x = train_x[mask]
        train_y_valence = train_y_valence[mask]
        train_y_arousal = train_y_arousal[mask]
        train_x = zscore(train_x, axis=2)
        scores = {name:{} for name, clf in clfs.items()}
    
        for i, (name, clf) in enumerate(clfs.items()):
            # first test localizer itself
            scores[name]['valence'] = cross_val_multiscore(clf, train_x.copy(), train_y_valence.copy(), 
                                                  n_jobs=1, cv=5, verbose=False)

            scores[name]['arousal'] = cross_val_multiscore(clf, train_x.copy(), train_y_arousal.copy(),
                                                  n_jobs=1, cv=5, verbose=False)        

            
            
            axs[i*2].set_ylabel(name)
            for j, (target, score) in enumerate(scores[name].items()):
                ax = axs[i*2+j]
                ax.clear()
                ax.plot(times, score.mean(0), color=colors[j])
                ax.hlines(score.mean(0)[times<0].mean(), times[0], times[-1], color='gray', linestyle='--')
                ax.vlines(0, *ax.get_ylim(), color='black')
                ax.legend([f'{target} pred', f'{target} pred pre-stim',])
                tqdm_loop.update()
                plt.pause(0.001)
                    
    
        
        fig.tight_layout(pad=2)
        fig.savefig(f'{folder}/predictions sfreq-{sfreq}.png')


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