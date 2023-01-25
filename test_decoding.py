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
from sklearn.model_selection import cross_val_score
import numpy as np
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
from data_loading import get_night, get_learning_type, get_subj
from data_loading import load_localizer, load_test
import yasa



#%% data loading
# this cell loads the data of the localizers into memory

# list all folders in the data_dir
# the location of the data_dir has to be defined in the settings.py
# for each individual host machine
folders_subj = ospath.list_folders(f'{settings.data_dir}/Raw_data/', pattern='PN*')

#  settings for data loading
sfreq = 100  # sampling frequency that signals will be downsampled to
tmin= -1     # time before stimulus onset that is loaded in seconds
tmax = 2     # time after stimulus onset that is loaded in seconds

data = {}   # this variable will hold all data of the localizer
data_test = {}  # dictionary to hold the data of testing

for folder in tqdm(folders_subj, desc='loading participant localizers'):
    # in each folder there are two separate recording days/nights
    nights_folders = ospath.list_folders(folder, pattern='*night*')
    # retrieve name of subject, i.e. "PN05"
    subj = get_subj(folder)
    # for each subject, load the two experiment days
    for folder_night in nights_folders:
        # retrieve which learning type we have: low or high arousal images
        night = get_learning_type(folder_night)
        # load the data of the localizer of that specific day
        times, data_x, data_y  = load_localizer(folder_night, sfreq=sfreq,
                             tmin=tmin, tmax=tmax, event_id=trigger.STIMULUS)

        # n_epochs = ie. ~96 minus rejected epochs
        # n_samples = ie. ~301 msamples, 3.01 seconds with 100 Hz
        # times  = array of size(n_samples, ) times in seconds for each sample
        # data_x = array of size(n_epochs, n_electrodes, n_samples)
        # data_y = dictionary with 5 different entries:
        #          each entry has size (n_epochs, )
        #           valence_subj: the ratings that this participant gave for the image shown in the specific epoch
        #           valence_mean: the mean rating of 100 OASIS participants for the image shown in the specific epoch
        #           arousal_subj: the ratings that this participant gave for the image shown in the specific epoch
        #           arousal_mean: the mean rating of 100 OASIS participants for the image shown in the specific epoch
        #           img_category: the image category of the images, i.e. object, animal, person, scene

        # save data in the data dictionary, for later access
        data[f'{subj}-{night}'] = times, data_x, data_y

        for which in ['before', 'after']: # also load the data of the test session
            times, data_x, data_y  = load_test(folder_night, which=which, sfreq=sfreq,
                                               tmin=tmin, tmax=tmax, event_id=trigger.STIMULUS)

            data_test[f'{subj}-{night}-{which}'] = times, data_x, data_y
    # except:
        # print(f'ERROR: {folder}')




#%% settings

# mse = make_scorer(mean_squared_error, greater_is_better=False)

# for now, use e.g. RandomForestClassifier or SVM classifiers or LogReg
# clf = RandomForestClassifier(200)
# clf = SVC()
clf = LogisticRegression(penalty='l1', C=1/1, solver='liblinear')


# parallelize the classifier to run on 3D arrays of (n_epochs, n_electrodes, n_samples)
# SlidingEstimator simply creates an individual classifier for each element
# in the last dimension (ie. time dimension)
clf = SlidingEstimator(clf, n_jobs=-1, scoring='f1_macro')

# these are the thresholds for binarization of the values,
# i.e. mapping from 0-4 => false/true, or in other words low/high
thresholds = {'valence_subj':2, 'arousal_subj':2,
              'valence_mean':3, 'arousal_mean':3}

# this is a stop sign to make the script not execute the rest of the file
stop

#%% try: raw electrode values
# the most simple approach: using raw sensor values without any
# feature extraction. This works well with another paradigm that we are using
# that uses MEG data and displays images. As the image onset is very sharp
# this works well for the image categories. But probably emotional reactions
# are not very time specific and can happen slower or faster, so we cannot
# use this approach for decoding arousal/valence

# open a plot with `axs` for the individual localizers, and a bottom plot summary
fig, axs, ax_b = plotting.make_fig(len(data), n_bottom=[0,0,1])

# save results of all participants in this data frame
df = pd.DataFrame()

# loop over all localizer data, i.e. "PN01-low" or "PN07-high"
for i, day_desc in enumerate(tqdm(data, desc='subj')):
    times, data_x, data_y = data[day_desc]

    subj = get_subj(day_desc)

    # zscore data across time
    train_x = zscore(data_x, axis=2)

    # save results of current participant and day in this dataframe
    df_subj = pd.DataFrame()
    for target, train_y in data_y.items():

        # binarize the train_y from 0-4 => low/high
        if target in thresholds:
            train_y = train_y.copy() # create copy to not overwrite anything
            train_y = train_y>thresholds[target]

        # sanity check: if there are fewer than 5 values in any class
        # skip this day. This can happen if a participant rates
        # all 96 epochs as 0 or 1
        if any(np.bincount(train_y)<5):
            print(f'Too few values in classes: {day_desc} {target=} '\
                  f'{np.bincount(train_y)}')
            continue

        # run classification
        # cross_val_multiscore runs a cross validation on the data with
        # 5 folds and returns the score, i.e. the accuracy
        cv = StratifiedKFold(5)  # stratified keeps classes in the same balance across folds
        # NB: using f1-score as classes are imbalanced
        scores = cross_val_multiscore(clf, train_x, train_y, cv=cv)

        # apply gaussian smoothing to make lines a bit less noisy.
        # this should not be done in the final classifier/paper
        # just to get a better feeling if something happens or not
        scores = gaussian_filter(scores, (0, 1.3))

        # create temporary dataframe with results
        df_tmp = pd.DataFrame({'f1_macro': scores.ravel(),
                               'timepoint': [x for x in times]*5,
                               'desc': day_desc,
                               'target': target})
        df_subj = pd.concat([df_subj, df_tmp], ignore_index=True)

    # plot current subject
    sns.lineplot(data=df_subj, x='timepoint', y='f1_macro',
                 hue='target', ax=axs[i], errorbar='sd')
    df = pd.concat([df, df_subj], ignore_index=True)
    plt.pause(0.1)

    # print mean of all subjects
    ax_b.clear()
    sns.lineplot(data=df, x='timepoint', y='f1_macro', hue='target', ax=ax_b)
    fig.tight_layout()
#%% use mne_features on data from TEST
# here we try to classify during the test whether the images have been seen before
# or whether the images were novel to the participant

from mne_features.feature_extraction import extract_features

fig, axs, ax_b = plotting.make_fig(len(data), n_bottom=[0,0,1])

df = pd.DataFrame()

# define the feature names that we want to extract
# these is the exhaustive list of all features that can be extracted
features = ['app_entropy', 'decorr_time', 'energy_freq_bands', 'higuchi_fd',
            'hjorth_complexity', 'hjorth_complexity_spect', 'hjorth_mobility',
            'hjorth_mobility_spect', 'hurst_exp', 'katz_fd', 'kurtosis',
            'line_length', 'mean', 'pow_freq_bands', 'ptp_amp', 'quantile',
            'rms', 'samp_entropy', 'skewness', 'spect_edge_freq',
            'spect_entropy', 'spect_slope', 'std', 'svd_entropy', 'svd_fisher_info',
            'teager_kaiser_energy', 'variance', 'wavelet_coef_energy', 'zero_crossings']

# these are the targets we are going to use here
# only categorical targets, that makes it a bit easier
targets = ['seen_before_truth', 'quad_truth', 'seen_before_response',
           'quad_resp', 'img_category']

# these are the frequency bands that we want to extract.
# is is only relevant for energy_freq_bands and pow_freq_bands
bands = np.array([0.5, 4, 8, 13, 30])

# use this classifier for now
clf = LogisticRegression(penalty='l1', C=1/1, solver='liblinear')

for i, which_test in enumerate(tqdm(data_test, desc='subj')):
    times, data_x, data_y = data_test[which_test]
    data_y = data_y.copy()

    # get name of subject

    # extract all univarate features that were defined above
    # will take a few seconds/minutes
    # this feature extraction will transform the data from a 3D matrix to 2D
    # that means, we will lose the time dimension
    train_x = extract_features(data_x, sfreq, features,
                                # weird format that the parameter needs to be passed by
                                funcs_params={'energy_freq_bands__freq_bands':bands,
                                              'pow_freq_bands__freq_bands': bands})
    # now we have a ton of features and need to do some feature reduction
    # currently we are extracting >3000 features

    # ... apply eg PCA() for feature reduction, insert code here


    df_subj = pd.DataFrame()
    for target in targets:
        train_y = data_y[target].copy()
        train_y[np.isnan(train_y)] = -1  # transform possible NaN values to -1
        cv = StratifiedKFold()
        scores = cross_val_score(clf, train_x, train_y, cv=cv, scoring='f1_macro')
        df_tmp = pd.DataFrame({'score': scores.ravel(),
                               'which_test': which_test,
                               'clf': str(clf),
                               'target': target})
        df_subj = pd.concat([df_subj, df_tmp], ignore_index=True)

    if len(df_subj)>0:
        sns.barplot(data=df_subj, x='target', y='score' ,
                     ax=axs[i], errorbar='sd')
        df = pd.concat([df, df_subj], ignore_index=True)
    plt.pause(0.1)

sns.barplot(data=df, x='target', y='score', ax=ax_b)
ax_b.hlines([0.5, 0.25], *ax_b.get_xlim())
#%% try frequency bands
# a bit more sophisticated approach: transform into frequency bands
# the most simple case here would be to transform into frequency space
# and to bin the results according to common brain bands delta beta alpha etc

import features
import mne_features

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
        stop
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