# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:07:18 2022

Dummy testing of decoders

@author: Simon Kern
"""
import ospath
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import features
import mne_features

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from mne import io, pick_types, read_events, Epochs, EvokedArray, create_info
from mne.datasets import sample
from mne.preprocessing import Xdawn
# from mne.decoding import Vectorizer

from mne_features.feature_extraction import extract_features

from scipy.stats import zscore
from scipy.ndimage import gaussian_filter

from settings import trigger
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

#%% Mondo: Input Data Preparation

# List of data names from all session
data_name = list(data_test.keys())

# Take example data from subj 1 (PN09-high-before)
X_times, X_PN01_hb, y = data_test[data_name[22]]

# Label dataset in dataframe format
y_df = pd.DataFrame(y)

# Some ways to get the column name (label name)
# y_col_df = sorted(y_df) #returns the label name alphabetically
# y_col_df = y_df.columns.values.tolist()
# y_col = list(y.keys()) #returns the label name directly from dict

# Separating new/old image class
# y_img_truth = data_label['seen_before_truth']
# y_img_resp = data_label['seen_before_response']
y_img_truth = y_df.seen_before_truth.replace({True: 1, False: 0})
y_img_resp = y_df.seen_before_response.replace({True: 1, False: 0})

count_0_truth = np.count_nonzero(y_img_truth==0)
count_1_truth = np.count_nonzero(y_img_truth==1)
count_0_resp = np.count_nonzero(y_img_resp==0)
count_1_resp = np.count_nonzero(y_img_resp==1)


#%% Assemble Classifier

# Feature Extraction
features_example = {'mean', 'ptp_amp', 'std'} # func_params
X_PN01_hb = extract_features(X_PN01_hb, sfreq, features_example) #returns (n_epochs, n_features)

# Need to balance classes (Undersampling)

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy='majority')
X, y_resp_rus = rus.fit_resample(X_PN01_hb, y_img_resp)
y_truth_rus = y_img_truth[y_resp_rus]

count_0 = np.count_nonzero(y_resp_rus==0)
count_1 = np.count_nonzero(y_resp_rus==1)
print(f"Number of 0s before undersammpling: {count_0_resp}\nNumber of 0s after undersammpling: {count_0}")
print(f"Number of 1s before undersammpling: {count_1_resp}\nNumber of 1s after undersammpling: {count_1}")


#%% Assemble Classifier

logreg = LogisticRegression(penalty='l1', C=1/1, solver='liblinear')
# lda = LinearDiscriminantAnalysis()
# csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# # Dimensionality Reduction (PCA)
# pca = PCA(n_components=0.95)
# X_PN01_hb = pca.fit_transform(X_PN01_hb)

# Make Pipeline
pipe_1 = Pipeline([('scaler', StandardScaler()), ('lr', logreg)])
# pipe_2 = Pipeline([('CSP', csp), ('LDA', lda)])
# pipe_3 = Pipeline([('pca', pca),('classifier', SVC)])
# pipe_4 = Pipeline([('pca', pca),('clf', LogisticRegression())])
# pipe_5 = Pipeline([('scaler', StandardScaler()),
#                    ('pca', pca,
#                     ('classifier', logreg)
#                     )])
# pipe = make_pipeline(StandardScaler(), logreg)
# pipe = make_pipeline(csp,lda)

# Initialize StratifiedKFold
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds,
                      shuffle=True,
                      random_state=0)

# Score temp for results
accuracy_resp = np.zeros(n_folds)
accuracy_truth = np.zeros(n_folds)

#%% Mondo: Cross-Validation

for i, (train_idx, test_idx) in enumerate(skf.split(X, y_resp_rus)):
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_resp_train, y_resp_test = y_resp_rus[train_idx], y_resp_rus[test_idx]
    y_truth_train, y_truth_test = y_img_truth[train_idx], y_img_truth[test_idx]
   
    # Fit pipeline on training data
    pipe_1.fit(X_train, y_resp_train)
   
    # Predict response on test data
    y_resp_pred = pipe_1.predict(X_test)
    accuracy_resp[i] = accuracy_score(y_resp_test, y_resp_pred)
   
    # Predict truth on test data
    y_truth_pred = pipe_1.predict(X_test)
    accuracy_truth[i] = accuracy_score(y_truth_test, y_truth_pred)

# Print mean accuracy for response and truth labels
print(".\n.\n.\n")
print("Mean accuracy for response:", np.mean(accuracy_resp))
print("Mean accuracy for truth:", np.mean(accuracy_truth))


stop


#%%
# # Split data 80:20
# # feature vector 'X_PN01_hb' is used as the input
# X_train, X_test, y_train_resp, y_test_resp = train_test_split(X_PN01_hb, y_img_resp, test_size=0.2, random_state=0)


# Perform PCA on the training data

# X_train = pca.fit_transform(X_train)

# Apply the same transformation to the test data
# X_test = pca.transform(X_test)
                    

# # Grid Search param to find best hyperparameters for logistic regression
# param_grid = {'clf__C': [0.1, 1, 10, 100],
#               'clf__penalty': ['l1','l2']}

# # perform grid-search with cross-validation
# grid_search = GridSearchCV(pipe_4, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # Fit pipeline to the X_train
# pipe.fit(X_train, y_train_resp)

# # cross-validation
# cv = StratifiedKFold(n_splits=5, random_state=42)


# # evaluate
# # scores = cross_val_score(grid_search.best_estimator_, X_test, y_test, cv=5)
# scores = cross_val_score(X_train, y_train_resp, cv=cv)
# print("cross-validation accuracy: %.2f +/- %.2f" % (scores.mean(), scores.std()))

# # make prediction on the X_test
# y_pred = pipe.predict(X_test)

# print(classification_report(y_test_resp. y_pred))

# print("accuracy between predicted labels and true labels: %0.2f" % np.mean(y_pred == ))






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