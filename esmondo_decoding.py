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
from sklearn.metrics import (mean_squared_error, make_scorer,
                             accuracy_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import (GridSearchCV, cross_val_score, 
                                     StratifiedKFold, train_test_split)
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from mne import io, pick_types, read_events, Epochs, EvokedArray, create_info
from mne.datasets import sample
# from mne.preprocessing import Xdawn
# from mne.time_frequency import psd_multitaper
# from mne.time_frequency import cwt_morlet


from mne_features.feature_extraction import extract_features

from scipy.stats import zscore
from scipy.ndimage import gaussian_filter

from settings import trigger
import settings
import plotting

from data_loading import get_night, get_learning_type, get_subj
from data_loading import load_localizer, load_test
import yasa



#%% Data Loading (also Preprocessing)
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

stop

#%% Data Storage (Predictor and Labels)

# List of data names from all session
data_name = list(data_test.keys())

# Take example data from subj 1 (PN09-high-before)
X_times, X, y = data_test[data_name[22]]

# Label dataset in dataframe format
y_df = pd.DataFrame(y)


# Separating new/old image class
y_truth = y_df.seen_before_truth.replace({True: 1, False: 0})
y_resp = y_df.seen_before_response.replace({True: 1, False: 0})


# indices of 0 and 1 from y_resp
class_idx_0 = np.where(y_resp == 0)[0]
class_idx_1 = np.where(y_resp == 1)[0]

# number of samples in each class 
n_class_0_resp = len(class_idx_0)
n_class_1_resp = len(class_idx_1)

# n_class_0_truth = np.sum(y_truth == 0)
# n_class_1_truth = np.sum(y_truth == 1)
# n_class_0_resp = np.sum(y_resp == 0)
# n_class_1_resp = np.sum(y_resp == 1)

# Find class with minimum samples in y_resp
min_class = 0 if n_class_0_resp < n_class_1_resp else 1

# Find the indices of the classes with minimum samples in y_resp
min_class_idx = class_idx_0 if min_class == 0 else class_idx_1
# min_class_idx = np.where(y_resp == min_class)[0] #alternative

# Randomly select the same number of samples from class with more samples
np.random.seed()
random_idx = np.random.choice(class_idx_1 if min_class == 0 else class_idx_0, size=n_class_0_resp, replace=False)

# Combine indices from the two classes
combined_idx = np.concatenate([min_class_idx, random_idx])

# Balanced data
X = X[combined_idx,:,:]
y_resp = y_resp[combined_idx]
y_truth = y_truth[combined_idx]



#%% Feature Extraction
features_example = {'mean', 'ptp_amp', 'std', 'var, ''mdian'} # func_params
def extrat_timewindows():
    wins = []
    for w in windows:
        X = extract_features(X, sfreq, features_example) #returns (n_epochs, n_features)
        wins.append(X_feats
    return []
#%% Assemble Classifier Pipeline


# PCA (Dimensionality Reduction)
pca = PCA(n_components=0.95)

# ANOVA (to select channels that contain significant stimulus-specific information)
# Selecting features according to the k highest score
anova = SelectKBest(f_classif, k=5)
# anova = f_classif

# Classifier
logreg = LogisticRegression(penalty='l1', C=1/1, solver='liblinear')
clf = RandomForestClassifier(100)

# Scaling method
scaler = StandardScaler()

# Pipelines
pipelines = {
    'None': make_pipeline(scaler, logreg),
    'PCA': make_pipeline(scaler, pca, logreg),
    'ANOVA': make_pipeline(scaler, anova, logreg)
    }

# pipelines1 = [('pca', pca),('anova',anova),('none',None)]
from pysnooper import snoop

@snoop()
def extract(X):
    for w in range(0,301, 50):
        X_win = X[w:w+50]
extract(X)

# Sliding estimator
time_decoding = SlidingEstimator(make_pipeline(scaler, pca, logreg), n_jobs=1, scoring='accuracy', verbose=True)





#%% Cross-Validation

# Initialize StratifiedKFold
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds,
                      shuffle=True,
                      random_state=0)

# Score temp for results
# accuracy_resp = np.zeros(n_folds)
# accuracy_truth = np.zeros(n_folds)
scores = []

for i, (train_idx, test_idx) in enumerate(skf.split(X, y_resp)):
    
    # X_train, y_train = X[train_idx], y_resp[train_idx]
    # X_test, y_test = X[test_idx], y_resp[test_idx]
       
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_resp[train_idx], y_resp[test_idx]
    y_truth_train, y_truth_test = y_truth[train_idx], y_truth[test_idx]
   
    # Fit pipeline on training data
    time_decoding.fit(X_train, y_train)
   
    # Predict response on test data
    y_pred = time_decoding.predict(X_test)
    # accuracy_resp[i] = accuracy_score(y_resp_test, y_resp_pred)
   
    # Predict truth on test data
    y_true = y_resp[test_idx]
    # accuracy_truth[i] = accuracy_score(y_truth_test, y_truth_pred)
    
    accuracy = np.mean(y_pred == y_true)
    recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
    sensitivity = np.sum((y_pred == 0) & (y_true == 0)) / np.sum(y_true == 0)
    
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Sensitivity: {sensitivity:.2f}')
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    
    # Decoding score
    score = cross_val_multiscore(time_decoding, X, y_resp, cv=skf, n_jobs=1)
    scores.append(score)

# # Print mean accuracy for response and truth labels
# print(".\n.\n.\n")
# print("Mean accuracy for response:", np.mean(accuracy_resp))
# print("Mean accuracy for truth:", np.mean(accuracy_truth))

# plot
fig, ax = plt.subplot()
ax.plot(times, np.mean(scores, axis=0))
ax.axhline(.5, color='k', linestyle='--', label='Chance')
ax.set_xlabel('Time (s)')
ax.set_ylabel('AUC')
plt.legend()
plt.show()

stop


