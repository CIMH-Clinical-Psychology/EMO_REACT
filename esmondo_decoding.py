# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:07:18 2022

Dummy testing of decoders

@author: Ardiansyah Esmondo and Simon Kern
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
import time
import eeglib
import pyeeg

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_squared_error, make_scorer,
                             accuracy_score, recall_score, precision_score,
                             f1_score,roc_curve, auc,
                             classification_report, confusion_matrix)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import (GridSearchCV, cross_val_score, 
                                     StratifiedKFold, train_test_split,
                                     cross_val_predict)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
from mne import io, pick_types, read_events, Epochs, EvokedArray, create_info
from mne.datasets import sample



from mne_features.feature_extraction import extract_features, FeatureExtractor


from scipy.stats import zscore
from scipy.ndimage import gaussian_filter

from settings import trigger
import settings
import plotting

from data_loading import get_night, get_learning_type, get_subj
from data_loading import load_localizer, load_test
import yasa

from joblib import Parallel, delayed

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
            
        


#%% Data Storage (Predictor and Labels)


# frontal = [ch for ch in settings.ch_names if ch.startswith('F')]


# List of data names from all session
data_name = list(data_test.keys())

high_before = []
high_after = []
low_before = []
low_after = []

for name in data_name:
    if "high-before" in name:
        high_before.append(name)
    elif "high-after" in name:
        high_after.append(name)
    elif "low-before" in name:
        low_before.append(name)
    elif "low-after" in name:
        low_after.append(name)

#%% Initiate X and y from specific dataset group

results_df = pd.DataFrame(columns=['dataset', 'accuracy'])

for subject in high_before:

    # Take data from the chosen dataset above
    X_times, X, y = data_test[subject]
    
    # Label dataset in dataframe format
    y_df = pd.DataFrame(y)
    
    
    # Separating new/old image class
    y_truth = y_df.seen_before_truth.replace({True: 1, False: 0})
    y_resp = y_df.seen_before_response.replace({True: 1, False: 0})
    y_truth = y_truth.values
    y_resp = y_resp.values
    
    # indices of 0 and 1 from y_resp
    class_idx_0 = np.where(y_resp == 0)[0]
    class_idx_1 = np.where(y_resp == 1)[0]
    
    # number of samples in each class 
    n_class_0_resp = len(class_idx_0)
    n_class_1_resp = len(class_idx_1)
    
    
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
    y_true = y_truth[combined_idx]
    
    
    #%% Standardize Data
    scaler = StandardScaler()
    
    n_epochs, n_channels, n_samples = X.shape
    
    # Reshape the data to a 2D array with shape (n_epochs * n_channels, n_samples)
    X_reshaped = X.reshape(-1, n_samples)
    
    # Scaling
    X_scaled = scaler.fit_transform(X_reshaped)
    
    # Reshape the scaled data back to its original shape
    X_scaled = X_scaled.reshape(n_epochs, n_channels, n_samples)
    
    #%% Time Window Extraction
    
    from sklearn.feature_extraction.image import _extract_patches
    def extract_time_windows(arr, wlen, step, axis=-1):
        """
            Parameters
        ----------
        arr : np.ndarray
            input array of arbitrary dimensionality
        wlen : int
            window length in sample points
        step : int
            steps in sample points between window starts
        axis : in, optional
            Along which axis to extract, e.g -1 if the time dimension is the last
            dimension.The default is -1.
    
        Returns
        -------
        windows : np.ndarray
            extracted windows. first dimension is the number of windows
        """
        patch_shape = list(arr.shape)
        patch_shape[axis] = wlen
        windows = _extract_patches(arr, patch_shape, extraction_step=step)
        windows = windows.squeeze()
        # arrays are views, so no changing of values allowed for safety
        windows.setflags(write=False) 
        
        return windows
    
    
    window_length = int(sfreq * 0.1)
    step = 1
    
    # extract time windows (n_windows, n_epochs, n_channels, n_samples)
    start_time = time.time()
    X_timewindows = extract_time_windows(X_scaled, window_length, step)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time of X_timewindows: {elapsed_time:.2f} seconds")
    
    
    
    #%% Feature Extraction without MNE
    
    freq_bands = np.array([0.5, 4., 8., 13., 30.])
    feature_names=['pow_freq_bands']
    
    n_timewindows, n_epochs, n_channels, n_samples = X_timewindows.shape
    n_features = len(freq_bands) * n_channels
    
    X_features = []
    
    for i in range(n_timewindows):
        X_features += [extract_features(X_timewindows[i,:,:,:], sfreq, feature_names, 
                                      {'pow_freq_bands__freq_bands':freq_bands})]
    
    # X_features shape is (n_windows, n_epochs, n_features)    
    X_features = np.array(X_features)
    
    
    
    #%% Dimensionality Reduction
    
    n_timewindows, n_epochs, n_features = X_features.shape
    n_reduced = 50
    # 95 percent of the variance
    
    pca = PCA(n_components=40)
    
    X_pca = []
    
    for i in range(X_features.shape[0]):
        
        X_window = X_features[i,:,:]
        
        X_window_pca = pca.fit_transform(X_window)
        
        X_pca += [X_window_pca]
        
    X_pca_con = np.array(X_pca)
    
    
    #%% Decoding Time Windows
    
    clf = LogisticRegression()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get the total number of windows etc.
    n_timewindows, n_epochs, n_features = X_pca_con.shape
    
    
    
    def decode(i, X, y_resp, y_true):
        y_pred = np.zeros_like(y_resp)
        
        for train, test in cv.split(X, y_resp):
            clf.fit(X[train], y_resp[train])
            y_pred[test] = clf.predict(X[test])
            
        results = {}    
        results['accuracy'] = accuracy_score(y_resp, y_pred)
        results['recall'] = recall_score(y_resp, y_pred)
        results['precision'] = precision_score(y_resp, y_pred)
        results['f1'] = f1_score(y_resp, y_pred)
        results['confmat'] = confusion_matrix(y_resp, y_pred)
        
        return results
    
    # Call the decode function for each window in parallel
    res = Parallel(40)(delayed(decode)(i, X, y_resp, y_true) for i, X in enumerate(tqdm(X_pca_con, total=n_timewindows)))
    
    
    results = {
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1': [],
        'confmat': []
    }
    
    # Aggregate the results across all windows
    for i, rr in enumerate(res):
        results['accuracy'].append(rr['accuracy'])
        results['recall'].append(rr['recall'])
        results['precision'].append(rr['precision'])
        results['f1'].append(rr['f1'])
        results['confmat'].append(rr['confmat'])
    
    # Convert the lists to numpy arrays
    for key in results.keys():
        results[key] = np.array(results[key])
    
    
 
    
    #%% Plotting
    
    # time_axis = np.arange(n_timewindows) * window_length + window_length / 2
    
    # time_start = times[0]
    # time_end = times[-1]
    
    # x_labels = np.linspace(time_start, time_end, n_timewindows, endpoint=False)
    
    # fig, ax = plt.subplots()
    
    # ax.plot(x_labels, gaussian_filter(results['accuracy'], 1), label='Accuracy')
    # ax.axvline(0, color='k', linestyle='--', label='Onset')
    # ax.axhline(0.5, color='r', linestyle='--', label='Chance')
    
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Decoding Accuracy')
    # ax.set_ylim([0.3, 0.7])
    # ax.legend(loc='lower right')
    
    # # plt.plot(x_labels, gaussian_filter(results['accuracy'], 1))
    # # plt.xlabel('Time (s)')
    # # plt.ylabel('Accuracy')
    # # onset_time = 0.0  # replace with your desired onset time
    # # plt.axvline(x=onset_time, linestyle='--', color='gray')
    # # plt.show()



    #%% after looping all subjects
    results_df = results_df.append({'dataset': subject,
                                   'accuracy': results['accuracy']},
                                   ignore_index=True)
