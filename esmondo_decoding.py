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
import time

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

# stop

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
y_truth = y_truth[combined_idx]

# stop


#%% Extract Window Function

# Function 1
def extract_time_windows1(X, window_size, step_size):
# def extract_time_windows1(X, sfreq, window_size, step_size):
    
    # n_epochs, n_channels, n_samples = X.shape
    n_samples = X.shape[-1]
    
    # Define the sliding time windows
    # window_samples = int(window_size * sfreq)
    # step_samples = int(step_size * sfreq) - 1
       
    windows = []
    n_windows = 0
    
    for i in range(0, n_samples - window_size + 1, step_size):
                
        if i == step_size:
            start = i + 1
        else:
            start = i   
        end = start + window_size - 1
        print("Extracting time window within {} and {} sample points".format(start, end))
        
       
        if len(list(range(start,end+1)))==window_size:
            
            print(f"Time window length is {window_size} ms")
        else:
            raise ValueError("start and end sample points might not equal to desired window size")
        
        X_window = X[:, :, start:end]
        # selected_features = {'mean', 'ptp_amp', 'std'}
        # X_features = extract_features(X_window, sfreq, selected_features)       
        # windows.append(X_features)
        windows.append(X_window)
        n_windows += 1
        print(f"Extracting time window {n_windows}")

    return np.array(windows)

# Function 2
from sklearn.feature_extraction.image import _extract_patches
def extract_time_windows2(arr, wlen, step, axis=-1):
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


# Function 3
def extract_time_windows3(eeg_data, window_size, step_size):
    num_trials, num_electrodes, num_samples = eeg_data.shape
    
    # Calculate the number of windows that will fit in the EEG data
    num_windows = int(np.floor((num_samples - window_size) / step_size) + 1)
    
    # Create an empty array to store the extracted windows
    extracted_windows = np.zeros((num_trials, num_electrodes, num_windows, window_size))
    
    # Loop through each trial and electrode
    for trial in range(num_trials):
        for electrode in range(num_electrodes):
            # Extract windows for the current trial and electrode
            for i in range(num_windows):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                extracted_windows[trial, electrode, i, :] = eeg_data[trial, electrode, start_idx:end_idx]
    
    return extracted_windows.swapaxes(0, 2).swapaxes(1,2)

#%% Example
array = np.random.rand(96, 64, 301)
wlen = 10
step = 1

start_time = time.time()
w1 = extract_time_windows2(array, wlen, step)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time of w1: {elapsed_time:.2f} seconds")

start_time = time.time()
w2 = extract_time_windows3(array, wlen, step)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time of w2: {elapsed_time:.2f} seconds")

np.testing.assert_array_equal(w1, w2)

#%% Time Windows Extraction

# Define the time window size and step size in seconds
# if the step is equal to the windos size, there will be no overlap between
# consecutive windows.
# if the step is smaller than window size, we could extract features at higher
# temporal resolution (more computational work).
# window_size = 100/1000  # 100 ms
# step_size = (11/1000)

window_length = int(sfreq * 0.1)
step = 1

# # extract time windows 1 (n_windows, n_epochs, n_channels, n_samples)
# start_time = time.time()
# X_timewindows = extract_time_windows1(X, window_length, step)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time of X_timewindows1: {elapsed_time:.2f} seconds")

# extract time windows 2 (n_windows, n_epochs, n_channels, n_samples)
start_time = time.time()
X_timewindows = extract_time_windows2(X, window_length, step)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time of X_timewindows2: {elapsed_time:.2f} seconds")

# # extract time windows 3 (n_windows, n_epochs, n_channels, n_samples)
# start_time = time.time()
# X_timewindows = extract_time_windows3(X, window_length, step)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time of X_timewindows3: {elapsed_time:.2f} seconds")

#%% Assemble Features and Classifiers


# Classifier and cross-validation
scaler = StandardScaler()
clf = LogisticRegression()
# pca = PCA(n_components=0.95)
# anova = SelectKBest(f_classif, k=5)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get the total number of windows etc.
n_timewindows, n_epochs, n_channels, n_samples = X_timewindows.shape
# n_timewindows1 = X_timewindows1.shape[0]

# Set up the results storage
# results = []
results = {
    'accuracy': np.zeros(n_timewindows),
    'recall': np.zeros(n_timewindows),
    'precision': np.zeros(n_timewindows),
    'f1': np.zeros(n_timewindows),
    'confmat': np.zeros((n_timewindows, 2, 2)),
}

freq_bands = {'delta': (0.5, 4),
              'theta': (4, 8),
              'alpha': (8, 13),
              'beta': (13, 30),
              'gamma': (30, 50)}

# selected_features = {'mean', 'ptp_amp', 'std'}
selected_features = {'app_entropy'}
# selected_features = list(freq_bands.keys())



stop

#%% Decoding Time Windows
    
for i, X_window in enumerate(X_timewindows):
    print(f"Processing time window {i+1} of {n_timewindows}")
    
    # each iteration represents each time window
    # in each iteration, the shape of X_window is (n_epochs, n_channels, n_samples)
    
    # Split the data into training and testing indices
    n_trials = X_window.shape[0]
    split_idx = np.random.permutation(n_trials)
    split_point = int(n_trials * 0.8)
    train_indices, test_indices = split_idx[:split_point], split_idx[split_point:]
    
    # Get the training data
    X_train = X_window[train_indices,:,:]
    y_train = y_resp[train_indices]
    
    # Get the testing data
    X_test = X_window[test_indices,:,:]
    y_test = y_resp[test_indices]
    
    # Get the true label for calculating performance metrics
    y_train_true = y_truth[train_indices]
    y_test_true = y_truth[test_indices]
    
    # Scale the training data
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_channels*n_samples))
    X_train_scaled = X_train_scaled.reshape(-1, n_channels, n_samples)

    # Scale the testing data
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_channels*n_samples))
    X_test_scaled = X_test_scaled.reshape(-1, n_channels, n_samples)
    
    # Extract features using entropy
    X_train_features = extract_features(X_train, sfreq, selected_features)
    X_test_features = extract_features(X_test, sfreq, selected_features)
    
    # Train the classifier on the training data
    clf.fit(X_train_features, y_train)
    
    # Make predictions on the testing data
    y_test_pred = clf.predict(X_test_features)


    # Perform classification using stratified cv on training data
    y_pred = cross_val_predict(clf, X_train_features, y_train, cv=cv)
    
    # Train the classifier on the full training set
    clf.fit(X_train_features, y_train)
    
    # Test the classifier on the testing set
    y_pred_test = clf.predict(X_test_features)
    
    # Evaluate performance on training set
    accuracy = accuracy_score(y_train_true, y_pred)
    recall = recall_score(y_train_true, y_pred)
    precision = precision_score(y_train_true, y_pred)
    f1 = f1_score(y_test_true, y_test_pred)
    confmat = confusion_matrix(y_train_true, y_pred)
    # results.append((accuracy,recall,precision,f1))
    
    # tp = np.sum((y_pred == 1) & (y_train_true == 1))
    # tn = np.sum((y_pred == 0) & (y_train_true == 0))
    # fp = np.sum((y_pred == 1) & (y_train_true == 0))
    # fn = np.sum((y_pred == 0) & (y_train_true == 1))
    
    # cm = np.array([[tn, fp], [fn, tp]])
    
    # Store the results for training data
    results['accuracy'][i] = accuracy
    results['recall'][i] = recall
    results['precision'][i] = precision
    results['f1'][i] = f1
    results['confmat'][i] = confmat
    


#%% Plotting
# Create a time axis in seconds
time_axis = np.arange(n_timewindows) * window_length + window_length / 2
window_starts = np.arange(0, n_timewindows * window_length, window_length) * sfreq

x_labels = times[window_starts]

acc = [r[0] for r in results]
# Plot the results
# plt.plot(range(0, n_timewindows), acc, label='Accuracy')
plt.plot(time_axis, results['accuracy'], label='Accuracy')
# plt.plot(time_axis, results['recall'], label='Recall')
# plt.plot(time_axis, results['precision'], label='Precision')

# Add labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Score')
plt.xticks(range(0, n_timewindows))
plt.legend()

# Show the plot
plt.show()