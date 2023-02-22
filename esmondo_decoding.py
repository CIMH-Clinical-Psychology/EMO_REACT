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
selected_features = {'mean', 'ptp_amp', 'std'}

def extract_time_windows(X, sfreq, window_size, step_size):
    
    # n_epochs, n_channels, n_samples = X.shape
    n_samples = X.shape[-1]
    
    # Define the sliding time windows
    window_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq) - 1
       
    windows = []
    n_windows = 0
    
    for i in range(0, n_samples - window_samples + 1, step_samples):
                
        if i == step_samples:
            start = i + 1
        else:
            start = i   
        end = start + window_samples - 1
        print("Extracting time window within {} and {} sample points".format(start, end))
        
       
        if len(list(range(start,end+1)))==window_samples:
            win_len = window*1000
            print(f"Time window length is {win_len} ms")
        else:
            raise ValueError("start and end sample points might not equal to desired window size")
        
        X_window = X[:, :, start:end]
        X_features = extract_features(X_window, sfreq, selected_features)       
        windows.append(X_features)
        n_windows += 1
        print(f"Extracting time window {n_windows}")

    return np.array(windows)

# Define the time window size and step size in seconds
window = 50/1000  # 50 ms
step = (50/1000)
"""
if the step is equal to the windos size, there will be no overlap between
consecutive windows.
if the step is smaller than window size, we could extract features at higher
temporal resolution (more computational work).
"""

# extract time windows
X_timewindows = extract_time_windows(X, sfreq, window, step)
"""
This returns the segmented X time windows (n_windows, n_epochs, n_features).
"""


#%% New decoding

# Classifier and cross-validation
scaler = StandardScaler()
clf = LogisticRegression()
# pca = PCA(n_components=0.95)
# anova = SelectKBest(f_classif, k=5)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Get the total number of windows
n_timewindows = X_timewindows.shape[0]

# Set up the results storage
results = {
    'accuracy': np.zeros(n_timewindows),
    'recall': np.zeros(n_timewindows),
    'precision': np.zeros(n_timewindows),
    'confmat': np.zeros((n_timewindows, 2, 2))
}



for i, X_window in enumerate(X_timewindows):
    print(f"Processing time window {i+1} of {n_timewindows}")
    
    # Split the data into training and testing indices
    n_trials = X_window.shape[0]
    split_idx = np.random.permutation(n_trials)
    split_point = int(n_trials * 0.8)
    train_indices, test_indices = split_idx[:split_point], split_idx[split_point:]
    
    # Get the training data
    X_train = X_window[train_indices]
    y_train = y_resp[train_indices]
    
    # Get the testing data
    X_test = X_window[test_indices]
    y_test = y_resp[test_indices]
    
    # Get the true label for calculating performance metrics
    y_train_true = y_truth[train_indices]
    y_test_true = y_truth[test_indices]
       
    # Scale the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    
    # Perform classification using stratified cv
    y_pred = cross_val_predict(clf, X_train, y_train, cv=cv)
        
        
    
    # Evaluate performance
    accuracy = accuracy_score(y_train_true, y_pred)
    recall = recall_score(y_train_true, y_pred)
    precision = precision_score(y_train_true, y_pred)
    # cm = sklearn.metrics.confusion_matrix(y_train_true, y_pred)
    
    tp = np.sum((y_pred == 1) & (y_train_true == 1))
    tn = np.sum((y_pred == 0) & (y_train_true == 0))
    fp = np.sum((y_pred == 1) & (y_train_true == 0))
    fn = np.sum((y_pred == 0) & (y_train_true == 1))
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Store the results
    results['accuracy'][i] = accuracy
    results['recall'][i] = recall
    results['precision'][i] = precision
    results['confmat'][i] = cm

stop

# Create a time axis in seconds
time_axis = np.arange(n_timewindows) * window + window / 2

# Plot the results
plt.plot(time_axis, results['accuracy'], label='Accuracy')
plt.plot(time_axis, results['recall'], label='Recall')
plt.plot(time_axis, results['precision'], label='Precision')

# Add labels and legend
plt.xlabel('Time (s)')
plt.ylabel('Score')
plt.legend()

# Show the plot
plt.show()





