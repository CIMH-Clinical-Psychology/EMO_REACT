# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:32:14 2022

This module contains functions for preprocessing EEG data and extracting
features from epochs

@author: Simon
"""
import yasa
import mne
# import braindecode
import numpy as np
from scipy.signal import welch
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA

def get_bands(data_x, wlen=0.5, tstep=0.25, sfreq=100, n_pca=30,
              **kwargs):
    
    nperseg = int(wlen*sfreq)
    power = []
   
    for trial in data_x:
        times, windows = yasa.sliding_window(trial, 100, wlen, tstep)
        freqs, psd = welch(windows, sfreq, nperseg=nperseg, axis=-1)
        # bandpower has shape (bands,windows, channels)
        bandpower = yasa.bandpower_from_psd_ndarray(psd, freqs, **kwargs)
        power.append(bandpower)
        
    # power is now of shape (trials, bands, windows, channels)
    power = np.array(power)
    
    power = power.swapaxes(-1, -2)
    power = power.reshape([power.shape[0], -1, power.shape[-1]])
    if n_pca>0:
        pca = PCA(n_pca)
        power = np.array([pca.fit_transform(X.T) for X in power.T])
        power = np.swapaxes(power.T, 0, 1)
        assert power.shape==(data_x.shape[0], n_pca, len(windows))

    return times, power