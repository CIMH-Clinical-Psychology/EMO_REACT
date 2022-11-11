# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:07:18 2022

Dummy testing of decoders

@author: Simon Kern
"""
import os
import logging
import ospath
import mne
import sklearn
from scipy.stats import kurtosis
import numpy as np
from joblib import Memory
from mne.preprocessing import ICA, read_ica
from autoreject import AutoReject, compute_thresholds, get_rejection_threshold
from autoreject import read_auto_reject
from joblib import dump, load
import hashlib

import settings

mem = Memory(settings.cache_dir)

def hash_array(arr, dtype=np.int64):
    arr = arr.astype(dtype)
    return hashlib.sha1(arr.flatten('C')).hexdigest()[:8]


@mem.cache
def mne_io_read_raw(vhdr_file, *args, sfreq=None,**kwargs):
    
    ica_fif = vhdr_file + '-ica.fif'
    kwargs['preload'] = True
    raw = mne.io.read_raw_brainvision(vhdr_fname=vhdr_file,
                                      eog=('HEOG', 'VEOG'),
                                      misc=('EMG', 'ECG'),
                                      *args, **kwargs)
    
    assert settings.ref_ch not in raw.ch_names
    raw.add_reference_channels(settings.ref_ch)
    
    montage = mne.channels.read_custom_montage(settings.montage_file)
    
    raw.set_montage(montage)   
    raw.set_eeg_reference()
    
    raw.notch_filter(50, verbose='WARNING')
    raw.filter(0.5, 45, verbose='WARNING')
    
    # resample if required
    if sfreq is not None and not sfreq==np.round(raw.info['sfreq']):
        raw.resample(sfreq)    
    
    picks_eeg = mne.pick_types(raw.info, eeg=True)
    ica_method = 'picard'
    n_components = len(picks_eeg)-1
    
    # filter data with lfreq 1, as recommended by MNE, to remove slow drifts

    # if we have computed this solution already, load it
    if ospath.isfile(ica_fif):
        ica = read_ica(ica_fif)
        assert ica.n_components==n_components, f'n components is not the same, please delete {ica_fif}'
        assert ica.method == ica_method, f'ica method is not the same, please delete {ica_fif}'
    else:
        raw_ica = raw.copy().filter(l_freq=1, h_freq=None, verbose='WARNING')
        ica = ICA(n_components=n_components, method=ica_method, verbose='WARNING',
                  fit_params=dict(ortho=True, extended=True), random_state=140)       
        ica.fit(raw_ica, picks=picks_eeg)
        ica.save(ica_fif, overwrite=True) # save ICA to file for later loading
        
    eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG', verbose=False)
    # emg_indices, emg_scores = ica.find_bads_muscle(raw, verbose=False)
    
    # sources = ica.get_sources(raw)
    # k =dict(zip(sources.ch_names, [kurtosis(sources.get_data(ch).squeeze()) for ch in sources.ch_names]))
    
    exclude = set(eog_indices + ecg_indices + [0,1])
    
    assert len(exclude)<n_components//2, 'more than 50% bad components, sure this is right?'
    logging.info(f'Found bad components via ICA: {exclude}')
    
    raw = ica.apply(raw.copy(), exclude=exclude)

    return raw



def repair_epochs_autoreject(epochs, epochs_file):
    
    if os.path.exists(epochs_file):
        logging.info(f'Loading repaired epochs from {epochs_file}')
        epochs_repaired, idx_removed = load(epochs_file)
        return epochs_repaired, idx_removed
    

    # apply autoreject on this data to automatically repair
    # artefacted data points
    picks_eeg = mne.pick_types(epochs.info, eeg=True)
    logging.info(f'Calculating autoreject pkl solution and saving epochs to {epochs_file}')
    clf = AutoReject(picks=picks_eeg, n_jobs=-1, random_state=0)
    clf.fit(epochs.copy())
    epochs_repaired, reject_log = clf.transform(epochs, return_log=True)
    idx_removed = np.where(reject_log.bad_epochs)[0]
    dump((epochs_repaired, idx_removed), epochs_file)
    return epochs_repaired, idx_removed
    

def load_localizer(folder, sfreq=None, event_id=2, tmin=-0.2, tmax=0.5,
                     return_epochs=False):
    """load the localizer of one participant    

    Parameters
    ----------
    folder : TYPE
        link to participant night.
    sfreq : TYPE, optional
        sampling frequency to downsample to. The default is None.
    event_id : TYPE, optional
        after which event to load the data. The default is 2.

    Returns
    -------
    tuple
        trials , valence, arousal
    """
    files_vhdr = ospath.list_files(folder, exts='vhdr')
    files_mat = ospath.list_files(folder, exts='mat')
    
    vhdr = [file for file in files_vhdr if 'localizer' in file.lower()]
    mat = [file for file in files_mat if file.lower().endswith('localizer.mat')]
    
    assert len(vhdr)==1, f'too many or too few localizer vhdr files: {vhdr=}'
    assert len(mat)==1, f'too many or too few localizer mat files: {mat=}'
    
    raw = mne_io_read_raw(vhdr[0], sfreq=sfreq)
    events, events_dict = mne.events_from_annotations(raw)
    
    resp = settings.loadmat_single(mat[0])
    
    # coding was changed after the first night, so make some exceptions
    idx_valence = 1 if settings.get_night(folder)=='PN04-night1' else 2
    idx_arousal = 4 if settings.get_night(folder)=='PN04-night1' else 5
    
    valence = resp[:, idx_valence].astype(int)-51
    arousal = resp[:, idx_arousal].astype(int)-49
    
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        preload=True, verbose='WARNING')
    
    events_hash = hash_array(epochs.events[:,0])
    epochs_file = f'{raw.filenames[0]}-{tmin}-{tmax}-{events_hash}-epochs.pkl.gz'
    epochs, idx_removed = repair_epochs_autoreject(epochs, epochs_file)
    
    data_x = epochs.get_data()
    valence = np.array([v for i,v in enumerate(valence) if i not in idx_removed])
    arousal = np.array([v for i,v in enumerate(arousal) if i not in idx_removed])

    assert np.all(valence<=2)
    assert np.all(valence>=-2)
    assert np.all(arousal<=4)
    assert np.all(arousal>=0)
    
    assert len(data_x)==len(valence), 'unequal number of trials and valence ratings'
    assert len(data_x)==len(arousal), 'unequal number of trials and arousal ratings'
    if return_epochs:
        return epochs, (valence, arousal)
    return epochs.times, data_x, (valence, arousal)


if __name__=='__main__':
    # set some values for easier debugging
    args = ()
    kwargs={}
    vhdr_file = 'z:/Emo_React/Raw_data/PN4/Experimental night 1/PN4_Emo_React_Exp1_Localizer.vhdr'
    sfreq = 100
    event_id = 2
    tmin=-0.1
    tmax=1.5
