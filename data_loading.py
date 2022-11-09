# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:07:18 2022

Dummy testing of decoders

@author: Simon Kern
"""
import ospath
import mne
import sklearn
import numpy as np
from joblib import Memory
from mne.preprocessing import ICA, read_ica

import settings

mem = Memory(settings.cache_dir)

@mem.cache
def mne_io_read_raw(vhdr_file, *args, sfreq=None,**kwargs):
    
    ica_fif = vhdr_file + '-ica.fif'
    kwargs['preload'] = True
    raw = mne.io.read_raw_brainvision(vhdr_fname=vhdr_file,
                                      eog=('HEOG', 'VEOG'),
                                      misc=('EMG', 'ECG'),
                                      *args, **kwargs)
    montage = mne.channels.read_custom_montage(settings.montage_file)
    raw.set_montage(montage)
    
    picks_eeg = mne.pick_types(raw.info, eeg=True)
    ica_method = 'picard'
    n_components = len(picks_eeg)
    
    # filter data with lfreq 1, as recommended by MNE, to remove slow drifts
    raw_ica = raw.copy().filter(l_freq=1, h_freq=40, verbose='WARNING')

    # if we have computed this solution already, load it
    if ospath.isfile(ica_fif):
        ica = read_ica(ica_fif)
        assert ica.n_components==n_components, f'n components is not the same, please delete {ica_fif}'
        assert ica.method == ica_method, f'ica method is not the same, please delete {ica_fif}'
    else:
        ica = ICA(n_components=n_components, method=ica_method)
        picks_eeg = mne.pick_types(raw.info, eeg=True)
        ica.fit(raw_ica, picks=picks_eeg)
        ica.save(ica_fif) # save ICA to file for later loading
        
        
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw_ica, ch_name='ECG')
    emg_indices, emg_scores = ica.find_bads_muscle(raw_ica)
    eog_indices, eog_scores = ica.find_bads_eog(raw_ica)
    
    exclude = ecg_indices + emg_indices + eog_indices
    
    assert len(exclude)<30, 'more than 40% bad components, sure this is right?'
    
    raw = ica.apply(raw, exclude=exclude)
    
    # resample if required
    if sfreq is not None and not sfreq==np.round(raw.info['sfreq']):
        raw.resample(sfreq)    
        
    return raw


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
    mat = [file for file in files_mat if 'localizer' in file.lower()]
    
    assert len(vhdr)==1, f'too many or too few localizer vhdr files: {vhdr=}'
    assert len(mat)==1, f'too many or too few localizer mat files: {mat=}'
    
    raw = mne_io_read_raw(vhdr[0], sfreq=sfreq)
    events, events_dict = mne.events_from_annotations(raw)
    
    resp = settings.loadmat_single(mat[0])
    valence = resp[:, 1].astype(int)-51
    arousal = resp[:, 4].astype(int)-49
    
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax)
    data_x = epochs.get_data()

    assert np.all(arousal<=4)
    assert np.all(arousal>=0)
    assert np.all(valence<=2)
    assert np.all(valence>=-2)
    
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
