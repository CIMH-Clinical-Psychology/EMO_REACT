# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:32:50 2022

Script to check that all trigger values are present
and that we can reconstruct the data as needed.

@author: Simon Kern
"""
import ospath
import mne
import numpy as np
import scipy.io
import settings

def get_counts(arr):
    uniques, counts = np.unique(arr, return_counts=True)
    return dict(zip(uniques, counts))


folders = [f'Z:/Emo_React/Raw_data/Dry run with changes']


logs = {}
for folder in folders:
    log = ['\n'+'#'*20]
    log.append(f'#### Subject: {ospath.basename(folder)} ####')
    files_vhdr = ospath.list_files(folder, exts='vhdr')
    files_mat = ospath.list_files(folder, exts='mat')
    
    #%% Check learning part
    # first check that all files are present
    vhdr = [file for file in files_vhdr if 'learn' in file.lower()]
    mat = [file for file in files_mat if 'arousal_data' in file.lower()]
    
    assert len(vhdr)==1, f'too many or too few learning vhdr files: {vhdr=}'
    assert len(mat)>6, f'too many or too few learning mat files: {mat=}'

    # load learning
    raw = mne.io.read_raw(vhdr[0])
    
    # extract events and make sure there are enough events per category
    events, events_dict = mne.events_from_annotations(raw)
    n_events = len(events)
    events_counts = get_counts(events[:,2])
    events_diff = {e: np.diff(events[:,0])[events[:-1, 2]==e]/raw.info['sfreq'] for e in np.unique(events[:-1, 2])}
    events_diff = {e:v[v<5] for e, v in events_diff.items() if e<100}
    events_length = {e:f'{np.mean(v):.3f}+-{np.std(v):.3f}' for e,v in events_diff.items()}
    assert n_events>1930, f'number of events in learning was not 1934 {n_events=}'
    
    resp = []
    for _matfile in mat:
        content = settings.loadmat_single(_matfile)
        resp.append(content[:,:3])      
    resp = np.vstack(resp)   
    
    resp_quadrant = get_counts(resp[:, 0].astype(int))
    resp_c2 = get_counts(resp[:, 1].astype(int))
    resp_image = get_counts(resp[:, 2].astype(int))
    imgs_learning = set(resp[:,2])
    assert set(resp_image.values())==set([16]), 'not all images are shown equally often'

    log.append(f'\nLearning file is {ospath.basename(vhdr[0])}')
    log.append(f'Localizer mats are {[ospath.basename(x) for x in mat]}')
    log.append(f'\tlength = {raw.times[-1]/60:.1f} minutes')
    log.append(f'\t{n_events = }')
    log.append(f'\t{events_counts = }')
    log.append(f'\t{events_length = }')
    log.append(f'\t{resp_quadrant = }')
    log.append(f'\t{resp_c2 = }')
    log.append(f'\t{resp_image = }')

 
    #%% Check localizer part
    # first check that all files are present
    vhdr = [file for file in files_vhdr if 'localizer' in file.lower()]
    mat = [file for file in files_mat if 'localizer' in file.lower()]
    
    assert len(vhdr)==1, f'too many or too few localizer vhdr files: {vhdr=}'
    assert len(mat)==1, f'too many or too few localizer mat files: {mat=}'

    # load localizer
    raw = mne.io.read_raw(vhdr[0])
    
    # extract events and make sure there are enough events per category
    events, events_dict = mne.events_from_annotations(raw)
    n_events = len(events)
    events_counts = get_counts(events[:,2])
    events_diff = {e: np.diff(events[:,0])[events[:-1, 2]==e]/raw.info['sfreq'] for e in np.unique(events[:-1, 2])}
    events_diff = {e:v[v<5] for e, v in events_diff.items() if e<100}
    events_length = {e:f'{np.mean(v):.3f}+-{np.std(v):.3f}' for e,v in events_diff.items()}
    
    
    assert n_events==387, f'number of events in localizer was not 387 {n_events=}'
    
    resp = settings.loadmat_single(mat[0])
    resp_valence = get_counts(resp[:, 1].astype(int))
    resp_intensity = get_counts(resp[:, 4].astype(int))
    imgs_localizer = set(resp[:,0])
    assert len(set(resp[:,0]))==len(resp), "not all localizer images were unique"

    log.append(f'\nLocalizer vhdr is {ospath.basename(vhdr[0])}')
    log.append(f'Localizer mat is {ospath.basename(mat[0])}')
    log.append(f'\tlength = {raw.times[-1]/60:.1f} minutes')
    log.append(f'\t{n_events = }')
    log.append(f'\t{events_counts = }')
    log.append(f'\t{events_length = }')
    log.append(f'\t{resp_valence = }')
    log.append(f'\t{resp_intensity = }')
    
    #%% Check sleep part
    # first check that all files are present
    vhdr = [file for file in files_vhdr if 'sleep' in file.lower() and not 'test' in file.lower()]
    
    assert len(vhdr)==1, f'too many or too few sleep vhdr files: {vhdr=}'

    # load localizer
    raw = mne.io.read_raw(vhdr[0])
    
    # extract events and make sure there are enough events per category
    events, events_dict = mne.events_from_annotations(raw)
    n_events = len(events)
    events_counts = get_counts(events[:,2])
    events_diff = {e: np.diff(events[:,0])[events[:-1, 2]==e]/raw.info['sfreq'] for e in np.unique(events[:-1, 2])}
    events_diff = {e:v[v<5] for e, v in events_diff.items()}
    events_length = {e:f'{np.mean(v):.3f}+-{np.std(v):.3f}' for e,v in events_diff.items()}
    
    assert n_events==11, f'number of events in localizer was not 11 {n_events=}'
    assert 10005 in events[:,2], 'Lights on marker not found'
    assert 10006 in events[:,2], 'Lights out marker not found'

    log.append(f'\nSleep file is {ospath.basename(vhdr[0])}')
    log.append(f'\tlength = {raw.times[-1]/60:.1f} minutes')
    log.append(f'\t{n_events = }')
    log.append(f'\t{events_length = }')
    log.append(f'\t{events_counts = }')

    #%% Check before-sleep-test
    # first check that all files are present
    vhdr = [file for file in files_vhdr if 'test_before_sleep' in file.lower()]
    mat = [file for file in files_mat if 'BS' in file]
    
    assert len(vhdr)==1, f'too many or too few localizer vhdr files: {vhdr=}'
    assert len(mat)==1, f'too many or too few localizer mat files: {mat=}'

    # load localizer
    raw = mne.io.read_raw(vhdr[0])
    
    # extract events and make sure there are enough events per category
    events, events_dict = mne.events_from_annotations(raw)
    n_events = len(events)
    events_counts = get_counts(events[:,2])
    events_diff = {e: np.diff(events[:,0])[events[:-1, 2]==e]/raw.info['sfreq'] for e in np.unique(events[:-1, 2])}
    events_diff = {e:v[v<5] for e, v in events_diff.items() if e<100}
    events_length = {e:f'{np.mean(v):.3f}+-{np.std(v):.3f}' for e,v in events_diff.items()}
    
    assert n_events>=527, f'number of events in post test was not 527 {n_events=}'
    
    resp = settings.loadmat_single(mat[0])
    resp_oldnew = get_counts(resp[:, 2])
    resp_quadrants = get_counts(resp[:, 5])
    
    imgs_bs = set(resp[:,0])
    assert resp_quadrants[0]==resp_oldnew[98], 'more votings for old images than empty quadrant votes'

    log.append(f'\n Test file is {ospath.basename(vhdr[0])}')
    log.append(f'Test mat is {ospath.basename(mat[0])}')
    log.append(f'\tlength = {raw.times[-1]/60:.1f} minutes')
    log.append(f'\t{n_events = }')
    log.append(f'\t{events_length = }')
    log.append(f'\t{events_counts = }')
    log.append(f'\t{resp_oldnew = }')
    log.append(f'\t{resp_quadrants = }')
    log.append(f'\t{resp_oldnew[97]} times selected "old image"')
    log.append(f'\t{resp_oldnew[98]} times selected "new image"')
    
    #%% Check after-sleep-test
    # first check that all files are present
    vhdr = [file for file in files_vhdr if 'test_after_sleep' in file.lower()]
    mat = [file for file in files_mat if 'AS' in file]
    
    assert len(vhdr)==1, f'too many or too few localizer vhdr files: {vhdr=}'
    assert len(mat)==1, f'too many or too few localizer mat files: {mat=}'

    # load localizer
    raw = mne.io.read_raw(vhdr[0])
    
    # extract events and make sure there are enough events per category
    events, events_dict = mne.events_from_annotations(raw)
    n_events = len(events)
    events_counts = get_counts(events[:,2])
    events_diff = {e: np.diff(events[:,0])[events[:-1, 2]==e]/raw.info['sfreq'] for e in np.unique(events[:-1, 2])}
    events_diff = {e:v[v<5] for e, v in events_diff.items() if e<100}
    events_length = {e:f'{np.mean(v):.3f}+-{np.std(v):.3f}' for e,v in events_diff.items()}
    
    assert n_events>=527, f'number of events in post test was not 527 {n_events=}'
    
    resp = settings.loadmat_single(mat[0])
    resp_oldnew = get_counts(resp[:, 2])
    resp_quadrants = get_counts(resp[:, 5])
    imgs_as = set(resp[:,0])
    assert resp_quadrants[0]==resp_oldnew[98], 'more votings for old images than empty quadrant votes'

    log.append(f'\n Test file is {ospath.basename(vhdr[0])}')
    log.append(f'Test mat is {ospath.basename(mat[0])}')
    log.append(f'\tlength = {raw.times[-1]/60:.1f} minutes')
    log.append(f'\t{n_events = }')
    log.append(f'\t{events_length = }')
    log.append(f'\t{events_counts = }')
    log.append(f'\t{resp_oldnew = }')
    log.append(f'\t{resp_quadrants = }')
    log.append(f'\t{resp_oldnew[97]} times selected "old image"')
    log.append(f'\t{resp_oldnew[98]} times selected "new image"')

    print('\n\n')
    print('\n'.join(log))
    logs[folder] = '\n'.join(log)