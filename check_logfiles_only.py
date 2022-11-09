# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:18:36 2022

@author: Simon
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:32:50 2022

Script to check that the logfiles contain everything we need

@author: Simon Kern
"""
import deepdiff
import ospath
import mne
import numpy as np
import scipy.io
import settings
from settings import VAL_OLD, VAL_NEW


def get_counts(arr):
    uniques, counts = np.unique(arr, return_counts=True)
    return dict(zip(uniques, counts))

def get_img_mapping(mat_file):
    content = settings.loadmat_single(mat_file)
    imgs = {}
    resp_info = {}
    for [name], quad, oldnew, *rest in content:
        name = name[:-4]
        assert name not in imgs
        imgs[name] = {'quad':int(quad[0]), 
                         'valence': int(oldnew[0])}
        if len(rest)>0:
            resp_info[name[0]] = {'valence': int(rest[1]),
                                  'arousal': int(rest[4])}
    return imgs, resp_info

def get_localizer_mapping(mat_file):
    content = settings.loadmat_single(mat_file)
    resp_info = {}
    for [name], _t1, valence, _t2, _t3, arousal, _t4 in content:
        name = name[:-4]
        assert name not in resp_info
        resp_info[name] = {'valence': int(valence),
                              'arousal': int(arousal)}
    return resp_info

def get_test_mapping(mat_file):
    content = settings.loadmat_single(mat_file)
    resp_info = {}
    for [name], _t1, oldnew_resp, _t2, _t3, quad_resp, _t4 in content:
        assert ('_old_' in name) or ('_new_' in name)
        oldnew = 0 if '_old_' in name else 1
        valence = name[-5:-4]
        assert name not in resp_info
        resp_info[name[:-10]] = {'oldnew': oldnew,
                           'oldnew_resp': int(oldnew_resp),
                           'valence': int(valence),
                           'quad_resp': int(quad_resp)}
    return resp_info


folders = [f'Z:/Emo_React/Raw_data/Final dry run']


logs = {}
for folder in folders:
    log = ['\n'+'#'*20]
    log.append(f'#### Subject: {ospath.basename(folder)} ####')
    files_mat = [f for f in ospath.list_files(folder, exts='mat') if not 'seed' in f]
    
    #%% Check learning part
    # first check that all files are present
    mat = [file for file in files_mat if 'arousal_data' in file.lower()]
  
    assert len(mat)==8, f'too many or too few learning mat files: {mat=}'

    # extract events and make sure there are enough events per category
    mapping_learning = []
    resp_learning = []
    for _matfile in mat:
        mapping, resp = get_img_mapping(_matfile)
        assert len(mapping)==96
        mapping_learning.append(mapping)
        if len(resp)>0:
            resp_learning.append(resp)
        
    assert not any([(list(mapping_learning[0].keys())==list(r.keys())) for r in mapping_learning[1:]]), 'order was the same in learning block'
    assert all([deepdiff.DeepDiff(mapping_learning[0], r)=={} for r in mapping_learning]), 'different images/quadrants/valence shown during learning'
    
    resp_quadrant = get_counts([d['quad'] for d in mapping.values()])
    resp_valence = get_counts([d['valence'] for d in mapping.values()])
    
    log.append(f'Learning mats are {[ospath.basename(x) for x in mat]}')
    log.append(f'\t{resp_quadrant = }')
    log.append(f'\t{resp_valence = }')

    
    #%% Check localizer part
    # first check that all files are present
    mat = [file for file in files_mat if 'localizer.mat' in file.lower()]
    assert len(mat)==1, f'too many or too few localizer mat files: {mat=}'
   
    resp_localizer = get_localizer_mapping(mat[0])
    resp_valence = get_counts([d['valence'] for d in resp_localizer.values()])
    resp_arousal = get_counts([d['arousal'] for d in resp_localizer.values()])
    
    assert len(set(resp_localizer) | set(mapping))==192, 'overlap between localizer and learning'
    
    log.append(f'Localizer mat is {ospath.basename(mat[0])}')
    log.append(f'\t{resp_valence = }')
    log.append(f'\t{resp_arousal = }')
    
    #%% Check before-sleep-test
    # first check that all files are present
    mat = [file for file in files_mat if 'BS' in file]
    assert len(mat)==1, f'too many or too few localizer mat files: {mat=}'

    resp_BS = get_test_mapping(mat[0])
    resp_oldnew = get_counts([d['oldnew'] for d in resp_BS.values()])
    resp_oldnew_resp = get_counts([d['oldnew_resp'] for d in resp_BS.values()])
    resp_quad_resp = get_counts([d['quad_resp'] for d in resp_BS.values()])
    
    assert resp_oldnew[0]==96
    assert resp_oldnew[1]==48
    
    for img, val in resp_BS.items():
        if val['oldnew']==VAL_OLD:
            assert img in mapping, 'old picture was not seen during learning'
        elif val['oldnew']==VAL_NEW:
            assert img not in mapping, 'new picture was seen during learning'
        else:
            raise Exception('something weird happened')

    assert len(set(resp_BS) | set(mapping))==144, 'overlap between testing and learning'
    assert len(set(resp_BS) - set(mapping))==48, 'overlap between testing and learning'
    assert len(set(mapping) - set(resp_BS))==0, 'some learning images not shown during testing'
    assert len(set(resp_localizer)- set(resp_BS))==96, 'localizer images shown during testing'
    assert resp_oldnew_resp[98] == resp_quad_resp[0], 'more votings for old images than empty quadrant votes'

    log.append(f'BS Test mat is {ospath.basename(mat[0])}')
    log.append(f'\t{resp_oldnew_resp = }')
    log.append(f'\t{resp_quad_resp = }')
    log.append(f'\t{resp_oldnew_resp[97]} times selected "old image"')
    log.append(f'\t{resp_oldnew_resp[98]} times selected "new image"')
    
    #%% Check after-sleep-test
    # first check that all files are present
    mat = [file for file in files_mat if 'AS' in file]
    assert len(mat)==1, f'too many or too few localizer mat files: {mat=}'

    resp_AS = get_test_mapping(mat[0])
    resp_oldnew = get_counts([d['oldnew'] for d in resp_AS.values()])
    resp_oldnew_resp = get_counts([d['oldnew_resp'] for d in resp_AS.values()])
    resp_quad_resp = get_counts([d['quad_resp'] for d in resp_AS.values()])
    
    assert resp_oldnew[0]==96
    assert resp_oldnew[1]==48
    
    for img, val in resp_AS.items():
        if val['oldnew']==VAL_OLD:
            assert img in mapping, 'old picture was not seen during learning'
        elif val['oldnew']==VAL_NEW:
            assert img not in mapping, 'new picture was seen during learning'
        else:
            raise Exception('something weird happened')

    assert len(set(resp_AS) | set(mapping))==144, 'overlap between testing and learning'
    assert len(set(resp_AS) - set(mapping))==48, 'overlap between testing and learning'
    assert len(set(mapping) - set(resp_AS))==0, 'some learning images not shown during testing'
    assert len(set(resp_localizer)- set(resp_AS))==96, 'localizer images shown during testing'
    assert resp_oldnew_resp[98] == resp_quad_resp[0], 'more votings for old images than empty quadrant votes'

    log.append(f'AS Test mat is {ospath.basename(mat[0])}')
    log.append(f'\t{resp_oldnew_resp = }')
    log.append(f'\t{resp_quad_resp = }')
    log.append(f'\t{resp_oldnew_resp[97]} times selected "old image"')
    log.append(f'\t{resp_oldnew_resp[98]} times selected "new image"')
    log.append('\n##### All checks passed')
    print('\n\n')
    print('\n'.join(log))
    logs[folder] = '\n'.join(log)