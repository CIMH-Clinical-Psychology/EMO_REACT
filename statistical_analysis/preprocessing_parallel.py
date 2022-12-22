#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 08:53:58 2022

This file preprocesses all data parallel, so it's available in the cache

@author: simon.kern
"""
import sys; sys.path.append('..')
import ospath
import settings
from data_loading import load_sleep, load_localizer, get_subj
from tqdm import tqdm
from joblib import Parallel, delayed

# check which data is loadable
folders_subj = ospath.list_folders(f'{settings.data_dir}/Raw_data/', pattern='PN*')
folders_nights = []  # store all night folders for which we have two in here

for folder_subj in tqdm(folders_subj, desc='loading participant responses'):
    subj = get_subj(folder_subj)
    nights_folders = ospath.list_folders(folder_subj, pattern='*night*')
    if len(nights_folders)<2:
        print(f'{len(nights_folders)} night(s) for {subj}, skipping')
        continue
    
    # for each subject, load the individual nights
    for folder_night in nights_folders:
        folders_nights.append(folder_night)

pool = Parallel(2)
pool(delayed(load_sleep)(f) for f in tqdm(folders_nights, desc=' loads sleep'))
pool(delayed(load_localizer)(f) for f in tqdm(folders_nights, desc='load localizer'))