# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:07:18 2022

Settings file of Sleep-EMO-React study

@author: Simon Kern
"""
import os
import ospath
import getpass
import platform
from enum import IntEnum, unique

def clear(locals, remove):
    """clear a set of variables from local namespace"""
    for var in remove:
        if var in locals:
            del locals[var]


@unique
class trigger(IntEnum):
    """Trigger values that are present in the recordings"""
    START = 1
    STIMULUS = 2
    VALENCE = 3
    INTENSITY = 4
    FIXATION = 5
    YESNO = 6
    QUADRANT = 7


@unique
class image_categories(IntEnum):
    """Trigger values that are present in the recordings"""
    Object = 0
    Scene = 1
    Animal = 2
    Person = 3

split_arousal = 3.73  # above this value, arousal was defined as "high", below as "low"
split_valence = 4.61  # above this value, high arousal images were defined as "positive", below "negative"

rating_map = {49:0,   # Very Happy  |  Very Arousing
              50:1,
              51:2,
              52:3,
              53:4,   # Very Sad    |  Not arousing at all
              54:4}   # weirdly enough this value appeared once

seen_before_map = {97: True,   # i have seen this image before
                   98: False}  # have not seen this image before

quad_map = {97: 4,    # lower left
            99: 3,    # lower right
            103: 1,   # upper left
            105: 2,   # upper right
            0: None}

stage_map = {1: 'S1', 2:'S2', 3:'SWS', 4:'REM', 0:'Wake'}

ref_ch = 'FCz'  # reference channel used in the sleep lab
ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
            'FC6', 'T7', 'C3', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6',
            'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2',
            'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9',
            'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7',
            'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3',
            'POz', 'PO4', 'PO8', 'ECG', 'HEOG', 'VEOG', 'EMG', 'FCz']

use_caching = False


username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = ospath.expanduser('~')



    # pass
cache_dir = None
data_dir = None

if username == 'simon' and host=='desktop-simon':
    data_dir = 'z:/Emo_React/'
    cache_dir = 'z:/cache/emo-react/'
    # plot_dir = ospath.expanduser('~/Nextcloud/ZI/2020.1 Pilotstudie/plots/')
    # log_dir = ospath.expanduser('~/Nextcloud/ZI/2020.1 Pilotstudie/plots/logs/')
    # results_dir = ospath.expanduser('~/Nextcloud/ZI/2020.1 Pilotstudie/results/')

elif username == 'simon.kern' and host=='zilxap29':
    data_dir = '/data/Emo_React/'
    cache_dir = f'{home}/joblib-cache/emo-react/'

elif username == 'ardiansyah.esmondo' and host=='zilxap29':
    data_dir = '/hobbes/Klips/Esmondo/Emo_React/'
    cache_dir = f'{home}/joblib-cache/emo-react/'

elif username == 'simon' and host=='simon-kubuntu':
    data_dir = '/data/Emo_React/'
    cache_dir = f'{home}/joblib-cache/emo-react/'


else:
    raise Exception(f'No profile found for {username} @ {host}')
# elif username == 'simon' and host=='laptop-simon':
#     data_dir = 'z:/desmrrest/'
#     cache_dir = 'z:/cache/'
#     plot_dir = 'z:/plots/'


montage_file = f'{data_dir}/montages/montage.bvef'
if not os.path.exists(montage_file):
    montage_file = './resources/montage.bvef'
if not os.path.exists(montage_file):
    montage_file = '../resources/montage.bvef'

if not os.path.exists(montage_file):
    raise FileNotFoundError('montage file `montage.bvef` not found')

os.makedirs(cache_dir, exist_ok=True)