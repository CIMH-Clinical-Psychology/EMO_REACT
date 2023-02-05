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
