# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:07:18 2022

Settings file of Sleep-EMO-React study

@author: Simon Kern
"""
import warnings
import ospath
import getpass
import platform
import scipy.io
from enum import IntEnum, unique

def loadmat_single(mat):
    data = scipy.io.loadmat(mat)
    keys = [key for key in data.keys() if not key.startswith('_')]
    assert len(keys)==1, f'more than one key ({keys=}) found in mat {mat}. please have a look'
    return data[keys[0]]

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

VAL_OLD = 0
VAL_NEW = 1
    
    

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
    cache_dir = 'z:/cache/'
    # plot_dir = ospath.expanduser('~/Nextcloud/ZI/2020.1 Pilotstudie/plots/')
    # log_dir = ospath.expanduser('~/Nextcloud/ZI/2020.1 Pilotstudie/plots/logs/')
    # results_dir = ospath.expanduser('~/Nextcloud/ZI/2020.1 Pilotstudie/results/')
    
elif username == 'simon.kern' and host=='zilxap29':
    data_dir = '/data/Emo_React/Raw_data'
    cache_dir = '~/joblib-cache/'
    
else:
    warnings.warn(f'No profile found for {username} @ {host}')
# elif username == 'simon' and host=='laptop-simon':
#     data_dir = 'z:/desmrrest/'
#     cache_dir = 'z:/cache/'
#     plot_dir = 'z:/plots/'
    

montage_file = f'{data_dir}/montages/CACS-74-X1_alle-Positionen.bvef'

