# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:07:18 2022

Settings file of Sleep-EMO-React study

@author: Simon Kern
"""
import re
import warnings
import ospath
import getpass
import platform
import scipy.io
from enum import IntEnum, unique

def get_subj(string, pattern=r"PN[^0-9]*(\d+)/*"):
    res = re.findall(pattern, string.upper() + '/')
    assert len(res)!=0, f'Pattern {pattern} not found in string {string}'
    assert len(res)==1, \
        f'Found more or less pattern matches for {pattern} in {string}'
    assert res[0].isdigit(), '{res} does not seem to be a digit?'
    return f'PN{int(res[0]):02d}'

def get_night(string):
    subj = get_subj(string)
    pattern = r"night[^0-9]*(\d+)/*"
    res = re.findall(pattern, string.lower() + '/')
    assert len(res)!=0, f'Pattern {pattern} not found in string {string}'
    assert len(res)==1, \
        f'Found more or less pattern matches for {pattern} in {string}'
    assert res[0].isdigit(), '{res} does not seem to be a digit?'
    
    
    return f'{subj}-night{res[0]}'


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

VAL_OLD = 0  # participant indicated he has seen the image before
VAL_NEW = 1  # participant indicated that the image is new to him/her
    
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

montage_file = './montage.bvef'
if not os.path.exists(montage_file):
    montage_file = f'{data_dir}/montages/all_combined.bvef'

